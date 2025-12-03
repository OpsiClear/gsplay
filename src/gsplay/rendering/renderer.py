"""
Rendering and autoplay loop for the Universal GSPlay.

This module handles the main render callback and autoplay loop logic.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING
from collections.abc import Callable

import numpy as np
import torch
from gsplat.rendering import rasterization as _rasterization

from src.domain.entities import SH_DEGREE_MAP
from src.infrastructure.processing_mode import ProcessingMode
from src.gsplay.nerfview import CameraState, RenderTabState
from src.gsplay.interaction.events import EventBus, EventType

if TYPE_CHECKING:
    from src.domain.entities import GSData, GSTensor
    from src.domain.interfaces import ModelInterface
    from src.gsplay.config.settings import UIHandles, GSPlayConfig

logger = logging.getLogger(__name__)


def create_render_function(
    model: ModelInterface,
    ui: UIHandles,
    device: str,
    apply_edits_fn: Callable[[GSData | GSTensor], GSTensor],
    config: "GSPlayConfig" = None,
    event_bus: EventBus | None = None,
) -> Callable:
    """
    Create the main render callback function for nerfview.

    Parameters
    ----------
    model : ModelInterface
        Model to render from
    ui : UIHandles
        UI handles for accessing slider values
    device : str
        Torch device
    apply_edits_fn : Callable
        Function to apply edits to gaussian data
    config : GSPlayConfig
        GSPlay configuration
    event_bus : EventBus | None
        Event bus for emitting render stats

    Returns
    -------
    Callable
        Render function with signature (camera_state, render_tab_state) -> np.ndarray
    """

    # CUDA stream for this render function instance (provides isolation between instances)
    _cuda_stream = torch.cuda.Stream() if device.startswith("cuda") else None

    @torch.no_grad()
    def render_fn(
        camera_state: CameraState, render_tab_state: RenderTabState
    ) -> np.ndarray:
        """
        Main render callback function.

        Called by nerfview for each frame render.

        Parameters
        ----------
        camera_state : CameraState
            Camera parameters
        render_tab_state : RenderTabState
            Render settings and state

        Returns
        -------
        np.ndarray
            Rendered image [H, W, 3] in range [0, 1]
        """
        # Allow modification of the outer CUDA stream in error recovery
        nonlocal _cuda_stream
        
        # Performance profiling
        _frame_start = time.perf_counter()
        _t_load = _t_render = 0.0

        # Get dimensions
        W, H = render_tab_state.viewer_width, render_tab_state.viewer_height

        if model is None:
            return np.zeros((H, W, 3), dtype=np.float32)

        # Get selected quality (default to 960 if not set)
        max_res = int(ui.render_quality.value) if ui.render_quality else 1280

        # Apply resolution limit (only scale if difference is significant)
        # Avoid upscaling overhead when window size is close to target quality
        if W > max_res or H > max_res:
            scale_factor = min(max_res / W, max_res / H)
            W_scaled, H_scaled = int(W * scale_factor), int(H * scale_factor)
        elif abs(W - max_res) / max(W, 1) > 0.15 or abs(H - max_res) / max(H, 1) > 0.15:
            # Window is significantly smaller than target - scale up to target
            scale_factor = min(max_res / W, max_res / H)
            W_scaled, H_scaled = int(W * scale_factor), int(H * scale_factor)
        else:
            # Window size is close to target - render at native resolution (avoid upscale overhead)
            W_scaled, H_scaled = W, H

        # Determine the current time to render (normal interactive mode)
        frame_index = ui.time_slider.value if ui.time_slider else 0
        total_frames = ui.time_slider.max + 1 if ui.time_slider else 1
        normalized_time = (
            frame_index / max(1, total_frames - 1) if total_frames > 1 else 0.0
        )

        # Get gaussians from model
        _checkpoint_load = time.perf_counter()

        # Apply volume filter processing mode if configured
        if (
            config
            and hasattr(config, "volume_filter")
            and hasattr(model, "processing_mode")
        ):
            try:
                edit_mode = ProcessingMode.from_string(
                    config.volume_filter.processing_mode
                )
                model.processing_mode = edit_mode.loader_mode
            except ValueError:
                logger.warning(
                    "Invalid processing mode '%s' on volume filter; defaulting loader to ALL_GPU",
                    getattr(config.volume_filter, "processing_mode", "unknown"),
                )
                model.processing_mode = ProcessingMode.ALL_GPU.value

        gaussians = model.get_gaussians_at_normalized_time(normalized_time)

        _t_load = (time.perf_counter() - _checkpoint_load) * 1000  # ms

        if gaussians is None or gaussians.means.shape[0] == 0:
            logger.warning("No gaussians available for rendering")
            return np.zeros((H, W, 3), dtype=np.float32)

        # Apply edits
        _checkpoint_edits = time.perf_counter()
        render_gaussians = apply_edits_fn(gaussians)
        _t_edits = (time.perf_counter() - _checkpoint_edits) * 1000  # ms

        # Emit render stats instead of updating UI directly
        if event_bus:
            # Get actual loaded frame filename and index from model
            frame_filename = "N/A"
            frame_index_str = "N/A"

            if hasattr(model, "_last_loaded_filename") and model._last_loaded_filename:
                # For PLY models - use tracked filename
                frame_filename = model._last_loaded_filename
                if (
                    hasattr(model, "_last_loaded_frame_index")
                    and model._last_loaded_frame_index is not None
                ):
                    frame_index_str = str(model._last_loaded_frame_index)
            elif hasattr(model, "get_current_frame_info"):
                # For other models that implement this method
                frame_filename = model.get_current_frame_info()

            gaussian_count = render_gaussians.means.shape[0] if render_gaussians else 0

            throughput = 0.0
            if hasattr(model, "get_frame_throughput_fps"):
                throughput = model.get_frame_throughput_fps()

            event_bus.emit(
                EventType.RENDER_STATS_UPDATED,
                source="render_function",
                frame_index=frame_index_str,
                frame_filename=frame_filename,
                gaussian_count=gaussian_count,
                throughput_fps=throughput,
            )

        # Check if edits removed all gaussians
        if render_gaussians is None or render_gaussians.means.shape[0] == 0:
            logger.debug("All gaussians filtered out after edits")
            return np.zeros((H, W, 3), dtype=np.float32)

        # Render
        _checkpoint_render = time.perf_counter()
        try:
            # Camera setup (batched transfer for 2x speedup - eliminates 1 kernel launch)
            _checkpoint_camera = time.perf_counter()
            # Batch camera matrices into single GPU transfer
            camera_params = np.concatenate(
                [
                    camera_state.c2w.flatten(),  # 16 values (4x4 matrix)
                    camera_state.get_K(
                        (W_scaled, H_scaled)
                    ).flatten(),  # 9 values (3x3 matrix)
                ],
                axis=0,
            )
            # Use dedicated stream for camera transfer to isolate from nvImageCodec
            # which runs on the default stream. This prevents CUDA errors from
            # nvImageCodec (e.g., during JPEG encoding failure) from corrupting
            # our render operations.
            if _cuda_stream is not None:
                with torch.cuda.stream(_cuda_stream):
                    camera_params_gpu = (
                        torch.from_numpy(camera_params).float().to(device, non_blocking=True)
                    )
                # Sync to ensure camera params are on GPU before use
                _cuda_stream.synchronize()
            else:
                camera_params_gpu = (
                    torch.from_numpy(camera_params).float().to(device)
                )

            # Unpack on GPU (creates views, no data copy)
            c2w = camera_params_gpu[:16].reshape(4, 4).contiguous()
            K = camera_params_gpu[16:25].reshape(3, 3).contiguous()
            _t_camera = (time.perf_counter() - _checkpoint_camera) * 1000

            # Prepare render arguments
            _checkpoint_prep = time.perf_counter()
            render_kwargs = {
                "means": render_gaussians.means,
                "quats": render_gaussians.quats,
                "scales": render_gaussians.scales,
                "opacities": render_gaussians.opacities,
                "viewmats": torch.linalg.inv(c2w[None]),
                "Ks": K[None],
                "width": W_scaled,
                "height": H_scaled,
            }

            # Use SH coefficients if available, otherwise RGB colors
            sh_degree = None
            if getattr(render_gaussians, "shN", None) is not None:
                sh_degree = SH_DEGREE_MAP.get(render_gaussians.shN.shape[1])

            if sh_degree is not None:
                # gsplat expects ALL SH coefficients (DC + higher-order) when sh_degree is specified
                # sh0: [N, 3] -> [N, 1, 3] (DC component)
                # shN: [N, K, 3] (higher-order components)
                # Combined: [N, K+1, 3] where K+1 = (sh_degree+1)^2
                sh0_expanded = render_gaussians.sh0.unsqueeze(1)  # [N, 3] -> [N, 1, 3]
                colors_all = torch.cat([sh0_expanded, render_gaussians.shN], dim=1)  # [N, K+1, 3]
                render_kwargs["colors"] = colors_all
                render_kwargs["sh_degree"] = sh_degree
            else:
                render_kwargs["colors"] = render_gaussians.sh0
            _t_prep = (time.perf_counter() - _checkpoint_prep) * 1000

            # Actual rasterization - use dedicated stream to isolate from other instances
            _checkpoint_rasterize = time.perf_counter()
            if _cuda_stream is not None:
                with torch.cuda.stream(_cuda_stream):
                    render_colors, render_alphas, _ = _rasterization(**render_kwargs)
                # Sync before accessing results to ensure rasterization is complete
                _cuda_stream.synchronize()
            else:
                render_colors, render_alphas, _ = _rasterization(**render_kwargs)
            
            # Note: Stream sync above already ensures rasterization is complete.
            # No additional global sync needed here - using stream-based isolation
            # allows multiple instances to run concurrently without blocking each other.
            
            # Normalize output layout to channels-last for downstream math
            # gsplat outputs: render_colors [N, H, W, C], render_alphas [N, H, W, 1]
            if render_colors.dim() == 4 and render_colors.shape[-1] not in (3, 4) and render_colors.shape[1] in (3, 4):
                # NCHW format detected - permute to NHWC
                render_colors = render_colors.permute(0, 2, 3, 1).contiguous()
                if render_alphas.dim() == 4:
                    render_alphas = render_alphas.permute(0, 2, 3, 1).contiguous()
                elif render_alphas.dim() == 3:
                    render_alphas = render_alphas.unsqueeze(-1).contiguous()
            elif render_colors.dim() == 4 and render_colors.shape[-1] in (3, 4):
                # Already NHWC format - ensure contiguity
                render_colors = render_colors.contiguous()
                if render_alphas.dim() == 4:
                    # Always make alpha contiguous regardless of shape[-1]
                    render_alphas = render_alphas.contiguous()
                elif render_alphas.dim() == 3:
                    render_alphas = render_alphas.unsqueeze(-1).contiguous()
            else:
                # Unexpected format - log warning and ensure contiguity
                logger.warning(
                    f"Unexpected render output format: colors={render_colors.shape}, "
                    f"alphas={render_alphas.shape}"
                )
                render_colors = render_colors.contiguous()
                render_alphas = render_alphas.contiguous()
            _t_rasterize = (time.perf_counter() - _checkpoint_rasterize) * 1000

            # Post-processing: GPU upscaling + async transfer
            _checkpoint_post = time.perf_counter()

            # Use alpha channel to mask uncovered pixels to black
            # render_alphas is [N, H, W, 1], so render_alphas[0] is [H, W, 1]
            # Composite: rgb * alpha (uncovered pixels become black)
            # Single clamp after multiplication - avoids redundant kernel launch
            alpha_mask = render_alphas[0]
            result_gpu = torch.clamp(render_colors[0] * alpha_mask, 0.0, 1.0)

            # Validate output - check for NaN/Inf which can corrupt JPEG encoding
            if torch.isnan(result_gpu).any() or torch.isinf(result_gpu).any():
                logger.warning(
                    "NaN/Inf detected in render output - replacing with zeros. "
                    f"Colors has_nan={torch.isnan(render_colors).any().item()}, "
                    f"Alpha has_nan={torch.isnan(render_alphas).any().item()}"
                )
                result_gpu = torch.nan_to_num(result_gpu, nan=0.0, posinf=1.0, neginf=0.0)

            # GPU-based upscaling (if needed) - faster than CPU
            _checkpoint_upscale = time.perf_counter()
            if W_scaled != W or H_scaled != H:
                # Upscale on GPU using torch interpolation
                result_gpu = (
                    torch.nn.functional.interpolate(
                        result_gpu.permute(2, 0, 1).unsqueeze(
                            0
                        ),  # [H, W, C] -> [1, C, H, W]
                        size=(H, W),
                        mode="bilinear",
                        align_corners=False,
                    )
                    .squeeze(0)
                    .permute(1, 2, 0)
                )  # [1, C, H, W] -> [H, W, C]
            _t_upscale = (time.perf_counter() - _checkpoint_upscale) * 1000

            # GPU->CPU transfer with synchronization
            # Note: We use synchronous transfer to avoid race conditions with viser's
            # async JPEG encoding. The overhead is minimal (~1-2ms) and prevents
            # frame cross-talk between instances.
            if _cuda_stream is not None:
                # Use dedicated stream for this instance's transfers
                with torch.cuda.stream(_cuda_stream):
                    # CRITICAL: Ensure tensor is contiguous before transfer!
                    # Permute operations (used in upscaling) can leave strides in a state
                    # that causes "stripes" or corruption when converted to numpy/image.
                    result_gpu = result_gpu.contiguous()

                    # Convert to uint8 on GPU (once) for both viser and streaming
                    # This avoids redundant float32→uint8 conversion in viser/PIL
                    gpu_frame_uint8 = (result_gpu * 255).clamp(0, 255).to(torch.uint8)

                    # Transfer uint8 to CPU (4x smaller than float32, no redundant conversion)
                    result = gpu_frame_uint8.cpu().numpy()

                # CRITICAL: Sync stream BEFORE JPEG encoding
                # This ensures the tensor data is complete and prevents corruption
                _cuda_stream.synchronize()

                # Encode to JPEG immediately and cache the bytes
                # This is more efficient than caching the tensor because:
                # 1. No clone needed (we encode while we still own the tensor)
                # 2. Less memory (JPEG bytes ~150KB vs tensor ~6MB for 1080p)
                from src.gsplay.rendering.jpeg_encoder import encode_and_cache
                encode_and_cache(gpu_frame_uint8)
            else:
                # Fallback for non-CUDA devices
                result_gpu = result_gpu.contiguous()

                # Convert to uint8 (once) for both viser and streaming
                gpu_frame_uint8 = (result_gpu * 255).clamp(0, 255).to(torch.uint8)

                # Transfer uint8 to CPU (4x smaller, no redundant conversion)
                result = gpu_frame_uint8.detach().cpu().numpy()

                # Encode to JPEG immediately and cache the bytes
                from src.gsplay.rendering.jpeg_encoder import encode_and_cache
                encode_and_cache(gpu_frame_uint8)

            _t_post = (time.perf_counter() - _checkpoint_post) * 1000

            # Performance logging
            _t_render = (time.perf_counter() - _checkpoint_render) * 1000  # ms
            _t_total = (time.perf_counter() - _frame_start) * 1000  # ms
            _t_other = (
                _t_total - _t_load - _t_edits - _t_render
            )  # Missing time (UI updates, etc.)

            if not hasattr(model, "_perf_frame_count"):
                model._perf_frame_count = 0
            model._perf_frame_count += 1

            if model._perf_frame_count % 90 == 0:
                _fps = 1000.0 / _t_total if _t_total > 0 else 0
                _n_gaussians = (
                    render_gaussians.means.shape[0] if render_gaussians else 0
                )
                perf_msg = (
                    f"[PERF] Frame {model._perf_frame_count}: "
                    f"Total={_t_total:.1f}ms, "
                    f"Load={_t_load:.1f}ms, "
                    f"Edits={_t_edits:.1f}ms, "
                    f"Render={_t_render:.1f}ms ("
                    f"Cam={_t_camera:.1f}ms, "
                    f"Prep={_t_prep:.1f}ms, "
                    f"Rasterize={_t_rasterize:.1f}ms, "
                    f"Upscale={_t_upscale:.1f}ms, "
                    f"Transfer={_t_post:.1f}ms), "
                    f"Other={_t_other:.1f}ms, "
                    f"FPS={_fps:.1f}, Res={W_scaled}x{H_scaled}, Gaussians={_n_gaussians:,}"
                )
                detail_segments: list[str] = []
                load_profile = getattr(model, "_last_load_profile", None)

                # Emit FPS update event
                if event_bus:
                    event_bus.emit(
                        EventType.RENDER_STATS_UPDATED,
                        source="render_function",
                        render_fps=_fps,
                    )

                if load_profile:
                    stage_entries = []
                    breakdown = load_profile.get("process_breakdown") or {}
                    for stage_name, duration in breakdown.items():
                        if duration is None:
                            continue
                        stage_entries.append(f"{stage_name}={duration:.1f}ms")
                    prefetch_state = "hit" if load_profile.get("prefetch_hit") else "miss"
                    stage_str = ", ".join(stage_entries)
                    load_detail = (
                        f"LoadDetail[mode={load_profile.get('mode')}, "
                        f"io={load_profile.get('io_ms', 0.0):.1f}ms, "
                        f"proc={load_profile.get('process_ms', 0.0):.1f}ms, "
                        f"prefetch={prefetch_state}"
                    )
                    if stage_str:
                        load_detail += f", stages={stage_str}"
                    load_detail += "]"
                    detail_segments.append(load_detail)
                edit_owner = getattr(apply_edits_fn, "__self__", None)
                edit_profile = getattr(edit_owner, "_last_edit_profile", None)
                if edit_profile:
                    stage_entries = []
                    for key, value in edit_profile.items():
                        if key.endswith("_ms") and isinstance(value, (int, float)):
                            stage_entries.append(f"{key[:-3]}={value:.1f}ms")
                    stage_str = ", ".join(stage_entries)
                    edit_detail = f"EditsDetail[mode={edit_profile.get('mode')}"
                    if stage_str:
                        edit_detail += f", stages={stage_str}"
                    edit_detail += "]"
                    detail_segments.append(edit_detail)
                if detail_segments:
                    perf_msg += " | " + " | ".join(detail_segments)
                logger.info(perf_msg)

            return result

        except Exception as e:
            # InterruptRenderException is normal - user moved camera, re-raise silently
            if e.__class__.__name__ == "InterruptRenderException":
                raise

            # CUDA error recovery: Clear corrupted state to prevent propagation
            # When nvImageCodec or other CUDA operations fail, they can corrupt the
            # CUDA context. We need to synchronize and clear the error state to allow
            # subsequent renders to succeed.
            if device.startswith("cuda"):
                try:
                    # Clear the GPU frame cache first - it may reference corrupted memory
                    try:
                        from src.gsplay.rendering.jpeg_encoder import get_service
                        get_service().clear_gpu_frame()
                    except Exception:
                        pass  # Best effort

                    # Reset the CUDA stream for this render function
                    # This abandons any pending async work on the corrupted stream
                    if _cuda_stream is not None:
                        try:
                            # Query stream status to check if it's in error state
                            _cuda_stream.synchronize()
                        except Exception:
                            pass  # Stream may be in error state
                        # Create a fresh stream to recover from any stream-level corruption
                        _cuda_stream = torch.cuda.Stream()

                    # Synchronize device to ensure all pending operations complete/fail
                    try:
                        torch.cuda.synchronize(device)
                    except Exception:
                        pass  # May fail if device is in bad state

                    # Reset CUDA error state by calling cudaGetLastError equivalent
                    # PyTorch doesn't expose this directly, but we can force a reset
                    # by doing a minimal CUDA operation that clears the error
                    try:
                        # This small allocation forces CUDA to report and clear last error
                        _probe = torch.zeros(1, device=device)
                        del _probe
                    except Exception:
                        # If even this fails, the device may need a full reset
                        logger.warning("CUDA device in unrecoverable state, attempting cache clear")

                    # Empty cache to release any corrupted allocations
                    torch.cuda.empty_cache()

                    logger.warning(
                        f"CUDA error recovered. Cleared error state, stream, and cache. "
                        f"Error was: {e.__class__.__name__}"
                    )
                except Exception as cleanup_error:
                    logger.error(
                        f"Failed to cleanup CUDA state after error: {cleanup_error}",
                        exc_info=False
                    )

            logger.error(f"Render failed: {e}", exc_info=True)
            return np.zeros((H, W, 3), dtype=np.float32)

    return render_fn
