"""
Optimized PLY Model Implementation using gsply v0.2.5 native methods.

This module implements the on-demand PLY loader using gsply's built-in format handling.
PLY files are always in PLY format (log scales, logit opacities), so we always use
denormalize() to convert to linear format for rendering.

Features:
- Uses gsply v0.2.5 denormalize() and to_rgb() methods directly
- In-place operations for better performance
- Calculates recommended max_scale threshold (percentile) for UI
"""

# --- REQUIRED IMPORTS ---
import logging
import time
import threading
from pathlib import Path

import torch
import numpy as np
import gsply
from src.domain.interfaces import ModelInterface, DataLoaderInterface
from src.domain.entities import GSData, GSTensor
from src.infrastructure.io.path_io import UniversalPath
from src.shared.perf import PerfMonitor
from src.infrastructure.io.discovery import discover_and_sort_ply_files
from src.infrastructure.processing.ply.format_loader import (
    FormatAwarePlyLoader,
    PlyFrameEncoding,
)
from src.shared.perf import (
    FrameLoadEvent,
    FrameLoadObserver,
    FrameProfilingBroadcaster,
    FrameThroughputObserver,
)

logger = logging.getLogger(__name__)
# --- End of Imports ---


# --- Model Implementation ---

class OptimizedPlyModel(ModelInterface):
    """
    ModelInterface implementation for loading a sequence of .ply files.
    Applies smart activations and forces RGB mode to fix artifacts.

    Features:
    - Clustered LUT for fast activation functions
    - Smart format detection for scales and opacities
    - Calculates recommended max_scale threshold (default percentile) for UI
    """
    def __init__(
        self,
        ply_files: list[str | Path | UniversalPath],
        device: str = "cuda",
        enable_concurrent_prefetch: bool = True,
        processing_mode: str = "all_gpu",  # Global processing mode from VolumeFilter
        opacity_threshold: float = 0.01,  # Quality filtering threshold
        scale_threshold: float = 1e-7,
        enable_quality_filtering: bool = True,
        format_loader: FormatAwarePlyLoader | None = None,
    ) -> None:
        super().__init__()
        # Convert all paths to UniversalPath for cloud storage support
        self.ply_files = [UniversalPath(f) for f in ply_files]
        self.device = device
        self.total_frames = len(ply_files)

        # Processing configuration (unified with VolumeFilter modes)
        self._processing_mode = processing_mode

        # Track last loaded frame for info display
        self._last_loaded_filename: str | None = None
        self._last_loaded_frame_index: int | None = None
        self._format_loader = format_loader or FormatAwarePlyLoader(device=self.device)
        self._format_loader.set_device(self.device)

        if self.total_frames == 0:
            raise ValueError("No .ply files were provided.")

        # Concurrent prefetching (optional for background frame loading)
        self.enable_concurrent_prefetch = enable_concurrent_prefetch
        self._prefetch_executor = None
        self._last_process_breakdown: dict[str, object] | None = None
        self._last_load_profile: dict[str, object] | None = None
        self._cpu_prefetch_results: dict[int, 'GSData | GSTensor'] = {}
        self._cpu_prefetch_inflight: set[int] = set()
        self._cpu_prefetch_lock = threading.Lock()
        self._calculated_max_scale_percentile: float | None = None
        self._profiling_bus = FrameProfilingBroadcaster()
        self._throughput_observer = FrameThroughputObserver(logger=logger)
        self._profiling_bus.subscribe(self._throughput_observer)
        self._last_requested_frame: int | None = None

        # Cached folder-level format (avoids per-file detection for homogeneous sequences)
        self._cached_folder_format: tuple[PlyFrameEncoding, int | None] | None = None

        # Current frame cache - avoids disk reload on camera rotation
        self._cached_frame_idx: int | None = None
        self._cached_frame_data: GSData | GSTensor | None = None

        if self.enable_concurrent_prefetch:
            from concurrent.futures import ThreadPoolExecutor
            # Use 2 threads (don't compete with rendering)
            self._prefetch_executor = ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="ply_prefetch"
            )
            logger.debug("[OptimizedPlyModel] Concurrent prefetching enabled (2 background threads)")

        logger.debug(f"[OptimizedPlyModel] Initialized with {self.total_frames} PLY files.")
        logger.debug(
            "[OptimizedPlyModel] Concurrent prefetch: %s",
            "enabled" if enable_concurrent_prefetch else "disabled",
        )

        # Log file order at initialization to verify correct sorting
        first_10 = [f.name for f in self.ply_files[:10]]
        last_10 = [f.name for f in self.ply_files[-10:]]
        logger.debug(f"[File Order at Init] First 10: {first_10}")
        logger.debug(f"[File Order at Init] Last 10: {last_10}")
        self._log_folder_format_hint()


    def get_total_frames(self) -> int:
        return self.total_frames

    def get_frame_time(self, frame_idx: int) -> float:
        if self.total_frames <= 1:
            return 0.0
        return frame_idx / (self.total_frames - 1)

    @property
    def processing_mode(self) -> str:
        """Get current processing mode."""
        return self._processing_mode

    @processing_mode.setter
    def processing_mode(self, value: str) -> None:
        """Set processing mode and invalidate frame cache if changed."""
        if value != self._processing_mode:
            self._processing_mode = value
            self.invalidate_frame_cache()

    def invalidate_frame_cache(self) -> None:
        """Clear the cached frame data, forcing reload on next request."""
        self._cached_frame_idx = None
        self._cached_frame_data = None

    def get_recommended_max_scale(self) -> float:
        """
        Get the recommended max_scale value based on percentile of first frame.

        This triggers loading of the first frame if not already loaded.

        Returns
        -------
        float
            Recommended max_scale threshold (percentile of scale values)
        """
        if self._calculated_max_scale_percentile is None:
            # Load first frame to calculate percentile
            logger.debug("[get_recommended_max_scale] Loading first frame to calculate percentile")
            self.get_gaussians_at_normalized_time(0.0)

        return self._calculated_max_scale_percentile or 0.0

    def get_last_profile(self) -> dict[str, object] | None:
        """Expose the latest frame load profile for diagnostics."""
        return self._last_load_profile

    def add_frame_observer(self, observer: FrameLoadObserver) -> None:
        """Allow external components to subscribe to frame-load events."""
        self._profiling_bus.subscribe(observer)


    def _log_folder_format_hint(self) -> None:
        """Emit a one-time log summarizing the detected folder encoding and cache it."""
        if not self.ply_files:
            return

        sample_count = min(5, self.total_frames)
        hint = self._format_loader.summarize_folder(self.ply_files[:sample_count])
        folder = self.ply_files[0].parent
        if hint is None:
            logger.info(
                "[OptimizedPlyModel] Unable to determine Gaussian PLY encoding for %s; resolving per frame.",
                folder,
            )
            return

        # Cache folder-level format to skip per-file detection
        self._cached_folder_format = hint
        encoding, sh_degree = hint

        # Set default format on loader to bypass per-file detection in load_for_gpu
        self._format_loader.set_default_format(encoding, sh_degree)

        if encoding == PlyFrameEncoding.COMPRESSED:
            logger.info(
                "[OptimizedPlyModel] Detected compressed Gaussian PLYs in %s (cached for all frames)",
                folder,
            )
        else:
            sh_msg = f"SH degree {sh_degree}" if sh_degree is not None else "SH degree unknown"
            logger.info(
                "[OptimizedPlyModel] Detected uncompressed Gaussian PLYs (%s) in %s (cached for all frames)",
                sh_msg,
                folder,
            )

    def _process_frame_gpu(
        self,
        gstensor_cpu: 'GSTensor',
        frame_idx: int
    ) -> 'GSTensor':
        """
        Process GPU frame using gsply v0.2.5 native methods.
        
        PLY files are always in PLY format (log/logit), so we always denormalize.
        """
        import torch.nn.functional as F
        from src.infrastructure.processing.gaussian_constants import GaussianConstants as GC
        
        monitor = PerfMonitor(label=f"gpu_frame_{frame_idx}")
        
        # CPU -> GPU transfer
        with monitor.track("transfer_ms"):
            gstensor_gpu = gstensor_cpu.to(self.device)

        with monitor.track("activations_ms"):
            # PLY files are always in PLY format, so always denormalize to linear
            gstensor_gpu = gstensor_gpu.denormalize(inplace=True)
            
            # Clamp scales and opacities to valid ranges (in-place)
            gstensor_gpu.scales = gstensor_gpu.scales.clamp(
                min=GC.Numerical.MIN_SCALE,
                max=GC.Numerical.MAX_SCALE,
            )
            gstensor_gpu.opacities = gstensor_gpu.opacities.squeeze().clamp(0.0, 1.0)
            
            # Normalize quaternions (in-place)
            gstensor_gpu.quats = F.normalize(gstensor_gpu.quats, p=2, dim=-1)
            
            # Calculate percentile for first frame
            if self._calculated_max_scale_percentile is None:
                with monitor.track("percentile_ms"):
                    self._calculated_max_scale_percentile = torch.quantile(
                        gstensor_gpu.scales.flatten(),
                        GC.Filtering.DEFAULT_PERCENTILE,
                    ).item()

        with monitor.track("color_ms"):
            # Only convert SH to RGB if no higher-order SH coefficients
            # When shN is present, gsplat will evaluate SH during rendering
            # which requires sh0 to remain in SH format
            if gstensor_gpu.shN is None or gstensor_gpu.shN.numel() == 0:
                # No higher-order SH - convert to RGB for direct color blending
                gstensor_gpu = gstensor_gpu.to_rgb(inplace=True)
                gstensor_gpu.sh0 = torch.clamp(gstensor_gpu.sh0, 0.0, 1.0)
            # else: Keep sh0 in SH format for gsplat SH evaluation

        stage_timings, total_ms = monitor.stop()
        self._last_process_breakdown = stage_timings.copy()
        return gstensor_gpu

    def _process_frame_cpu_from_gsdata(
        self,
        gsdata: 'GSData',
        frame_idx: int,
    ) -> 'GSData':
        """
        Process CPU frame using gsply v0.2.5 native methods.
        
        PLY files are always in PLY format (log/logit), so we always denormalize.
        
        Parameters
        ----------
        gsdata : GSData
            Input GSData in PLY format
        frame_idx : int
            Frame index for performance monitoring
        
        Returns
        -------
        GSData
            Processed GSData in linear format with RGB colors
        """
        import numpy as np
        from src.infrastructure.processing.gaussian_constants import GaussianConstants as GC
        
        monitor = PerfMonitor(label=f"cpu_frame_{frame_idx}")

        with monitor.track("activations_ms"):
            # Validate input data before processing
            n_input = gsdata.means.shape[0]
            if n_input == 0:
                logger.error(f"[OptimizedPlyModel] Input GSData is empty for frame {frame_idx}")
                return gsdata
            
            # PLY files are always in PLY format, so always denormalize to linear
            processed = gsdata.denormalize(inplace=True)
            
            # Validate after denormalize
            n_after_denorm = processed.means.shape[0]
            if n_after_denorm == 0:
                logger.error(
                    f"[OptimizedPlyModel] Data became empty after denormalize for frame {frame_idx} "
                    f"(input had {n_input} gaussians)"
                )
                return processed
            
            # Clamp scales after denormalization
            processed.scales = np.clip(
                processed.scales,
                GC.Numerical.MIN_SCALE,
                GC.Numerical.MAX_SCALE
            )
            
            # Calculate percentile for first frame
            if self._calculated_max_scale_percentile is None:
                with monitor.track("percentile_ms"):
                    scales = processed.scales.flatten()
                    self._calculated_max_scale_percentile = float(
                        np.percentile(scales, GC.Filtering.DEFAULT_PERCENTILE * 100)
                    )

        with monitor.track("color_ms"):
            # Only convert SH to RGB if no higher-order SH coefficients
            # When shN is present, gsplat will evaluate SH during rendering
            # which requires sh0 to remain in SH format
            if processed.shN is None or processed.shN.size == 0:
                # No higher-order SH - convert to RGB for direct color blending
                processed = processed.to_rgb(inplace=True)
                
                # Validate after to_rgb
                n_after_rgb = processed.means.shape[0]
                if n_after_rgb == 0:
                    logger.error(
                        f"[OptimizedPlyModel] Data became empty after to_rgb for frame {frame_idx} "
                        f"(had {n_after_denorm} gaussians before to_rgb)"
                    )
                    return processed
                
                processed.sh0 = np.clip(processed.sh0, 0.0, 1.0)
            # else: Keep sh0 in SH format for gsplat SH evaluation

        stage_timings, total_ms = monitor.stop()
        self._last_process_breakdown = stage_timings.copy()
        return processed

    @torch.no_grad()
    def get_gaussians_at_normalized_time(
        self,
        normalized_time: float,
    ) -> 'GSData | GSTensor | None':
        """
        Load and process Gaussian data at given time.

        Returns GSData (NumPy) for CPU modes or GSTensor (PyTorch, GPU) for GPU mode.

        Supports two processing modes:
        - "gpu": Transfer to GPU immediately, process on GPU (fastest, ~2-3ms load time)
        - "cpu": Process on CPU with NumPy, transfer at end (slower ~10ms, lower GPU memory)

        Args:
            normalized_time: Normalized time in [0.0, 1.0]

        Returns:
            Processed GSTensor (GSTensor) on GPU, ready for rendering
        """
        frame_idx = int(round(normalized_time * (self.total_frames - 1)))
        frame_idx = max(0, min(frame_idx, self.total_frames - 1))

        # Fast path: return cached data if same frame (e.g., camera rotation)
        if frame_idx == self._cached_frame_idx and self._cached_frame_data is not None:
            return self._cached_frame_data

        previous_frame_idx = self._last_requested_frame
        self._last_requested_frame = frame_idx
        ply_path = self.ply_files[frame_idx]
        load_start = time.perf_counter()
        io_ms = 0.0

        # Track what we're actually loading for info display
        filename = ply_path.name
        self._last_loaded_filename = filename
        self._last_loaded_frame_index = frame_idx

        # Use cached folder format if available (skips per-file detection overhead)
        if self._cached_folder_format is not None:
            frame_encoding, sh_degree = self._cached_folder_format
        else:
            frame_encoding, sh_degree = self._format_loader.detect_format(ply_path)
        sh_info = f"sh={sh_degree}" if sh_degree is not None else "sh=unknown"
        encoding_label = frame_encoding.value

        # Comprehensive diagnostic logging for frame loading
        logger.debug(
            f"[Frame Load] time={normalized_time:.4f} -> idx={frame_idx}/{self.total_frames-1} -> file={filename} "
            f"(mode={self.processing_mode}, format={encoding_label}, {sh_info})"
        )

        # Periodically verify file order hasn't changed (every 30 frames)
        if frame_idx % 30 == 0:
            first_5 = [f.name for f in self.ply_files[:5]]
            last_5 = [f.name for f in self.ply_files[-5:]]
            logger.debug(
                f"[File Order Check] First 5: {first_5}, Last 5: {last_5}"
            )

        try:
            # Determine processing mode early
            current_mode = self.processing_mode
            is_cpu_mode = current_mode in ("all_cpu", "color_transform_gpu", "transform_gpu", "color_gpu")

            prefetched_data = self._pop_cpu_prefetch(frame_idx) if is_cpu_mode else None

            # Step 1: Load raw data - optimize path based on processing mode
            from src.infrastructure.processing.ply.loader import load_ply_as_gsdata

            processed_data: GSData | GSTensor | None = None
            pipeline_type = "unknown"

            if prefetched_data is not None:
                processed_data = prefetched_data
                pipeline_type = "cpu_prefetch"
            elif is_cpu_mode:
                gsdata_raw = load_ply_as_gsdata(ply_path)
                if gsdata_raw.means.shape[0] == 0:
                    logger.error(f"[OptimizedPlyModel] Loaded PLY file is empty: {ply_path}")
                    return None
                processed_data = self._process_frame_cpu_from_gsdata(gsdata_raw, frame_idx)
                pipeline_type = "cpu"
            else:
                load_result = self._format_loader.load_for_gpu(ply_path)
                frame_encoding = load_result.encoding
                sh_degree = load_result.sh_degree
                gstensor_raw = load_result.gstensor
                gsdata_raw = load_result.gsdata

                if gstensor_raw is None:
                    gsdata_source = gsdata_raw or load_ply_as_gsdata(ply_path)
                    if gsdata_source is None:
                        logger.error(f"[OptimizedPlyModel] No data source for GPU processing frame {frame_idx}")
                        return None
                    gstensor_raw = gsply.GSTensor.from_gsdata(gsdata_source, device="cuda")
                processed_data = self._process_frame_gpu(gstensor_raw, frame_idx)
                pipeline_type = "gpu"

            io_ms = (time.perf_counter() - load_start) * 1000

            # Validate processed data before returning
            if processed_data is None:
                logger.error(f"[OptimizedPlyModel] Processing returned None for frame {frame_idx}")
                return None

            if hasattr(processed_data, "means"):
                n_gaussians = processed_data.means.shape[0]
                if n_gaussians == 0:
                    logger.warning(
                        f"[OptimizedPlyModel] Processed data is empty for frame {frame_idx} "
                        f"(prefetched={prefetched_data is not None}, pipeline={pipeline_type})"
                    )
                    return None
                logger.debug(
                    f"[OptimizedPlyModel] Frame {frame_idx}: {n_gaussians} gaussians "
                    f"(prefetched={prefetched_data is not None}, pipeline={pipeline_type})"
                )
            else:
                logger.error(
                    f"[OptimizedPlyModel] Processed data missing 'means' attribute for frame {frame_idx}"
                )
                return None

            process_ms = sum(self._last_process_breakdown.values()) if self._last_process_breakdown else 0.0
            total_ms = (time.perf_counter() - load_start) * 1000
            io_ms = max(io_ms - process_ms, 0.0) if process_ms else io_ms
            logger.debug(
                "[Frame Load Timing][%s] frame=%d io=%.2fms process=%.2fms total=%.2fms prefetch_hit=%s",
                pipeline_type.upper(),
                frame_idx,
                io_ms,
                process_ms,
                total_ms,
                prefetched_data is not None,
            )
            combined_timings = {"io_ms": io_ms}
            if self._last_process_breakdown:
                combined_timings.update(self._last_process_breakdown)
            self._last_load_profile = {
                "frame_idx": frame_idx,
                "mode": current_mode,
                "io_ms": io_ms,
                "process_ms": process_ms,
                "total_ms": total_ms,
                "prefetch_hit": prefetched_data is not None,
                "process_breakdown": self._last_process_breakdown,
                "throughput_fps": self.get_frame_throughput_fps(),
            }
            self._profiling_bus.notify(
                FrameLoadEvent(
                    frame_idx=frame_idx,
                    pipeline=pipeline_type,
                    total_ms=total_ms,
                    stage_timings=combined_timings,
                    prefetch_hit=prefetched_data is not None,
                )
            )
            if is_cpu_mode:
                self._schedule_cpu_prefetch(frame_idx, previous_frame_idx)

            # Cache the loaded frame for fast retrieval on camera rotation
            self._cached_frame_idx = frame_idx
            self._cached_frame_data = processed_data

            return processed_data

        except Exception as e:
            # InterruptRenderException is normal - user moved camera, re-raise without logging
            if e.__class__.__name__ == "InterruptRenderException":
                raise

            logger.error(f"Error loading PLY file {ply_path}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _schedule_cpu_prefetch(self, current_frame: int, previous_frame: int | None) -> None:
        """Prefetch up to two CPU frames based on playback direction."""
        if not self.enable_concurrent_prefetch or self._prefetch_executor is None:
            return

        direction = 1
        if previous_frame is not None and current_frame < previous_frame:
            direction = -1
        for offset in (1, 2):
            next_frame = current_frame + direction * offset
            if next_frame < 0 or next_frame >= self.total_frames:
                continue

            with self._cpu_prefetch_lock:
                if next_frame in self._cpu_prefetch_results or next_frame in self._cpu_prefetch_inflight:
                    continue
                self._cpu_prefetch_inflight.add(next_frame)

            self._prefetch_executor.submit(self._prefetch_frame_cpu, next_frame)

    def _prefetch_frame_cpu(self, frame_idx: int) -> None:
        """Load and process a frame on a background thread for CPU pipelines."""
        try:
            from src.infrastructure.processing.ply.loader import load_ply_as_gsdata

            ply_path = self.ply_files[frame_idx]
            gsdata = load_ply_as_gsdata(ply_path)
            if gsdata.means.shape[0] == 0:
                return

            processed = self._process_frame_cpu_from_gsdata(gsdata, frame_idx)
            self._store_cpu_prefetch(frame_idx, processed)
            logger.debug("[OptimizedPlyModel] Prefetched CPU frame %d", frame_idx)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("[OptimizedPlyModel] CPU prefetch failed for frame %d: %s", frame_idx, exc)
        finally:
            with self._cpu_prefetch_lock:
                self._cpu_prefetch_inflight.discard(frame_idx)

    def __del__(self):
        """Cleanup: Shutdown ThreadPoolExecutor if enabled."""
        if self._prefetch_executor is not None:
            logger.info("[OptimizedPlyModel] Shutting down concurrent prefetch executor")
            self._prefetch_executor.shutdown(wait=True, cancel_futures=False)

    def _pop_cpu_prefetch(self, frame_idx: int) -> 'GSData | None':
        with self._cpu_prefetch_lock:
            return self._cpu_prefetch_results.pop(frame_idx, None)

    def _store_cpu_prefetch(self, frame_idx: int, processed_data: 'GSData') -> None:
        with self._cpu_prefetch_lock:
            self._cpu_prefetch_results[frame_idx] = processed_data
            if len(self._cpu_prefetch_results) > 4:
                # Drop the smallest index (approximate LRU by traversal order)
                for drop_idx in sorted(self._cpu_prefetch_results.keys()):
                    if drop_idx != frame_idx:
                        self._cpu_prefetch_results.pop(drop_idx, None)
                        break

    def get_frame_throughput_fps(self) -> float:
        if hasattr(self, "_throughput_observer") and self._throughput_observer:
            return self._throughput_observer.latest_fps
        return 0.0

# --- DataLoader Implementation ---
class OptimizedPlyDataLoader(DataLoaderInterface):
    """
    Data loader for OptimizedPLYModel.
    Loads the first frame's means for initialization.
    Supports local filesystem and cloud storage.
    """
    def __init__(self, ply_files: list[str | Path | UniversalPath]) -> None:
        # Convert all paths to UniversalPath for cloud storage support
        self.ply_files = [UniversalPath(f) for f in ply_files]
        self.initial_points: np.ndarray | None = None
        if not self.ply_files:
            raise ValueError("No .ply files provided.")
        logger.debug(f"[OptimizedPlyDataLoader] Initialized with {len(self.ply_files)} PLY files.")


    def get_camera_data(self) -> dict[str, any] | None:
        return None

    def get_points_for_initialization(self) -> np.ndarray:
        """
        Loads and caches the points from the first PLY file.
        """
        if self.initial_points is not None:
            return self.initial_points

        try:
            first_ply_path = self.ply_files[0]
            logger.info(f"[OptimizedPlyDataLoader] Loading initial points from: {first_ply_path.name}")
            # Use load_ply_as_gsdata to get GSData (NumPy), then extract means
            from src.infrastructure.processing.ply.loader import load_ply_as_gsdata
            gsdata = load_ply_as_gsdata(first_ply_path)
            self.initial_points = gsdata.means  # Already NumPy
            logger.info(f"[OptimizedPlyDataLoader] Successfully loaded {self.initial_points.shape[0]} initial points.")
            return self.initial_points
        except Exception as e:
            logger.warning(f"Error loading first PLY for initialization ({self.ply_files[0].name}): {e}")
            self.initial_points = np.zeros((1, 3)) # Cache fallback
            return self.initial_points

# --- Factory Functions ---
def create_optimized_ply_model_from_folder(
    ply_folder: str | Path | UniversalPath,
    device: str = "cuda",
    enable_concurrent_prefetch: bool = True,
    processing_mode: str = "all_gpu",  # Unified with VolumeFilter modes
    opacity_threshold: float = 0.01,
    scale_threshold: float = 1e-7,
    enable_quality_filtering: bool = True,
) -> OptimizedPlyModel:
    # Use SINGLE authoritative function for file discovery and sorting
    # discover_and_sort_ply_files now returns UniversalPath objects
    ply_files = discover_and_sort_ply_files(ply_folder)
    logger.info(
        f"[Factory] Found {len(ply_files)} .ply files "
        f"(sorted by discover_and_sort_ply_files, supports cloud storage)"
    )

    # Note: DataLoader is instantiated implicitly here, but the factory functions
    # in universal_viewer.py will create them separately.
    return OptimizedPlyModel(
        ply_files,
        device=device,
        enable_concurrent_prefetch=enable_concurrent_prefetch,
        processing_mode=processing_mode,
        opacity_threshold=opacity_threshold,
        scale_threshold=scale_threshold,
        enable_quality_filtering=enable_quality_filtering,
    )

def create_optimized_ply_data_loader_from_folder(ply_folder: str | Path | UniversalPath) -> OptimizedPlyDataLoader:
    # Use SINGLE authoritative function for file discovery and sorting
    # discover_and_sort_ply_files now returns UniversalPath objects
    ply_files = discover_and_sort_ply_files(ply_folder)
    logger.info(
        f"[Factory] Found {len(ply_files)} .ply files for data loader "
        f"(sorted by discover_and_sort_ply_files, supports cloud storage)"
    )
    return OptimizedPlyDataLoader(ply_files)
