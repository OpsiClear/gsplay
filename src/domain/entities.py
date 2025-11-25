"""
Core domain entities for Gaussian Splatting.

Uses gsply containers directly based on processing stage:
- GSData: NumPy arrays for CPU processing (load, gspro operations)
- GSTensor: PyTorch tensors for GPU processing (rendering, GPU edits)

Extended with gsmod Pro types for enhanced processing:
- GSDataPro: CPU processing with color/transform/filter methods
- GSTensorPro: GPU processing with optimized color/transform/filter methods
"""

from dataclasses import dataclass, asdict
import gsply

# Re-export gsply containers for convenience
# Use GSData for CPU operations, GSTensor for GPU operations
GSData = gsply.GSData
GSTensor = gsply.GSTensor

# Import gsmod Pro types for enhanced processing
from gsmod import GSDataPro
from gsmod.torch import GSTensorPro

# Import gsmod configuration values
from gsmod.config.values import ColorValues, FilterValues, TransformValues


@dataclass
class SceneBounds:
    """Scene bounding box and volume filter geometry."""

    min_coords: tuple[float, float, float] = (0.0, 0.0, 0.0)
    max_coords: tuple[float, float, float] = (1.0, 1.0, 1.0)
    center: tuple[float, float, float] = (0.5, 0.5, 0.5)
    size: tuple[float, float, float] = (1.0, 1.0, 1.0)


# ============================================================================
# GSTensor Helper Functions
# ============================================================================
# Use gsply.GSTensor directly as the primary data container.
# Constructor signature: GSTensor(means, scales, quats, opacities, sh0, shN=None)
#   - sh0: RGB colors [N, 3] in LINEAR space [0, 1]
#   - shN: Higher-order SH coefficients [N, K, 3] or None
#   - All other fields are standard Gaussian parameters
# For concatenation: Use native GSTensor.add() or sum([gstensor1, gstensor2, ...])
# For slicing: Use gstensor[mask] with boolean torch.Tensor masks (same device)


def get_sh_degree(gaussian_data: GSTensor) -> int | None:
    """
    Get the spherical harmonics degree from a GSTensor (GSTensor).

    Args:
        gaussian_data: GSTensor containing Gaussian data

    Returns:
        SH degree (0, 1, 2, 3) or None if no higher-order SH
    """
    if gaussian_data.shN is None or gaussian_data.shN.numel() == 0:
        return None

    # Degree is derived from shN shape: [N, K, 3] where K = (degree+1)^2 - 1
    # degree 1: K=3, degree 2: K=8, degree 3: K=15
    K = gaussian_data.shN.shape[1]
    degree_map = {3: 1, 8: 2, 15: 3}
    return degree_map.get(K, None)


@dataclass
class GaussianLayer:
    """
    A single layer of Gaussians with metadata for multi-asset rendering.

    This structure allows multiple Gaussian datasets to be composed together
    with individual control over visibility, ordering, and blending.

    Attributes:
        data: The Gaussian data for this layer
        layer_id: Unique identifier for this layer
        visible: Whether this layer should be rendered
        z_order: Rendering order (higher values render on top)
        opacity_multiplier: Global opacity adjustment for this layer (0.0 to 1.0)
    """
    data: GSTensor
    layer_id: str
    visible: bool = True
    z_order: int = 0
    opacity_multiplier: float = 1.0


@dataclass
class CompositeGSTensor:
    """
    Container for multiple layers of Gaussians for multi-asset rendering.

    This structure enables composition of multiple Gaussian datasets
    (e.g., static background + dynamic foreground, multiple PLY sequences).
    Layers can be independently controlled and merged for rendering.

    Attributes:
        layers: List of GaussianLayer objects to compose
    """
    layers: list[GaussianLayer]

    def merge(self, filter_invisible: bool = True) -> GSTensor:
        """
        Merge all layers into a single GSTensor for rendering.

        This performs tensor concatenation of visible layers, sorted by z_order.
        The merged result can be passed directly to the renderer.

        Args:
            filter_invisible: If True, only merge visible layers

        Returns:
            A single GSTensor containing all merged Gaussians

        Raises:
            ValueError: If no visible layers exist
        """
        # Filter and sort layers
        layers_to_merge = [
            layer for layer in self.layers
            if not filter_invisible or layer.visible
        ]

        if not layers_to_merge:
            raise ValueError("No visible layers to merge")

        # Sort by z_order (lower values render first)
        layers_to_merge.sort(key=lambda x: x.z_order)

        # Apply opacity multipliers before concatenation
        processed_layers = []
        for layer in layers_to_merge:
            if layer.opacity_multiplier != 1.0:
                # Clone and modify opacity in-place
                adjusted_data = layer.data.clone()
                adjusted_data.opacities = layer.data.opacities * layer.opacity_multiplier
                processed_layers.append(adjusted_data)
            else:
                processed_layers.append(layer.data)

        # Use native GSTensor concatenation (GPU-optimized with _base support)
        # sum() uses __radd__ which calls .add() for efficient merging
        return sum(processed_layers)

    def get_layer(self, layer_id: str) -> GaussianLayer | None:
        """Get a layer by its ID."""
        for layer in self.layers:
            if layer.layer_id == layer_id:
                return layer
        return None

    def set_layer_visibility(self, layer_id: str, visible: bool) -> None:
        """Set the visibility of a specific layer."""
        layer = self.get_layer(layer_id)
        if layer:
            layer.visible = visible

    def get_layer_ids(self) -> list[str]:
        """Get all layer IDs in order."""
        return [layer.layer_id for layer in self.layers]
