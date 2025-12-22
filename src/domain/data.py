"""
Unified GaussianData abstraction for CPU/GPU data IO.

This module provides a single interface for data loaders and exporters,
with lazy CPU/GPU conversion and integration with gsply containers.

Loaders produce GaussianData, exporters consume GaussianData.
The viewer converts to/from gsply types for processing.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from src.domain.entities import SH_DEGREE_MAP


if TYPE_CHECKING:
    from gsply import GSData
    from gsply.torch import GSTensor

logger = logging.getLogger(__name__)


@dataclass
class FormatInfo:
    """Format tracking for Gaussian data.

    Tracks the encoding state of scales, opacities, and colors
    to ensure correct processing.
    """

    is_scales_ply: bool = True  # True = log-space, False = linear
    is_opacities_ply: bool = True  # True = logit-space, False = linear
    is_sh0_rgb: bool = False  # True = RGB [0,1], False = SH coefficients
    sh_degree: int | None = None  # SH degree (0, 1, 2, 3) or None


@dataclass
class GaussianData:
    """Unified abstraction for Gaussian data with lazy CPU/GPU conversion.

    Single interface for data IO - loaders produce this, exporters consume this.
    Internally wraps either CPU (numpy) or GPU (torch) data with lazy conversion.

    Attributes:
        means: Gaussian centers [N, 3]
        scales: Gaussian scales [N, 3] (may be log or linear)
        quats: Quaternion rotations [N, 4]
        opacities: Opacity values [N] or [N, 1] (may be logit or linear)
        sh0: DC color coefficients [N, 3]
        shN: Higher-order SH coefficients [N, K, 3] or None
        format_info: Format tracking
        n_gaussians: Number of Gaussians
        source_path: Original file path (for metadata)
    """

    # Core Gaussian attributes - numpy arrays (CPU)
    means: np.ndarray | None = None
    scales: np.ndarray | None = None
    quats: np.ndarray | None = None
    opacities: np.ndarray | None = None
    sh0: np.ndarray | None = None
    shN: np.ndarray | None = None

    # GPU tensors (lazy-populated)
    _gpu_means: Any = field(default=None, repr=False)
    _gpu_scales: Any = field(default=None, repr=False)
    _gpu_quats: Any = field(default=None, repr=False)
    _gpu_opacities: Any = field(default=None, repr=False)
    _gpu_sh0: Any = field(default=None, repr=False)
    _gpu_shN: Any = field(default=None, repr=False)
    _gpu_device: str | None = field(default=None, repr=False)

    # Metadata
    format_info: FormatInfo = field(default_factory=FormatInfo)
    source_path: str | None = None

    @property
    def n_gaussians(self) -> int:
        """Get number of Gaussians."""
        if self.means is not None:
            return self.means.shape[0]
        if self._gpu_means is not None:
            return self._gpu_means.shape[0]
        return 0

    @property
    def is_on_cpu(self) -> bool:
        """Check if CPU data is available."""
        return self.means is not None

    @property
    def is_on_gpu(self) -> bool:
        """Check if GPU data is available."""
        return self._gpu_means is not None

    @classmethod
    def from_gsdata(cls, gsdata: GSData, source_path: str | None = None) -> GaussianData:
        """Create GaussianData from gsply.GSData.

        Parameters
        ----------
        gsdata : GSData
            gsply GSData container with numpy arrays
        source_path : str | None
            Optional source file path for metadata

        Returns
        -------
        GaussianData
            New GaussianData instance with CPU data
        """
        # Extract format info from gsdata if available
        format_info = FormatInfo()
        if hasattr(gsdata, "_format"):
            fmt = gsdata._format
            format_info.is_scales_ply = fmt.get("scales", "ply") == "ply"
            format_info.is_opacities_ply = fmt.get("opacities", "ply") == "ply"
            format_info.is_sh0_rgb = fmt.get("sh0", "sh") == "rgb"
        elif hasattr(gsdata, "is_scales_ply"):
            format_info.is_scales_ply = gsdata.is_scales_ply
            format_info.is_opacities_ply = gsdata.is_opacities_ply
            format_info.is_sh0_rgb = gsdata.is_sh0_rgb

        # Determine SH degree from shN shape
        sh_degree = None
        if gsdata.shN is not None and gsdata.shN.size > 0:
            k = gsdata.shN.shape[1] if len(gsdata.shN.shape) > 1 else 0
            sh_degree = SH_DEGREE_MAP.get(k)
        format_info.sh_degree = sh_degree

        return cls(
            means=gsdata.means,
            scales=gsdata.scales,
            quats=gsdata.quats,
            opacities=gsdata.opacities,
            sh0=gsdata.sh0,
            shN=gsdata.shN if gsdata.shN is not None and gsdata.shN.size > 0 else None,
            format_info=format_info,
            source_path=source_path,
        )

    @classmethod
    def from_gstensor(cls, gstensor: GSTensor, source_path: str | None = None) -> GaussianData:
        """Create GaussianData from gsply.torch.GSTensor.

        Parameters
        ----------
        gstensor : GSTensor
            gsply GSTensor container with PyTorch tensors
        source_path : str | None
            Optional source file path for metadata

        Returns
        -------
        GaussianData
            New GaussianData instance with GPU data
        """
        # Extract format info from gstensor if available
        format_info = FormatInfo()
        if hasattr(gstensor, "_format"):
            fmt = gstensor._format
            format_info.is_scales_ply = fmt.get("scales", "ply") == "ply"
            format_info.is_opacities_ply = fmt.get("opacities", "ply") == "ply"
            format_info.is_sh0_rgb = fmt.get("sh0", "sh") == "rgb"
        elif hasattr(gstensor, "is_scales_ply"):
            format_info.is_scales_ply = gstensor.is_scales_ply
            format_info.is_opacities_ply = gstensor.is_opacities_ply
            format_info.is_sh0_rgb = gstensor.is_sh0_rgb

        # Determine SH degree from shN shape
        sh_degree = None
        if gstensor.shN is not None and gstensor.shN.numel() > 0:
            k = gstensor.shN.shape[1] if len(gstensor.shN.shape) > 1 else 0
            sh_degree = SH_DEGREE_MAP.get(k)
        format_info.sh_degree = sh_degree

        device = str(gstensor.means.device)

        instance = cls(
            format_info=format_info,
            source_path=source_path,
        )
        # Store GPU tensors
        instance._gpu_means = gstensor.means
        instance._gpu_scales = gstensor.scales
        instance._gpu_quats = gstensor.quats
        instance._gpu_opacities = gstensor.opacities
        instance._gpu_sh0 = gstensor.sh0
        instance._gpu_shN = (
            gstensor.shN if gstensor.shN is not None and gstensor.shN.numel() > 0 else None
        )
        instance._gpu_device = device

        return instance

    def to_gsdata(self) -> GSData:
        """Convert to gsply.GSData for CPU processing.

        Returns
        -------
        GSData
            gsply GSData container with numpy arrays
        """
        from gsply import GSData

        # Ensure CPU data is available
        self._ensure_cpu()

        gsdata = GSData(
            means=self.means,
            scales=self.scales,
            quats=self.quats,
            opacities=self.opacities,
            sh0=self.sh0,
            shN=self.shN,
        )

        # Copy format info
        if hasattr(gsdata, "_format"):
            gsdata._format = {
                "scales": "ply" if self.format_info.is_scales_ply else "linear",
                "opacities": "ply" if self.format_info.is_opacities_ply else "linear",
                "sh0": "rgb" if self.format_info.is_sh0_rgb else "sh",
            }

        return gsdata

    def to_gstensor(self, device: str = "cuda:0") -> GSTensor:
        """Convert to gsply.torch.GSTensor for GPU processing.

        Parameters
        ----------
        device : str
            Target device (e.g., "cuda:0", "cuda", "cpu")

        Returns
        -------
        GSTensor
            gsply GSTensor container with PyTorch tensors
        """
        from gsply.torch import GSTensor

        # If already on the target device, use cached tensors
        if self._gpu_device == device and self._gpu_means is not None:
            gstensor = GSTensor(
                means=self._gpu_means,
                scales=self._gpu_scales,
                quats=self._gpu_quats,
                opacities=self._gpu_opacities,
                sh0=self._gpu_sh0,
                shN=self._gpu_shN,
            )
        else:
            # Ensure CPU data is available for conversion
            self._ensure_cpu()

            # Create GSTensor from numpy arrays
            gstensor = GSTensor.from_gsdata(self.to_gsdata(), device=device)

            # Cache GPU tensors
            self._gpu_means = gstensor.means
            self._gpu_scales = gstensor.scales
            self._gpu_quats = gstensor.quats
            self._gpu_opacities = gstensor.opacities
            self._gpu_sh0 = gstensor.sh0
            self._gpu_shN = gstensor.shN if gstensor.shN is not None else None
            self._gpu_device = device

        # Copy format info
        if hasattr(gstensor, "_format"):
            gstensor._format = {
                "scales": "ply" if self.format_info.is_scales_ply else "linear",
                "opacities": "ply" if self.format_info.is_opacities_ply else "linear",
                "sh0": "rgb" if self.format_info.is_sh0_rgb else "sh",
            }

        return gstensor

    def _ensure_cpu(self) -> None:
        """Ensure CPU data is available (lazy conversion from GPU)."""
        if self.means is not None:
            return  # Already have CPU data

        if self._gpu_means is None:
            raise ValueError("No data available (neither CPU nor GPU)")

        # Convert GPU tensors to numpy
        self.means = self._gpu_means.cpu().numpy()
        self.scales = self._gpu_scales.cpu().numpy()
        self.quats = self._gpu_quats.cpu().numpy()
        self.opacities = self._gpu_opacities.cpu().numpy()
        self.sh0 = self._gpu_sh0.cpu().numpy()
        if self._gpu_shN is not None:
            self.shN = self._gpu_shN.cpu().numpy()

    def clear_gpu_cache(self) -> None:
        """Clear cached GPU tensors to free memory."""
        self._gpu_means = None
        self._gpu_scales = None
        self._gpu_quats = None
        self._gpu_opacities = None
        self._gpu_sh0 = None
        self._gpu_shN = None
        self._gpu_device = None

    def clone(self) -> GaussianData:
        """Create a deep copy of this GaussianData.

        Returns
        -------
        GaussianData
            New independent GaussianData instance
        """
        return GaussianData(
            means=self.means.copy() if self.means is not None else None,
            scales=self.scales.copy() if self.scales is not None else None,
            quats=self.quats.copy() if self.quats is not None else None,
            opacities=self.opacities.copy() if self.opacities is not None else None,
            sh0=self.sh0.copy() if self.sh0 is not None else None,
            shN=self.shN.copy() if self.shN is not None else None,
            format_info=copy.copy(self.format_info),
            source_path=self.source_path,
        )

    def __len__(self) -> int:
        """Return number of Gaussians."""
        return self.n_gaussians

    def __repr__(self) -> str:
        """Return string representation."""
        location = []
        if self.is_on_cpu:
            location.append("CPU")
        if self.is_on_gpu:
            location.append(f"GPU({self._gpu_device})")
        loc_str = "+".join(location) if location else "empty"

        return (
            f"GaussianData(n={self.n_gaussians}, "
            f"location={loc_str}, "
            f"sh_degree={self.format_info.sh_degree})"
        )
