"""
Embedded version of the nerfview viewer primitives.

This module vendors the viewer utilities from the upstream nerfview project so
that we can tweak them locally and avoid depending on an external package.
"""

from .render_panel import (
    Colormaps,
    RenderTabState,
    apply_float_colormap,
)
from .version import __version__
from .viewer import VIEWER_LOCK, CameraState, GSPlay, with_viewer_lock

__all__ = [
    "CameraState",
    "RenderTabState",
    "GSPlay",
    "VIEWER_LOCK",
    "with_viewer_lock",
    "__version__",
    "Colormaps",
    "apply_float_colormap",
]
