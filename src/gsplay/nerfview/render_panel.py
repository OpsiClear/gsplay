# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Minimal render panel utilities for nerfview.

The camera path trajectory editor has been removed. This module now only contains
the basic RenderTabState dataclass and colormap utilities.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import torch
from torch import Tensor


try:
    import matplotlib

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclasses.dataclass
class RenderTabState:
    """Useful GUI handles exposed by the render tab."""

    num_train_rays_per_sec: float | None = None
    num_view_rays_per_sec: float = 100000.0
    viewer_res: int = 2048
    viewer_width: int = 1280
    viewer_height: int = 960
    render_width: int = 1280
    render_height: int = 960
    jpeg_quality: int = 85  # Current JPEG quality for streaming


Colormaps = Literal["turbo", "viridis", "magma", "inferno", "cividis", "gray"]


def apply_float_colormap(image: Tensor, colormap: Colormaps = "viridis") -> Tensor:
    """Copied from nerfstudio/utils/colormaps.py
    Convert single channel to a color image.

    Args:
        image: Single channel image.
        colormap: Colormap for image.

    Returns:
        Tensor: Colored image with colors in [0, 1]
    """

    image = torch.nan_to_num(image, 0)
    if colormap == "gray" or not MATPLOTLIB_AVAILABLE:
        # Fallback: grayscale if matplotlib not available
        return image.repeat(1, 1, 3)
    image_long = (image * 255).long()
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"
    return torch.tensor(matplotlib.colormaps[colormap].colors, device=image.device)[
        image_long[..., 0]
    ]
