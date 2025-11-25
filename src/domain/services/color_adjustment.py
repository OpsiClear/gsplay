"""
Color adjustment services for Gaussian Splatting data.

Provides pure-functional transformations for color correction that remain
in the domain layer to keep viewer and infrastructure concerns separated.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from gsmod.config.values import ColorValues

logger = logging.getLogger(__name__)


class ColorAdjustmentService:
    """
    Service for applying color adjustments to Gaussian data.

    All methods are pure functions that return new tensors without modifying inputs.
    """

    @staticmethod
    def apply_temperature(colors: torch.Tensor, temperature: float) -> torch.Tensor:
        """
        Apply color temperature adjustment to Gaussian colors.

        :param colors: Input colors [N, 3] in RGB format, range [0, 1]
        :param temperature: Temperature adjustment (-1.0=cool, 0.0=neutral, 1.0=warm)
        :return: Adjusted colors [N, 3], clamped to [0, 1]
        """
        if temperature == 0.0:  # Neutral temperature, no adjustment
            return colors

        try:
            device = colors.device
            dtype = colors.dtype

            # Define color shifts
            cool_shift = torch.tensor([0.0, 0.2, 0.8], device=device, dtype=dtype)
            warm_shift = torch.tensor([0.8, 0.4, 0.0], device=device, dtype=dtype)

            factor = abs(temperature)
            if temperature < 0.0:
                shift = cool_shift * factor
            else:
                shift = warm_shift * factor

            adjusted = torch.clamp(colors + shift.unsqueeze(0), 0.0, 1.0)
            return adjusted

        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Temperature adjustment failed: %s", exc, exc_info=True)
            return colors

    @staticmethod
    def apply_brightness(colors: torch.Tensor, brightness: float) -> torch.Tensor:
        """
        Apply brightness adjustment to Gaussian colors.

        :param colors: Input colors [N, 3]
        :param brightness: Brightness multiplier (1.0 = no change)
        :return: Adjusted colors [N, 3], clamped to [0, 1]
        """
        if brightness == 1.0:
            return colors

        try:
            adjusted = torch.clamp(colors * brightness, 0.0, 1.0)
            return adjusted
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Brightness adjustment failed: %s", exc, exc_info=True)
            return colors

    @staticmethod
    def apply_contrast(colors: torch.Tensor, contrast: float) -> torch.Tensor:
        """
        Apply contrast adjustment to Gaussian colors.

        :param colors: Input colors [N, 3]
        :param contrast: Contrast multiplier (1.0 = no change, >1.0 = more contrast)
        :return: Adjusted colors [N, 3], clamped to [0, 1]
        """
        if contrast == 1.0:
            return colors

        try:
            # Pivot around 0.5 (middle gray)
            adjusted = torch.clamp((colors - 0.5) * contrast + 0.5, 0.0, 1.0)
            return adjusted
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Contrast adjustment failed: %s", exc, exc_info=True)
            return colors

    @staticmethod
    def apply_saturation(colors: torch.Tensor, saturation: float) -> torch.Tensor:
        """
        Apply saturation adjustment to Gaussian colors.

        :param colors: Input colors [N, 3]
        :param saturation: Saturation multiplier (0.0 = grayscale, 1.0 = no change)
        :return: Adjusted colors [N, 3], clamped to [0, 1]
        """
        if saturation == 1.0:
            return colors

        try:
            # Calculate luminance using standard weights (Rec. 709)
            luminance = (
                0.299 * colors[:, 0] + 0.587 * colors[:, 1] + 0.114 * colors[:, 2]
            ).unsqueeze(1)

            # Interpolate between grayscale and color
            adjusted = torch.clamp(
                luminance + (colors - luminance) * saturation, 0.0, 1.0
            )
            return adjusted

        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Saturation adjustment failed: %s", exc, exc_info=True)
            return colors

    @staticmethod
    def apply_gamma(colors: torch.Tensor, gamma: float) -> torch.Tensor:
        """
        Apply gamma correction to Gaussian colors.

        :param colors: Input colors [N, 3]
        :param gamma: Gamma value (1.0 = no change, <1.0 = brighter, >1.0 = darker)
        :return: Adjusted colors [N, 3], clamped to [0, 1]
        """
        if gamma == 1.0:
            return colors

        try:
            adjusted = torch.clamp(torch.pow(colors, gamma), 0.0, 1.0)
            return adjusted
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Gamma correction failed: %s", exc, exc_info=True)
            return colors

    @staticmethod
    def apply_hue_shift(colors: torch.Tensor, hue_shift: float) -> torch.Tensor:
        """
        Apply hue shift to Gaussian colors.

        :param colors: Input colors [N, 3]
        :param hue_shift: Hue shift in degrees (0.0 to 360.0)
        :return: Adjusted colors [N, 3], clamped to [0, 1]
        """
        if hue_shift == 0.0:
            return colors

        try:
            device = colors.device
            dtype = colors.dtype

            # Convert degrees to radians
            hue_rad = hue_shift * (np.pi / 180.0)
            cos_hue = torch.cos(torch.tensor(hue_rad, device=device, dtype=dtype))
            sin_hue = torch.sin(torch.tensor(hue_rad, device=device, dtype=dtype))

            # Rotation matrix for hue shift in RGB space
            sqrt3 = 3**0.5
            rotation_matrix = torch.tensor(
                [
                    [
                        cos_hue + (1 - cos_hue) / 3,
                        (1 - cos_hue) / 3 - sin_hue / sqrt3,
                        (1 - cos_hue) / 3 + sin_hue / sqrt3,
                    ],
                    [
                        (1 - cos_hue) / 3 + sin_hue / sqrt3,
                        cos_hue + (1 - cos_hue) / 3,
                        (1 - cos_hue) / 3 - sin_hue / sqrt3,
                    ],
                    [
                        (1 - cos_hue) / 3 - sin_hue / sqrt3,
                        (1 - cos_hue) / 3 + sin_hue / sqrt3,
                        cos_hue + (1 - cos_hue) / 3,
                    ],
                ],
                device=device,
                dtype=dtype,
            )

            adjusted = torch.clamp(torch.matmul(colors, rotation_matrix.T), 0.0, 1.0)
            return adjusted

        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Hue shift failed: %s", exc, exc_info=True)
            return colors

    @staticmethod
    def apply_vibrance(colors: torch.Tensor, vibrance: float) -> torch.Tensor:
        """
        Apply vibrance adjustment (selective saturation boost for muted colors).

        :param colors: Input colors [N, 3]
        :param vibrance: Vibrance multiplier (1.0 = no change)
        :return: Adjusted colors [N, 3], clamped to [0, 1]
        """
        if vibrance == 1.0:
            return colors

        try:
            # Calculate saturation of each pixel
            max_vals, _ = torch.max(colors, dim=1, keepdim=True)
            min_vals, _ = torch.min(colors, dim=1, keepdim=True)
            saturation = max_vals - min_vals

            # Apply vibrance boost more to less saturated colors
            saturation_boost = (1.0 - saturation) * (vibrance - 1.0)
            mid_tone = (max_vals + min_vals) / 2

            adjusted = torch.clamp(
                colors + (colors - mid_tone) * saturation_boost, 0.0, 1.0
            )
            return adjusted

        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Vibrance adjustment failed: %s", exc, exc_info=True)
            return colors

    @staticmethod
    def apply_shadows_highlights(
        colors: torch.Tensor, shadows: float, highlights: float
    ) -> torch.Tensor:
        """
        Apply separate adjustments to shadows and highlights.

        :param colors: Input colors [N, 3]
        :param shadows: Shadow adjustment (1.0 = no change)
        :param highlights: Highlight adjustment (1.0 = no change)
        :return: Adjusted colors [N, 3], clamped to [0, 1]
        """
        if shadows == 0.0 and highlights == 0.0:
            return colors

        try:
            # Calculate luminance
            luminance = 0.299 * colors[:, 0] + 0.587 * colors[:, 1] + 0.114 * colors[:, 2]

            # Create masks for shadows and highlights
            shadow_mask = (luminance < 0.5).float().unsqueeze(1)
            highlight_mask = (luminance >= 0.5).float().unsqueeze(1)

            # Apply adjustments
            shadow_factor = 1.0 + shadows
            highlight_factor = 1.0 + highlights

            shadow_adjustment = colors * shadow_mask * (shadow_factor - 1.0)
            highlight_adjustment = colors * highlight_mask * (highlight_factor - 1.0)

            adjusted = torch.clamp(
                colors + shadow_adjustment + highlight_adjustment, 0.0, 1.0
            )
            return adjusted

        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Shadows/highlights adjustment failed: %s", exc, exc_info=True)
            return colors

    @staticmethod
    def apply_all_adjustments(
        colors: torch.Tensor, adjustments: "ColorValues"
    ) -> torch.Tensor:
        """
        Apply all color adjustments in the correct order.

        The order of operations is carefully chosen to produce natural-looking results:
        1. Temperature (color balance)
        2. Hue shift (color rotation)
        3. Saturation (color intensity)
        4. Vibrance (selective saturation)
        5. Contrast (tonal range)
        6. Brightness (exposure)
        7. Shadows/Highlights (tonal regions)
        8. Gamma (overall curve)

        :param colors: Input colors [N, 3]
        :param adjustments: Color adjustment parameters
        :return: Adjusted colors [N, 3], clamped to [0, 1]
        """
        # Early exit if no adjustments needed
        if adjustments.is_neutral():
            return colors

        try:
            # Chain operations efficiently - each method handles early exit internally
            # All operations are vectorized and run entirely on GPU
            adjusted = colors

            # 1. Temperature (vectorized GPU operation)
            if adjustments.temperature != 0.0:
                adjusted = ColorAdjustmentService.apply_temperature(adjusted, adjustments.temperature)

            # 2. Hue shift (vectorized GPU operation)
            if adjustments.hue_shift != 0.0:
                logger.debug("Applying hue shift: %s degrees", adjustments.hue_shift)
                adjusted = ColorAdjustmentService.apply_hue_shift(adjusted, adjustments.hue_shift)

            # 3. Saturation (vectorized GPU operation)
            if adjustments.saturation != 1.0:
                adjusted = ColorAdjustmentService.apply_saturation(adjusted, adjustments.saturation)

            # 4. Vibrance (vectorized GPU operation)
            if adjustments.vibrance != 1.0:
                adjusted = ColorAdjustmentService.apply_vibrance(adjusted, adjustments.vibrance)

            # 5. Contrast (vectorized GPU operation)
            if adjustments.contrast != 1.0:
                adjusted = ColorAdjustmentService.apply_contrast(adjusted, adjustments.contrast)

            # 6. Brightness (vectorized GPU operation)
            if adjustments.brightness != 1.0:
                adjusted = ColorAdjustmentService.apply_brightness(adjusted, adjustments.brightness)

            # 7. Shadows/Highlights (vectorized GPU operation)
            if adjustments.shadows != 0.0 or adjustments.highlights != 0.0:
                adjusted = ColorAdjustmentService.apply_shadows_highlights(
                    adjusted, adjustments.shadows, adjustments.highlights
                )

            # 8. Gamma (applied last for final tonal curve, vectorized GPU operation)
            if adjustments.gamma != 1.0:
                adjusted = ColorAdjustmentService.apply_gamma(adjusted, adjustments.gamma)

            # Final clamp (vectorized, single GPU kernel)
            return torch.clamp(adjusted, 0.0, 1.0)

        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Comprehensive color adjustment failed: %s", exc, exc_info=True)
            return colors

