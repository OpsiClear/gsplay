"""
Color preset definitions and composition utilities.

This module provides pure functions for color profile presets and
color value composition. These are extracted from app.py for better
modularity and testability.
"""

from __future__ import annotations

from gsmod import ColorValues


# Color profile presets (relative to neutral)
COLOR_PRESETS: dict[str, ColorValues] = {
    "vibrant": ColorValues(
        saturation=1.3,
        vibrance=1.2,
        contrast=1.05,
    ),
    "dramatic": ColorValues(
        contrast=1.3,
        shadows=-0.15,
        highlights=0.05,
        saturation=1.1,
    ),
    "bright": ColorValues(
        brightness=1.2,
        gamma=0.9,
    ),
    "dark": ColorValues(
        brightness=0.8,
        gamma=1.1,
    ),
    "warm": ColorValues(
        temperature=0.3,
        tint=0.05,
        saturation=1.05,
    ),
    "cool": ColorValues(
        temperature=-0.3,
        tint=-0.05,
        saturation=1.05,
    ),
    "cinematic": ColorValues(
        contrast=1.15,
        saturation=0.9,
        fade=0.08,
        shadows=-0.1,
        highlights=-0.05,
        shadow_tint_hue=0.6,
        shadow_tint_sat=0.15,
        highlight_tint_hue=0.1,
        highlight_tint_sat=0.1,
    ),
    "muted": ColorValues(
        saturation=0.7,
        vibrance=0.8,
        contrast=0.9,
        fade=0.05,
    ),
    "punchy": ColorValues(
        contrast=1.25,
        saturation=1.25,
        vibrance=1.15,
        shadows=-0.1,
        highlights=0.1,
    ),
}


# Unified adjustment options for the Color tab dropdown
# Maps: display label -> (category, handler_key)
# Categories: "correction" (gsmod 0.1.4), "stylize" (presets), "advanced" (histogram learning)
ADJUSTMENT_OPTIONS: dict[str, tuple[str, str]] = {
    # Correction section (gsmod 0.1.4 auto-correction)
    "Auto Enhance": ("correction", "auto_enhance"),
    "Auto Contrast": ("correction", "auto_contrast"),
    "Auto Exposure": ("correction", "auto_exposure"),
    "Auto WB (Gray World)": ("correction", "auto_wb_gray"),
    "Auto WB (White Patch)": ("correction", "auto_wb_white"),
    # Stylize section (existing presets)
    "Vibrant": ("stylize", "vibrant"),
    "Dramatic": ("stylize", "dramatic"),
    "Bright": ("stylize", "bright"),
    "Dark": ("stylize", "dark"),
    "Warm": ("stylize", "warm"),
    "Cool": ("stylize", "cool"),
    "Cinematic": ("stylize", "cinematic"),
    "Muted": ("stylize", "muted"),
    "Punchy": ("stylize", "punchy"),
    # Advanced section (legacy histogram learning)
    "Auto Fit (Basic)": ("advanced", "basic"),
    "Auto Fit (Standard)": ("advanced", "standard"),
    "Auto Fit (Full)": ("advanced", "full"),
}

# Cached dropdown options list (created once at module load)
_DROPDOWN_OPTIONS: list[str] = list(ADJUSTMENT_OPTIONS.keys())


def get_dropdown_options() -> list[str]:
    """
    Get ordered list of dropdown options for UI.

    Returns
    -------
    list[str]
        All adjustment option labels in display order (cached)
    """
    return _DROPDOWN_OPTIONS


def get_adjustment_type(option: str) -> tuple[str, str]:
    """
    Get adjustment category and key for dropdown option.

    Parameters
    ----------
    option : str
        The dropdown option label (e.g., "Auto Enhance", "Vibrant")

    Returns
    -------
    tuple[str, str]
        (category, key) where category is "correction", "stylize", or "advanced"
        Returns ("stylize", "neutral") if option not found
    """
    return ADJUSTMENT_OPTIONS.get(option, ("stylize", "neutral"))


def get_preset_values(profile: str) -> ColorValues:
    """
    Get preset color adjustments for a profile (relative to neutral).

    Parameters
    ----------
    profile : str
        Profile name (case-insensitive): 'vibrant', 'dramatic', 'bright',
        'dark', 'warm', 'cool', 'cinematic', 'muted', 'punchy'

    Returns
    -------
    ColorValues
        Preset color values for the profile, or neutral if not found
    """
    return COLOR_PRESETS.get(profile.lower(), ColorValues())


def compose_color_values(base: ColorValues, style: ColorValues) -> ColorValues:
    """
    Compose two ColorValues: apply base normalization, then style adjustments.

    Composition rules:
    - Multiplicative params (brightness, contrast, saturation, vibrance, gamma): multiply
    - Additive params (temperature, tint, shadows, highlights, hue_shift, fade): add
    - Tint params: take style values if non-zero, else keep base

    Parameters
    ----------
    base : ColorValues
        Base color values (e.g., from auto-normalization)
    style : ColorValues
        Style adjustments to apply (e.g., from preset)

    Returns
    -------
    ColorValues
        Composed color values
    """
    return ColorValues(
        # Multiplicative
        brightness=base.brightness * style.brightness,
        contrast=base.contrast * style.contrast,
        saturation=base.saturation * style.saturation,
        vibrance=base.vibrance * style.vibrance,
        gamma=base.gamma * style.gamma,
        # Additive
        temperature=base.temperature + style.temperature,
        tint=base.tint + style.tint,
        shadows=base.shadows + style.shadows,
        highlights=base.highlights + style.highlights,
        hue_shift=base.hue_shift + style.hue_shift,
        fade=base.fade + style.fade,
        # Tint colors (use style if set)
        shadow_tint_hue=style.shadow_tint_hue if style.shadow_tint_hue != 0 else base.shadow_tint_hue,
        shadow_tint_sat=style.shadow_tint_sat if style.shadow_tint_sat != 0 else base.shadow_tint_sat,
        highlight_tint_hue=style.highlight_tint_hue if style.highlight_tint_hue != 0 else base.highlight_tint_hue,
        highlight_tint_sat=style.highlight_tint_sat if style.highlight_tint_sat != 0 else base.highlight_tint_sat,
    )
