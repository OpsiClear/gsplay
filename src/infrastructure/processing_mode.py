"""
Processing mode configuration for CPU/GPU pipeline optimization.

This module defines the 5 processing modes for the edit pipeline:
1. All GPU - Everything on GPU (fastest, default)
2. Color+Transform GPU - Filter on CPU, rest on GPU
3. Transform GPU - Filter+Color on CPU, Transform on GPU
4. Color GPU - Filter+Transform on CPU, Color on GPU
5. All CPU - Everything on CPU (max GPU memory savings)
"""

from enum import Enum


class ProcessingMode(Enum):
    """Processing mode for edit pipeline stages."""

    ALL_GPU = "all_gpu"
    COLOR_TRANSFORM_GPU = "color_transform_gpu"
    TRANSFORM_GPU = "transform_gpu"
    COLOR_GPU = "color_gpu"
    ALL_CPU = "all_cpu"

    @property
    def filter_on_cpu(self) -> bool:
        """Whether to run volume filtering on CPU (gspro)."""
        return self != ProcessingMode.ALL_GPU

    @property
    def color_on_cpu(self) -> bool:
        """Whether to run color adjustments on CPU (gspro)."""
        return self in (ProcessingMode.ALL_CPU, ProcessingMode.TRANSFORM_GPU)

    @property
    def transform_on_cpu(self) -> bool:
        """Whether to run scene transforms on CPU (gspro)."""
        return self in (ProcessingMode.ALL_CPU, ProcessingMode.COLOR_GPU)

    @property
    def transfer_count(self) -> int:
        """Expected number of CPU->GPU transfers per frame."""
        if self == ProcessingMode.ALL_GPU:
            return 0  # Data already on GPU from model loader
        else:
            return 1  # All CPU modes do exactly 1 batched CPU->GPU transfer

    @property
    def loader_mode(self) -> str:
        """
        Processing mode to use for model/data loading.

        Only the ALL_CPU edit profile benefits from forcing the loader to CPU;
        every other mode can keep the model in its fast all_gpu activation path.
        """
        return (
            ProcessingMode.ALL_CPU.value
            if self == ProcessingMode.ALL_CPU
            else ProcessingMode.ALL_GPU.value
        )

    @property
    def description(self) -> str:
        """Human-readable description of this mode."""
        descriptions = {
            ProcessingMode.ALL_GPU: "All stages on GPU (fastest, default)",
            ProcessingMode.COLOR_TRANSFORM_GPU: "Filter on CPU, color+transform on GPU (saves GPU memory)",
            ProcessingMode.TRANSFORM_GPU: "Filter+color on CPU, transform on GPU",
            ProcessingMode.COLOR_GPU: "Filter+transform on CPU, color on GPU",
            ProcessingMode.ALL_CPU: "All stages on CPU (max GPU memory savings)",
        }
        return descriptions[self]

    @classmethod
    def from_string(cls, value: str) -> "ProcessingMode":
        """Convert string to ProcessingMode enum.

        Supports both snake_case and display names.
        """
        # Try direct match (snake_case)
        value_lower = value.lower().replace(" ", "_").replace("+", "_")
        for mode in cls:
            if mode.value == value_lower:
                return mode

        # Try mapping from UI strings
        ui_to_mode = {
            "all gpu": cls.ALL_GPU,
            "color+transform gpu": cls.COLOR_TRANSFORM_GPU,
            "color transform gpu": cls.COLOR_TRANSFORM_GPU,
            "transform gpu": cls.TRANSFORM_GPU,
            "color gpu": cls.COLOR_GPU,
            "all cpu": cls.ALL_CPU,
        }

        if value.lower() in ui_to_mode:
            return ui_to_mode[value.lower()]

        raise ValueError(
            f"Unknown processing mode: {value}. "
            f"Valid options: {[m.value for m in cls]}"
        )

    def to_display_string(self) -> str:
        """Convert to UI display string."""
        display_map = {
            ProcessingMode.ALL_GPU: "All GPU",
            ProcessingMode.COLOR_TRANSFORM_GPU: "Color+Transform GPU",
            ProcessingMode.TRANSFORM_GPU: "Transform GPU",
            ProcessingMode.COLOR_GPU: "Color GPU",
            ProcessingMode.ALL_CPU: "All CPU",
        }
        return display_map[self]

    @classmethod
    def get_display_options(cls) -> list[str]:
        """Get list of display strings for UI dropdowns."""
        return [mode.to_display_string() for mode in cls]

    @classmethod
    def get_default_display(cls) -> str:
        """Get default mode display string."""
        return cls.ALL_GPU.to_display_string()
