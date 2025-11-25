"""
Data format validation for Gaussian splatting data.

This module provides validation functions to ensure data is in the correct
format before processing, catching errors early and providing helpful messages.
"""
import logging
import torch
import numpy as np

from src.infrastructure.processing.gaussian_constants import GaussianConstants as GC
from src.domain.entities import GSTensor

logger = logging.getLogger(__name__)


class DataFormatValidator:
    """Validator for Gaussian splatting data formats."""

    @staticmethod
    def validate_gaussian_data(
        data: GSTensor,
        check_activations: bool = True
    ) -> tuple[bool, list[str]]:
        """
        Validate a GSTensor object for correctness.

        Parameters
        ----------
        data : GSTensor
            The Gaussian data to validate
        check_activations : bool
            If True, check that data is in physical/linear space (post-activation)

        Returns
        -------
        tuple[bool, list[str]]
            (is_valid, list_of_error_messages)
        """
        errors = []

        # Check required fields
        if data.means is None:
            errors.append("Missing means (positions)")
        elif data.means.ndim != 2 or data.means.shape[1] != 3:
            errors.append(f"Invalid means shape: {data.means.shape}, expected (N, 3)")

        if data.scales is None:
            errors.append("Missing scales")
        elif data.scales.ndim != 2 or data.scales.shape[1] != 3:
            errors.append(f"Invalid scales shape: {data.scales.shape}, expected (N, 3)")

        if data.quats is None:
            errors.append("Missing quats (rotations)")
        elif data.quats.ndim != 2 or data.quats.shape[1] != 4:
            errors.append(f"Invalid quats shape: {data.quats.shape}, expected (N, 4)")

        if data.opacities is None:
            errors.append("Missing opacities")
        elif data.opacities.ndim != 1:
            errors.append(f"Invalid opacities shape: {data.opacities.shape}, expected (N,)")

        if data.sh0 is None:
            errors.append("Missing colors")
        elif data.sh0.ndim not in [2, 3]:
            errors.append(f"Invalid colors shape: {data.sh0.shape}, expected (N, 3) or (N, K, 3)")

        # Check consistent sizes
        if not errors:
            n_gaussians = data.means.shape[0]
            if data.scales.shape[0] != n_gaussians:
                errors.append(f"Scales count mismatch: {data.scales.shape[0]} != {n_gaussians}")
            if data.quats.shape[0] != n_gaussians:
                errors.append(f"Quats count mismatch: {data.quats.shape[0]} != {n_gaussians}")
            if data.opacities.shape[0] != n_gaussians:
                errors.append(f"Opacities count mismatch: {data.opacities.shape[0]} != {n_gaussians}")
            if data.sh0.shape[0] != n_gaussians:
                errors.append(f"Colors count mismatch: {data.sh0.shape[0]} != {n_gaussians}")

        # Check activation status if requested
        if check_activations and not errors:
            # Check scales are in physical space (positive)
            if (data.scales < 0).any():
                errors.append("Scales contain negative values - likely in log space, need exp()")

            # Check opacities are in linear space [0, 1]
            op_min = data.opacities.min().item()
            op_max = data.opacities.max().item()
            if op_min < -0.1 or op_max > 1.1:
                errors.append(
                    f"Opacities out of [0,1] range ({op_min:.2f}, {op_max:.2f}) - "
                    f"likely in logit space, need sigmoid()"
                )

            # Check quaternions are normalized
            quat_norms = torch.norm(data.quats, dim=1)
            if (torch.abs(quat_norms - 1.0) > 0.1).any():
                errors.append("Quaternions not normalized - need F.normalize()")

        return len(errors) == 0, errors

    @staticmethod
    def validate_scale_format(
        scales: torch.Tensor | np.ndarray
    ) -> str:
        """
        Detect whether scales are in log or linear space.

        Parameters
        ----------
        scales : torch.Tensor | np.ndarray
            Scale values to check

        Returns
        -------
        str
            "log" if scales appear to be in log space
            "linear" if scales appear to be in linear space
            "unknown" if unclear
        """
        if torch.is_tensor(scales):
            min_val = scales.min().item()
            max_val = scales.max().item()
        else:
            min_val = scales.min()
            max_val = scales.max()

        # Check for log space indicators
        if min_val < GC.Format.LOG_SCALE_THRESHOLD:
            return "log"

        # Check for linear space indicators
        if min_val >= 0 and max_val < 100:
            return "linear"

        return "unknown"

    @staticmethod
    def validate_opacity_format(
        opacities: torch.Tensor | np.ndarray
    ) -> str:
        """
        Detect whether opacities are in logit or linear space.

        Parameters
        ----------
        opacities : torch.Tensor | np.ndarray
            Opacity values to check

        Returns
        -------
        str
            "logit" if opacities appear to be in logit space
            "linear" if opacities appear to be in linear [0,1] space
            "unknown" if unclear
        """
        if torch.is_tensor(opacities):
            min_val = opacities.min().item()
            max_val = opacities.max().item()
        else:
            min_val = opacities.min()
            max_val = opacities.max()

        # Check if in [0, 1] range (with small tolerance)
        if min_val >= -0.01 and max_val <= 1.01:
            return "linear"

        # Outside [0, 1] suggests logit space
        if min_val < -0.1 or max_val > 1.1:
            return "logit"

        return "unknown"

    @staticmethod
    def validate_sh_format(
        sh_data: torch.Tensor | np.ndarray,
        expected_bands: int | None = None
    ) -> tuple[bool, str]:
        """
        Validate spherical harmonics data format.

        Parameters
        ----------
        sh_data : torch.Tensor | np.ndarray
            SH coefficients to validate
        expected_bands : int | None
            Expected number of SH bands (degree + 1)^2

        Returns
        -------
        tuple[bool, str]
            (is_valid, error_message_or_info)
        """
        if sh_data.ndim == 2:
            # Shape (N, 3) for DC only
            if sh_data.shape[1] == 3:
                return True, "DC coefficients only (degree 0)"
            # Shape (N, K*3) for higher order
            if sh_data.shape[1] % 3 == 0:
                num_coeffs = sh_data.shape[1] // 3
                degree = int(np.sqrt(num_coeffs) - 1)
                if (degree + 1) ** 2 == num_coeffs:
                    if expected_bands and num_coeffs != expected_bands:
                        return False, f"Expected {expected_bands} bands, got {num_coeffs}"
                    return True, f"SH degree {degree} ({num_coeffs} bands)"
                return False, f"Invalid SH coefficient count: {num_coeffs}"
            return False, f"SH coefficients not divisible by 3: {sh_data.shape[1]}"
        elif sh_data.ndim == 3:
            # Shape (N, K, 3)
            num_coeffs = sh_data.shape[1]
            degree = int(np.sqrt(num_coeffs) - 1)
            if (degree + 1) ** 2 == num_coeffs:
                if expected_bands and num_coeffs != expected_bands:
                    return False, f"Expected {expected_bands} bands, got {num_coeffs}"
                return True, f"SH degree {degree} ({num_coeffs} bands)"
            return False, f"Invalid SH coefficient count: {num_coeffs}"
        else:
            return False, f"Invalid SH shape: {sh_data.shape}"

    @staticmethod
    def auto_fix_formats(
        data: GSTensor,
        apply_activations: bool = True
    ) -> GSTensor:
        """
        Attempt to automatically fix common format issues.

        Parameters
        ----------
        data : GSTensor
            Data to fix
        apply_activations : bool
            If True, use gsply v0.2.5 denormalize() to convert PLY format (log/logit) to linear

        Returns
        -------
        GSTensor
            Fixed data (may be same object if no fixes needed)
        """
        modified = False

        # Check and fix scales/opacities using gsply v0.2.5 denormalize() method
        if apply_activations:
            scale_format = DataFormatValidator.validate_scale_format(data.scales)
            opacity_format = DataFormatValidator.validate_opacity_format(data.opacities)
            
            if scale_format == "log" or opacity_format == "logit":
                logger.info(
                    f"Auto-fixing: Converting PLY format to linear using gsply.denormalize() "
                    f"(scales: {scale_format}, opacities: {opacity_format})"
                )
                # Use inplace=True for better performance
                data = data.denormalize(inplace=True)
                # Clamp scales to valid range after denormalization (in-place)
                data.scales = data.scales.clamp(
                    min=GC.Numerical.MIN_SCALE,
                    max=GC.Numerical.MAX_SCALE
                )
                modified = True

        # Normalize quaternions
        quat_norms = torch.norm(data.quats, dim=1)
        if (torch.abs(quat_norms - 1.0) > 0.01).any():
            logger.info("Auto-fixing: Normalizing quaternions")
            data.quats = torch.nn.functional.normalize(data.quats, p=2, dim=1)
            modified = True

        if modified:
            logger.info("Applied automatic format fixes to GSTensor")

        return data


def validate_before_export(
    gaussian_data: GSTensor,
    target_format: str = "ply"
) -> None:
    """
    Validate data before exporting to file.

    Parameters
    ----------
    gaussian_data : GSTensor
        Data to validate
    target_format : str
        Target export format ("ply", "compressed", etc.)

    Raises
    ------
    ValueError
        If data is invalid for export
    """
    validator = DataFormatValidator()

    # Validate structure and activation status
    is_valid, errors = validator.validate_gaussian_data(
        gaussian_data,
        check_activations=True
    )

    if not is_valid:
        error_msg = f"Cannot export to {target_format}, data validation failed:\n"
        error_msg += "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)

    # Check specific format requirements
    if target_format in ["ply", "compressed"]:
        # PLY expects physical scales and linear opacities
        scale_format = validator.validate_scale_format(gaussian_data.scales)
        if scale_format == "log":
            raise ValueError(
                "PLY export requires physical scales (positive values), "
                "but data contains log scales. Apply torch.exp() first."
            )

        opacity_format = validator.validate_opacity_format(gaussian_data.opacities)
        if opacity_format == "logit":
            raise ValueError(
                "PLY export requires linear opacities [0,1], "
                "but data contains logit opacities. Apply torch.sigmoid() first."
            )

    logger.debug(f"Data validation passed for {target_format} export")


# Export convenience functions
__all__ = [
    "DataFormatValidator",
    "validate_before_export",
]
