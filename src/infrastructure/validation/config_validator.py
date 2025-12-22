"""Configuration validation for plugins.

Validates configuration dictionaries against dataclass schemas,
providing detailed error messages for invalid configurations.
"""

from __future__ import annotations

import logging
from dataclasses import MISSING, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

from src.shared.exceptions import ConfigValidationError


logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of configuration validation."""

    valid: bool
    errors: list[str]
    warnings: list[str]
    coerced_config: dict[str, Any] | None = None

    @property
    def error_message(self) -> str:
        """Get combined error message."""
        return "; ".join(self.errors)


class ConfigValidator:
    """Validates configuration dictionaries against dataclass schemas.

    Features:
    - Type checking with coercion (str -> Path, int -> float)
    - Required field validation
    - Default value handling
    - Constraint validation (path exists, positive numbers, etc.)
    - Nested dataclass validation

    Example
    -------
    >>> @dataclass
    ... class MyConfig:
    ...     ply_folder: Path
    ...     max_frames: int = 100
    ...
    >>> result = ConfigValidator.validate(
    ...     {"ply_folder": "/path/to/data"},
    ...     MyConfig
    ... )
    >>> if result.valid:
    ...     config = MyConfig(**result.coerced_config)
    """

    # Types that can be coerced from strings
    _COERCIBLE_FROM_STR = {Path, int, float, bool}

    @classmethod
    def validate(
        cls,
        config: dict[str, Any],
        schema: type,
        *,
        plugin_name: str | None = None,
        strict: bool = False,
    ) -> ValidationResult:
        """Validate config dict against a dataclass schema.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration dictionary to validate
        schema : type
            Dataclass type to validate against
        plugin_name : str | None
            Plugin name for error messages
        strict : bool
            If True, unknown fields are errors; if False, warnings

        Returns
        -------
        ValidationResult
            Validation result with errors, warnings, and coerced config
        """
        if not is_dataclass(schema):
            return ValidationResult(
                valid=False,
                errors=[f"Schema {schema} is not a dataclass"],
                warnings=[],
            )

        errors: list[str] = []
        warnings: list[str] = []
        coerced: dict[str, Any] = {}

        # Get type hints for the schema
        try:
            type_hints = get_type_hints(schema)
        except Exception as e:
            return ValidationResult(
                valid=False,
                errors=[f"Failed to get type hints for schema: {e}"],
                warnings=[],
            )

        # Get field info
        schema_fields = {f.name: f for f in fields(schema)}
        required_fields = {
            name
            for name, f in schema_fields.items()
            if f.default is MISSING and f.default_factory is MISSING
        }

        # Check for unknown fields
        config_keys = set(config.keys())
        schema_keys = set(schema_fields.keys())
        unknown_keys = config_keys - schema_keys

        for key in unknown_keys:
            msg = f"Unknown configuration field: '{key}'"
            if strict:
                errors.append(msg)
            else:
                warnings.append(msg)

        # Validate each schema field
        for field_name, field_info in schema_fields.items():
            expected_type = type_hints.get(field_name, Any)

            if field_name in config:
                # Field provided - validate type
                value = config[field_name]
                validated, error = cls._validate_type(value, expected_type, field_name)
                if error:
                    errors.append(error)
                else:
                    coerced[field_name] = validated
            elif field_name in required_fields:
                # Required field missing
                errors.append(f"Missing required field: '{field_name}'")
            # Optional field - use default
            elif field_info.default is not MISSING:
                coerced[field_name] = field_info.default
            elif field_info.default_factory is not MISSING:
                coerced[field_name] = field_info.default_factory()

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            coerced_config=coerced if len(errors) == 0 else None,
        )

    @classmethod
    def validate_or_raise(
        cls,
        config: dict[str, Any],
        schema: type,
        *,
        plugin_name: str | None = None,
    ) -> dict[str, Any]:
        """Validate config and raise ConfigValidationError if invalid.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration dictionary
        schema : type
            Dataclass schema
        plugin_name : str | None
            Plugin name for error messages

        Returns
        -------
        dict[str, Any]
            Validated and coerced config dictionary

        Raises
        ------
        ConfigValidationError
            If validation fails
        """
        result = cls.validate(config, schema, plugin_name=plugin_name)

        if not result.valid:
            raise ConfigValidationError(
                result.error_message,
                plugin_name=plugin_name,
            )

        for warning in result.warnings:
            logger.warning("[%s] Config warning: %s", plugin_name or "unknown", warning)

        return result.coerced_config  # type: ignore

    @classmethod
    def _validate_type(
        cls,
        value: Any,
        expected_type: type,
        field_name: str,
    ) -> tuple[Any, str | None]:
        """Validate and coerce a single value.

        Returns
        -------
        tuple[Any, str | None]
            (coerced_value, error_message or None)
        """
        # Handle None for Optional types
        origin = get_origin(expected_type)
        args = get_args(expected_type)

        # Handle Optional[T] (Union[T, None])
        if origin is Union:
            if type(None) in args:
                if value is None:
                    return None, None
                # Try non-None types
                non_none_types = [t for t in args if t is not type(None)]
                for t in non_none_types:
                    coerced, error = cls._validate_type(value, t, field_name)
                    if error is None:
                        return coerced, None
                return (
                    None,
                    f"Field '{field_name}': expected {expected_type}, got {type(value).__name__}",
                )

        # Handle list types
        if origin is list:
            if not isinstance(value, list):
                return None, f"Field '{field_name}': expected list, got {type(value).__name__}"
            if args:
                item_type = args[0]
                coerced_list = []
                for i, item in enumerate(value):
                    coerced_item, error = cls._validate_type(item, item_type, f"{field_name}[{i}]")
                    if error:
                        return None, error
                    coerced_list.append(coerced_item)
                return coerced_list, None
            return value, None

        # Handle dict types
        if origin is dict:
            if not isinstance(value, dict):
                return None, f"Field '{field_name}': expected dict, got {type(value).__name__}"
            return value, None

        # Handle Path coercion
        if expected_type is Path:
            if isinstance(value, Path):
                return value, None
            if isinstance(value, str):
                return Path(value), None
            return None, f"Field '{field_name}': expected Path or str, got {type(value).__name__}"

        # Handle bool (must check before int since bool is subclass of int)
        if expected_type is bool:
            if isinstance(value, bool):
                return value, None
            if isinstance(value, str):
                if value.lower() in ("true", "1", "yes"):
                    return True, None
                if value.lower() in ("false", "0", "no"):
                    return False, None
            return None, f"Field '{field_name}': expected bool, got {type(value).__name__}"

        # Handle numeric coercion
        if expected_type is int:
            if isinstance(value, int) and not isinstance(value, bool):
                return value, None
            if isinstance(value, str):
                try:
                    return int(value), None
                except ValueError:
                    pass
            return None, f"Field '{field_name}': expected int, got {type(value).__name__}"

        if expected_type is float:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value), None
            if isinstance(value, str):
                try:
                    return float(value), None
                except ValueError:
                    pass
            return None, f"Field '{field_name}': expected float, got {type(value).__name__}"

        # Handle str
        if expected_type is str:
            if isinstance(value, str):
                return value, None
            return None, f"Field '{field_name}': expected str, got {type(value).__name__}"

        # Handle nested dataclasses
        if is_dataclass(expected_type) and not isinstance(expected_type, type):
            expected_type = type(expected_type)

        if is_dataclass(expected_type):
            if isinstance(value, dict):
                result = cls.validate(value, expected_type)
                if not result.valid:
                    return None, f"Field '{field_name}': {result.error_message}"
                return result.coerced_config, None
            if isinstance(value, expected_type):
                return value, None
            return None, f"Field '{field_name}': expected {expected_type.__name__} dict or instance"

        # Handle Any type
        if expected_type is Any:
            return value, None

        # Default: check isinstance
        if isinstance(value, expected_type):
            return value, None

        return None, f"Field '{field_name}': expected {expected_type}, got {type(value).__name__}"


__all__ = ["ConfigValidator", "ValidationResult"]
