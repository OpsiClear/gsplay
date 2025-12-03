"""
Factory for creating exporter instances based on format.

Provides centralized registration and creation of exporters,
enabling configuration-driven format selection with capabilities.

Now integrates with DataSinkRegistry for unified sink discovery.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING
from enum import Enum, auto
from dataclasses import dataclass

from src.domain.interfaces import ExporterInterface
from src.infrastructure.exporters.ply_exporter import PlyExporter
from src.infrastructure.exporters.compressed_ply_exporter import CompressedPlyExporter

if TYPE_CHECKING:
    from src.domain.interfaces import DataSinkProtocol

logger = logging.getLogger(__name__)


def _ensure_registry_initialized() -> None:
    """Ensure the data sink registry is initialized with defaults."""
    from src.infrastructure.registry import register_defaults
    register_defaults()


class ExportCapability(Enum):
    """Capabilities that exporters can support."""
    COMPRESSION = auto()
    STREAMING = auto()
    CLOUD_STORAGE = auto()
    BATCH_EXPORT = auto()
    SH_COEFFICIENTS = auto()
    CUSTOM_PROPERTIES = auto()


class ExportFormat(Enum):
    """Standard export formats."""
    PLY = "ply"
    COMPRESSED_PLY = "compressed-ply"
    # Future formats can be added here
    # GLTF = "gltf"
    # OBJ = "obj"


@dataclass
class ExporterInfo:
    """Information about a registered exporter."""
    format: str
    exporter_class: type[ExporterInterface]
    capabilities: set[ExportCapability]
    description: str
    file_extension: str


class ExporterFactory:
    """
    Factory for creating exporter instances with capability discovery.

    Supports registration of custom exporters and provides
    type-safe exporter creation based on format string or enum.
    """

    _exporters: dict[str, ExporterInfo] = {}

    # Register built-in exporters
    @classmethod
    def _register_builtin_exporters(cls) -> None:
        """Register built-in exporters."""
        if not cls._exporters:  # Only register once
            cls.register(
                "ply",
                PlyExporter,
                capabilities={ExportCapability.CLOUD_STORAGE, ExportCapability.SH_COEFFICIENTS},
                description="Standard PLY format",
                file_extension=".ply"
            )
            cls.register(
                "compressed-ply",
                CompressedPlyExporter,
                capabilities={
                    ExportCapability.COMPRESSION,
                    ExportCapability.CLOUD_STORAGE,
                    ExportCapability.SH_COEFFICIENTS
                },
                description="Compressed PLY format (16 bytes/splat)",
                file_extension=".compressed.ply"
            )

    @classmethod
    def create(cls, format: str | ExportFormat, **config: Any) -> ExporterInterface:
        """
        Create exporter instance for specified format.

        Parameters
        ----------
        format : str | ExportFormat
            Export format (string or enum)
        **config : Any
            Configuration options passed to exporter constructor

        Returns
        -------
        ExporterInterface
            Configured exporter instance

        Raises
        ------
        ValueError
            If format is not registered

        Examples
        --------
        >>> exporter = ExporterFactory.create("ply")
        >>> exporter = ExporterFactory.create(ExportFormat.COMPRESSED_PLY)
        """
        cls._register_builtin_exporters()

        # Handle enum
        if isinstance(format, ExportFormat):
            format = format.value

        format_lower = format.lower()

        if format_lower not in cls._exporters:
            available = ", ".join(cls._exporters.keys())
            raise ValueError(
                f"Unknown export format: '{format}'. "
                f"Available formats: {available}"
            )

        exporter_info = cls._exporters[format_lower]
        exporter_class = exporter_info.exporter_class
        logger.debug(f"Creating {format_lower.upper()} exporter with config: {config}")

        return exporter_class(**config)

    @classmethod
    def register(
        cls,
        format: str,
        exporter_class: type[ExporterInterface],
        capabilities: set[ExportCapability] | None = None,
        description: str = "",
        file_extension: str = ""
    ) -> None:
        """
        Register custom exporter for a format.

        Parameters
        ----------
        format : str
            Format name (e.g., "gltf", "binary")
        exporter_class : type[ExporterInterface]
            Exporter class implementing ExporterInterface
        capabilities : set[ExportCapability] | None
            Set of capabilities this exporter supports
        description : str
            Human-readable description
        file_extension : str
            File extension (e.g., ".ply")

        Examples
        --------
        >>> ExporterFactory.register(
        ...     "gltf",
        ...     GltfExporter,
        ...     capabilities={ExportCapability.COMPRESSION},
        ...     description="GLTF format",
        ...     file_extension=".gltf"
        ... )
        """
        format_lower = format.lower()
        logger.info(f"Registering exporter for format: {format_lower}")

        exporter_info = ExporterInfo(
            format=format_lower,
            exporter_class=exporter_class,
            capabilities=capabilities or set(),
            description=description or f"{format_lower.upper()} exporter",
            file_extension=file_extension or f".{format_lower}"
        )

        cls._exporters[format_lower] = exporter_info

    @classmethod
    def get_available_formats(cls) -> list[str]:
        """
        Get list of available export formats.

        Returns
        -------
        list[str]
            Sorted list of registered format names
        """
        cls._register_builtin_exporters()
        return sorted(cls._exporters.keys())

    @classmethod
    def supports_format(cls, format: str | ExportFormat) -> bool:
        """
        Check if format is supported.

        Parameters
        ----------
        format : str | ExportFormat
            Format name to check

        Returns
        -------
        bool
            True if format is registered
        """
        cls._register_builtin_exporters()

        if isinstance(format, ExportFormat):
            format = format.value

        return format.lower() in cls._exporters

    @classmethod
    def get_formats_with_capability(cls, capability: ExportCapability) -> list[str]:
        """
        Get formats that support a specific capability.

        Parameters
        ----------
        capability : ExportCapability
            Capability to query

        Returns
        -------
        list[str]
            List of format names supporting this capability

        Examples
        --------
        >>> compressed_formats = ExporterFactory.get_formats_with_capability(
        ...     ExportCapability.COMPRESSION
        ... )
        """
        cls._register_builtin_exporters()

        return [
            fmt
            for fmt, info in cls._exporters.items()
            if capability in info.capabilities
        ]

    @classmethod
    def get_exporter_info(cls, format: str | ExportFormat) -> ExporterInfo | None:
        """
        Get detailed information about an exporter.

        Parameters
        ----------
        format : str | ExportFormat
            Format to query

        Returns
        -------
        ExporterInfo | None
            Exporter information, or None if not found
        """
        cls._register_builtin_exporters()

        if isinstance(format, ExportFormat):
            format = format.value

        return cls._exporters.get(format.lower())

    @classmethod
    def has_capability(
        cls,
        format: str | ExportFormat,
        capability: ExportCapability
    ) -> bool:
        """
        Check if a format supports a specific capability.

        Parameters
        ----------
        format : str | ExportFormat
            Format to check
        capability : ExportCapability
            Capability to query

        Returns
        -------
        bool
            True if format supports capability
        """
        info = cls.get_exporter_info(format)
        return info is not None and capability in info.capabilities

    @classmethod
    def get_exporter(cls, format: str | ExportFormat, **config: Any) -> ExporterInterface:
        """
        Alias for create() for backward compatibility.

        Parameters
        ----------
        format : str | ExportFormat
            Export format
        **config : Any
            Configuration options

        Returns
        -------
        ExporterInterface
            Configured exporter instance
        """
        return cls.create(format, **config)

    # =========================================================================
    # DataSinkRegistry Integration (New API)
    # =========================================================================

    @classmethod
    def get_sink(cls, format_name: str) -> DataSinkProtocol:
        """Get a data sink from the registry.

        This is the new API that returns DataSinkProtocol implementations.

        Parameters
        ----------
        format_name : str
            Sink identifier (e.g., "ply", "compressed-ply")

        Returns
        -------
        DataSinkProtocol
            Sink instance

        Raises
        ------
        ValueError
            If format is not registered
        """
        _ensure_registry_initialized()

        from src.infrastructure.registry import DataSinkRegistry
        sink_class = DataSinkRegistry.get(format_name)

        if sink_class is None:
            available = DataSinkRegistry.names()
            raise ValueError(
                f"Unknown sink format: '{format_name}'. "
                f"Available: {', '.join(available)}"
            )

        return sink_class()

    @classmethod
    def list_sink_formats(cls) -> list[str]:
        """List available sink formats from registry.

        Returns
        -------
        list[str]
            List of registered sink names
        """
        _ensure_registry_initialized()

        from src.infrastructure.registry import DataSinkRegistry
        return DataSinkRegistry.names()

    @classmethod
    def get_sink_metadata(cls) -> list:
        """Get metadata for all registered sinks.

        Returns
        -------
        list[DataSinkMetadata]
            List of sink metadata
        """
        _ensure_registry_initialized()

        from src.infrastructure.registry import DataSinkRegistry
        return DataSinkRegistry.list_all()
