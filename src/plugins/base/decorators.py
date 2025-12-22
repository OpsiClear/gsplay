"""Decorators for plugin development.

Provides convenient decorators for defining plugin metadata.
"""

from __future__ import annotations

from typing import TypeVar

from src.domain.interfaces import SourceMetadata


T = TypeVar("T")


def source_metadata(
    name: str,
    description: str,
    file_extensions: list[str] | None = None,
    config_schema: type | None = None,
    supports_streaming: bool = True,
    supports_seeking: bool = True,
    version: str = "1.0.0",
) -> callable:
    """Decorator to define metadata for a data source class.

    This decorator sets up the `metadata()` classmethod automatically,
    simplifying plugin development.

    Example
    -------
    >>> @source_metadata(
    ...     name="My Format",
    ...     description="Load .myformat files",
    ...     file_extensions=[".myformat"],
    ...     version="1.0.0",
    ... )
    ... class MySource(BaseDataSource):
    ...     ...

    Parameters
    ----------
    name : str
        Display name for the source
    description : str
        Brief description of what this source does
    file_extensions : list[str] | None
        File extensions this source can handle
    config_schema : type | None
        Dataclass for config validation
    supports_streaming : bool
        Whether frames can be loaded on-demand
    supports_seeking : bool
        Whether random frame access is supported
    version : str
        Plugin version string
    """

    def decorator(cls: type[T]) -> type[T]:
        # Create the metadata object
        metadata = SourceMetadata(
            name=name,
            description=description,
            file_extensions=file_extensions or [],
            config_schema=config_schema,
            supports_streaming=supports_streaming,
            supports_seeking=supports_seeking,
            version=version,
        )

        # Store on class attribute for BaseDataSource.metadata() to find
        cls._source_metadata = metadata

        # Also define the classmethod directly for protocol compliance
        original_metadata = getattr(cls, "metadata", None)
        if original_metadata is None or not callable(original_metadata):

            @classmethod
            def metadata_method(cls_inner: type) -> SourceMetadata:
                return metadata

            cls.metadata = metadata_method

        return cls

    return decorator


__all__ = ["source_metadata"]
