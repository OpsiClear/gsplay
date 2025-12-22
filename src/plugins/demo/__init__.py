"""
Demo Plugin Package - Reference Implementation for Plugin Development.

This package provides example implementations of the plugin system to help
developers understand how to create custom loaders (DataSource) and exporters
(DataSink) for the GSPlay.

PLUGIN ARCHITECTURE OVERVIEW
============================

The GSPlay uses a registry-based plugin system with two main
extension points:

1. **DataSource (Loader)** - Loads Gaussian data from files/sources
   - Protocol: DataSourceProtocol
   - Registry: DataSourceRegistry
   - Output: GaussianData

2. **DataSink (Exporter)** - Saves Gaussian data to files/formats
   - Protocol: DataSinkProtocol
   - Registry: DataSinkRegistry
   - Input: GaussianData


DATA FLOW
=========

    [Files] --> DataSource --> GaussianData --> [GSPlay/Processing]
                                    |
                                    v
                               DataSink --> [Output Files]

The key abstraction is `GaussianData` - a unified data container that:
- Supports both CPU (numpy) and GPU (torch) data with lazy conversion
- Tracks format information (scales in log-space, opacities in logit-space, etc.)
- Provides conversion to/from gsply containers (GSData, GSTensor)


QUICK START
===========

1. Create a DataSource (loader):

    >>> from src.domain.interfaces import DataSourceProtocol, DataSourceMetadata
    >>> from src.domain.data import GaussianData
    >>>
    >>> class MySource(DataSourceProtocol):
    ...     @classmethod
    ...     def metadata(cls) -> DataSourceMetadata:
    ...         return DataSourceMetadata(
    ...             name="My Format",
    ...             description="Loads .myformat files",
    ...             file_extensions=[".myformat"],
    ...         )
    ...
    ...     @classmethod
    ...     def can_load(cls, path: str) -> bool:
    ...         return path.endswith(".myformat")
    ...
    ...     @property
    ...     def total_frames(self) -> int:
    ...         return self._frame_count
    ...
    ...     def get_frame(self, index: int) -> GaussianData:
    ...         # Load and return frame data
    ...         pass
    ...
    ...     def get_frame_at_time(self, normalized_time: float) -> GaussianData:
    ...         idx = int(normalized_time * (self.total_frames - 1))
    ...         return self.get_frame(idx)

2. Register the source:

    >>> from src.infrastructure.registry import DataSourceRegistry
    >>> DataSourceRegistry.register("my-format", MySource)

3. Use via config:

    {
        "module": "my-format",
        "config": {
            "path": "/path/to/data.myformat"
        }
    }


CONTENTS
========

- demo_source.py: Complete DataSource (loader) reference implementation
- demo_sink.py: Complete DataSink (exporter) reference implementation

See each module for detailed documentation and implementation guidance.
"""

from src.plugins.demo.demo_sink import DemoJsonSink
from src.plugins.demo.demo_source import DemoRandomSource, DemoRandomSourceConfig


__all__ = [
    "DemoJsonSink",
    "DemoRandomSource",
    "DemoRandomSourceConfig",
]
