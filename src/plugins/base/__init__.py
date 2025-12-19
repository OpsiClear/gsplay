"""Base classes and decorators for plugin development.

This module provides convenience classes that simplify plugin development:
- BaseDataSource: Base class with sensible defaults for data sources
- source_metadata: Decorator for defining source metadata

Example
-------
>>> from src.plugins.base import BaseDataSource, source_metadata
>>>
>>> @source_metadata(
...     name="My Format",
...     description="Load .myformat files",
...     file_extensions=[".myformat"],
... )
>>> class MySource(BaseDataSource):
...     def __init__(self, config: dict) -> None:
...         super().__init__(config)
...         self._files = self._discover_files()
...
...     @property
...     def total_frames(self) -> int:
...         return len(self._files)
...
...     def get_frame_at_time(self, normalized_time: float) -> GaussianData:
...         index = self._time_to_index(normalized_time)
...         return self._load_frame(self._files[index])
...
...     @classmethod
...     def can_load(cls, path: str) -> bool:
...         return path.endswith(".myformat")
"""

from src.plugins.base.data_source_base import BaseDataSource
from src.plugins.base.decorators import source_metadata

__all__ = ["BaseDataSource", "source_metadata"]
