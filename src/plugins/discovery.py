"""Plugin discovery utilities.

Provides functions for discovering plugins from:
- Entry points in pyproject.toml
- Directory scanning
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points
from pathlib import Path
from typing import Type

from src.domain.interfaces import BaseGaussianSource

logger = logging.getLogger(__name__)


def discover_plugins(group: str = "gsplay.plugins") -> dict[str, Type[BaseGaussianSource]]:
    """Discover plugins from entry_points.

    Parameters
    ----------
    group : str
        Entry point group to search

    Returns
    -------
    dict[str, Type[BaseGaussianSource]]
        Map of plugin name to class

    Example
    -------
    In pyproject.toml:
    ```toml
    [project.entry-points."gsplay.plugins"]
    load-ply = "src.models.ply.optimized_model:OptimizedPlyModel"
    my-format = "my_plugin:MyFormatSource"
    ```

    >>> plugins = discover_plugins()
    >>> print(plugins.keys())
    dict_keys(['load-ply', 'my-format'])
    """
    discovered: dict[str, Type[BaseGaussianSource]] = {}

    try:
        eps = entry_points(group=group)
    except TypeError:
        # Python < 3.10 compatibility
        all_eps = entry_points()
        eps = all_eps.get(group, [])

    for ep in eps:
        try:
            plugin_class = ep.load()
            discovered[ep.name] = plugin_class
            logger.debug("Discovered plugin: %s -> %s", ep.name, plugin_class)
        except Exception as e:
            logger.warning("Failed to load plugin '%s': %s", ep.name, e)

    return discovered


def discover_sinks(group: str = "gsplay.sinks") -> dict[str, type]:
    """Discover sink plugins from entry_points.

    Parameters
    ----------
    group : str
        Entry point group to search

    Returns
    -------
    dict[str, type]
        Map of sink name to class
    """
    discovered: dict[str, type] = {}

    try:
        eps = entry_points(group=group)
    except TypeError:
        all_eps = entry_points()
        eps = all_eps.get(group, [])

    for ep in eps:
        try:
            sink_class = ep.load()
            discovered[ep.name] = sink_class
            logger.debug("Discovered sink: %s -> %s", ep.name, sink_class)
        except Exception as e:
            logger.warning("Failed to load sink '%s': %s", ep.name, e)

    return discovered


def discover_from_directory(
    directory: str | Path,
    pattern: str = "*.py",
) -> list[str]:
    """Discover potential plugin modules from a directory.

    This function finds Python files that might contain plugins.
    Useful for development/testing without installing packages.

    Parameters
    ----------
    directory : str | Path
        Directory to search
    pattern : str
        Glob pattern for files

    Returns
    -------
    list[str]
        List of module paths found
    """
    directory = Path(directory)
    if not directory.is_dir():
        logger.warning("Plugin directory not found: %s", directory)
        return []

    modules = []
    for file in directory.glob(pattern):
        if file.name.startswith("_"):
            continue
        module_name = file.stem
        modules.append(str(file))
        logger.debug("Found potential plugin module: %s", file)

    return modules


def get_plugin_info() -> dict[str, dict]:
    """Get information about all discovered plugins.

    Returns
    -------
    dict[str, dict]
        Map of plugin name to info dict containing:
        - class: The plugin class
        - metadata: SourceMetadata if available
        - module: Module name
    """
    info: dict[str, dict] = {}

    plugins = discover_plugins()
    for name, cls in plugins.items():
        plugin_info = {
            "class": cls.__name__,
            "module": cls.__module__,
        }

        try:
            if hasattr(cls, "metadata"):
                meta = cls.metadata()
                plugin_info["metadata"] = {
                    "name": meta.name,
                    "description": meta.description,
                    "file_extensions": meta.file_extensions,
                    "version": meta.version,
                }
        except Exception as e:
            plugin_info["metadata_error"] = str(e)

        info[name] = plugin_info

    return info


__all__ = [
    "discover_plugins",
    "discover_sinks",
    "discover_from_directory",
    "get_plugin_info",
]
