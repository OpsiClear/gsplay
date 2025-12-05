"""FastAPI dependency injection for GSPlay Launcher.

This module provides proper dependency injection following the Dependency Inversion
Principle (DIP). Instead of using global state with setter functions, we use a
state container that gets configured at app startup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fastapi import Depends, Request

if TYPE_CHECKING:
    from gsplay_launcher.services.file_browser import FileBrowserService
    from gsplay_launcher.services.instance_manager import InstanceManager


@dataclass
class AppState:
    """Application state container for dependency injection.

    This replaces global mutable state with a proper state container
    that gets attached to the FastAPI app during lifespan.
    """

    instance_manager: InstanceManager | None = None
    file_browser: FileBrowserService | None = None


# State key for FastAPI app.state
STATE_KEY = "gsplay_state"


def get_app_state(request: Request) -> AppState:
    """Get application state from request."""
    state = getattr(request.app.state, STATE_KEY, None)
    if state is None:
        raise RuntimeError("Application state not initialized")
    return state


def get_instance_manager(
    state: AppState = Depends(get_app_state),
) -> "InstanceManager":
    """Get instance manager from application state."""
    if state.instance_manager is None:
        raise RuntimeError("Instance manager not initialized")
    return state.instance_manager


def get_file_browser(
    state: AppState = Depends(get_app_state),
) -> "FileBrowserService | None":
    """Get file browser service (may be None if disabled)."""
    return state.file_browser


def get_external_url(
    manager: "InstanceManager" = Depends(get_instance_manager),
) -> str | None:
    """Get external URL from manager config."""
    return manager.config.external_url


def get_network_url(
    manager: "InstanceManager" = Depends(get_instance_manager),
) -> str | None:
    """Get network URL from manager config."""
    return manager.config.network_url
