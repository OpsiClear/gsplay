"""Event dispatcher for centralized event routing.

This module extracts event subscription logic from the main app,
providing a clean interface for registering event handlers.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

from src.gsplay.interaction.events import EventBus, EventType, Event

if TYPE_CHECKING:
    from src.gsplay.core.app import UniversalGSPlay

logger = logging.getLogger(__name__)


class EventDispatcher:
    """Centralized event subscription and routing.

    Handles registration of all event handlers for the viewer application.
    Provides a single place to see all event subscriptions.
    """

    def __init__(self, event_bus: EventBus):
        """Initialize dispatcher with event bus.

        Parameters
        ----------
        event_bus : EventBus
            Event bus instance for subscriptions
        """
        self.event_bus = event_bus
        self._subscriptions: list[tuple[EventType, Callable]] = []
        logger.debug("EventDispatcher initialized")

    def register_app_handlers(self, app: "UniversalGSPlay") -> None:
        """Register all application event handlers.

        Parameters
        ----------
        app : UniversalGSPlay
            The viewer application instance
        """
        handlers = [
            # Model events
            (EventType.MODEL_LOADED, app._on_model_loaded),
            (EventType.FRAME_CHANGED, app._on_frame_changed),

            # Command events
            (EventType.LOAD_DATA_REQUESTED, app._on_load_data_requested),
            (EventType.EXPORT_REQUESTED, app._on_export_requested),
            (EventType.RESET_COLORS_REQUESTED, app._on_reset_colors_requested),
            (EventType.RESET_TRANSFORM_REQUESTED, app._on_reset_transform_requested),
            (EventType.RESET_FILTER_REQUESTED, app._on_reset_filter_requested),
            (EventType.CENTER_REQUESTED, app._on_center_requested),
            (EventType.RERENDER_REQUESTED, app._on_rerender_requested),
            (EventType.TERMINATE_REQUESTED, app._on_terminate_requested),
        ]

        for event_type, handler in handlers:
            self.event_bus.subscribe(event_type, handler)
            self._subscriptions.append((event_type, handler))

        logger.info(f"Registered {len(handlers)} event handlers")

    def unregister_all(self) -> None:
        """Unregister all previously registered handlers."""
        for event_type, handler in self._subscriptions:
            self.event_bus.unsubscribe(event_type, handler)

        count = len(self._subscriptions)
        self._subscriptions.clear()
        logger.info(f"Unregistered {count} event handlers")

    def get_subscription_count(self) -> int:
        """Get number of active subscriptions."""
        return len(self._subscriptions)
