"""
Event bus system for decoupling UI components from handlers.

This module provides a simple event bus pattern for loose coupling between
UI components and business logic handlers.
"""
import logging
from collections.abc import Callable
from typing import Any
from dataclasses import dataclass
from enum import Enum, auto

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Standard event types for the viewer."""

    # Color adjustment events
    BRIGHTNESS_CHANGED = auto()
    CONTRAST_CHANGED = auto()
    SATURATION_CHANGED = auto()
    GAMMA_CHANGED = auto()

    # Transform events
    SCALE_CHANGED = auto()
    ROTATION_CHANGED = auto()
    TRANSLATION_CHANGED = auto()

    # Volume filter events
    FILTER_TYPE_CHANGED = auto()
    FILTER_PARAMS_CHANGED = auto()
    MAX_SCALE_CHANGED = auto()

    # Playback events
    FRAME_CHANGED = auto()
    PLAY_PAUSE_TOGGLED = auto()
    FPS_CHANGED = auto()

    # Export events
    EXPORT_REQUESTED = auto()
    EXPORT_COMPLETED = auto()
    EXPORT_FAILED = auto()
    EXPORT_PROGRESS = auto()    # Progress during multi-frame export: {"current": int, "total": int, "source_time": float}
    EXPORT_CANCELLED = auto()   # Export was cancelled

    # Edit events
    EDIT_APPLIED = auto()
    EDIT_UNDONE = auto()
    EDIT_REDONE = auto()

    # Model events
    MODEL_LOAD_STARTED = auto()
    MODEL_LOADED = auto()
    MODEL_LOAD_FAILED = auto()
    MODEL_UNLOADED = auto()
    MODEL_CHANGED = auto()

    # GSPlay events
    VIEWER_CREATED = auto()
    VIEWER_DESTROYED = auto()

    # Render events
    RENDER_RESOLUTION_CHANGED = auto()
    RENDER_STATS_UPDATED = auto()

    # Camera events
    CAMERA_MOVED = auto()
    CAMERA_RESET = auto()

    # Command events (Requests)
    LOAD_DATA_REQUESTED = auto()
    RESET_COLORS_REQUESTED = auto()
    RESET_TRANSFORM_REQUESTED = auto()
    RESET_FILTER_REQUESTED = auto()
    CENTER_REQUESTED = auto()
    BAKE_VIEW_REQUESTED = auto()
    ALIGN_UP_REQUESTED = auto()
    RERENDER_REQUESTED = auto()
    TERMINATE_REQUESTED = auto()


@dataclass
class Event:
    """Event data container."""
    type: EventType | str
    data: dict[str, Any]
    source: str | None = None


class EventBus:
    """
    Simple event bus for pub/sub pattern.

    Allows components to emit events and subscribe to them without
    direct coupling.
    """

    def __init__(self, name: str = "default"):
        """
        Initialize event bus.

        Parameters
        ----------
        name : str
            Name of this event bus instance
        """
        self.name = name
        self._subscribers: dict[EventType | str, list[Callable]] = {}
        self._event_history: list[Event] = []
        self._max_history = 100
        logger.debug(f"Created EventBus: {name}")

    def subscribe(
        self,
        event_type: EventType | str,
        callback: Callable[[Event], None],
        priority: int = 0
    ) -> None:
        """
        Subscribe to an event type.

        Parameters
        ----------
        event_type : EventType | str
            Event type to subscribe to
        callback : Callable[[Event], None]
            Function to call when event is emitted
        priority : int
            Priority for callback execution (higher = earlier)
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []

        # Insert by priority (higher priority first)
        callbacks = self._subscribers[event_type]
        inserted = False
        for i, (existing_priority, _) in enumerate(callbacks):
            if priority > existing_priority:
                callbacks.insert(i, (priority, callback))
                inserted = True
                break

        if not inserted:
            callbacks.append((priority, callback))

        logger.debug(
            f"[{self.name}] Subscribed to {event_type}: "
            f"{callback.__name__} (priority={priority})"
        )

    def unsubscribe(
        self,
        event_type: EventType | str,
        callback: Callable[[Event], None]
    ) -> bool:
        """
        Unsubscribe from an event type.

        Parameters
        ----------
        event_type : EventType | str
            Event type to unsubscribe from
        callback : Callable[[Event], None]
            Callback to remove

        Returns
        -------
        bool
            True if callback was found and removed
        """
        if event_type not in self._subscribers:
            return False

        callbacks = self._subscribers[event_type]
        for i, (_, cb) in enumerate(callbacks):
            if cb == callback:
                del callbacks[i]
                logger.debug(
                    f"[{self.name}] Unsubscribed from {event_type}: "
                    f"{callback.__name__}"
                )
                return True

        return False

    def emit(
        self,
        event_type: EventType | str,
        source: str | None = None,
        **data
    ) -> None:
        """
        Emit an event.

        Parameters
        ----------
        event_type : EventType | str
            Type of event to emit
        source : str | None
            Component emitting the event
        **data
            Event data as keyword arguments
        """
        event = Event(type=event_type, data=data, source=source)

        # Add to history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)

        # Call subscribers
        if event_type in self._subscribers:
            subscribers = self._subscribers[event_type]
            logger.debug(
                f"[{self.name}] Emitting {event_type} from {source or 'unknown'} "
                f"to {len(subscribers)} subscribers"
            )

            for priority, callback in subscribers:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(
                        f"[{self.name}] Error in event handler {callback.__name__} "
                        f"for {event_type}: {e}",
                        exc_info=True
                    )
        else:
            logger.debug(
                f"[{self.name}] Emitted {event_type} from {source or 'unknown'} "
                f"(no subscribers)"
            )

    def clear_subscribers(self, event_type: (EventType | str) | None = None) -> None:
        """
        Clear subscribers.

        Parameters
        ----------
        event_type : (EventType | str) | None
            If provided, clear only for this event type.
            If None, clear all subscribers.
        """
        if event_type is None:
            self._subscribers.clear()
            logger.debug(f"[{self.name}] Cleared all subscribers")
        elif event_type in self._subscribers:
            del self._subscribers[event_type]
            logger.debug(f"[{self.name}] Cleared subscribers for {event_type}")

    def get_history(
        self,
        event_type: (EventType | str) | None = None,
        limit: int | None = None
    ) -> list[Event]:
        """
        Get event history.

        Parameters
        ----------
        event_type : (EventType | str) | None
            If provided, filter by event type
        limit : int | None
            Maximum number of events to return

        Returns
        -------
        list[Event]
            Event history (most recent last)
        """
        history = self._event_history

        if event_type is not None:
            history = [e for e in history if e.type == event_type]

        if limit is not None:
            history = history[-limit:]

        return history

    def has_subscribers(self, event_type: EventType | str) -> bool:
        """Check if an event type has any subscribers."""
        return event_type in self._subscribers and len(self._subscribers[event_type]) > 0


# Global event bus instance
_global_event_bus = EventBus(name="global")


def get_event_bus() -> EventBus:
    """Get the global event bus instance."""
    return _global_event_bus


def emit_event(event_type: EventType | str, source: str | None = None, **data) -> None:
    """Emit event on global bus (convenience function)."""
    _global_event_bus.emit(event_type, source=source, **data)


def subscribe_event(
    event_type: EventType | str,
    callback: Callable[[Event], None],
    priority: int = 0
) -> None:
    """Subscribe to event on global bus (convenience function)."""
    _global_event_bus.subscribe(event_type, callback, priority=priority)


def unsubscribe_event(
    event_type: EventType | str,
    callback: Callable[[Event], None]
) -> bool:
    """Unsubscribe from event on global bus (convenience function)."""
    return _global_event_bus.unsubscribe(event_type, callback)


# Export public API
__all__ = [
    "EventBus",
    "Event",
    "EventType",
    "get_event_bus",
    "emit_event",
    "subscribe_event",
    "unsubscribe_event",
]