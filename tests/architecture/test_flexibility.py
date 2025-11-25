#!/usr/bin/env python3
"""
Test script to verify flexibility improvements work correctly.

Run with: python test_flexibility.py
"""

import sys
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Removed: test_config_validation - config_schema was deleted as dead code


def test_event_bus():
    """Test event bus system."""
    logger.info("[TEST] Testing event bus system...")

    try:
        from src.viewer.interaction.events import EventBus, EventType, Event

        bus = EventBus(name="test")

        # Track events
        received_events = []

        def handler1(event: Event):
            received_events.append(("handler1", event.data))

        def handler2(event: Event):
            received_events.append(("handler2", event.data))

        # Subscribe
        bus.subscribe(EventType.BRIGHTNESS_CHANGED, handler1)
        bus.subscribe(
            EventType.BRIGHTNESS_CHANGED, handler2, priority=10
        )  # Higher priority

        # Emit event
        bus.emit(EventType.BRIGHTNESS_CHANGED, source="test", value=0.8)

        # Check both handlers received it
        assert len(received_events) == 2, (
            f"Expected 2 events, got {len(received_events)}"
        )

        # Check priority (handler2 should be first due to higher priority)
        assert received_events[0][0] == "handler2", "Priority not working"
        assert received_events[1][0] == "handler1", "Priority not working"

        # Check data
        assert received_events[0][1]["value"] == 0.8, "Event data not passed correctly"

        # Test unsubscribe
        bus.unsubscribe(EventType.BRIGHTNESS_CHANGED, handler1)
        received_events.clear()
        bus.emit(EventType.BRIGHTNESS_CHANGED, source="test", value=0.5)

        assert len(received_events) == 1, "Unsubscribe not working"
        assert received_events[0][0] == "handler2", "Wrong handler called"

        # Test event history
        history = bus.get_history(EventType.BRIGHTNESS_CHANGED)
        assert len(history) == 2, f"Expected 2 events in history, got {len(history)}"

        logger.info("[PASS] Event bus working correctly")
        return True

    except Exception as e:
        logger.error(f"[FAIL] Event bus test failed: {e}", exc_info=True)
        return False


def test_export_component():
    """Test export component."""
    logger.info("[TEST] Testing export component...")

    try:
        from src.viewer.core.components import ExportComponent
        from src.viewer.interaction.events import EventBus, EventType

        bus = EventBus(name="test")
        export_component = ExportComponent(event_bus=bus)

        # Track events
        events_received = []

        def track_event(event):
            events_received.append(event.type)

        bus.subscribe(EventType.EXPORT_REQUESTED, track_event)
        bus.subscribe(EventType.EXPORT_COMPLETED, track_event)
        bus.subscribe(EventType.EXPORT_FAILED, track_event)

        # Check component exists and has methods
        assert hasattr(export_component, "export_frame_sequence"), (
            "Missing export_frame_sequence"
        )
        assert hasattr(export_component, "export_single_frame"), (
            "Missing export_single_frame"
        )
        assert hasattr(export_component, "set_default_output_dir"), (
            "Missing set_default_output_dir"
        )

        # Test default output dir
        export_component.set_default_output_dir("./test_export")
        assert export_component.default_output_dir is not None, "Default dir not set"

        logger.info("[PASS] Export component structure correct")
        return True

    except Exception as e:
        logger.error(f"[FAIL] Export component test failed: {e}", exc_info=True)
        return False


def test_exporter_registry():
    """Test enhanced exporter registry."""
    logger.info("[TEST] Testing exporter registry...")

    try:
        from src.infrastructure.exporters.factory import (
            ExporterFactory,
            ExportCapability,
            ExportFormat,
        )

        # Test format enumeration
        formats = ExporterFactory.get_available_formats()
        assert "ply" in formats, "PLY format not registered"
        assert "compressed-ply" in formats, "Compressed PLY not registered"

        # Test capability queries
        compressed_formats = ExporterFactory.get_formats_with_capability(
            ExportCapability.COMPRESSION
        )
        assert "compressed-ply" in compressed_formats, (
            "Compressed PLY missing COMPRESSION capability"
        )

        cloud_formats = ExporterFactory.get_formats_with_capability(
            ExportCapability.CLOUD_STORAGE
        )
        assert "ply" in cloud_formats, "PLY missing CLOUD_STORAGE capability"

        # Test capability check
        assert ExporterFactory.has_capability(
            "compressed-ply", ExportCapability.COMPRESSION
        ), "Capability check not working"

        # Test exporter info
        info = ExporterFactory.get_exporter_info("ply")
        assert info is not None, "Exporter info not found"
        assert info.format == "ply", "Format name incorrect"
        assert info.file_extension == ".ply", "File extension incorrect"
        assert len(info.capabilities) > 0, "No capabilities listed"

        # Test enum-based creation
        exporter1 = ExporterFactory.create(ExportFormat.PLY)
        assert exporter1 is not None, "Enum-based creation failed"

        # Test string-based creation (backward compat)
        exporter2 = ExporterFactory.create("ply")
        assert exporter2 is not None, "String-based creation failed"

        # Test invalid format
        try:
            ExporterFactory.create("invalid-format")
            logger.error("[FAIL] Should have raised error for invalid format")
            return False
        except ValueError:
            pass  # Expected

        logger.info("[PASS] Exporter registry working correctly")
        return True

    except Exception as e:
        logger.error(f"[FAIL] Exporter registry test failed: {e}", exc_info=True)
        return False


def test_model_component():
    """Test model component structure."""
    logger.info("[TEST] Testing model component...")

    try:
        from src.viewer.core.components import ModelComponent
        from src.viewer.interaction.events import EventBus

        bus = EventBus(name="test")
        model_comp = ModelComponent(device="cpu", event_bus=bus)

        # Check component exists and has methods
        assert hasattr(model_comp, "load_from_config"), "Missing load_from_config"
        assert hasattr(model_comp, "load_from_path"), "Missing load_from_path"
        assert hasattr(model_comp, "get_model"), "Missing get_model"
        assert hasattr(model_comp, "get_metadata"), "Missing get_metadata"
        assert hasattr(model_comp, "is_loaded"), "Missing is_loaded"

        # Check initial state
        assert not model_comp.is_loaded(), "Model should not be loaded initially"
        assert model_comp.get_model() is None, "Model should be None initially"

        logger.info("[PASS] Model component structure correct")
        return True

    except Exception as e:
        logger.error(f"[FAIL] Model component test failed: {e}", exc_info=True)
        return False


def test_render_component():
    """Test render component structure."""
    logger.info("[TEST] Testing render component...")

    try:
        from src.viewer.core.components import RenderComponent

        # Create a mock server (can't actually create without running server)
        # Just test the component can be imported and has required methods
        assert hasattr(RenderComponent, "create_render_function"), (
            "Missing create_render_function"
        )
        assert hasattr(RenderComponent, "create_viewer"), "Missing create_viewer"
        assert hasattr(RenderComponent, "setup_viewer"), "Missing setup_viewer"
        assert hasattr(RenderComponent, "configure_quality"), (
            "Missing configure_quality"
        )
        assert hasattr(RenderComponent, "get_viewer"), "Missing get_viewer"
        assert hasattr(RenderComponent, "rerender"), "Missing rerender"

        logger.info("[PASS] Render component structure correct")
        return True

    except Exception as e:
        logger.error(f"[FAIL] Render component test failed: {e}", exc_info=True)
        return False


def test_flexibility_integration():
    """Test that all flexibility components work together."""
    logger.info("[TEST] Testing flexibility integration...")

    try:
        # Import all components
        from src.viewer.interaction.events import EventBus
        from src.viewer.core.components import ExportComponent, ModelComponent
        from src.infrastructure.exporters.factory import ExporterFactory

        # Create instances
        bus = EventBus(name="integration")
        model_comp = ModelComponent(device="cpu", event_bus=bus)
        export_comp = ExportComponent(event_bus=bus)
        formats = ExporterFactory.get_available_formats()

        # Verify they exist
        assert bus is not None, "EventBus not created"
        assert model_comp is not None, "ModelComponent not created"
        assert export_comp is not None, "ExportComponent not created"
        assert len(formats) > 0, "No export formats available"

        logger.info("[PASS] All flexibility components integrate correctly")
        return True

    except Exception as e:
        logger.error(f"[FAIL] Integration test failed: {e}", exc_info=True)
        return False


def main():
    """Run all flexibility tests."""
    logger.info("=" * 60)
    logger.info("FLEXIBILITY IMPROVEMENTS TEST SUITE")
    logger.info("=" * 60)

    tests = [
        test_event_bus,
        test_export_component,
        test_exporter_registry,
        test_model_component,
        test_render_component,
        test_flexibility_integration,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        logger.info("")
        if test_func():
            passed += 1
        else:
            failed += 1

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"FLEXIBILITY RESULTS: {passed} passed, {failed} failed")

    if failed == 0:
        logger.info("SUCCESS: All flexibility improvements working!")
    else:
        logger.info("FAILURE: Some improvements need attention")

    logger.info("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
