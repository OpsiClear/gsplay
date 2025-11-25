#!/usr/bin/env python3
"""
Test backward compatibility after component refactoring.

This test verifies that the refactored UniversalViewer maintains
the same public API and behavior as before.
"""

import sys
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def test_properties_are_accessible():
    """Test that model, data_loader, source_path, viewer are accessible as properties."""
    logger.info("[TEST] Testing property accessibility...")

    try:
        from src.viewer.core.app import UniversalViewer
        from src.viewer.config.settings import ViewerConfig

        # Create a minimal viewer config
        config = ViewerConfig(
            port=6020, device="cpu"
        )  # Use different port to avoid conflicts

        # Create viewer (should not fail)
        viewer = UniversalViewer(config)

        # Check that properties exist and return None initially
        assert hasattr(viewer, "model"), "Missing 'model' attribute"
        assert hasattr(viewer, "data_loader"), "Missing 'data_loader' attribute"
        assert hasattr(viewer, "source_path"), "Missing 'source_path' attribute"
        assert hasattr(viewer, "viewer"), "Missing 'viewer' attribute"

        # Check initial values (should all be None before loading)
        assert viewer.model is None, "model should be None initially"
        assert viewer.data_loader is None, "data_loader should be None initially"
        assert viewer.source_path is None, "source_path should be None initially"
        assert viewer.viewer is None, "viewer should be None initially"

        logger.info("[PASS] All properties are accessible")
        return True

    except Exception as e:
        logger.error(f"[FAIL] Property accessibility test failed: {e}", exc_info=True)
        return False


def test_components_exist():
    """Test that new components are accessible."""
    logger.info("[TEST] Testing component accessibility...")

    try:
        from src.viewer.core.app import UniversalViewer
        from src.viewer.config.settings import ViewerConfig

        config = ViewerConfig(port=6021, device="cpu")
        viewer = UniversalViewer(config)

        # Check that components exist
        assert hasattr(viewer, "model_component"), "Missing model_component"
        assert hasattr(viewer, "render_component"), "Missing render_component"
        assert hasattr(viewer, "export_component"), "Missing export_component"
        assert hasattr(viewer, "event_bus"), "Missing event_bus"

        # Check components are not None
        assert viewer.model_component is not None, "model_component is None"
        assert viewer.render_component is not None, "render_component is None"
        assert viewer.export_component is not None, "export_component is None"
        assert viewer.event_bus is not None, "event_bus is None"

        logger.info("[PASS] All components are accessible")
        return True

    except Exception as e:
        logger.error(f"[FAIL] Component accessibility test failed: {e}", exc_info=True)
        return False


def test_api_compatibility():
    """Test that ViewerAPI still works with refactored viewer."""
    logger.info("[TEST] Testing ViewerAPI compatibility...")

    try:
        from src.viewer.core.app import UniversalViewer
        from src.viewer.config.settings import ViewerConfig
        from src.viewer.core.api import ViewerAPI

        config = ViewerConfig(port=6022, device="cpu")
        viewer = UniversalViewer(config)

        # Create API (should not fail)
        api = ViewerAPI(viewer)

        # Check that API can access viewer.model without error
        # (It should be None, but shouldn't raise an error)
        try:
            _ = api._viewer.model
        except Exception as e:
            logger.error(f"API cannot access viewer.model: {e}")
            return False

        logger.info("[PASS] ViewerAPI is compatible")
        return True

    except Exception as e:
        logger.error(f"[FAIL] API compatibility test failed: {e}", exc_info=True)
        return False


def test_method_signatures():
    """Test that key methods still have the same signatures."""
    logger.info("[TEST] Testing method signatures...")

    try:
        from src.viewer.core.app import UniversalViewer
        from src.viewer.config.settings import ViewerConfig
        import inspect

        config = ViewerConfig(port=6023, device="cpu")
        viewer = UniversalViewer(config)

        # Check that key methods exist
        required_methods = [
            "load_model_from_config",
            "setup_viewer",
            "run",
            "_handle_load_data",
            "_handle_export_ply",
            "_handle_color_reset",
            "_handle_pose_reset",
            "_handle_filter_reset",
        ]

        for method_name in required_methods:
            if not hasattr(viewer, method_name):
                logger.error(f"Missing method: {method_name}")
                return False

        # Check load_model_from_config signature
        sig = inspect.signature(viewer.load_model_from_config)
        params = list(sig.parameters.keys())
        assert "config_dict" in params, (
            "load_model_from_config missing config_dict param"
        )
        assert "config_file" in params, (
            "load_model_from_config missing config_file param"
        )

        logger.info("[PASS] All method signatures are preserved")
        return True

    except Exception as e:
        logger.error(f"[FAIL] Method signature test failed: {e}", exc_info=True)
        return False


def test_no_breaking_changes():
    """Test that properties don't break when accessed."""
    logger.info("[TEST] Testing no breaking changes in property access...")

    try:
        from src.viewer.core.app import UniversalViewer
        from src.viewer.config.settings import ViewerConfig

        config = ViewerConfig(port=6024, device="cpu")
        viewer = UniversalViewer(config)

        # These should all work without raising exceptions
        _ = viewer.model
        _ = viewer.data_loader
        _ = viewer.source_path
        _ = viewer.viewer

        # Multiple accesses should work
        for _ in range(5):
            _ = viewer.model

        logger.info("[PASS] No breaking changes in property access")
        return True

    except Exception as e:
        logger.error(f"[FAIL] Breaking change detected: {e}", exc_info=True)
        return False


def main():
    """Run all backward compatibility tests."""
    logger.info("=" * 60)
    logger.info("BACKWARD COMPATIBILITY TEST SUITE")
    logger.info("=" * 60)

    tests = [
        test_properties_are_accessible,
        test_components_exist,
        test_api_compatibility,
        test_method_signatures,
        test_no_breaking_changes,
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
    logger.info(f"COMPATIBILITY RESULTS: {passed} passed, {failed} failed")

    if failed == 0:
        logger.info("SUCCESS: No breaking changes detected!")
    else:
        logger.info("FAILURE: Breaking changes detected - review needed")

    logger.info("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
