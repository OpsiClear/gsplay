#!/usr/bin/env python3
"""
Test script to verify Phase 2 architecture improvements work correctly.

Run with: python test_phase2_improvements.py
"""

import sys
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def test_model_factory_integration():
    """Test that ModelFactory is properly integrated."""
    logger.info("[TEST] Testing ModelFactory integration in viewer...")

    try:
        # Check that viewer uses ModelFactory
        import inspect
        from src.viewer.core.app import UniversalViewer

        source = inspect.getsource(UniversalViewer.load_model_from_config)

        # Should import and use ModelFactory
        assert "from src.infrastructure.model_factory import ModelFactory" in source, (
            "Viewer doesn't import ModelFactory"
        )
        assert "ModelFactory.create(" in source, (
            "Viewer doesn't use ModelFactory.create()"
        )

        # Should NOT have model-specific imports
        assert "from src.models.ply.optimized_model import" not in source, (
            "Viewer still has PLY model imports"
        )
        assert "from src.models.streaming.model import" not in source, (
            "Viewer still has streaming model imports"
        )

        # Count lines in the method (should be much shorter)
        lines = source.split("\n")
        line_count = len([line for line in lines if line.strip()])
        logger.info(f"  load_model_from_config is now {line_count} lines (was 136)")
        assert line_count < 60, f"Method still too long: {line_count} lines"

        logger.info("[PASS] ModelFactory properly integrated")
        return True
    except AssertionError as e:
        logger.error(f"[FAIL] ModelFactory integration: {e}")
        return False
    except Exception as e:
        logger.error(f"[FAIL] ModelFactory test failed: {e}")
        return False


def test_protocol_based_checks():
    """Test that isinstance() checks have been replaced with protocols."""
    logger.info("[TEST] Testing protocol-based type checking...")

    try:
        import inspect
        from src.viewer.core.app import UniversalViewer

        # Check _setup_layer_controls method
        source = inspect.getsource(UniversalViewer._setup_layer_controls)

        # Should use hasattr checks or protocol
        assert "isinstance(" not in source or "CompositeModelInterface" in source, (
            "Still using isinstance with concrete type"
        )
        assert "hasattr(self.model" in source or "CompositeModelInterface" in source, (
            "Not using protocol-based checking"
        )

        # Check that CompositeModelInterface exists
        from src.domain.interfaces import CompositeModelInterface

        assert hasattr(CompositeModelInterface, "get_layer_info"), (
            "CompositeModelInterface missing get_layer_info"
        )
        assert hasattr(CompositeModelInterface, "set_layer_visibility"), (
            "CompositeModelInterface missing set_layer_visibility"
        )

        logger.info("[PASS] Protocol-based checking implemented")
        return True
    except AssertionError as e:
        logger.error(f"[FAIL] Protocol check: {e}")
        return False
    except Exception as e:
        logger.error(f"[FAIL] Protocol test failed: {e}")
        return False


def test_gaussian_constants_usage():
    """Test that models use GaussianConstants instead of hardcoded values."""
    logger.info("[TEST] Testing GaussianConstants usage in models...")

    try:
        import inspect
        import src.models.ply.optimized_model as ply_module

        # Get module source, not just class source
        source = inspect.getsource(ply_module)

        # Should import GaussianConstants (might use alias)
        assert (
            "from src.infrastructure.processing.gaussian_constants import GaussianConstants"
            in source
            or "from src.infrastructure.processing.gaussian_constants import GaussianConstants as GC"
            in source
        ), "OptimizedPlyModel doesn't import GaussianConstants"

        # Should use GC constants
        assert "GC.Format.LOG_SCALE_THRESHOLD" in source, (
            "Not using GC.Format.LOG_SCALE_THRESHOLD"
        )
        assert "GC.Numerical.MIN_SCALE" in source, "Not using GC.Numerical.MIN_SCALE"
        assert "GC.Filtering.DEFAULT_PERCENTILE" in source, (
            "Not using GC.Filtering.DEFAULT_PERCENTILE"
        )

        # Should NOT have hardcoded constants
        assert "LOG_SCALE_THRESHOLD = -5.0" not in source, (
            "Still has hardcoded LOG_SCALE_THRESHOLD"
        )
        assert "C0 = 0.28209479177387814" not in source, "Still has hardcoded C0"

        logger.info("[PASS] GaussianConstants properly used in models")
        return True
    except AssertionError as e:
        logger.error(f"[FAIL] GaussianConstants usage: {e}")
        return False
    except Exception as e:
        logger.error(f"[FAIL] GaussianConstants test failed: {e}")
        return False


def test_no_circular_imports():
    """Test that there are no circular import issues."""
    logger.info("[TEST] Testing for circular imports...")

    try:
        # Try importing all key modules
        from src.infrastructure.model_factory import ModelFactory

        # Try creating a factory (without real config)
        try:
            ModelFactory.create("invalid", {}, "cpu")
        except ValueError as e:
            if "Unknown module type" in str(e):
                pass  # Expected error
            else:
                raise

        logger.info("[PASS] No circular import issues")
        return True
    except ImportError as e:
        logger.error(f"[FAIL] Import error (possible circular dependency): {e}")
        return False
    except Exception as e:
        logger.error(f"[FAIL] Circular import test failed: {e}")
        return False


def test_reduced_code_size():
    """Test that code size has been reduced through refactoring."""
    logger.info("[TEST] Testing code size reduction...")

    try:
        import inspect
        from src.viewer.core.app import UniversalViewer

        # Check load_model_from_config size
        load_model_source = inspect.getsource(UniversalViewer.load_model_from_config)
        load_model_lines = len(
            [line for line in load_model_source.split("\n") if line.strip()]
        )

        # Original was 136 lines, should be much smaller now
        assert load_model_lines < 60, (
            f"load_model_from_config still too large: {load_model_lines} lines"
        )

        logger.info(
            f"  load_model_from_config: {load_model_lines} lines (reduced from 136)"
        )
        logger.info(f"  Reduction: {100 * (136 - load_model_lines) / 136:.1f}%")

        logger.info("[PASS] Code size successfully reduced")
        return True
    except AssertionError as e:
        logger.error(f"[FAIL] Code size: {e}")
        return False
    except Exception as e:
        logger.error(f"[FAIL] Code size test failed: {e}")
        return False


def test_clean_architecture_boundaries():
    """Test that Clean Architecture boundaries are maintained."""
    logger.info("[TEST] Testing Clean Architecture boundaries...")

    try:
        import inspect

        # Domain should not import from infrastructure or models
        from src.domain import interfaces, entities, services

        for module in [interfaces, entities, services]:
            source = inspect.getsource(module)
            assert "from src.infrastructure" not in source, (
                f"Domain module {module.__name__} imports from infrastructure"
            )
            assert "from src.models" not in source, (
                f"Domain module {module.__name__} imports from models"
            )

        # Infrastructure should not import from viewer or models (except model_factory)
        from src.infrastructure import gaussian_constants, config

        for module in [gaussian_constants, config]:
            source = inspect.getsource(module)
            assert "from src.viewer" not in source, (
                f"Infrastructure module {module.__name__} imports from viewer"
            )

        logger.info("[PASS] Clean Architecture boundaries maintained")
        return True
    except AssertionError as e:
        logger.error(f"[FAIL] Architecture boundary violation: {e}")
        return False
    except Exception as e:
        logger.error(f"[FAIL] Architecture boundary test failed: {e}")
        return False


def main():
    """Run all Phase 2 tests."""
    logger.info("=" * 60)
    logger.info("PHASE 2 ARCHITECTURE IMPROVEMENTS TEST SUITE")
    logger.info("=" * 60)

    tests = [
        test_model_factory_integration,
        test_protocol_based_checks,
        test_gaussian_constants_usage,
        test_no_circular_imports,
        test_reduced_code_size,
        test_clean_architecture_boundaries,
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
    logger.info(f"PHASE 2 RESULTS: {passed} passed, {failed} failed")

    if failed == 0:
        logger.info("SUCCESS: All Phase 2 improvements working correctly!")
    else:
        logger.info("FAILURE: Some improvements need attention")

    logger.info("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
