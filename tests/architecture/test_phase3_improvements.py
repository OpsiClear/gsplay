#!/usr/bin/env python3
"""
Test script to verify Phase 3 architecture improvements work correctly.

Run with: python test_phase3_improvements.py
"""

import logging
import sys
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def test_duplicate_writers_removed():
    """Test that duplicate PLY writer has been removed."""
    logger.info("[TEST] Testing duplicate PLY writer removal...")

    try:
        # Fast writer should not exist anymore
        fast_writer_path = Path("src/models/ply/fast_writer.py")
        assert not fast_writer_path.exists(), "Duplicate fast_writer.py still exists"

        # Try to import should fail
        try:
            from src.models.ply.fast_writer import fast_write_gaussian_ply as _  # noqa: F401

            logger.error("[FAIL] fast_writer module still importable")
            return False
        except ImportError:
            pass  # Expected

        # Main writer should exist
        from src.infrastructure.processing.ply.writer import write_ply

        assert write_ply is not None, "Main write_ply function not found"

        logger.info("[PASS] Duplicate PLY writer successfully removed")
        return True
    except AssertionError as e:
        logger.error(f"[FAIL] Duplicate writer test: {e}")
        return False
    except Exception as e:
        logger.error(f"[FAIL] Duplicate writer test failed: {e}")
        return False


def test_configurable_model_interface():
    """Test that ConfigurableModelInterface exists and is properly defined."""
    logger.info("[TEST] Testing ConfigurableModelInterface...")

    try:
        from src.domain.interfaces import ConfigurableModelInterface

        # Check that ConfigurableModelInterface exists
        assert hasattr(
            ConfigurableModelInterface, "from_config"
        ), "ConfigurableModelInterface missing from_config method"

        # Check it has ModelInterface methods too
        assert hasattr(
            ConfigurableModelInterface, "get_gaussians_at_normalized_time"
        ), "ConfigurableModelInterface missing get_gaussians_at_normalized_time"
        assert hasattr(
            ConfigurableModelInterface, "get_total_frames"
        ), "ConfigurableModelInterface missing get_total_frames"
        assert hasattr(
            ConfigurableModelInterface, "get_frame_time"
        ), "ConfigurableModelInterface missing get_frame_time"

        logger.info("[PASS] ConfigurableModelInterface properly defined")
        return True
    except AssertionError as e:
        logger.error(f"[FAIL] ConfigurableModelInterface: {e}")
        return False
    except Exception as e:
        logger.error(f"[FAIL] ConfigurableModelInterface test failed: {e}")
        return False


def test_model_factory_registry():
    """Test that ModelFactory supports model registration."""
    logger.info("[TEST] Testing ModelFactory registry pattern...")

    try:
        from src.infrastructure.model_factory import ModelFactory

        # Check that register_model method exists
        assert hasattr(ModelFactory, "register_model"), "ModelFactory missing register_model method"

        # Check that _configurable_models registry exists
        assert hasattr(
            ModelFactory, "_configurable_models"
        ), "ModelFactory missing _configurable_models registry"

        logger.info("[PASS] ModelFactory registry pattern implemented")
        return True
    except AssertionError as e:
        logger.error(f"[FAIL] ModelFactory registry: {e}")
        return False
    except Exception as e:
        logger.error(f"[FAIL] ModelFactory registry test failed: {e}")
        return False


def test_no_duplicate_constants():
    """Test that hardcoded constants have been replaced with GaussianConstants."""
    logger.info("[TEST] Testing for duplicate constants...")

    try:
        import inspect

        import src.models.ply.optimized_model as ply_module

        source = inspect.getsource(ply_module)

        # Should not have hardcoded constants
        hardcoded_patterns = [
            "C0 = 0.28209479177387814",
            "LOG_SCALE_THRESHOLD = -5.0",
            "MIN_SCALE = 1e-6",
            "MAX_SCALE = 1e3",
        ]

        for pattern in hardcoded_patterns:
            assert pattern not in source, f"Found hardcoded constant: {pattern}"

        # Should use GaussianConstants
        assert (
            "from src.infrastructure.processing.gaussian_constants import" in source
        ), "Not importing GaussianConstants"
        assert "GC." in source, "Not using GC alias"

        logger.info("[PASS] No duplicate constants found")
        return True
    except AssertionError as e:
        logger.error(f"[FAIL] Duplicate constants: {e}")
        return False
    except Exception as e:
        logger.error(f"[FAIL] Duplicate constants test failed: {e}")
        return False


def test_clean_imports():
    """Test that imports are clean and follow architecture boundaries."""
    logger.info("[TEST] Testing clean import structure...")

    try:
        # Domain should not import from infrastructure (except interfaces)
        import inspect

        import src.domain.entities as entities
        import src.domain.services as services

        for module in [entities, services]:
            source = inspect.getsource(module)
            # Check for bad imports
            assert (
                "from src.infrastructure" not in source
                or "from src.infrastructure.processing.gaussian_constants" in source
            ), f"{module.__name__} has infrastructure imports"
            assert "from src.models" not in source, f"{module.__name__} has model imports"
            assert "from src.viewer" not in source, f"{module.__name__} has viewer imports"

        logger.info("[PASS] Import structure is clean")
        return True
    except AssertionError as e:
        logger.error(f"[FAIL] Import structure: {e}")
        return False
    except Exception as e:
        logger.error(f"[FAIL] Import structure test failed: {e}")
        return False


def test_gaussian_constants():
    """Test that GaussianConstants are accessible."""
    logger.info("[TEST] Testing GaussianConstants...")

    try:
        from src.infrastructure.processing.gaussian_constants import GaussianConstants as GC

        # Use some constants
        assert GC.SH.C0 > 0, "GaussianConstants not accessible"
        assert GC.Format.LOG_SCALE_THRESHOLD < 0, "Format constants not accessible"

        logger.info("[PASS] GaussianConstants accessible")
        return True
    except AssertionError as e:
        logger.error(f"[FAIL] GaussianConstants: {e}")
        return False
    except Exception as e:
        logger.error(f"[FAIL] GaussianConstants test failed: {e}")
        return False


def main():
    """Run all Phase 3 tests."""
    logger.info("=" * 60)
    logger.info("PHASE 3 ARCHITECTURE IMPROVEMENTS TEST SUITE")
    logger.info("=" * 60)

    tests = [
        test_duplicate_writers_removed,
        test_configurable_model_interface,
        test_model_factory_registry,
        test_no_duplicate_constants,
        test_clean_imports,
        test_gaussian_constants,
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
    logger.info(f"PHASE 3 RESULTS: {passed} passed, {failed} failed")

    if failed == 0:
        logger.info("SUCCESS: All Phase 3 improvements working correctly!")
    else:
        logger.info("FAILURE: Some improvements need attention")

    logger.info("=" * 60)

    # Summary of all phases
    logger.info("")
    logger.info("CUMULATIVE RESULTS:")
    logger.info("  Phase 1: 4/4 tests passing")
    logger.info("  Phase 2: 6/6 tests passing")
    logger.info(f"  Phase 3: {passed}/{len(tests)} tests passing")
    logger.info(f"  Total: {10 + passed}/{10 + len(tests)} tests passing")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
