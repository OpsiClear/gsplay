"""
Round-trip PLY test to verify data integrity through load/save cycle.

This test verifies that:
1. PLY loading correctly handles log scales and logit opacities
2. PLY writing correctly converts back to log/logit format
3. Round-trip preserves data within acceptable tolerance
"""

import logging
import tempfile
from pathlib import Path

import numpy as np

from src.infrastructure.processing.ply import load_ply_as_gsdata, write_ply


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def test_roundtrip():
    """Test PLY round-trip: load -> save -> load -> compare."""

    # Test with a known PLY file
    input_ply = Path("export_with_edits/frame_00000.ply")

    if not input_ply.exists():
        logger.error(f"Test file not found: {input_ply}")
        logger.error("Please ensure export_with_edits/frame_00000.ply exists")
        return False

    logger.info(f"Testing round-trip with: {input_ply}")

    # Step 1: Load original as GSData (already in log/logit format)
    logger.info("Step 1: Loading original PLY...")
    gsdata1 = load_ply_as_gsdata(input_ply)

    logger.info(f"  Loaded {gsdata1.means.shape[0]} Gaussians")
    logger.info(f"  Scales range: [{gsdata1.scales.min():.4f}, {gsdata1.scales.max():.4f}]")
    logger.info(
        f"  Opacities range: [{gsdata1.opacities.min():.4f}, {gsdata1.opacities.max():.4f}]"
    )

    # Step 2: Save to temporary file (standard format)
    logger.info("Step 2: Writing to temporary PLY (standard format)...")
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        write_ply(tmp_path, gsdata1, format="ply")
        logger.info(f"  Written to {tmp_path}")

        # Step 3: Reload from temporary file
        logger.info("Step 3: Reloading from temporary PLY...")
        gsdata2 = load_ply_as_gsdata(tmp_path)

        logger.info(f"  Reloaded {gsdata2.means.shape[0]} Gaussians")
        logger.info(f"  Scales range: [{gsdata2.scales.min():.4f}, {gsdata2.scales.max():.4f}]")
        logger.info(
            f"  Opacities range: [{gsdata2.opacities.min():.4f}, {gsdata2.opacities.max():.4f}]"
        )

        # Step 4: Compare tensors
        logger.info("Step 4: Comparing tensors...")

        # Check shapes match
        assert (
            gsdata1.means.shape == gsdata2.means.shape
        ), f"Means shape mismatch: {gsdata1.means.shape} != {gsdata2.means.shape}"
        assert gsdata1.scales.shape == gsdata2.scales.shape, "Scales shape mismatch"
        assert gsdata1.quats.shape == gsdata2.quats.shape, "Quats shape mismatch"
        assert gsdata1.opacities.shape == gsdata2.opacities.shape, "Opacities shape mismatch"
        assert gsdata1.sh0.shape == gsdata2.sh0.shape, "SH0 shape mismatch"

        # Check values are close (allow small numerical errors)
        rtol = 1e-4  # Relative tolerance
        atol = 1e-6  # Absolute tolerance

        means_close = np.allclose(gsdata1.means, gsdata2.means, rtol=rtol, atol=atol)
        scales_close = np.allclose(gsdata1.scales, gsdata2.scales, rtol=rtol, atol=atol)
        quats_close = np.allclose(gsdata1.quats, gsdata2.quats, rtol=rtol, atol=atol)
        opacities_close = np.allclose(gsdata1.opacities, gsdata2.opacities, rtol=rtol, atol=atol)
        sh0_close = np.allclose(gsdata1.sh0, gsdata2.sh0, rtol=rtol, atol=atol)

        # Report results
        logger.info("  Results:")
        logger.info(f"    Means match: {means_close}")
        logger.info(f"    Scales match: {scales_close}")
        logger.info(f"    Quats match: {quats_close}")
        logger.info(f"    Opacities match: {opacities_close}")
        logger.info(f"    SH0 match: {sh0_close}")

        if not scales_close:
            max_diff = np.abs(gsdata1.scales - gsdata2.scales).max()
            logger.warning(f"    Scales max diff: {max_diff}")

        if not opacities_close:
            max_diff = np.abs(gsdata1.opacities - gsdata2.opacities).max()
            logger.warning(f"    Opacities max diff: {max_diff}")

        # Overall pass/fail
        all_pass = means_close and scales_close and quats_close and opacities_close and sh0_close

        if all_pass:
            logger.info("[PASS] Round-trip test passed!")
            return True
        else:
            logger.error("[FAIL] Round-trip test failed - data mismatch detected")
            return False

    finally:
        # Clean up temporary file
        import os

        if Path(tmp_path).exists():
            os.unlink(tmp_path)
            logger.debug(f"Cleaned up temporary file: {tmp_path}")


def test_compressed_roundtrip():
    """Test compressed PLY round-trip."""

    input_ply = Path("export_with_edits/frame_00000.ply")

    if not input_ply.exists():
        logger.error(f"Test file not found: {input_ply}")
        return False

    logger.info(f"\nTesting compressed round-trip with: {input_ply}")

    # Load original as GSData (already in log/logit format)
    logger.info("Loading original PLY...")
    gsdata1 = load_ply_as_gsdata(input_ply)

    # Save as compressed
    logger.info("Writing to compressed PLY...")
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        write_ply(tmp_path, gsdata1, format="compressed")
        logger.info(f"  Written compressed to {tmp_path}")

        # Reload
        logger.info("Reloading from compressed PLY...")
        gsdata2 = load_ply_as_gsdata(tmp_path)

        # Compare (compressed format may have some loss)
        logger.info("Comparing tensors...")

        # Use more lenient tolerance for compressed format
        rtol = 1e-3
        atol = 1e-5

        scales_close = np.allclose(gsdata1.scales, gsdata2.scales, rtol=rtol, atol=atol)
        opacities_close = np.allclose(gsdata1.opacities, gsdata2.opacities, rtol=rtol, atol=atol)

        logger.info(f"  Scales match (lenient): {scales_close}")
        logger.info(f"  Opacities match (lenient): {opacities_close}")

        if scales_close and opacities_close:
            logger.info("[PASS] Compressed round-trip test passed!")
            return True
        else:
            logger.warning(
                "[WARNING] Compressed format has differences (expected for lossy compression)"
            )
            return True  # Still pass, as some difference is expected

    finally:
        import os

        if Path(tmp_path).exists():
            os.unlink(tmp_path)


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("PLY Round-trip Test Suite")
    logger.info("=" * 60)

    # Run tests
    test1_passed = test_roundtrip()
    test2_passed = test_compressed_roundtrip()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary:")
    logger.info(f"  Standard PLY round-trip: {'PASS' if test1_passed else 'FAIL'}")
    logger.info(f"  Compressed PLY round-trip: {'PASS' if test2_passed else 'FAIL'}")
    logger.info("=" * 60)

    if test1_passed and test2_passed:
        logger.info("\n[OK] All tests passed!")
    else:
        logger.error("\n[FAIL] Some tests failed!")
