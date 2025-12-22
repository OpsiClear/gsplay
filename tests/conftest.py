"""Pytest configuration and shared fixtures."""

import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from src.domain.entities import GSData, GSTensor


# Provide a lightweight viser stub for environments without the dependency installed.
if "viser" not in sys.modules:
    viser_stub = types.ModuleType("viser")
    viser_stub.GuiTextHandle = object
    viser_stub.GuiButtonHandle = object
    viser_stub.GuiMarkdownHandle = object
    viser_stub.GuiSliderHandle = object
    viser_stub.GuiButtonGroupHandle = object
    viser_stub.GuiDropdownHandle = object
    viser_stub.GuiCheckboxHandle = object
    viser_stub.ViserServer = object

    transforms_stub = types.ModuleType("viser.transforms")
    viser_stub.transforms = transforms_stub

    sys.modules["viser"] = viser_stub
    sys.modules["viser.transforms"] = transforms_stub


@pytest.fixture
def device():
    """Provide CUDA device if available, else CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_sh0():
    """Create sample SH0 coefficients."""
    num_gaussians = 1000
    sh0 = torch.randn(num_gaussians, 1, 3) * 0.5
    return sh0


@pytest.fixture
def sample_scales_raw():
    """Create sample scale values (log space)."""
    num_gaussians = 1000
    # Mix of positive and negative values (log space)
    scales = torch.randn(num_gaussians, 3) * 2.0 - 6.0
    return scales


@pytest.fixture
def sample_opacities_raw():
    """Create sample opacity values (logit space)."""
    num_gaussians = 1000
    # Logit space: range roughly [-5, 5]
    opacities = torch.randn(num_gaussians, 1) * 3.0
    return opacities


@pytest.fixture
def sample_quats_raw():
    """Create sample quaternion values (unnormalized)."""
    num_gaussians = 1000
    quats = torch.randn(num_gaussians, 4)
    return quats


@pytest.fixture
def ply_test_dir():
    """Return path to export_with_edits test data."""
    ply_path = Path("./export_with_edits")
    if not ply_path.exists():
        pytest.skip("Test PLY data not found at ./export_with_edits")
    return ply_path


@pytest.fixture
def sample_ply_files(ply_test_dir):
    """Get list of sample PLY files for testing."""
    ply_files = sorted(ply_test_dir.glob("*.ply"))
    if len(ply_files) < 5:
        pytest.skip("Not enough PLY files for testing")
    return [str(f) for f in ply_files[:5]]  # Use first 5 files


@pytest.fixture
def sample_gsdata():
    """Create a minimal GSData payload for CPU-side testing."""
    num_gaussians = 8
    means = np.zeros((num_gaussians, 3), dtype=np.float32)
    scales = np.ones((num_gaussians, 3), dtype=np.float32)
    quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (num_gaussians, 1))
    opacities = np.full((num_gaussians,), 0.5, dtype=np.float32)
    sh0 = np.full((num_gaussians, 3), 0.75, dtype=np.float32)
    return GSData(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh0=sh0,
        shN=None,
    )


@pytest.fixture
def fake_gs_bridge():
    """Fake GSBridge implementation for unit tests."""

    class _FakeBridge:
        def __init__(self) -> None:
            self.ensure_gsdata_calls = 0
            self.ensure_tensor_calls = 0

        def ensure_gsdata(self, gaussians: GSData | GSTensor) -> GSData:
            self.ensure_gsdata_calls += 1
            return gaussians if isinstance(gaussians, GSData) else gaussians.to_gsdata()

        def ensure_tensor_on_device(
            self,
            gaussians: GSData | GSTensor,
            device: str,
        ) -> tuple[GSTensor, float]:
            self.ensure_tensor_calls += 1
            if isinstance(gaussians, GSTensor):
                tensor = gaussians
            else:
                tensor = GSTensor.from_gsdata(gaussians, device="cpu")
            if tensor.means.device.type != device:
                tensor = tensor.to(device)
            return tensor, 0.0

    return _FakeBridge()
