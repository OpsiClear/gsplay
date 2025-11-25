"""Unit tests for the GSBridge abstraction."""

from src.domain.entities import GSData, GSTensor
from src.viewer.processing.gs_bridge import DefaultGSBridge


def test_default_gs_bridge_noop_for_gsdata(sample_gsdata):
    bridge = DefaultGSBridge()
    result = bridge.ensure_gsdata(sample_gsdata)
    assert result is sample_gsdata


def test_default_gs_bridge_converts_tensor_to_gsdata(sample_gsdata):
    bridge = DefaultGSBridge()
    gstensor = GSTensor.from_gsdata(sample_gsdata, device="cpu")
    converted = bridge.ensure_gsdata(gstensor)
    assert isinstance(converted, GSData)
    assert converted.means.shape == sample_gsdata.means.shape


def test_default_gs_bridge_transfers_to_target_device(sample_gsdata, device):
    bridge = DefaultGSBridge()
    tensor, transfer_ms = bridge.ensure_tensor_on_device(sample_gsdata, device)
    assert isinstance(tensor, GSTensor)
    assert tensor.means.device.type == device
    assert transfer_ms >= 0.0
