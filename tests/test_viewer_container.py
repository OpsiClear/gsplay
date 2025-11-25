"""Tests for the viewer processing container wiring."""

from src.domain.entities import GSTensor
from src.infrastructure.processing_mode import ProcessingMode
from src.viewer.config.settings import ViewerConfig
from src.viewer.core.container import (
    build_default_processing_providers,
    create_edit_manager,
)


def test_build_default_processing_providers_covers_all_modes():
    providers = build_default_processing_providers()
    assert set(providers.strategies.keys()) == set(ProcessingMode)


def test_create_edit_manager_uses_custom_gs_bridge(sample_gsdata, fake_gs_bridge):
    config = ViewerConfig()
    config.edits_active = True
    config.volume_filter.processing_mode = "all_cpu"

    providers = build_default_processing_providers()
    providers.gs_bridge = fake_gs_bridge

    edit_manager = create_edit_manager(config, "cpu", providers)
    result = edit_manager.apply_edits(sample_gsdata)

    assert isinstance(result, GSTensor)
    assert fake_gs_bridge.ensure_gsdata_calls > 0


def test_all_gpu_mode_uses_tensor_bridge(sample_gsdata, fake_gs_bridge):
    config = ViewerConfig()
    config.edits_active = True
    config.volume_filter.processing_mode = "all_gpu"

    providers = build_default_processing_providers()
    providers.gs_bridge = fake_gs_bridge

    edit_manager = create_edit_manager(config, "cpu", providers)
    edit_manager.apply_edits(sample_gsdata)

    assert fake_gs_bridge.ensure_tensor_calls > 0
