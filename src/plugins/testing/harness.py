"""Test harness for plugin development.

Provides utilities for testing plugin implementations without
running the full viewer.
"""

from __future__ import annotations

import logging
from typing import Any

from src.domain.data import GaussianData
from src.domain.interfaces import (
    BaseGaussianSource,
    HealthStatus,
    PluginState,
)


logger = logging.getLogger(__name__)


class PluginTestHarness:
    """Test harness for validating plugin implementations.

    Provides automated testing of:
    - Protocol compliance (required methods exist)
    - Lifecycle management
    - Frame loading
    - Health checks

    Example
    -------
    >>> harness = PluginTestHarness(MySource)
    >>> config = {"ply_folder": "/path/to/test/data"}
    >>> results = harness.run_all_tests(config)
    >>> assert all(r["passed"] for r in results.values())
    """

    def __init__(self, plugin_class: type[BaseGaussianSource]) -> None:
        """Initialize test harness.

        Parameters
        ----------
        plugin_class : Type[BaseGaussianSource]
            Plugin class to test
        """
        self.plugin_class = plugin_class
        self.results: dict[str, dict[str, Any]] = {}

    def run_all_tests(
        self,
        config: dict[str, Any],
        device: str = "cuda",
    ) -> dict[str, dict[str, Any]]:
        """Run all tests against the plugin.

        Parameters
        ----------
        config : dict
            Config to pass to plugin constructor
        device : str
            Device for testing

        Returns
        -------
        dict
            Test results mapping test name to result dict
        """
        self.results = {}

        # Protocol compliance tests
        self.results["protocol_compliance"] = self._test_protocol_compliance()

        # Instantiation test
        instance_result = self._test_instantiation(config, device)
        self.results["instantiation"] = instance_result

        if not instance_result["passed"]:
            return self.results

        instance = instance_result["instance"]

        # Frame loading tests
        self.results["frame_loading"] = self._test_frame_loading(instance)

        # Health check tests
        self.results["health_check"] = self._test_health_check(instance)

        # Lifecycle tests
        self.results["lifecycle"] = self._test_lifecycle(instance)

        return self.results

    def _test_protocol_compliance(self) -> dict[str, Any]:
        """Test that plugin class has required methods."""
        required_class_methods = ["metadata", "can_load"]
        required_instance_methods = ["get_frame_at_time"]
        required_properties = ["total_frames"]

        missing = []

        for method in required_class_methods:
            if not hasattr(self.plugin_class, method):
                missing.append(f"classmethod {method}")
            elif not callable(getattr(self.plugin_class, method)):
                missing.append(f"classmethod {method} (not callable)")

        for method in required_instance_methods:
            if not hasattr(self.plugin_class, method):
                missing.append(f"method {method}")

        for prop in required_properties:
            if not hasattr(self.plugin_class, prop):
                missing.append(f"property {prop}")

        if missing:
            return {
                "passed": False,
                "message": f"Missing: {', '.join(missing)}",
            }

        # Test metadata() returns SourceMetadata
        try:
            meta = self.plugin_class.metadata()
            if not hasattr(meta, "name") or not hasattr(meta, "description"):
                return {
                    "passed": False,
                    "message": "metadata() must return SourceMetadata",
                }
        except Exception as e:
            return {
                "passed": False,
                "message": f"metadata() raised: {e}",
            }

        return {"passed": True, "message": "All protocol methods present"}

    def _test_instantiation(
        self,
        config: dict[str, Any],
        device: str,
    ) -> dict[str, Any]:
        """Test plugin instantiation."""
        config_with_device = {**config, "device": device}

        try:
            instance = self.plugin_class(config_with_device)
            return {
                "passed": True,
                "message": "Instantiation successful",
                "instance": instance,
            }
        except Exception as e:
            return {
                "passed": False,
                "message": f"Instantiation failed: {e}",
                "instance": None,
            }

    def _test_frame_loading(self, instance: BaseGaussianSource) -> dict[str, Any]:
        """Test frame loading at various times."""
        test_times = [0.0, 0.5, 1.0]
        results = []

        for t in test_times:
            try:
                data = instance.get_frame_at_time(t)

                if data is None:
                    results.append(f"t={t}: returned None")
                elif not isinstance(data, GaussianData):
                    results.append(f"t={t}: wrong type {type(data)}")
                elif data.n_gaussians == 0:
                    results.append(f"t={t}: empty data")
                else:
                    results.append(f"t={t}: OK ({data.n_gaussians} gaussians)")
            except Exception as e:
                results.append(f"t={t}: exception {e}")

        all_ok = all("OK" in r for r in results)
        return {
            "passed": all_ok,
            "message": "; ".join(results),
        }

    def _test_health_check(self, instance: BaseGaussianSource) -> dict[str, Any]:
        """Test health check functionality."""
        if not hasattr(instance, "health_check"):
            return {"passed": True, "message": "No health_check method (optional)"}

        try:
            result = instance.health_check()
            if result.status == HealthStatus.HEALTHY:
                return {
                    "passed": True,
                    "message": f"Health check passed: {result.message}",
                }
            else:
                return {
                    "passed": False,
                    "message": f"Health check failed: {result.status.name} - {result.message}",
                }
        except Exception as e:
            return {
                "passed": False,
                "message": f"Health check raised: {e}",
            }

    def _test_lifecycle(self, instance: BaseGaussianSource) -> dict[str, Any]:
        """Test lifecycle management."""
        if not hasattr(instance, "state"):
            return {"passed": True, "message": "No lifecycle support (optional)"}

        try:
            state = instance.state
            if state != PluginState.READY:
                return {
                    "passed": False,
                    "message": f"Expected READY state, got {state.name}",
                }

            # Test shutdown if available
            if hasattr(instance, "on_shutdown"):
                instance.on_shutdown()
                if instance.state != PluginState.TERMINATED:
                    return {
                        "passed": False,
                        "message": f"After shutdown, expected TERMINATED, got {instance.state.name}",
                    }

            return {"passed": True, "message": "Lifecycle management OK"}
        except Exception as e:
            return {
                "passed": False,
                "message": f"Lifecycle test raised: {e}",
            }

    def print_results(self) -> None:
        """Print test results to console."""
        print(f"\n{'=' * 60}")
        print(f"Plugin Test Results: {self.plugin_class.__name__}")
        print(f"{'=' * 60}")

        for test_name, result in self.results.items():
            status = "PASS" if result.get("passed") else "FAIL"
            message = result.get("message", "")
            print(f"  [{status}] {test_name}: {message}")

        total = len(self.results)
        passed = sum(1 for r in self.results.values() if r.get("passed"))
        print(f"\n  Total: {passed}/{total} tests passed")
        print(f"{'=' * 60}\n")


__all__ = ["PluginTestHarness"]
