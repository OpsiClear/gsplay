"""
Universal 4D Gaussian Splatting GSPlay - Main Entry Point.

This is the CLI entry point that uses tyro for argument parsing.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Union

import tyro


logger = logging.getLogger(__name__)


def _init_heavy_imports() -> None:
    """Initialize heavy imports (torch, viser) after dependency check."""
    # --- Monkeypatch for viser on Windows with uv/pip entry points ---
    try:
        import viser._tunnel

        _original_is_multiprocess_ok = viser._tunnel._is_multiprocess_ok

        def _safe_is_multiprocess_ok() -> bool:
            try:
                return _original_is_multiprocess_ok()
            except (FileNotFoundError, OSError):
                return False

        viser._tunnel._is_multiprocess_ok = _safe_is_multiprocess_ok
    except ImportError:
        pass

    # Pre-import torchvision to avoid circular import issues when imported from threads
    import torchvision  # noqa: F401


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("gsply.torch.compression").setLevel(logging.WARNING)
    logging.getLogger("gsply.torch").setLevel(logging.WARNING)


# ============================================================================
# CLI Commands
# ============================================================================


@dataclass
class ViewCmd:
    """Launch the GSPlay viewer."""

    config: Annotated[Path, tyro.conf.Positional]
    """Path to model configuration (JSON file or PLY folder)."""

    port: int = 6019
    """Viser server port."""

    host: str = "0.0.0.0"
    """Host to bind to."""

    stream_port: int = -1
    """WebSocket stream port. Default: -1 (auto). Set to 0 to disable."""

    log_level: str = "INFO"
    """Logging level: DEBUG, INFO, WARNING, ERROR."""

    gpu: int | None = None
    """GPU device number. If None, uses GPU 0 if available."""

    view_only: bool = False
    """Hide input path, config save, and export options from UI."""

    compact: bool = False
    """Mobile-friendly compact UI."""


@dataclass
class SetupCmd:
    """Install GPU dependencies (PyTorch, gsplat, triton)."""

    force: bool = False
    """Force reinstall even if dependencies are present."""

    yes: bool = False
    """Skip confirmation prompts (for CI/automated use)."""


@dataclass
class DoctorCmd:
    """Diagnose environment and check dependencies."""

    verbose: bool = False
    """Show detailed diagnostic information."""


@dataclass
class PluginListCmd:
    """List all discovered plugins."""

    verbose: bool = False
    """Show detailed plugin information."""


@dataclass
class PluginInfoCmd:
    """Show information about a specific plugin."""

    name: Annotated[str, tyro.conf.Positional]
    """Plugin name to show info for."""


@dataclass
class PluginTestCmd:
    """Test a plugin with the test harness."""

    name: Annotated[str, tyro.conf.Positional]
    """Plugin name to test."""

    config: Path | None = None
    """Optional JSON config file for the plugin."""

    device: str = "cuda"
    """Device to use for testing."""


# Main command union with flat subcommands
MainCommand = Union[
    Annotated[ViewCmd, tyro.conf.subcommand("view", default=True)],
    Annotated[SetupCmd, tyro.conf.subcommand("setup")],
    Annotated[DoctorCmd, tyro.conf.subcommand("doctor")],
    Annotated[PluginListCmd, tyro.conf.subcommand("plugins-list")],
    Annotated[PluginInfoCmd, tyro.conf.subcommand("plugins-info")],
    Annotated[PluginTestCmd, tyro.conf.subcommand("plugins-test")],
]


# ============================================================================
# Command Handlers
# ============================================================================


def run_doctor(cmd: DoctorCmd) -> None:
    """Run environment diagnostics."""
    from src.gsplay.core.dependency_check import (
        check_environment,
        check_gsplat,
        check_python,
        check_torch,
        check_triton,
        check_viser,
        detect_system_cuda,
    )

    print("GSPlay Doctor")
    print("=" * 50)
    print()

    # System CUDA
    cuda = detect_system_cuda()
    if cuda:
        print(f"System CUDA:    [OK] {cuda}")
    else:
        print("System CUDA:    [X] Not detected (nvidia-smi not found)")

    print()

    # Check each dependency
    checks = [
        ("Python", check_python),
        ("PyTorch", check_torch),
        ("gsplat", check_gsplat),
        ("triton", check_triton),
        ("viser", check_viser),
    ]

    all_ok = True
    for name, check_fn in checks:
        status = check_fn()
        if status.available:
            print(f"{name:15} [OK] {status.message}")
        else:
            print(f"{name:15} [X] {status.message}")
            if status.fix_command:
                print(f"{'':15}   Fix: {status.fix_command}")
            all_ok = False

    # Verbose mode - show more details
    if cmd.verbose:
        print()
        print("Detailed Environment:")
        print("-" * 50)
        env = check_environment()

        if env.gpu_name:
            print(f"GPU:            {env.gpu_name}")
        if env.gpu_memory:
            print(f"GPU Memory:     {env.gpu_memory}")
        if env.cuda_version:
            print(f"CUDA (PyTorch): {env.cuda_version}")

        print(f"Platform:       {sys.platform}")
        print(f"Python Path:    {sys.executable}")

    print()
    print("=" * 50)
    if all_ok:
        print("[OK] Environment is ready!")
        print("  Run: gsplay <path-to-ply-folder>")
    else:
        print("[X] Issues found. Run 'gsplay setup' to fix.")

    sys.exit(0 if all_ok else 1)


def run_setup(cmd: SetupCmd) -> None:
    """Run the setup wizard."""
    from src.gsplay.core.setup import run_setup as do_setup

    success = do_setup(force=cmd.force, yes=cmd.yes)
    sys.exit(0 if success else 1)


def run_view(cmd: ViewCmd) -> None:
    """Run the viewer with the given configuration."""
    from src.gsplay.core.dependency_check import check_environment_fast

    # Fast dependency check (fail-fast)
    is_ready, issues = check_environment_fast()

    if not is_ready:
        print("=" * 50)
        print("GSPlay - Missing Dependencies")
        print("=" * 50)
        print()
        for issue in issues:
            print(f"  [X] {issue}")
        print()
        print("To fix, run one of:")
        print("  gsplay setup     # Interactive setup wizard")
        print("  gsplay doctor    # Detailed diagnostics")
        print()
        print("Or install manually:")
        print("  ./install.sh     # Linux/macOS")
        print("  ./install.ps1    # Windows")
        sys.exit(1)

    # Now safe to import torch and heavy modules
    _init_heavy_imports()
    import torch

    from src.gsplay.config.settings import GSPlayConfig
    from src.gsplay.core.app import UniversalGSPlay

    setup_logging(cmd.log_level)

    # Convert GPU number to device string
    if cmd.gpu is not None:
        device = f"cuda:{cmd.gpu}"
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    logger.info("=== GSPlay ===")
    logger.info(f"Config: {cmd.config}")
    logger.info(f"Host: {cmd.host}")
    logger.info(f"Port: {cmd.port}")
    if cmd.stream_port != 0:
        logger.info(f"Stream port: {cmd.port + 1} (auto)")
    logger.info(f"Device: {device}")

    # Create viewer config
    viewer_config = GSPlayConfig(
        port=cmd.port,
        host=cmd.host,
        stream_port=cmd.stream_port,
        device=device,
        model_config_path=cmd.config,
        view_only=cmd.view_only,
        compact_ui=cmd.compact,
    )

    # Create viewer
    viewer = UniversalGSPlay(viewer_config)

    # Load model
    if cmd.config.is_dir():
        # PLY folder
        logger.info(f"Loading PLY files from directory: {cmd.config}")
        config_dict = {
            "module": "load-ply",
            "config": {
                "ply_folder": str(cmd.config),
                "device": device,
            },
        }
        viewer.load_model_from_config(config_dict, config_file=str(cmd.config))

    elif cmd.config.is_file() and cmd.config.suffix == ".json":
        # JSON config
        logger.info(f"Loading configuration from: {cmd.config}")
        with open(cmd.config) as f:
            config_dict = json.load(f)
        viewer.load_model_from_config(config_dict, config_file=str(cmd.config))

    else:
        logger.error(f"Invalid config: {cmd.config}")
        logger.error("Config must be either a PLY folder or JSON file")
        return

    # Setup viewer
    viewer.setup_viewer()

    # Run
    viewer.run()


def run_plugin_list(cmd: PluginListCmd) -> None:
    """List all discovered plugins."""
    from src.plugins.discovery import discover_plugins, get_plugin_info

    plugins = discover_plugins()
    if not plugins:
        print("No plugins discovered.")
        return

    print(f"\nDiscovered {len(plugins)} plugin(s):\n")
    print(f"{'Name':<20} {'Class':<30} {'Module'}")
    print("-" * 80)

    for name, cls in plugins.items():
        print(f"{name:<20} {cls.__name__:<30} {cls.__module__}")

    if cmd.verbose:
        print("\n" + "=" * 80)
        info = get_plugin_info()
        for name, details in info.items():
            print(f"\n{name}:")
            if "metadata" in details:
                meta = details["metadata"]
                print(f"  Name: {meta.get('name', 'N/A')}")
                print(f"  Description: {meta.get('description', 'N/A')}")
                print(f"  Extensions: {meta.get('file_extensions', [])}")
                print(f"  Version: {meta.get('version', 'N/A')}")
            elif "metadata_error" in details:
                print(f"  Metadata error: {details['metadata_error']}")


def run_plugin_info(cmd: PluginInfoCmd) -> None:
    """Show information about a specific plugin."""
    from src.plugins.discovery import discover_plugins

    plugins = discover_plugins()
    if cmd.name not in plugins:
        print(f"Plugin '{cmd.name}' not found.")
        print(f"Available plugins: {', '.join(plugins.keys())}")
        return

    cls = plugins[cmd.name]
    print(f"\nPlugin: {cmd.name}")
    print(f"  Class: {cls.__name__}")
    print(f"  Module: {cls.__module__}")

    if hasattr(cls, "metadata"):
        try:
            meta = cls.metadata()
            print("\nMetadata:")
            print(f"  Display Name: {meta.name}")
            print(f"  Description: {meta.description}")
            print(f"  File Extensions: {meta.file_extensions}")
            print(f"  Version: {meta.version}")
            if meta.config_schema:
                print(f"  Config Schema: {meta.config_schema.__name__}")
        except Exception as e:
            print(f"\nMetadata error: {e}")

    # Show required methods
    print("\nProtocol compliance:")
    for method in ["metadata", "can_load", "total_frames", "get_frame_at_time"]:
        has_method = hasattr(cls, method)
        status = "+" if has_method else "-"
        print(f"  {status} {method}")


def run_plugin_test(cmd: PluginTestCmd) -> None:
    """Test a plugin with the test harness."""
    from src.plugins.discovery import discover_plugins
    from src.plugins.testing import PluginTestHarness

    plugins = discover_plugins()
    if cmd.name not in plugins:
        print(f"Plugin '{cmd.name}' not found.")
        print(f"Available plugins: {', '.join(plugins.keys())}")
        return

    cls = plugins[cmd.name]

    # Load config
    config: dict = {}
    if cmd.config:
        if cmd.config.exists():
            with open(cmd.config) as f:
                config = json.load(f)
        else:
            print(f"Config file not found: {cmd.config}")
            return

    config["device"] = cmd.device

    print(f"\nTesting plugin: {cmd.name}")
    print(f"Config: {config}")
    print()

    harness = PluginTestHarness(cls)
    harness.run_all_tests(config, device=cmd.device)
    harness.print_results()


# ============================================================================
# Entry Point
# ============================================================================


def cli() -> None:
    """Entry point for the installed script."""
    # Check if first arg looks like a path (not a subcommand)
    # This allows `gsplay /path/to/ply` to work without `view` prefix
    if len(sys.argv) > 1:
        first_arg = sys.argv[1]
        subcommands = {
            "view",
            "setup",
            "doctor",
            "plugins-list",
            "plugins-info",
            "plugins-test",
            "-h",
            "--help",
        }
        if first_arg not in subcommands and not first_arg.startswith("-"):
            # Insert 'view' as the subcommand
            sys.argv.insert(1, "view")

    cmd = tyro.cli(MainCommand)

    if isinstance(cmd, ViewCmd):
        run_view(cmd)
    elif isinstance(cmd, SetupCmd):
        run_setup(cmd)
    elif isinstance(cmd, DoctorCmd):
        run_doctor(cmd)
    elif isinstance(cmd, PluginListCmd):
        run_plugin_list(cmd)
    elif isinstance(cmd, PluginInfoCmd):
        run_plugin_info(cmd)
    elif isinstance(cmd, PluginTestCmd):
        run_plugin_test(cmd)


if __name__ == "__main__":
    cli()
