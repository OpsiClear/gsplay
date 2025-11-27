#!/usr/bin/env python3
"""CLI utility to discover and stop orphaned GSPlay instances.

This script finds GSPlay processes that may have been orphaned (e.g., after
a launcher restart) and provides options to stop them.

Usage:
    # List all GSPlay processes
    uv run python -m gsplay_launcher.cli.cleanup

    # Stop all orphaned GSPlay processes
    uv run python -m gsplay_launcher.cli.cleanup --stop

    # Stop specific process by PID
    uv run python -m gsplay_launcher.cli.cleanup --stop --pid 12345

    # Force kill (no graceful shutdown)
    uv run python -m gsplay_launcher.cli.cleanup --stop --force
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import psutil


@dataclass
class GSPlayProcess:
    """Information about a GSPlay process."""

    pid: int
    port: int | None
    config_path: str | None
    memory_mb: float
    cpu_percent: float
    status: str
    cmdline: list[str]


def find_gsplay_processes() -> list[GSPlayProcess]:
    """Find all running GSPlay processes.

    Returns
    -------
    list[GSPlayProcess]
        List of GSPlay process information.
    """
    processes = []

    for proc in psutil.process_iter(["pid", "name", "cmdline", "status"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            cmdline_str = " ".join(cmdline).lower()

            # Look for gsplay processes
            # They typically have "gsplay" in the command or run main.py from gsplay
            is_gsplay = (
                "gsplay" in cmdline_str
                and "cleanup" not in cmdline_str  # Exclude this script
                and "launcher" not in cmdline_str  # Exclude launcher
            )

            if not is_gsplay:
                continue

            # Extract port from command line arguments
            port = None
            config_path = None
            for i, arg in enumerate(cmdline):
                if arg == "--port" and i + 1 < len(cmdline):
                    try:
                        port = int(cmdline[i + 1])
                    except ValueError:
                        pass
                # Config path is usually a positional argument (path ending in / or .json)
                if (arg.endswith("/") or arg.endswith(".json")) and "/" in arg:
                    config_path = arg

            # Get process stats
            try:
                memory_mb = proc.memory_info().rss / (1024 * 1024)
                cpu_percent = proc.cpu_percent(interval=0.1)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                memory_mb = 0.0
                cpu_percent = 0.0

            processes.append(
                GSPlayProcess(
                    pid=proc.pid,
                    port=port,
                    config_path=config_path,
                    memory_mb=memory_mb,
                    cpu_percent=cpu_percent,
                    status=proc.info.get("status", "unknown"),
                    cmdline=cmdline,
                )
            )

        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    return processes


def stop_process(pid: int, force: bool = False, timeout: float = 10.0) -> bool:
    """Stop a process by PID.

    This is a wrapper that uses the shared stop_process from process_manager,
    with CLI-friendly output.

    Parameters
    ----------
    pid : int
        Process ID to stop.
    force : bool
        If True, force kill immediately without graceful shutdown.
    timeout : float
        Timeout for graceful shutdown before force kill.

    Returns
    -------
    bool
        True if process was stopped successfully.
    """
    import sys
    from gsplay_launcher.services.process_manager import stop_process as _stop_process

    action = "Force killing" if force else "Terminating"
    print(f"  {action} PID {pid}...", end=" ", flush=True)

    success = _stop_process(pid, force=force, timeout=timeout)

    if success:
        print("✓ stopped")
    else:
        print("✗ failed")
    sys.stdout.flush()

    return success


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Discover and stop orphaned GSPlay instances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    List all GSPlay processes
  %(prog)s --stop             Stop all GSPlay processes
  %(prog)s --stop --pid 1234  Stop specific process
  %(prog)s --stop --force     Force kill without graceful shutdown
        """,
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop discovered processes (default: list only)",
    )
    parser.add_argument(
        "--pid",
        type=int,
        help="Stop only this specific PID",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force kill without graceful shutdown",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Timeout for graceful shutdown (default: 10s)",
    )

    args = parser.parse_args()

    print("Scanning for GSPlay processes...\n")

    processes = find_gsplay_processes()

    if not processes:
        print("No GSPlay processes found.")
        return 0

    # Filter by PID if specified
    if args.pid:
        processes = [p for p in processes if p.pid == args.pid]
        if not processes:
            print(f"No GSPlay process found with PID {args.pid}")
            return 1

    # Display processes
    print(f"Found {len(processes)} GSPlay process(es):\n")
    print(f"{'PID':<8} {'Port':<8} {'Memory':<10} {'Status':<12} {'Config'}")
    print("-" * 70)

    for proc in processes:
        port_str = str(proc.port) if proc.port else "-"
        memory_str = f"{proc.memory_mb:.1f} MB"
        config_str = proc.config_path or "-"
        if len(config_str) > 30:
            config_str = "..." + config_str[-27:]
        print(f"{proc.pid:<8} {port_str:<8} {memory_str:<10} {proc.status:<12} {config_str}")

    print()

    # Stop processes if requested
    if args.stop:
        print("Stopping processes...\n")
        stopped = 0
        failed = 0

        for proc in processes:
            success = stop_process(proc.pid, force=args.force, timeout=args.timeout)
            if success:
                stopped += 1
            else:
                failed += 1

        print()
        print(f"Summary: {stopped} stopped, {failed} failed")
        return 0 if failed == 0 else 1

    else:
        print("Use --stop to terminate these processes.")
        print("Use --stop --force for immediate termination.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
