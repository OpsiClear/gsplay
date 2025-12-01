"""Instance management service - orchestrates all operations."""

from __future__ import annotations

import logging
from pathlib import Path

import psutil

from gsplay_launcher.config import LauncherConfig
from gsplay_launcher.models import InstanceStatus, LauncherState, GSPlayInstance
from gsplay_launcher.services.port_allocator import PortAllocator
from gsplay_launcher.services.process_manager import ProcessManager, ProcessStartError
from gsplay_launcher.services.state_persistence import StatePersistence

logger = logging.getLogger(__name__)


class InstanceNotFoundError(Exception):
    """Raised when an instance is not found."""

    def __init__(self, instance_id: str) -> None:
        self.instance_id = instance_id
        super().__init__(f"Instance not found: {instance_id}")


class PortInUseError(Exception):
    """Raised when a port is already in use."""

    def __init__(self, port: int, by_instance: str | None = None) -> None:
        self.port = port
        self.by_instance = by_instance
        msg = f"Port {port} is already in use"
        if by_instance:
            msg += f" by instance {by_instance}"
        super().__init__(msg)


class ConfigPathError(Exception):
    """Raised when config path is invalid."""

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"Invalid config path '{path}': {reason}")


class InstanceManager:
    """Manages gsplay instance lifecycle.

    This service orchestrates process management, port allocation,
    and state persistence for GSPlay instances.

    Parameters
    ----------
    config : LauncherConfig
        Launcher configuration.
    """

    def __init__(self, config: LauncherConfig) -> None:
        self.config = config
        self._persistence = StatePersistence(config.state_file)
        self._port_allocator = PortAllocator(
            start_port=config.gsplay_port_start,
            end_port=config.gsplay_port_end,
            host=config.host,
        )
        self._process_manager = ProcessManager(
            gsplay_script=config.gsplay_script,
            python_cmd="uv",
            stop_timeout=config.process_stop_timeout,
        )
        self._state: LauncherState = LauncherState()

    def initialize(self) -> None:
        """Initialize manager: load state and reconcile with running processes."""
        self.config.ensure_data_dir()
        self._state = self._persistence.load()
        self._reconcile_state()
        logger.info("Instance manager initialized with %d instances", len(self._state.instances))

    def _reconcile_state(self) -> None:
        """Reconcile persisted state with actual process states.

        Called on startup to detect orphaned or dead processes.
        """
        changed = False

        for instance in list(self._state.instances.values()):
            if instance.status in (InstanceStatus.RUNNING, InstanceStatus.STARTING):
                if instance.pid and self._process_manager.is_running(instance.pid):
                    # Process still running - mark as orphaned
                    instance.mark_orphaned()
                    logger.info(
                        "Found orphaned instance: %s (PID: %d)",
                        instance.id,
                        instance.pid,
                    )
                    changed = True
                else:
                    # Process died
                    instance.mark_stopped()
                    logger.info("Instance %s no longer running", instance.id)
                    changed = True

        if changed:
            self._persistence.save(self._state)

    def create_and_start(
        self,
        config_path: str,
        name: str = "",
        port: int | None = None,
        host: str | None = None,
        stream_port: int = 0,
        gpu: int | None = None,
        cache_size: int = 100,
        view_only: bool = False,
        compact: bool = False,
        log_level: str = "INFO",
        custom_ip: str | None = None,
    ) -> GSPlayInstance:
        """Create and start a new gsplay instance.

        Parameters
        ----------
        config_path : str
            Path to PLY folder or JSON config.
        name : str
            Human-readable name.
        port : int | None
            Port number (auto-assigned if None).
        host : str | None
            Host to bind to (uses config.gsplay_host if None).
        stream_port : int
            WebSocket stream port (0 = disabled).
        gpu : int | None
            GPU device number.
        cache_size : int
            Frame cache size.
        view_only : bool
            Hide editing UI.
        compact : bool
            Use compact/mobile UI.
        log_level : str
            Logging level.

        Returns
        -------
        GSPlayInstance
            Created and started instance.

        Raises
        ------
        ConfigPathError
            If config path doesn't exist.
        PortInUseError
            If requested port is in use.
        ProcessStartError
            If gsplay fails to start.
        """
        # Validate config path (handle both absolute and relative paths)
        path = Path(config_path).resolve()
        if not path.exists():
            raise ConfigPathError(config_path, "Path does not exist")

        # Determine port
        if port is not None:
            # User specified port - must be even (odd ports reserved for streaming)
            if port % 2 != 0:
                logger.warning(
                    "Requested odd port %d adjusted to %d (odd ports reserved for streaming)",
                    port,
                    port - 1,
                )
                port = port - 1
            # Check availability of both the viser port and stream port (port+1)
            if not self._port_allocator.is_available(port):
                existing = self._find_instance_by_port(port)
                raise PortInUseError(port, existing.id if existing else None)
            if not self._port_allocator.is_available(port + 1):
                raise PortInUseError(
                    port + 1,
                    f"Port {port + 1} (stream port for {port}) is not available",
                )
            assigned_port = port
        else:
            # Auto-assign port (allocator returns even ports with port+1 available)
            used_ports = {
                inst.port
                for inst in self._state.instances.values()
                if inst.is_active
            }
            assigned_port = self._port_allocator.find_available(
                exclude=used_ports,
                start_hint=self._state.next_port_hint,
            )
            if assigned_port is None:
                raise PortInUseError(0, "No available ports in range")
            # Hint for next allocation: skip to next even port (+2)
            self._state.next_port_hint = assigned_port + 2

        # Determine host (use config default if not specified)
        assigned_host = host if host is not None else self.config.gsplay_host

        # Create instance
        instance = GSPlayInstance(
            name=name or f"GSPlay-{assigned_port}",
            config_path=str(path.resolve()),
            port=assigned_port,
            host=assigned_host,
            stream_port=stream_port,
            gpu=gpu,
            cache_size=cache_size,
            view_only=view_only,
            compact=compact,
            log_level=log_level,
            custom_ip=custom_ip,
        )

        # Start the process
        instance.mark_starting()
        try:
            pid = self._process_manager.start(instance)
            instance.mark_running(pid)
            logger.info(
                "Started instance %s on port %d (PID: %d)",
                instance.id,
                assigned_port,
                pid,
            )
        except ProcessStartError as e:
            instance.mark_failed(str(e))
            self._state.instances[instance.id] = instance
            self._persistence.save(self._state)
            raise

        # Save state
        self._state.instances[instance.id] = instance
        self._persistence.save(self._state)

        return instance

    def stop(self, instance_id: str) -> GSPlayInstance:
        """Stop a running instance.

        Parameters
        ----------
        instance_id : str
            Instance ID to stop.

        Returns
        -------
        GSPlayInstance
            Updated instance.

        Raises
        ------
        InstanceNotFoundError
            If instance not found.
        """
        instance = self._state.instances.get(instance_id)
        if instance is None:
            raise InstanceNotFoundError(instance_id)

        if instance.pid and self._process_manager.is_running(instance.pid):
            instance.mark_stopping()
            self._persistence.save(self._state)

            self._process_manager.stop(instance.pid)

        instance.mark_stopped()
        self._persistence.save(self._state)

        logger.info("Stopped instance %s", instance_id)
        return instance

    def delete(self, instance_id: str) -> bool:
        """Delete an instance (stops it first if running).

        Parameters
        ----------
        instance_id : str
            Instance ID to delete.

        Returns
        -------
        bool
            True if deleted.

        Raises
        ------
        InstanceNotFoundError
            If instance not found.
        """
        instance = self._state.instances.get(instance_id)
        if instance is None:
            raise InstanceNotFoundError(instance_id)

        # Stop if running
        if instance.is_active and instance.pid:
            self._process_manager.stop(instance.pid)

        del self._state.instances[instance_id]
        self._persistence.save(self._state)

        logger.info("Deleted instance %s", instance_id)
        return True

    def get(self, instance_id: str) -> GSPlayInstance:
        """Get instance by ID.

        Parameters
        ----------
        instance_id : str
            Instance ID.

        Returns
        -------
        GSPlayInstance
            Instance.

        Raises
        ------
        InstanceNotFoundError
            If instance not found.
        """
        instance = self._state.instances.get(instance_id)
        if instance is None:
            raise InstanceNotFoundError(instance_id)

        # Sync status with actual process
        if self._sync_instance_status(instance):
            self._persistence.save(self._state)
        return instance

    def list_all(self) -> list[GSPlayInstance]:
        """List all instances, pruning any that have already stopped."""
        changed = False
        to_delete: list[str] = []

        # Work on a copy so we can safely prune terminated entries
        for instance_id, instance in list(self._state.instances.items()):
            if self._sync_instance_status(instance):
                changed = True

            # Automatically remove cleanly stopped instances to keep the UI in sync
            if instance.status == InstanceStatus.STOPPED:
                to_delete.append(instance_id)

        if to_delete:
            for instance_id in to_delete:
                del self._state.instances[instance_id]
            changed = True
            logger.info("Pruned stopped instances: %s", ", ".join(to_delete))

        if changed:
            self._persistence.save(self._state)

        return list(self._state.instances.values())

    def get_next_available_port(self) -> int | None:
        """Get next available port.

        Returns
        -------
        int | None
            Available port or None.
        """
        used_ports = {
            inst.port
            for inst in self._state.instances.values()
            if inst.is_active
        }
        return self._port_allocator.find_available(
            exclude=used_ports,
            start_hint=self._state.next_port_hint,
        )

    def _find_instance_by_port(self, port: int) -> GSPlayInstance | None:
        """Find active instance using a specific port."""
        for instance in self._state.instances.values():
            if instance.port == port and instance.is_active:
                return instance
        return None

    def _sync_instance_status(self, instance: GSPlayInstance) -> bool:
        """Sync instance status with actual process state.

        Returns
        -------
        bool
            True if the instance state changed.
        """
        if instance.pid is None:
            return False

        if self._process_manager.is_running(instance.pid):
            return False

        exit_code: int | None = None
        try:
            proc = psutil.Process(instance.pid)
            exit_code = proc.wait(timeout=0)
        except psutil.NoSuchProcess:
            exit_code = 0
        except psutil.TimeoutExpired:
            return False
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("Could not determine exit code for %s: %s", instance.id, e)

        if exit_code and exit_code != 0:
            instance.mark_failed(f"Process exited with code {exit_code}")
            logger.warning("Instance %s exited unexpectedly (code %s)", instance.id, exit_code)
        else:
            instance.mark_stopped()
            logger.info("Instance %s stopped (process exited)", instance.id)

        return True
