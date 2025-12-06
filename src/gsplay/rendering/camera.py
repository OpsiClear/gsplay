"""
SuperSplat-style camera controls for the universal viewer.

Provides practical enhancements to viser's built-in orbit controls:
- Grid toggle
- Focus on content (auto-frame scene)
- Preset camera views (top, front, side, isometric)
- FOV adjustment
- Continuous rotation using quaternions (no gimbal lock)

Uses quaternion-based state management for smooth rotation through all orientations.
"""

import logging
import threading
import time

import numpy as np
import viser

# Import CameraState from its new home
from src.gsplay.rendering.camera_state import CameraState

# Import quaternion utilities
from src.gsplay.rendering.quaternion_utils import (
    quat_from_axis_angle,
    quat_from_look_at,
    quat_multiply,
    quat_normalize,
    quat_to_rotation_matrix,
    rotation_matrix_to_quat,
)

# Re-export factory functions from camera_ui for backward compatibility
from src.gsplay.rendering.camera_ui import (
    PlaybackButton,
    create_fps_control,
    create_playback_controls,
    create_quality_controls,
    create_supersplat_camera_controls,
    create_view_controls,
)

logger = logging.getLogger(__name__)

# Re-export CameraState for backward compatibility
__all__ = [
    "CameraState",
    "SuperSplatCamera",
    "create_view_controls",
    "create_fps_control",
    "create_quality_controls",
    "create_playback_controls",
    "create_supersplat_camera_controls",
    "PlaybackButton",
]


class SuperSplatCamera:
    """
    Camera enhancements for viser, inspired by SuperSplat.

    Uses quaternion-based state for smooth rotation without gimbal lock.
    """

    def __init__(
        self,
        server: viser.ViserServer,
        scene_bounds: dict | None = None,
    ):
        """
        Initialize camera controller.

        Parameters
        ----------
        server : viser.ViserServer
            Viser server instance
        scene_bounds : dict | None
            Scene bounds dictionary with keys: 'min_coords', 'max_coords', 'center', 'size'
        """
        self.server = server
        self.scene_bounds = scene_bounds

        # Grid
        self.grid_handle: viser.SceneNodeHandle | None = None
        self.grid_visible = False

        # World axis
        self.world_axis_handle: viser.SceneNodeHandle | None = None
        self.world_axis_visible = False

        # Auto-rotation state
        self._rotation_thread: threading.Thread | None = None
        self._rotation_active = False
        self._rotation_speed = 20.0  # degrees per second
        self._rotation_axis = "y"  # "y" for azimuth, "x" for elevation

        # Callback to trigger rerender (set by app during initialization)
        self._rerender_callback: callable | None = None

        # Explicit camera state (single source of truth)
        self.state: CameraState | None = None
        self.state_lock = threading.Lock()

        # Track which clients have been initialized with our camera state
        # Clients not in this set will have their on_update events ignored
        # (to prevent viser's default camera from overwriting our state)
        self._initialized_clients: set[int] = set()

        # Initialize
        self._setup_grid()
        self._setup_world_axis()
        self._initialize_state_from_camera()

        logger.info("SuperSplatCamera initialized (event-driven sync)")

    def _setup_grid(self) -> None:
        """Create the ground plane grid."""
        if self.scene_bounds is not None and "size" in self.scene_bounds:
            size = self.scene_bounds["size"]
            extent = float(np.max(size))
            grid_size = extent * 2
        else:
            grid_size = 20.0

        self.grid_handle = self.server.scene.add_grid(
            name="/supersplat_grid",
            width=grid_size,
            height=grid_size,
            plane="xz",
            cell_size=grid_size / 20,
            cell_color=(200, 200, 200),
            cell_thickness=1.0,
            section_size=grid_size / 4,
            section_color=(100, 100, 100),
            section_thickness=2.0,
            visible=False,
        )

        logger.debug(f"Created grid with size {grid_size}")

    def _setup_world_axis(self) -> None:
        """Create the world axis frame."""
        if self.scene_bounds is not None and "size" in self.scene_bounds:
            size = self.scene_bounds["size"]
            extent = float(np.max(size))
            axis_size = extent * 0.1
        else:
            axis_size = 2.0

        self.world_axis_handle = self.server.scene.add_frame(
            name="/supersplat_world_axis",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.0),
            axes_length=axis_size,
            axes_radius=axis_size * 0.05,
            visible=False,
        )

        logger.debug(f"Created world axis with size {axis_size}")

    def _initialize_state_from_camera(self) -> None:
        """
        Initialize camera state from current viser camera or scene bounds.
        """
        # Get first connected client
        first_client = None
        for client in self.server.get_clients().values():
            first_client = client
            break

        if first_client is None:
            # No clients - use default state based on scene bounds
            if self.scene_bounds is not None and "center" in self.scene_bounds:
                center = np.array(self.scene_bounds["center"])
                size = np.array(self.scene_bounds["size"])
                extent = float(np.linalg.norm(size))
                distance = extent * 2.5
            else:
                center = np.array([0.0, 0.0, 0.0])
                distance = 10.0

            # Default isometric view (45 deg azimuth, 30 deg elevation)
            state = CameraState(distance=distance, look_at=center)
            state.set_from_euler(45.0, 30.0, 0.0)

            with self.state_lock:
                self.state = state

            logger.debug("Initialized camera state with defaults (no clients)")
            return

        # Read from viser camera
        camera_pos = np.array(first_client.camera.position)
        look_at = np.array(first_client.camera.look_at)
        up_dir = np.array(first_client.camera.up_direction)

        # Calculate distance
        distance = float(np.linalg.norm(camera_pos - look_at))

        # Compute orientation from position/look_at/up_direction
        # NOTE: Do NOT use camera.wxyz directly! Viser uses OpenCV convention
        # (look=+Z, up=-Y) which differs from our convention (look=-Z, up=+Y).
        orientation = quat_from_look_at(camera_pos, look_at, up_dir)

        with self.state_lock:
            self.state = CameraState(
                orientation=orientation,
                distance=distance,
                look_at=look_at.copy(),
            )

        logger.debug(
            f"Initialized camera state from viser: distance={distance:.2f}"
        )

    def apply_state_to_camera(self, clients: list | None = None) -> None:
        """
        Apply the current camera state to viser camera(s).

        Uses quaternion orientation directly - no Euler conversion needed.

        Parameters
        ----------
        clients : list[viser.ClientHandle] | None
            Specific clients to update. If None, updates all connected clients.
        """
        if self.state is None:
            logger.warning("Cannot apply state - state not initialized")
            return

        with self.state_lock:
            orientation = self.state.orientation.copy()
            distance = self.state.distance
            look_at = self.state.look_at.copy()

        # Calculate camera position from orientation and distance
        # Camera looks down -Z in local space, so forward = R @ [0, 0, -1]
        R = quat_to_rotation_matrix(orientation)
        forward = R @ np.array([0.0, 0.0, -1.0])
        camera_pos = look_at - forward * distance

        # Calculate up direction from orientation
        up_dir = R @ np.array([0.0, 1.0, 0.0])

        # Apply to viser cameras
        # NOTE: Do NOT set camera.wxyz directly! Viser uses OpenCV convention (look=+Z, up=-Y)
        # which differs from our convention (look=-Z, up=+Y). Setting wxyz would cause viser
        # to recompute position/look_at/up_direction with wrong convention, causing 180° flip.
        # Instead, only set position/look_at/up_direction which are unambiguous.
        target_clients = clients or list(self.server.get_clients().values())
        for client in target_clients:
            with client.atomic():
                client.camera.position = tuple(camera_pos)
                client.camera.look_at = tuple(look_at)
                client.camera.up_direction = tuple(up_dir)

        logger.debug(f"Applied state to camera: distance={distance:.2f}")

    def sync_state_from_client(self, client) -> None:
        """
        Sync camera state from a specific client's camera.

        This should be called from client.camera.on_update to update
        camera.state when the user interacts with the camera.

        Parameters
        ----------
        client : viser.ClientHandle
            The client whose camera to read from
        """
        if self.state is None:
            return

        # Only sync from initialized clients (ignore viser's default camera)
        if client.client_id not in self._initialized_clients:
            return

        # Skip sync during auto-rotation - rotation loop owns the state
        if self._rotation_active:
            return

        try:
            camera_pos = np.array(client.camera.position)
            look_at = np.array(client.camera.look_at)
            up_dir = np.array(client.camera.up_direction)

            distance = float(np.linalg.norm(camera_pos - look_at))
            orientation = quat_from_look_at(camera_pos, look_at, up_dir)

            with self.state_lock:
                self.state.orientation = quat_normalize(orientation)
                self.state.distance = distance
                self.state.look_at = look_at.copy()

        except Exception as e:
            logger.debug(f"Error syncing state from client: {e}")

    def mark_client_initialized(self, client) -> None:
        """
        Mark a client as initialized (camera state has been applied).

        After this, on_update events from this client will update camera.state.

        Parameters
        ----------
        client : viser.ClientHandle
            The client to mark as initialized
        """
        self._initialized_clients.add(client.client_id)
        logger.debug(f"Client {client.client_id} marked as initialized")

    def toggle_grid(self) -> None:
        """Toggle grid visibility."""
        if self.grid_handle is not None:
            self.grid_visible = not self.grid_visible
            self.grid_handle.visible = self.grid_visible
            logger.info(f"Grid {'visible' if self.grid_visible else 'hidden'}")

    def set_preset_view(self, view: str) -> None:
        """
        Set camera to a preset view.

        Parameters
        ----------
        view : str
            One of: "top", "bottom", "front", "back", "left", "right", "iso"
        """
        if self.scene_bounds is None or "center" not in self.scene_bounds:
            logger.warning("No scene bounds available for preset views")
            return

        center = np.array(self.scene_bounds["center"])
        size = np.array(self.scene_bounds["size"])
        extent = float(np.linalg.norm(size))
        distance = extent * 1.5

        # Define preset angles (azimuth, elevation)
        presets = {
            "top": (0.0, 90.0, 0.0),
            "bottom": (0.0, -90.0, 0.0),
            "front": (0.0, 0.0, 0.0),
            "back": (180.0, 0.0, 0.0),
            "left": (270.0, 0.0, 0.0),
            "right": (90.0, 0.0, 0.0),
            "iso": (45.0, 30.0, 0.0),
        }

        if view not in presets:
            logger.error(f"Unknown view: {view}")
            return

        azimuth, elevation, roll = presets[view]

        # Update state
        if self.state is not None:
            with self.state_lock:
                self.state.set_from_euler(azimuth, elevation, roll)
                self.state.distance = distance
                self.state.look_at = center.copy()

            self.apply_state_to_camera()

        logger.info(f"Set camera to {view} view")

    def focus_on_bounds(
        self, bounds: dict | None = None, duration: float = 0.5
    ) -> None:
        """
        Focus camera on scene bounds.

        Parameters
        ----------
        bounds : dict | None
            Bounds dictionary to focus on. If None, uses scene_bounds.
        duration : float
            Animation duration in seconds (not currently implemented)
        """
        if bounds is None:
            bounds = self.scene_bounds

        if bounds is None or "center" not in bounds:
            logger.warning("No bounds available to focus on")
            return

        center = np.array(bounds["center"])
        size = np.array(bounds["size"])
        extent = float(np.linalg.norm(size))
        distance = extent * 1.5

        # Update state - keep current orientation
        if self.state is not None:
            with self.state_lock:
                self.state.distance = distance
                self.state.look_at = center.copy()

            self.apply_state_to_camera()

        logger.info(f"Focused camera on bounds: center={center}, distance={distance}")

    def start_auto_rotation(self, axis: str = "y", speed: float = 20.0) -> None:
        """
        Start continuous camera rotation around the scene.

        Uses quaternion multiplication for smooth rotation without gimbal lock.

        Parameters
        ----------
        axis : str
            Rotation axis: "y" (horizontal/azimuth) or "x" (vertical/elevation)
        speed : float
            Rotation speed in degrees per second
        """
        if self._rotation_active:
            logger.debug("Stopping existing rotation to start new one")
            self.stop_auto_rotation()

        if self.state is None:
            logger.warning("Cannot start rotation - state not initialized")
            return

        self._rotation_active = True
        self._rotation_axis = axis
        self._rotation_speed = speed

        def rotation_loop():
            """Continuous rotation using quaternion multiplication."""
            last_time = time.time()

            # Define rotation axis in world coordinates
            world_axis = np.array([0.0, 1.0, 0.0]) if axis == "y" else np.array([1.0, 0.0, 0.0])

            while self._rotation_active:
                try:
                    current_time = time.time()
                    dt = current_time - last_time
                    last_time = current_time

                    # Calculate rotation angle for this frame
                    angle_rad = np.radians(self._rotation_speed * dt)

                    # Create rotation quaternion
                    delta_quat = quat_from_axis_angle(world_axis, angle_rad)

                    # Apply rotation to current orientation
                    with self.state_lock:
                        self.state.orientation = quat_normalize(
                            quat_multiply(delta_quat, self.state.orientation)
                        )

                    # Apply to camera
                    self.apply_state_to_camera()

                    # Trigger rerender to update streaming output
                    if self._rerender_callback is not None:
                        try:
                            self._rerender_callback()
                        except Exception as e:
                            logger.debug(f"Rerender callback failed: {e}")

                    time.sleep(0.05)  # 20 FPS

                except Exception as e:
                    logger.error(f"Error in rotation loop: {e}")
                    self._rotation_active = False
                    break

        self._rotation_thread = threading.Thread(target=rotation_loop, daemon=True)
        self._rotation_thread.start()
        logger.info(f"Started auto-rotation around {axis} axis at {speed} deg/s")

    def stop_auto_rotation(self) -> None:
        """Stop continuous camera rotation."""
        if not self._rotation_active:
            return

        self._rotation_active = False
        if self._rotation_thread:
            self._rotation_thread.join(timeout=1.0)
            self._rotation_thread = None

        logger.info("Stopped auto-rotation")

    def set_rerender_callback(self, callback: callable) -> None:
        """Set callback to trigger rerender during auto-rotation.

        Parameters
        ----------
        callback : callable
            Function to call after each rotation frame update
        """
        self._rerender_callback = callback

    def update_scene_bounds(self, bounds: dict | None) -> None:
        """
        Update scene bounds and recreate grid if needed.

        Parameters
        ----------
        bounds : dict | None
            New scene bounds dictionary
        """
        self.scene_bounds = bounds

        if self.grid_handle is not None:
            self.grid_handle.remove()

        self._setup_grid()
        logger.info("Updated scene bounds and grid")

    # === Backwards Compatibility Methods ===
    # These methods are kept for compatibility with existing code but use
    # quaternions internally.

    def sync_state_from_camera(self) -> None:
        """
        Sync the camera state from the first initialized client.

        Backwards compatibility method. Prefers sync_state_from_client().
        """
        if self.state is None:
            logger.warning("Cannot sync state - state not initialized")
            return

        # Find first initialized client
        for client in self.server.get_clients().values():
            if client.client_id in self._initialized_clients:
                self.sync_state_from_client(client)
                return

        logger.debug("No initialized clients to sync from")

    def orbit_to_angle(
        self,
        azimuth: float,
        elevation: float = 0.0,
        roll: float = 0.0,
        preserve_distance: bool = True,
        preserve_lookat: bool = True,
        explicit_distance: float | None = None,
        explicit_lookat: np.ndarray | None = None,
    ) -> None:
        """
        Set camera to a specific angle (backwards compatibility).

        Now uses quaternion state internally.

        Parameters
        ----------
        azimuth : float
            Azimuth angle in degrees (0-360)
        elevation : float
            Elevation angle in degrees (-90 to 90)
        roll : float
            Roll angle in degrees
        preserve_distance : bool
            If True, maintains current camera distance
        preserve_lookat : bool
            If True, maintains current look-at point
        explicit_distance : float | None
            If provided, uses this exact distance
        explicit_lookat : np.ndarray | None
            If provided, uses this exact look-at point
        """
        if self.state is None:
            logger.warning("Cannot orbit - state not initialized")
            return

        with self.state_lock:
            # Set orientation from Euler angles
            self.state.set_from_euler(azimuth, elevation, roll)

            # Handle distance
            if explicit_distance is not None:
                self.state.distance = explicit_distance
            elif not preserve_distance and self.scene_bounds is not None:
                size = np.array(self.scene_bounds["size"])
                extent = float(np.linalg.norm(size))
                self.state.distance = extent * 1.5

            # Handle look-at
            if explicit_lookat is not None:
                self.state.look_at = explicit_lookat.copy()
            elif not preserve_lookat and self.scene_bounds is not None:
                self.state.look_at = np.array(self.scene_bounds["center"])

        self.apply_state_to_camera()

        logger.debug(f"Orbited to azimuth={azimuth}, elevation={elevation}")

    def calculate_azimuth_elevation(
        self,
        camera_pos: np.ndarray,
        look_at: np.ndarray,
        up_direction: np.ndarray | None = None,
        detect_flip: bool = True,
    ) -> tuple[float, float]:
        """
        Calculate azimuth and elevation from camera position (backwards compatibility).

        With quaternions, this is derived from state properties.

        Parameters
        ----------
        camera_pos : np.ndarray
            Camera position (not used - reads from state)
        look_at : np.ndarray
            Look-at point (not used - reads from state)
        up_direction : np.ndarray | None
            Camera up direction (not used)
        detect_flip : bool
            Whether to detect flip (not used with quaternions)

        Returns
        -------
        tuple[float, float]
            (azimuth, elevation) in degrees
        """
        if self.state is None:
            return (0.0, 0.0)

        with self.state_lock:
            azimuth = self.state.azimuth
            elevation = self.state.elevation

        return (azimuth, elevation)

    def _calculate_roll_from_camera(
        self,
        camera_pos: np.ndarray,
        look_at: np.ndarray,
        up_direction: np.ndarray,
        elevation: float,
    ) -> float:
        """
        Calculate roll angle (backwards compatibility).

        With quaternions, roll is derived from state.

        Returns
        -------
        float
            Roll angle in degrees
        """
        if self.state is None:
            return 0.0

        with self.state_lock:
            return self.state.roll

    def validate_state(self, state: CameraState) -> CameraState:
        """
        Validate camera state (backwards compatibility).

        With quaternion state, validation is minimal since quaternions
        auto-normalize and don't have Euler angle range issues.

        Parameters
        ----------
        state : CameraState
            State to validate

        Returns
        -------
        CameraState
            Validated state (copy)
        """
        return state.copy()
