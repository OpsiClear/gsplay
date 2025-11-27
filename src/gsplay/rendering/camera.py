"""
SuperSplat-style camera controls for the universal viewer.

Provides practical enhancements to viser's built-in orbit controls:
- Grid toggle
- Focus on content (auto-frame scene)
- Preset camera views (top, front, side, isometric)
- FOV adjustment
- Continuous rotation (no gimbal lock)

Note: Viser v1.0.15 doesn't support keyboard events or double-click natively.
For full SuperSplat controls, consider upgrading viser or using GUI buttons as alternatives.
"""

import logging
import threading
import time

import numpy as np
import viser

# Import CameraState from its new home
from src.gsplay.rendering.camera_state import CameraState

# Re-export factory functions from camera_ui for backward compatibility
from src.gsplay.rendering.camera_ui import (
    create_view_controls,
    create_render_controls,
    create_fps_control,
    create_quality_controls,
    create_playback_controls,
    create_supersplat_camera_controls,
    PlaybackButton,
)

logger = logging.getLogger(__name__)

# Re-export CameraState for backward compatibility
__all__ = [
    "CameraState",
    "SuperSplatCamera",
    "create_view_controls",
    "create_render_controls",
    "create_fps_control",
    "create_quality_controls",
    "create_playback_controls",
    "create_supersplat_camera_controls",
    "PlaybackButton",
]


class SuperSplatCamera:
    """
    Camera enhancements for viser, inspired by SuperSplat.

    Provides convenient camera controls that work within viser's current API:
    - Grid display
    - Focus on content (auto-frame)
    - Preset camera positions
    - FOV adjustment
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

        # Mouse drag state (for unlimited elevation orbit)
        self._last_camera_state = {}  # Track camera state per client
        self._intercept_thread: threading.Thread | None = None
        self._intercept_active = False

        # Hysteresis/cooldown for boundary push (prevents oscillation)
        self._last_push_time: dict[int, float] = {}  # Per-client cooldown tracking
        self._push_cooldown = 0.25  # 250ms cooldown after push

        # Explicit camera state (single source of truth)
        self.state: CameraState | None = None
        self.state_lock = threading.Lock()

        # Initialize
        self._setup_grid()
        self._setup_world_axis()
        self._initialize_state_from_camera()  # Initialize state before starting threads
        self._start_orbit_intercept()

        logger.info("SuperSplat-style camera controller initialized")

    def _setup_grid(self) -> None:
        """Create the ground plane grid."""
        # Determine grid size from scene bounds
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
            plane="xz",  # Ground plane
            cell_size=grid_size / 20,
            cell_color=(200, 200, 200),
            cell_thickness=1.0,
            section_size=grid_size / 4,
            section_color=(100, 100, 100),
            section_thickness=2.0,
            visible=False,  # Off by default
        )

        logger.debug(f"Created grid with size {grid_size}")

    def _setup_world_axis(self) -> None:
        """Create the world axis frame."""
        # Determine axis size from scene bounds
        if self.scene_bounds is not None and "size" in self.scene_bounds:
            size = self.scene_bounds["size"]
            extent = float(np.max(size))
            axis_size = extent * 0.1  # 10% of scene extent
        else:
            axis_size = 2.0

        # Create world axis frame at origin
        self.world_axis_handle = self.server.scene.add_frame(
            name="/supersplat_world_axis",
            wxyz=(1.0, 0.0, 0.0, 0.0),  # Identity quaternion (no rotation)
            position=(0.0, 0.0, 0.0),
            axes_length=axis_size,
            axes_radius=axis_size * 0.05,
            visible=False,  # Off by default
        )

        logger.debug(f"Created world axis with size {axis_size}")

    def _calculate_roll_from_camera(
        self,
        camera_pos: np.ndarray,
        look_at: np.ndarray,
        up_direction: np.ndarray,
        elevation: float,
    ) -> float:
        """
        Calculate camera roll angle from camera properties.

        Roll is determined by comparing the actual up vector to the expected
        base up vector (without roll) for the given elevation.

        Parameters
        ----------
        camera_pos : np.ndarray
            Camera position (3,)
        look_at : np.ndarray
            Look-at point (3,)
        up_direction : np.ndarray
            Camera up direction (3,)
        elevation : float
            Current elevation angle in degrees

        Returns
        -------
        float
            Roll angle in degrees (-180 to 180)
        """
        # Normalize up direction first (defensive)
        up_direction = up_direction / (np.linalg.norm(up_direction) + 1e-8)

        # Calculate view direction
        view_dir = look_at - camera_pos
        view_norm = np.linalg.norm(view_dir)
        if view_norm < 1e-6:
            return 0.0  # Camera at look_at point - no valid roll
        view_dir = view_dir / view_norm

        # Determine expected base up (without roll) based on elevation
        if abs(elevation) > 85:  # Near poles
            if abs(elevation) > 90:  # Flipped
                base_up = np.array([0, 0, 1]) if elevation > 0 else np.array([0, 0, -1])
            else:
                base_up = np.array([0, 0, -1]) if elevation > 0 else np.array([0, 0, 1])
        else:  # Away from poles
            base_up = (
                np.array([0, -1, 0]) if abs(elevation) > 90 else np.array([0, 1, 0])
            )

        # Project both up vectors onto plane perpendicular to view direction
        actual_up_proj = up_direction - view_dir * np.dot(up_direction, view_dir)
        base_up_proj = base_up - view_dir * np.dot(base_up, view_dir)

        # Check projection magnitudes (near-zero indicates degenerate case)
        actual_mag = np.linalg.norm(actual_up_proj)
        base_mag = np.linalg.norm(base_up_proj)
        if actual_mag < 1e-6 or base_mag < 1e-6:
            return 0.0  # Degenerate case - up aligned with view

        # Normalize projections
        actual_up_norm = actual_up_proj / actual_mag
        base_up_norm = base_up_proj / base_mag

        # Calculate roll angle using dot product (gives angle magnitude)
        cos_roll = np.clip(np.dot(actual_up_norm, base_up_norm), -1.0, 1.0)
        roll = np.degrees(np.arccos(cos_roll))

        # Apply deadband for small roll values (reduces jitter)
        if abs(roll) < 0.5:
            return 0.0

        # Determine roll sign using cross product
        cross = np.cross(base_up_norm, actual_up_norm)
        if np.dot(cross, view_dir) < 0:
            roll = -roll

        return roll

    def _initialize_state_from_camera(self) -> None:
        """
        Initialize camera state from current viser camera properties.

        Reads the first client's camera position, look_at, and up_direction,
        then derives azimuth, elevation, roll, and distance to populate
        the authoritative CameraState.
        """
        # Get first connected client (or use default if none)
        first_client = None
        for client in self.server.get_clients().values():
            first_client = client
            break

        if first_client is None:
            # No clients yet - use default state
            if self.scene_bounds is not None and "center" in self.scene_bounds:
                center = np.array(self.scene_bounds["center"])
                size = np.array(self.scene_bounds["size"])
                extent = float(np.linalg.norm(size))
                distance = extent * 2.5  # Increased from 1.5 for better default zoom
            else:
                center = np.array([0.0, 0.0, 0.0])
                distance = 10.0

            # Default: isometric view
            with self.state_lock:
                self.state = CameraState(
                    azimuth=45.0,
                    elevation=30.0,
                    roll=0.0,
                    distance=distance,
                    look_at=center,
                )
            logger.debug(
                "Initialized camera state with defaults (no clients connected)"
            )
            return

        # Read current camera properties
        camera_pos = np.array(first_client.camera.position)
        look_at = np.array(first_client.camera.look_at)
        up_direction = np.array(first_client.camera.up_direction)

        # Calculate distance
        distance = float(np.linalg.norm(camera_pos - look_at))

        # Calculate azimuth and elevation (with flip detection)
        azimuth, elevation = self.calculate_azimuth_elevation(
            camera_pos, look_at, up_direction
        )

        # Calculate roll using helper method
        roll = self._calculate_roll_from_camera(
            camera_pos, look_at, up_direction, elevation
        )

        # Create and store state
        with self.state_lock:
            self.state = CameraState(
                azimuth=azimuth,
                elevation=elevation,
                roll=roll,
                distance=distance,
                look_at=look_at.copy(),
            )

        logger.debug(
            f"Initialized camera state: azimuth={azimuth:.1f}°, elevation={elevation:.1f}°, roll={roll:.1f}°, distance={distance:.2f}"
        )

    def validate_state(self, state: CameraState) -> CameraState:
        """
        Validate and normalize camera state values to ensure they're in valid ranges.

        Parameters
        ----------
        state : CameraState
            State to validate

        Returns
        -------
        CameraState
            Validated state with normalized values
        """
        # Wrap azimuth to [0, 360)
        azimuth = state.azimuth % 360.0
        if azimuth < 0:
            azimuth += 360.0

        # Clamp elevation to [-180, 180]
        elevation = np.clip(state.elevation, -180.0, 180.0)

        # Wrap roll to [-180, 180]
        roll = state.roll
        while roll > 180.0:
            roll -= 360.0
        while roll < -180.0:
            roll += 360.0

        # Ensure distance is positive
        distance = max(state.distance, 0.1)

        return CameraState(
            azimuth=azimuth,
            elevation=elevation,
            roll=roll,
            distance=distance,
            look_at=state.look_at.copy(),
        )

    def apply_state_to_camera(self, clients: list | None = None) -> None:
        """
        Apply the current camera state to viser camera(s).

        This is the single method for translating our explicit state
        to viser's camera representation. All camera updates should
        go through this method to maintain consistency.

        Parameters
        ----------
        clients : list[viser.ClientHandle] | None
            Specific clients to update. If None, updates all connected clients.
        """
        if self.state is None:
            logger.warning("Cannot apply state - state not initialized")
            return

        # Validate state before applying
        with self.state_lock:
            validated_state = self.validate_state(self.state)
            # Update stored state with validated values
            self.state = validated_state

            # Read state values (still holding lock for consistent read)
            azimuth = validated_state.azimuth
            elevation = validated_state.elevation
            roll = validated_state.roll
            distance = validated_state.distance
            look_at = validated_state.look_at.copy()

        # Apply to viser cameras using existing orbit_to_angle method
        # This method already handles all the complex math (spherical coords, roll, flips)
        # We're just using it as our "state to camera" translator now
        # Pass explicit distance and look_at from state to preserve zoom/pan
        self.orbit_to_angle(
            azimuth=azimuth,
            elevation=elevation,
            roll=roll,
            preserve_distance=False,  # Ignored - using explicit_distance
            preserve_lookat=False,  # Ignored - using explicit_lookat
            explicit_distance=distance,  # From state - preserves user's zoom
            explicit_lookat=look_at,  # From state - preserves user's pan
        )

        logger.debug(
            f"Applied state to camera: azimuth={azimuth:.1f}°, elevation={elevation:.1f}°, roll={roll:.1f}°, distance={distance:.2f}"
        )

    def sync_state_from_camera(self) -> None:
        """
        Sync the camera state from the current viser camera position.

        This updates the internal state to match the current camera position,
        useful before starting operations like auto-rotation to ensure we
        start from the current viewpoint.
        """
        if self.state is None:
            logger.warning("Cannot sync state - state not initialized")
            return

        # Get first connected client
        clients = list(self.server.get_clients().values())
        if not clients:
            logger.debug("No clients connected, cannot sync state from camera")
            return

        try:
            client = clients[0]
            camera_pos = np.array(client.camera.position)
            look_at = np.array(client.camera.look_at)
            up_dir = np.array(client.camera.up_direction)

            # Calculate distance
            distance = float(np.linalg.norm(camera_pos - look_at))

            # Calculate angles from current camera position
            azimuth, elevation = self.calculate_azimuth_elevation(
                camera_pos, look_at, up_dir
            )

            # Calculate roll
            roll = self._calculate_roll_from_camera(
                camera_pos, look_at, up_dir, elevation
            )

            # Update state from camera
            with self.state_lock:
                self.state.azimuth = azimuth
                self.state.elevation = elevation
                self.state.roll = roll
                self.state.distance = distance
                self.state.look_at = look_at.copy()

            logger.debug(
                f"Synced state from camera: azimuth={azimuth:.1f}°, elevation={elevation:.1f}°, distance={distance:.2f}"
            )

        except Exception as e:
            logger.warning(f"Error syncing state from camera: {e}")

    def _start_orbit_intercept(self) -> None:
        """Start intercepting orbit controls to allow unlimited elevation."""
        self._intercept_active = True

        def intercept_loop():
            """Monitor camera changes and reapply with unlimited elevation."""
            cleanup_counter = 0
            while self._intercept_active:
                try:
                    if self.scene_bounds is None or "center" not in self.scene_bounds:
                        time.sleep(0.1)
                        continue

                    # Periodic cleanup of stale client entries (every ~5 seconds)
                    cleanup_counter += 1
                    if cleanup_counter >= 100:  # 100 iterations * 50ms = 5s
                        cleanup_counter = 0
                        active_ids = {id(c) for c in self.server.get_clients().values()}
                        stale_ids = [
                            cid
                            for cid in self._last_camera_state
                            if cid not in active_ids
                        ]
                        for cid in stale_ids:
                            del self._last_camera_state[cid]
                            self._last_push_time.pop(cid, None)

                    for client in self.server.get_clients().values():
                        client_id = id(client)

                        # Get current camera state
                        current_pos = np.array(client.camera.position)
                        current_lookat = np.array(client.camera.look_at)
                        current_up = np.array(client.camera.up_direction)

                        # Check if we have previous state
                        if client_id not in self._last_camera_state:
                            # Initialize
                            self._last_camera_state[client_id] = {
                                "position": current_pos.copy(),
                                "lookat": current_lookat.copy(),
                                "up": current_up.copy(),
                            }
                            continue

                        last_state = self._last_camera_state[client_id]

                        # Check if camera changed (user dragged with mouse)
                        # Use relative tolerance for large distances to avoid jitter
                        distance_from_origin = np.linalg.norm(current_pos)
                        atol = max(0.01, distance_from_origin * 0.0005)  # 0.05% of distance
                        pos_changed = not np.allclose(
                            current_pos, last_state["position"], atol=atol
                        )
                        lookat_changed = not np.allclose(
                            current_lookat, last_state["lookat"], atol=atol
                        )

                        if pos_changed or lookat_changed:
                            # Check if roll is present - if so, skip orbit intercept
                            # (Roll makes boundary detection unreliable)
                            elevation_rad = np.arctan2(
                                current_pos[1] - current_lookat[1],
                                np.sqrt(
                                    (current_pos[0] - current_lookat[0]) ** 2
                                    + (current_pos[2] - current_lookat[2]) ** 2
                                ),
                            )
                            elevation_deg = np.degrees(elevation_rad)

                            has_roll = False
                            if abs(abs(elevation_deg) - 90) < 5:  # Near poles
                                off_axis = np.sqrt(
                                    current_up[0] ** 2 + current_up[1] ** 2
                                )
                                has_roll = off_axis > 0.15  # Increased threshold for noise
                            else:  # Away from poles
                                off_axis = np.sqrt(
                                    current_up[0] ** 2 + current_up[2] ** 2
                                )
                                has_roll = off_axis > 0.15  # Increased threshold for noise

                            if has_roll:
                                # Roll is present - don't interfere with orbit intercept
                                self._last_camera_state[client_id] = {
                                    "position": current_pos.copy(),
                                    "lookat": current_lookat.copy(),
                                    "up": current_up.copy(),
                                }
                                continue

                            # Camera was moved by user - check if elevation is clamped
                            azimuth, elevation = self.calculate_azimuth_elevation(
                                current_pos, current_lookat, current_up
                            )

                            # Calculate movement delta
                            last_azimuth, last_elevation = (
                                self.calculate_azimuth_elevation(
                                    last_state["position"],
                                    last_state["lookat"],
                                    last_state["up"],
                                )
                            )
                            elevation_delta = elevation - last_elevation

                            # Check for stuck at boundaries
                            needs_push = False
                            new_elevation = elevation

                            # Check cooldown first - skip if recently pushed
                            current_time = time.time()
                            in_cooldown = (
                                client_id in self._last_push_time
                                and current_time - self._last_push_time[client_id]
                                < self._push_cooldown
                            )

                            if not in_cooldown:
                                # Case 1: Near ±90° poles (top/bottom)
                                # Detect at 3° but push to 96° (past detection zone)
                                if abs(abs(elevation) - 90) < 3:  # Within 3° of pole
                                    if abs(elevation) > 87 and abs(elevation_delta) < 0.8:
                                        # User is stuck at pole - push them over
                                        new_elevation = 96 if elevation > 0 else -96
                                        needs_push = True

                                # Case 2: Near ±180° boundary (upside-down horizon)
                                # This happens when elevation is near 180 or -180 (flipped at horizon)
                                elif (
                                    abs(abs(elevation) - 180) < 3
                                ):  # Within 3° of flipped horizon
                                    # Check if camera is flipped (up.y is negative)
                                    if current_up[1] < -0.5 and abs(elevation_delta) < 0.8:
                                        # User is stuck at 180/-180 boundary - push through
                                        if elevation > 0:
                                            new_elevation = -176  # Push to negative side
                                        else:
                                            new_elevation = 176  # Push to positive side
                                        needs_push = True

                                # Case 3: Near 0° but flipped (shouldn't normally happen, but handle it)
                                elif (
                                    abs(elevation) < 3
                                    and current_up[1] < -0.5
                                    and abs(elevation_delta) < 0.8
                                ):
                                    # Stuck at flipped zero - push to either side based on direction
                                    new_elevation = 4 if elevation_delta >= 0 else -4
                                    needs_push = True

                            if needs_push:
                                # Double-check camera hasn't moved since calculation
                                check_pos = np.array(client.camera.position)
                                if not np.allclose(check_pos, current_pos, atol=0.02):
                                    # Camera moved during calculation - skip push
                                    needs_push = False

                            if needs_push:
                                # Apply push with wraparound
                                logger.debug(
                                    f"Pushing camera past boundary: {elevation:.1f}° -> {new_elevation:.1f}°"
                                )

                                # Record cooldown time
                                self._last_push_time[client_id] = current_time

                                # Update state first, then apply to camera
                                if self.state is not None:
                                    with self.state_lock:
                                        self.state.azimuth = azimuth
                                        self.state.elevation = new_elevation
                                        # Keep existing roll, distance, and look_at from state
                                    self.apply_state_to_camera()

                            # Update last state
                            self._last_camera_state[client_id] = {
                                "position": current_pos.copy(),
                                "lookat": current_lookat.copy(),
                                "up": current_up.copy(),
                            }

                    time.sleep(0.05)  # Check at 20 FPS

                except Exception as e:
                    logger.error(f"Error in orbit intercept: {e}", exc_info=True)
                    time.sleep(0.1)

        self._intercept_thread = threading.Thread(target=intercept_loop, daemon=True)
        self._intercept_thread.start()
        logger.debug("Started orbit intercept thread for unlimited elevation")

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

        # Define view directions and up vectors
        views = {
            "top": (np.array([0, distance, 0]), np.array([0, 0, -1])),
            "bottom": (np.array([0, -distance, 0]), np.array([0, 0, 1])),
            "front": (np.array([0, 0, distance]), np.array([0, 1, 0])),
            "back": (np.array([0, 0, -distance]), np.array([0, 1, 0])),
            "left": (np.array([-distance, 0, 0]), np.array([0, 1, 0])),
            "right": (np.array([distance, 0, 0]), np.array([0, 1, 0])),
            "iso": (np.array([distance, distance, distance]), np.array([0, 1, 0])),
        }

        if view not in views:
            logger.error(f"Unknown view: {view}")
            return

        offset, up_dir = views[view]
        camera_pos = center + offset

        # Update camera for all clients
        for client in self.server.get_clients().values():
            with client.atomic():
                client.camera.position = tuple(camera_pos)
                client.camera.look_at = tuple(center)
                client.camera.up_direction = tuple(up_dir)

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
            Animation duration in seconds
        """
        if bounds is None:
            bounds = self.scene_bounds

        if bounds is None or "center" not in bounds:
            logger.warning("No bounds available to focus on")
            return

        center = np.array(bounds["center"])
        size = np.array(bounds["size"])
        extent = float(np.linalg.norm(size))

        # Calculate camera position (view from 45 degree angle)
        distance = extent * 1.5
        camera_pos = center + np.array([distance, distance, distance])

        # Update camera for all clients
        for client in self.server.get_clients().values():
            with client.atomic():
                client.camera.position = tuple(camera_pos)
                client.camera.look_at = tuple(center)

        logger.info(f"Focused camera on bounds: center={center}, distance={distance}")

    def start_auto_rotation(self, axis: str = "y", speed: float = 20.0) -> None:
        """
        Start continuous camera rotation around the scene.

        Parameters
        ----------
        axis : str
            Rotation axis: "y" (horizontal/azimuth) or "x" (vertical/elevation)
        speed : float
            Rotation speed in degrees per second
        """
        # Stop any existing rotation before starting new one (allows override)
        if self._rotation_active:
            logger.debug("Stopping existing rotation to start new one")
            self.stop_auto_rotation()

        if self.scene_bounds is None or "center" not in self.scene_bounds:
            logger.warning("No scene bounds available for auto-rotation")
            return

        # Sync state from current camera position before starting rotation
        # This ensures we start from the user's current viewpoint
        self.sync_state_from_camera()

        self._rotation_active = True
        self._rotation_axis = axis
        self._rotation_speed = speed

        def rotation_loop():
            """
            Continuous rotation loop running in separate thread.

            Uses explicit state management: increments azimuth or elevation
            in state each frame, then applies state to camera.
            """
            if self.state is None:
                logger.error("Cannot start auto-rotation - state not initialized")
                return

            # Capture initial state (distance, look_at, roll are preserved)
            with self.state_lock:
                initial_azimuth = self.state.azimuth
                initial_elevation = self.state.elevation
                initial_roll = self.state.roll

            logger.debug(
                f"Auto-rotation starting from state: azimuth={initial_azimuth:.1f}°, elevation={initial_elevation:.1f}°, roll={initial_roll:.1f}°"
            )

            while self._rotation_active:
                try:
                    # Calculate rotation delta for this frame
                    angle_delta_degrees = (
                        self._rotation_speed * 0.05
                    )  # 50ms updates (20 FPS)

                    # Update state with rotation
                    with self.state_lock:
                        if self._rotation_axis == "y":
                            # Rotate around Y axis (increment azimuth)
                            self.state.azimuth += angle_delta_degrees
                            # Wrap azimuth to [0, 360)
                            self.state.azimuth = self.state.azimuth % 360.0
                        else:
                            # Rotate around X axis (increment elevation)
                            self.state.elevation += angle_delta_degrees
                            # Clamp elevation to [-180, 180] (unlimited with wraparound)
                            if self.state.elevation > 180.0:
                                self.state.elevation = -180.0 + (
                                    self.state.elevation - 180.0
                                )
                            elif self.state.elevation < -180.0:
                                self.state.elevation = 180.0 + (
                                    self.state.elevation + 180.0
                                )

                    # Apply state to camera (handles all the complex math)
                    self.apply_state_to_camera()

                    time.sleep(0.05)  # 20 FPS updates

                except Exception as e:
                    logger.error(f"Error in rotation loop: {e}")
                    self._rotation_active = False
                    break

        self._rotation_thread = threading.Thread(target=rotation_loop, daemon=True)
        self._rotation_thread.start()
        logger.info(f"Started auto-rotation around {axis} axis at {speed}°/s")

    def stop_auto_rotation(self) -> None:
        """Stop continuous camera rotation."""
        if not self._rotation_active:
            return

        self._rotation_active = False
        if self._rotation_thread:
            self._rotation_thread.join(timeout=1.0)
            self._rotation_thread = None

        logger.info("Stopped auto-rotation")

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
        Smoothly orbit camera to a specific angle.

        Parameters
        ----------
        azimuth : float
            Azimuth angle in degrees (0-360, 0 = front)
        elevation : float
            Elevation angle in degrees (-180 to 180, continuous)
        roll : float
            Roll angle in degrees (-180 to 180, camera tilt)
        preserve_distance : bool
            If True, maintains current camera distance from look-at point
            (ignored if explicit_distance is provided)
        preserve_lookat : bool
            If True, maintains current look-at point (for panning)
            (ignored if explicit_lookat is provided)
        explicit_distance : float | None
            If provided, uses this exact distance instead of preserve logic
        explicit_lookat : np.ndarray | None
            If provided, uses this exact look-at point instead of preserve logic
        """
        if self.scene_bounds is None or "center" not in self.scene_bounds:
            logger.warning("No scene bounds available for orbit")
            return

        # Get current camera state from first client
        current_client = None
        for client in self.server.get_clients().values():
            current_client = client
            break

        if current_client is None:
            return

        # Determine look-at point and distance
        if explicit_lookat is not None:
            # Use explicit look-at from caller (e.g., from state)
            center = explicit_lookat.copy()
        elif preserve_lookat:
            center = np.array(current_client.camera.look_at)
        else:
            center = np.array(self.scene_bounds["center"])

        if explicit_distance is not None:
            # Use explicit distance from caller (e.g., from state)
            distance = explicit_distance
        elif preserve_distance:
            # Calculate current distance
            camera_pos = np.array(current_client.camera.position)
            distance = float(np.linalg.norm(camera_pos - center))
        else:
            # Use default distance
            size = np.array(self.scene_bounds["size"])
            extent = float(np.linalg.norm(size))
            distance = extent * 1.5

        # Normalize elevation to -90° to +90° range (with wraparound)
        # This allows continuous rotation past the poles
        normalized_elevation = elevation
        normalized_azimuth = azimuth
        flipped_up = False

        if elevation > 90:
            # Going over the top - flip to other side
            normalized_elevation = 180 - elevation
            normalized_azimuth = (azimuth + 180) % 360
            flipped_up = True
        elif elevation < -90:
            # Going under the bottom - flip to other side
            normalized_elevation = -180 - elevation
            normalized_azimuth = (azimuth + 180) % 360
            flipped_up = True

        # Convert to radians
        azimuth_rad = np.radians(normalized_azimuth)
        elevation_rad = np.radians(normalized_elevation)

        # Calculate position on sphere (spherical coordinates)
        # Y is up, rotate around Y for azimuth
        x = distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
        y = distance * np.sin(elevation_rad)
        z = distance * np.cos(elevation_rad) * np.cos(azimuth_rad)

        target_pos = center + np.array([x, y, z])

        # Calculate base up direction (before roll)
        if abs(normalized_elevation) > 85:
            # Near poles, adjust up direction
            if flipped_up:
                # Flipped over pole - invert up direction
                base_up = (
                    np.array([0, 0, 1])
                    if normalized_elevation > 0
                    else np.array([0, 0, -1])
                )
            else:
                base_up = (
                    np.array([0, 0, -1])
                    if normalized_elevation > 0
                    else np.array([0, 0, 1])
                )
        else:
            # Normal orientation
            base_up = np.array([0, -1, 0]) if flipped_up else np.array([0, 1, 0])

        # Apply roll rotation with improved thresholds
        # Use 0.5° deadband to prevent jitter from numerical noise
        if abs(roll) > 0.5:
            # Calculate view direction
            view_dir = center - target_pos
            view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-8)

            # Rotate up vector around view direction by roll angle
            roll_rad = np.radians(roll)
            # Rodrigues' rotation formula
            cos_roll = np.cos(roll_rad)
            sin_roll = np.sin(roll_rad)
            up_dir = (
                base_up * cos_roll
                + np.cross(view_dir, base_up) * sin_roll
                + view_dir * np.dot(view_dir, base_up) * (1 - cos_roll)
            )
            # Normalize to ensure unit vector (numerical stability)
            up_dir = up_dir / (np.linalg.norm(up_dir) + 1e-8)
        else:
            up_dir = base_up

        # Update camera for all clients
        for client in self.server.get_clients().values():
            with client.atomic():
                client.camera.position = tuple(target_pos)
                client.camera.look_at = tuple(center)
                client.camera.up_direction = tuple(up_dir)

        logger.debug(
            f"Orbited to azimuth={azimuth}°, elevation={elevation}° (distance={distance:.2f})"
        )

    def calculate_azimuth_elevation(
        self,
        camera_pos: np.ndarray,
        look_at: np.ndarray,
        up_direction: np.ndarray | None = None,
    ) -> tuple[float, float]:
        """
        Calculate azimuth and elevation from camera position.

        Parameters
        ----------
        camera_pos : np.ndarray
            Camera position
        look_at : np.ndarray
            Look-at point (scene center)
        up_direction : np.ndarray | None
            Camera up direction (to detect flipped orientation)

        Returns
        -------
        tuple[float, float]
            (azimuth, elevation) in degrees (-180 to 180 for continuous rotation)
        """
        # Vector from look-at to camera
        to_camera = camera_pos - look_at

        # Calculate azimuth (rotation around Y axis)
        # atan2(x, z) gives angle in XZ plane
        azimuth_rad = np.arctan2(to_camera[0], to_camera[2])
        azimuth = np.degrees(azimuth_rad)
        if azimuth < 0:
            azimuth += 360

        # Calculate elevation (angle above/below horizontal plane)
        horizontal_dist = np.sqrt(to_camera[0] ** 2 + to_camera[2] ** 2)
        elevation_rad = np.arctan2(to_camera[1], horizontal_dist)
        elevation = np.degrees(elevation_rad)

        # Check if camera is flipped (up direction is inverted)
        if up_direction is not None:
            # First, detect if roll is present
            # Roll is present if up vector has significant off-axis components
            # Near poles: check X and Y components (up should be along Z)
            # Away from poles: check X and Z components (up should be along Y)
            # Use 0.15 threshold to filter numerical noise
            has_roll = False
            if abs(abs(elevation) - 90) < 5:  # Near poles
                # Up should be purely along Z axis [0, 0, ±1]
                off_axis_magnitude = np.sqrt(
                    up_direction[0] ** 2 + up_direction[1] ** 2
                )
                has_roll = off_axis_magnitude > 0.15
            else:  # Away from poles
                # Up should be purely along Y axis [0, ±1, 0]
                off_axis_magnitude = np.sqrt(
                    up_direction[0] ** 2 + up_direction[2] ** 2
                )
                has_roll = off_axis_magnitude > 0.15

            # Only apply flip detection if roll is not present
            # (Roll rotates the up vector, making flip detection unreliable)
            if not has_roll:
                # Detect flip based on up direction
                # Normal up is [0, 1, 0] (or [0, 0, -1] near top pole, [0, 0, 1] near bottom pole)
                # Flipped up is [0, -1, 0] (or [0, 0, 1] near top pole, [0, 0, -1] near bottom pole)

                # Near poles (elevation near ±90°), up is along Z-axis
                if abs(abs(elevation) - 90) < 5:  # Within 5° of poles
                    # Check Z component - flipped if sign is opposite to expected
                    if elevation > 85:  # Near top pole (+90°)
                        # Normal: up=[0,0,-1], Flipped: up=[0,0,+1]
                        is_flipped = up_direction[2] > 0.5
                    elif elevation < -85:  # Near bottom pole (-90°)
                        # Normal: up=[0,0,+1], Flipped: up=[0,0,-1]
                        is_flipped = up_direction[2] < -0.5
                    else:
                        # Shouldn't reach here, but fallback to Y check
                        is_flipped = up_direction[1] < -0.5
                else:
                    # Away from poles, check Y component
                    # Normal: up=[0,+1,0], Flipped: up=[0,-1,0]
                    is_flipped = up_direction[1] < -0.5

                if is_flipped:
                    # Camera is flipped - adjust angles to extended range
                    if elevation > 0:
                        elevation = 180 - elevation
                    else:
                        elevation = -180 - elevation
                    azimuth = (azimuth + 180) % 360

        return azimuth, elevation

    def update_scene_bounds(self, bounds: dict | None) -> None:
        """
        Update scene bounds and recreate grid if needed.

        Parameters
        ----------
        bounds : dict | None
            New scene bounds dictionary
        """
        self.scene_bounds = bounds

        # Recreate grid with new size
        if self.grid_handle is not None:
            self.grid_handle.remove()

        self._setup_grid()
        logger.info("Updated scene bounds and grid")
