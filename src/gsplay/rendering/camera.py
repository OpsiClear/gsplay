"""
Camera controller for the GSPlay viewer.

DESIGN: Explicit Mode-Based Camera Ownership
============================================

The camera can be in one of two modes:

1. USER_MODE: Viser owns the camera
   - User can orbit/pan/zoom with mouse
   - We sync FROM viser to update our spherical state
   - Slider changes are processed normally

2. APP_MODE: We own the camera
   - During rotation, presets, programmatic changes
   - We push TO viser, ignore callbacks FROM viser
   - Slider callbacks are blocked

This explicit ownership model eliminates race conditions and makes
the control flow easy to understand.

Mode transitions:
- start_auto_rotation() → APP_MODE
- stop_auto_rotation()  → USER_MODE (with brief cooldown)
- set_preset_view()     → APP_MODE briefly, then USER_MODE
- slider interaction    → Only in USER_MODE
"""

import logging
import threading
import time
from enum import Enum
from typing import TYPE_CHECKING, Callable

import numpy as np
import viser

from .camera_state import CameraState
from .camera_ui import (
    PlaybackButton,
    create_fps_control,
    create_playback_controls,
    create_quality_controls,
    create_supersplat_camera_controls,
    create_view_controls,
)
from .quaternion_utils import quat_from_axis_angle, quat_from_euler_deg, quat_multiply, quat_to_rotation_matrix

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class CameraMode(Enum):
    """Camera ownership mode."""
    USER = "user"  # Viser owns camera, we sync from it
    APP = "app"    # We own camera, we push to viser

__all__ = [
    "CameraState",
    "CameraController",
    "SuperSplatCamera",  # Alias for backwards compatibility
    "create_view_controls",
    "create_fps_control",
    "create_quality_controls",
    "create_playback_controls",
    "create_supersplat_camera_controls",
    "PlaybackButton",
]

# Default distance multiplier for camera positioning
# distance = scene_diagonal * DEFAULT_DISTANCE_MULTIPLIER
DEFAULT_DISTANCE_MULTIPLIER = 2.5


class CameraController:
    """
    Camera controller with proper thread safety and state management.

    Uses spherical coordinates as primary representation.
    Rotation is driven by render loop calling rotation_step().
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
            Scene bounds with keys: 'min_coords', 'max_coords', 'center', 'size'
        """
        self.server = server
        self.scene_bounds = scene_bounds

        # Camera state - single source of truth
        self._state: CameraState | None = None
        self._lock = threading.Lock()  # ONE lock for all state access

        # Mode-based ownership (replaces scattered flags)
        self._mode = CameraMode.USER
        self._mode_until: float = 0.0  # APP mode extends until this time (for cooldown)

        # Rotation parameters (only used when mode == APP during rotation)
        self._rotation_active = False  # Specific flag for render loop to check
        self._rotation_speed = 20.0  # degrees per second (positive=CW, negative=CCW)
        self._rotation_axis = "y"
        self._last_rotation_time: float = 0.0

        # Quaternion-based rotation state (avoids gimbal lock)
        # These capture viser's exact state at rotation start
        self._rotation_base_wxyz: np.ndarray | None = None
        self._rotation_base_position: np.ndarray | None = None
        self._rotation_look_at: np.ndarray | None = None
        self._rotation_accumulated_angle: float = 0.0

        # Headless rotation state (for rendering without viser clients)
        # Updated by rotation_step() so streaming continues when browser closes
        self._headless_wxyz: np.ndarray | None = None
        self._headless_position: np.ndarray | None = None
        self._headless_fov: float = 0.82  # Default ~47 degrees
        self._headless_aspect: float = 16.0 / 9.0

        # Client tracking
        self._initialized_clients: set[int] = set()

        # UI slider sync callback (set by camera_ui.py)
        self._slider_sync_callback: Callable[[], None] | None = None
        # Rerender callback (set by app.py to trigger re-render after camera changes)
        self._rerender_callback: Callable[[], None] | None = None

        # Grid and axis handles
        self.grid_handle: viser.SceneNodeHandle | None = None
        self.grid_visible = False
        self.world_axis_handle: viser.SceneNodeHandle | None = None
        self.world_axis_visible = False

        # Initialize
        self._setup_grid()
        self._setup_world_axis()
        self._initialize_state()

        logger.info("CameraController initialized")

    # =========================================================================
    # State Access (Thread-Safe)
    # =========================================================================

    @property
    def state(self) -> CameraState | None:
        """Get current camera state (for compatibility)."""
        return self._state

    @property
    def state_lock(self) -> threading.Lock:
        """Get state lock (for compatibility)."""
        return self._lock

    def get_state(self) -> CameraState | None:
        """Get thread-safe copy of current state."""
        with self._lock:
            return self._state.copy() if self._state else None

    # =========================================================================
    # Mode Management (Camera Ownership)
    # =========================================================================

    def is_app_controlled(self) -> bool:
        """Check if app currently owns the camera (blocks viser sync)."""
        if self._mode == CameraMode.APP:
            return True
        # Also check timed extension (cooldown after APP mode ends)
        if time.perf_counter() < self._mode_until:
            return True
        return False

    def _enter_app_mode(self) -> None:
        """Take ownership of camera (during rotation, presets, etc.)."""
        self._mode = CameraMode.APP

    def _enter_user_mode(self, cooldown: float = 0.3) -> None:
        """Release camera to user with brief cooldown to prevent race conditions."""
        self._mode = CameraMode.USER
        self._mode_until = time.perf_counter() + cooldown

    # =========================================================================
    # Viser Synchronization
    # =========================================================================

    def update_from_viser(self, client) -> None:
        """
        Sync state from viser camera (called on client.camera.on_update).

        Only runs in USER mode - when app owns camera, viser callbacks are ignored.

        Includes pole-crossing detection: if user orbits past a pole (elevation
        hits ±89° while moving toward it), the camera flips through to the other
        side (azimuth +180°, elevation inverts) for trackball-like behavior.

        Parameters
        ----------
        client : viser.ClientHandle
            The client whose camera changed
        """
        if self._state is None:
            return

        # Only sync from viser in USER mode
        if self.is_app_controlled():
            return

        # Skip uninitialized clients (prevents viser default from overwriting)
        if client.client_id not in self._initialized_clients:
            return

        try:
            with self._lock:
                self._state.set_from_viser(
                    client.camera.position,
                    client.camera.look_at,
                    client.camera.up_direction,
                )
                self._state.fov = client.camera.fov
                self._state.aspect = client.camera.aspect

            # Check for pole crossing (trackball-like behavior)
            should_cross, _ = self._should_cross_pole()
            if should_cross:
                self._execute_pole_cross()
                self.apply_to_viser([client])

        except Exception as e:
            logger.debug(f"Error syncing from viser: {e}")

    def apply_to_viser(self, clients=None) -> None:
        """
        Push current state to viser cameras.

        Parameters
        ----------
        clients : list | None
            Specific clients to update. If None, updates all.
        """
        if self._state is None:
            return

        with self._lock:
            # Get position, look_at, and up from state
            position = tuple(float(x) for x in self._state.position)
            look_at = tuple(float(x) for x in self._state.look_at)
            # Use computed up direction to apply roll (not hardcoded Y-up)
            up_direction = tuple(float(x) for x in self._state.up)

        target_clients = clients or list(self.server.get_clients().values())

        for client in target_clients:
            try:
                with client.atomic():
                    # DO NOT set wxyz explicitly - let viser compute it from position/look_at/up
                    # This ensures consistency with viser_look_at_matrix in _bake_camera_view
                    # Setting wxyz via quat_from_euler_deg uses a different convention than viser's
                    # internal look-at computation, which causes bake view issues.
                    client.camera.position = position
                    client.camera.look_at = look_at
                    client.camera.up_direction = up_direction
            except Exception as e:
                logger.debug(f"Error applying to viser client: {e}")

    def mark_client_initialized(self, client) -> None:
        """Mark a client as initialized (safe to sync from)."""
        self._initialized_clients.add(client.client_id)
        logger.debug(f"Client {client.client_id} initialized")

    def remove_client(self, client) -> None:
        """Remove client from tracking (call on disconnect)."""
        self._initialized_clients.discard(client.client_id)
        logger.debug(f"Client {client.client_id} removed")

    # =========================================================================
    # Pole Crossing (Trackball-like behavior)
    # =========================================================================

    def _should_cross_pole(self) -> tuple[bool, int]:
        """
        Check if camera should cross a pole.

        Returns (should_cross, direction) where direction is +1 for north pole,
        -1 for south pole, 0 for no crossing.
        """
        if self._state is None:
            return False, 0

        elev = self._state._elevation
        prev_elev = self._state._prev_elevation
        delta = elev - prev_elev

        # Threshold for "at the pole" - use 88° to catch before hard clamp at 89°
        POLE_THRESHOLD = 88.0

        # If stuck at pole (both current and prev at pole), reset prev_elevation
        # to allow crossing on next movement. This handles the case where user
        # slowly drags to pole and releases mouse - without this, delta=0 forever.
        if abs(elev) >= POLE_THRESHOLD and abs(prev_elev) >= POLE_THRESHOLD:
            # Reset prev_elevation to just below threshold so next drag can trigger crossing
            if elev >= POLE_THRESHOLD:
                self._state._prev_elevation = POLE_THRESHOLD - 1.0  # 87°
            else:
                self._state._prev_elevation = -POLE_THRESHOLD + 1.0  # -87°
            # Don't cross yet - let next update handle it with proper delta
            return False, 0

        if elev >= POLE_THRESHOLD and delta > 0.5:  # Moving up, hit north pole
            return True, 1
        elif elev <= -POLE_THRESHOLD and delta < -0.5:  # Moving down, hit south pole
            return True, -1

        return False, 0

    def _execute_pole_cross(self) -> None:
        """Flip camera through the pole (azimuth +180°, elevation inverts)."""
        if self._state is None:
            return

        with self._lock:
            new_azimuth = (self._state._azimuth + 180.0) % 360.0
            new_elevation = -self._state._elevation
            self._state.set_from_orbit(
                new_azimuth,
                new_elevation,
                self._state._roll,
                self._state._distance,
                self._state.look_at,
            )
        logger.debug(f"Pole crossed: az={new_azimuth:.1f}, el={new_elevation:.1f}")

    # =========================================================================
    # UI Callbacks
    # =========================================================================

    def set_slider_sync_callback(self, callback: Callable[[bool], None]) -> None:
        """Set callback to sync UI sliders with state.

        Callback signature: (force: bool) -> None
        When force=True, bypass is_app_controlled() check.
        """
        self._slider_sync_callback = callback

    def trigger_slider_sync(self, force: bool = False) -> None:
        """Trigger slider sync callback if set.

        Parameters
        ----------
        force : bool
            If True, bypass is_app_controlled() check (used by preset views)
        """
        if self._slider_sync_callback is not None:
            try:
                self._slider_sync_callback(force)
            except Exception as e:
                logger.debug(f"Slider sync error: {e}")

    def set_rerender_callback(self, callback: Callable[[], None]) -> None:
        """Set callback to trigger re-render after camera changes."""
        self._rerender_callback = callback

    def trigger_rerender(self) -> None:
        """Trigger rerender callback if set."""
        if self._rerender_callback is not None:
            try:
                self._rerender_callback()
            except Exception as e:
                logger.debug(f"Rerender callback error: {e}")

    # =========================================================================
    # Rotation Control (Flag-based, no thread)
    # =========================================================================

    def start_auto_rotation(self, axis: str = "y", speed: float = 20.0) -> None:
        """
        Start continuous camera rotation using quaternions (no euler conversion).

        This captures viser's exact camera state (wxyz, position, look_at) and rotates
        directly using quaternion multiplication. This avoids:
        - Gimbal lock issues with euler angles
        - Convention mismatches between our euler and viser's quaternion
        - Roll/azimuth jumps when starting rotation

        Parameters
        ----------
        axis : str
            "y" for horizontal orbit, "x" for vertical orbit
        speed : float
            Degrees per second (positive=CW, negative=CCW)
        """
        clients = list(self.server.get_clients().values())

        if clients:
            # Capture viser's EXACT state at rotation start
            client = clients[0]
            self._rotation_base_wxyz = np.array(client.camera.wxyz, dtype=np.float64)
            self._rotation_base_position = np.array(client.camera.position, dtype=np.float64)
            self._rotation_look_at = np.array(client.camera.look_at, dtype=np.float64)
            # Also initialize headless state
            self._headless_wxyz = self._rotation_base_wxyz.copy()
            self._headless_position = self._rotation_base_position.copy()
            self._headless_fov = client.camera.fov
            self._headless_aspect = client.camera.aspect
        elif self._headless_wxyz is not None:
            # No clients but have headless state - use it
            logger.info("Starting rotation from headless state (no clients)")
            self._rotation_base_wxyz = self._headless_wxyz.copy()
            self._rotation_base_position = self._headless_position.copy()
            if self._state is not None:
                with self._lock:
                    self._rotation_look_at = self._state.look_at.copy()
            else:
                # Estimate look_at from position and forward direction
                R = quat_to_rotation_matrix(self._rotation_base_wxyz)
                forward = -R[:, 2]
                self._rotation_look_at = self._rotation_base_position + forward * 5.0
        elif self._state is not None:
            # No clients, no headless state - build from CameraState
            logger.info("Starting rotation from CameraState (no clients)")
            with self._lock:
                self._rotation_base_wxyz = quat_from_euler_deg(
                    self._state._azimuth,
                    self._state._elevation,
                    self._state._roll
                )
                R = quat_to_rotation_matrix(self._rotation_base_wxyz)
                forward = -R[:, 2]
                position = self._state.look_at.astype(np.float64) - forward * self._state._distance
                self._rotation_base_position = position
                self._rotation_look_at = self._state.look_at.copy()
                # Initialize headless state
                self._headless_wxyz = self._rotation_base_wxyz.copy()
                self._headless_position = self._rotation_base_position.copy()
                self._headless_fov = self._state.fov if self._state.fov > 0.1 else 0.82
                self._headless_aspect = self._state.aspect if self._state.aspect > 0.1 else 16.0 / 9.0
        else:
            logger.warning("Cannot rotate - no camera state available")
            return

        self._rotation_accumulated_angle = 0.0
        self._enter_app_mode()  # Take ownership
        self._rotation_active = True
        self._rotation_axis = axis
        self._rotation_speed = speed
        self._last_rotation_time = time.perf_counter()
        logger.info(f"Started rotation: axis={axis}, speed={speed}")

    def stop_auto_rotation(self) -> None:
        """Stop continuous rotation."""
        if not self._rotation_active:
            return

        self._rotation_active = False
        # No need to apply_to_viser - rotation_step already pushed to viser
        # Just sync sliders from viser's current state
        self.trigger_slider_sync()
        self._enter_user_mode()  # Release with cooldown
        logger.info("Stopped rotation")

    def rotation_step(self) -> bool:
        """
        Advance rotation by one step using quaternion math. Called by render loop.

        This rotates directly from the captured base state using quaternion
        multiplication, completely avoiding euler angle conversion.

        Returns True if rotation occurred.
        """
        if not self._rotation_active:
            return False
        if self._rotation_base_wxyz is None or self._rotation_base_position is None:
            return False

        current_time = time.perf_counter()
        dt = current_time - self._last_rotation_time
        self._last_rotation_time = current_time

        # Clamp dt to avoid jumps (e.g., after tab switch)
        dt = min(dt, 0.1)

        # Accumulate rotation angle
        delta = self._rotation_speed * dt
        self._rotation_accumulated_angle += delta
        angle_rad = np.radians(self._rotation_accumulated_angle)

        # Extract rotation axis from viewer's orientation (captured at rotation start)
        # This makes rotation follow the camera's local frame
        R_base = quat_to_rotation_matrix(self._rotation_base_wxyz)

        if self._rotation_axis == "y":
            # Orbit around viewer's UP direction (local +Y axis)
            rotation_axis = R_base[:, 1]
        else:
            # Orbit around viewer's RIGHT direction (local +X axis)
            rotation_axis = R_base[:, 0]

        rot_quat = quat_from_axis_angle(rotation_axis, angle_rad)

        # Rotate position around look_at target
        offset = self._rotation_base_position - self._rotation_look_at
        R = quat_to_rotation_matrix(rot_quat)
        new_offset = R @ offset
        new_position = self._rotation_look_at + new_offset

        # Rotate orientation (quaternion multiplication)
        new_wxyz = quat_multiply(rot_quat, self._rotation_base_wxyz)

        # Store headless state (for rendering when no clients connected)
        self._headless_wxyz = new_wxyz.copy()
        self._headless_position = new_position.copy()

        # Push to viser clients if they exist
        # Use atomic() to prevent race conditions - wxyz and position must
        # be written together to avoid jittering from callbacks firing between them
        clients = list(self.server.get_clients().values())
        if clients:
            for client in clients:
                with client.atomic():
                    client.camera.wxyz = tuple(new_wxyz)
                    client.camera.position = tuple(new_position)
            # Capture fov/aspect for headless rendering
            self._headless_fov = clients[0].camera.fov
            self._headless_aspect = clients[0].camera.aspect
        # Note: rotation continues even without clients (headless mode)

        return True

    def get_rotation_state(self) -> dict:
        """Get current rotation state."""
        if not self._rotation_active:
            direction = "stopped"
        elif self._rotation_speed > 0:
            direction = "cw"
        else:
            direction = "ccw"

        return {
            "active": self._rotation_active,
            "speed": abs(self._rotation_speed),
            "axis": self._rotation_axis,
            "direction": direction,
        }

    def set_rotation_speed(self, speed: float) -> None:
        """Set rotation speed without changing active state."""
        self._rotation_speed = speed

    # =========================================================================
    # Preset Views
    # =========================================================================

    def set_preset_view(self, view: str) -> None:
        """
        Set camera to a preset view.

        Parameters
        ----------
        view : str
            One of: "top", "bottom", "front", "back", "left", "right", "iso"
        """
        # Stop rotation if active (will release to user mode)
        if self._rotation_active:
            self._rotation_active = False

        # Take brief ownership for the transition
        self._enter_app_mode()

        if self.scene_bounds is None or "size" not in self.scene_bounds:
            logger.warning("No scene bounds for preset views")
            self._enter_user_mode()
            return

        # look_at defaults to origin (0, 0, 0), distance from scene size
        look_at = np.zeros(3, dtype=np.float32)
        size = np.array(self.scene_bounds["size"])
        extent = float(np.linalg.norm(size))
        distance = extent * DEFAULT_DISTANCE_MULTIPLIER

        # Note: top/bottom use 89/-89 to avoid exact poles
        presets = {
            "top": (0.0, 89.0, 0.0),
            "bottom": (0.0, -89.0, 0.0),
            "front": (0.0, 0.0, 0.0),
            "back": (180.0, 0.0, 0.0),
            "left": (270.0, 0.0, 0.0),
            "right": (90.0, 0.0, 0.0),
            "iso": (45.0, 30.0, 0.0),
        }

        if view not in presets:
            logger.error(f"Unknown view: {view}")
            self._enter_user_mode()
            return

        azimuth, elevation, roll = presets[view]

        with self._lock:
            self._state.set_from_orbit(azimuth, elevation, roll, distance, look_at)

        self.apply_to_viser()
        self.trigger_slider_sync(force=True)  # Force sync even in APP mode
        self.trigger_rerender()  # Force re-render after camera change
        self._enter_user_mode()  # Release with cooldown
        logger.info(f"Set {view} view")

    def focus_on_bounds(self, bounds: dict | None = None) -> None:
        """Focus camera on scene bounds."""
        # Stop rotation if active
        if self._rotation_active:
            self._rotation_active = False

        # Take brief ownership
        self._enter_app_mode()

        if bounds is None:
            bounds = self.scene_bounds

        if bounds is None or "center" not in bounds:
            logger.warning("No bounds to focus on")
            self._enter_user_mode()
            return

        center = np.array(bounds["center"])
        size = np.array(bounds["size"])
        extent = float(np.linalg.norm(size))
        distance = extent * DEFAULT_DISTANCE_MULTIPLIER

        with self._lock:
            # Update look_at and distance, keep current orientation
            self._state.look_at = center.astype(np.float32)
            self._state._distance = distance
            self._state._invalidate_c2w()

        self.apply_to_viser()
        self.trigger_slider_sync(force=True)  # Force sync even in APP mode
        self.trigger_rerender()  # Force re-render after camera change
        self._enter_user_mode()  # Release with cooldown
        logger.info(f"Focused on bounds: center={center}")

    # =========================================================================
    # Grid and Axis
    # =========================================================================

    def _setup_grid(self) -> None:
        """Create ground plane grid."""
        if self.scene_bounds is not None and "size" in self.scene_bounds:
            size = self.scene_bounds["size"]
            grid_size = float(np.max(size)) * 2
        else:
            grid_size = 20.0

        self.grid_handle = self.server.scene.add_grid(
            name="/camera_grid",
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

    def _setup_world_axis(self) -> None:
        """Create world axis frame."""
        if self.scene_bounds is not None and "size" in self.scene_bounds:
            axis_size = float(np.max(self.scene_bounds["size"])) * 0.1
        else:
            axis_size = 2.0

        self.world_axis_handle = self.server.scene.add_frame(
            name="/camera_world_axis",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.0),
            axes_length=axis_size,
            axes_radius=axis_size * 0.05,
            visible=False,
        )

    def toggle_grid(self) -> None:
        """Toggle grid visibility."""
        if self.grid_handle is not None:
            self.grid_visible = not self.grid_visible
            self.grid_handle.visible = self.grid_visible

    def update_scene_bounds(self, bounds: dict | None) -> None:
        """Update scene bounds and recreate grid."""
        self.scene_bounds = bounds
        if self.grid_handle is not None:
            self.grid_handle.remove()
        self._setup_grid()

    # =========================================================================
    # Initialization
    # =========================================================================

    def _initialize_state(self) -> None:
        """Initialize camera state from scene bounds or defaults."""
        # Determine distance from scene bounds, but look_at defaults to origin
        if self.scene_bounds is not None and "size" in self.scene_bounds:
            size = np.array(self.scene_bounds["size"])
            distance = float(np.linalg.norm(size)) * DEFAULT_DISTANCE_MULTIPLIER
        else:
            distance = 10.0

        # Default look_at is always origin (0, 0, 0)
        # Users can adjust via View panel sliders or Center button
        look_at = np.zeros(3, dtype=np.float32)

        # Create state with default isometric view using new spherical-primary constructor
        self._state = CameraState(
            _azimuth=45.0,
            _elevation=30.0,
            _roll=0.0,
            _distance=distance,
            look_at=look_at,
        )

        logger.debug(f"Initialized state: distance={distance:.2f}, look_at=(0,0,0)")

    # =========================================================================
    # Legacy Compatibility
    # =========================================================================

    def apply_state_to_camera(self, clients=None) -> None:
        """Legacy alias for apply_to_viser."""
        self.apply_to_viser(clients)

    def sync_state_from_client(self, client) -> None:
        """Legacy alias for update_from_viser."""
        self.update_from_viser(client)

    def sync_state_from_camera(self) -> None:
        """Legacy: sync from first initialized client."""
        for client in self.server.get_clients().values():
            if client.client_id in self._initialized_clients:
                self.update_from_viser(client)
                return

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
        """Legacy: set camera to specific angle."""
        if self._state is None:
            return

        with self._lock:
            distance = explicit_distance or (self._state.distance if preserve_distance else None)
            look_at = explicit_lookat if explicit_lookat is not None else (
                self._state.look_at if preserve_lookat else None
            )

            if distance is None and self.scene_bounds is not None:
                size = np.array(self.scene_bounds["size"])
                distance = float(np.linalg.norm(size)) * DEFAULT_DISTANCE_MULTIPLIER

            if look_at is None:
                # Default look_at is origin (0, 0, 0)
                look_at = np.zeros(3, dtype=np.float32)

            self._state.set_from_orbit(azimuth, elevation, roll, distance, look_at)

        self.apply_to_viser()

    def calculate_azimuth_elevation(
        self,
        camera_pos: np.ndarray = None,
        look_at: np.ndarray = None,
        up_direction: np.ndarray = None,
        detect_flip: bool = True,
    ) -> tuple[float, float]:
        """Legacy: get azimuth/elevation from state."""
        if self._state is None:
            return (0.0, 0.0)
        with self._lock:
            return (self._state.azimuth, self._state.elevation)

    def _calculate_roll_from_camera(self, *args, **kwargs) -> float:
        """Legacy: get roll from state."""
        if self._state is None:
            return 0.0
        with self._lock:
            return self._state.roll

    def validate_state(self, state: CameraState) -> CameraState:
        """Legacy: validation (no-op)."""
        return state.copy()


# Alias for backwards compatibility
SuperSplatCamera = CameraController
