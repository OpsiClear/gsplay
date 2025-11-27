"""
Info panel component for displaying viewer statistics.

Provides a compact markdown-based display of frame info, gaussian counts,
and performance metrics.
"""

from __future__ import annotations

import logging

import viser

logger = logging.getLogger(__name__)


class InfoPanel:
    """Compact info panel using markdown for efficient display."""

    def __init__(self, server: viser.ViserServer):
        self._server = server
        self._markdown = server.gui.add_markdown("*Loading...*")

        # Current values
        self._frame_index = "N/A"
        self._total_frames = "N/A"
        self._file_name = "N/A"
        self._gaussian_count = "N/A"
        self._loader_fps = "N/A"
        self._render_fps = "N/A"
        self._status = None  # Optional status message (e.g., "Idle")

        self._update_display()

    def _update_display(self) -> None:
        """Update the markdown display with current values."""
        # Compact 2-line format
        if self._status:
            # Show status prominently when set
            content = (
                f"**{self._status}**  \n"
                f"Frame {self._frame_index}/{self._total_frames} | "
                f"Gaussians {self._gaussian_count}"
            )
        else:
            content = (
                f"**Frame** {self._frame_index}/{self._total_frames} | "
                f"**Gaussians** {self._gaussian_count}  \n"
                f"**Render** {self._render_fps} | **Load** {self._loader_fps}"
            )
        self._markdown.content = content

    def set_frame_index(self, index: int | str, total: int | str | None = None) -> None:
        """Set current frame index."""
        self._frame_index = str(index)
        if total is not None:
            self._total_frames = str(total)
        self._update_display()

    def set_total_frames(self, total: int | str) -> None:
        """Set total frame count."""
        self._total_frames = str(total)
        self._update_display()

    def set_file_name(self, name: str) -> None:
        """Set current file name (not displayed in compact mode)."""
        self._file_name = name
        # Not shown in compact display, but stored for potential tooltip/expansion

    def set_gaussian_count(self, count: int | str) -> None:
        """Set gaussian count."""
        if isinstance(count, int):
            if count >= 1_000_000:
                self._gaussian_count = f"{count / 1_000_000:.1f}M"
            elif count >= 1_000:
                self._gaussian_count = f"{count / 1_000:.1f}K"
            else:
                self._gaussian_count = str(count)
        else:
            self._gaussian_count = str(count)
        self._update_display()

    def set_loader_fps(self, fps: float | str) -> None:
        """Set loader throughput FPS."""
        if isinstance(fps, (int, float)):
            self._loader_fps = f"{fps:.0f} FPS"
        else:
            self._loader_fps = str(fps)
        self._update_display()

    def set_render_fps(self, fps: float | str) -> None:
        """Set render FPS."""
        if isinstance(fps, (int, float)):
            self._render_fps = f"{fps:.0f} FPS"
        else:
            self._render_fps = str(fps)
        self._update_display()

    def set_status(self, status: str | None) -> None:
        """Set status message (e.g., 'Idle - move to resume').

        Parameters
        ----------
        status : str | None
            Status message to display, or None to clear status
        """
        self._status = status
        self._update_display()


def create_info_panel(server: viser.ViserServer) -> InfoPanel:
    """
    Create compact info display panel using markdown.

    Parameters
    ----------
    server : viser.ViserServer
        Viser server instance

    Returns
    -------
    InfoPanel
        Info panel object with setter methods
    """
    logger.debug("Created compact info panel")
    return InfoPanel(server)
