"""Port allocation utilities for GSPlay instances."""

from __future__ import annotations

import logging
import socket

logger = logging.getLogger(__name__)


class PortAllocator:
    """Manages port allocation for GSPlay instances.

    Parameters
    ----------
    start_port : int
        Start of port range for allocation.
    end_port : int
        End of port range for allocation.
    host : str
        Host to check port availability on.
    """

    def __init__(
        self,
        start_port: int = 6020,
        end_port: int = 6100,
        host: str = "127.0.0.1",
    ) -> None:
        self.start_port = start_port
        self.end_port = end_port
        self.host = host

    def is_available(self, port: int) -> bool:
        """Check if a port is available for binding.

        Parameters
        ----------
        port : int
            Port number to check.

        Returns
        -------
        bool
            True if port is available.
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((self.host, port))
                return True
        except OSError:
            return False

    def find_available(
        self,
        exclude: set[int] | None = None,
        start_hint: int | None = None,
        even_only: bool = True,
    ) -> int | None:
        """Find first available port in range.

        Parameters
        ----------
        exclude : set[int] | None
            Ports to exclude from search.
        start_hint : int | None
            Port to start searching from (optimization hint).
        even_only : bool
            If True, only return even ports (default). This reserves odd ports
            for stream servers (stream_port = viser_port + 1).

        Returns
        -------
        int | None
            Available port number or None if none found.
        """
        exclude = exclude or set()
        start = start_hint if start_hint else self.start_port

        # Ensure start is in valid range
        if start < self.start_port or start >= self.end_port:
            start = self.start_port

        # If even_only, ensure start is even
        if even_only and start % 2 != 0:
            start += 1

        # Search from start to end
        port = start
        while port < self.end_port:
            if port not in exclude and self.is_available(port):
                # Also check that port+1 is available for streaming
                if not even_only or self.is_available(port + 1):
                    logger.debug("Found available port: %d", port)
                    return port
            port += 2 if even_only else 1

        # Wrap around: search from beginning to start
        port = self.start_port if not even_only or self.start_port % 2 == 0 else self.start_port + 1
        while port < start:
            if port not in exclude and self.is_available(port):
                # Also check that port+1 is available for streaming
                if not even_only or self.is_available(port + 1):
                    logger.debug("Found available port (wrapped): %d", port)
                    return port
            port += 2 if even_only else 1

        logger.warning(
            "No available ports in range [%d, %d)",
            self.start_port,
            self.end_port,
        )
        return None

    def get_used_in_range(self) -> list[int]:
        """Get list of ports in use within the range.

        Returns
        -------
        list[int]
            Ports that are currently in use.
        """
        used = []
        for port in range(self.start_port, self.end_port):
            if not self.is_available(port):
                used.append(port)
        return used
