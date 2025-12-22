"""Instance ID encoding/decoding for secure URLs.

This module provides functions to encode instance IDs into URL-safe tokens
that can't be easily guessed, and decode them back to original IDs.

The encoding uses HMAC-SHA256 to create a signature, then combines the
instance ID with a truncated signature for verification.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from gsplay_launcher.config import LauncherConfig

logger = logging.getLogger(__name__)

# Global config reference (set by app.py)
_config: LauncherConfig | None = None


def set_config(config: LauncherConfig) -> None:
    """Set the config for ID encoding.

    Parameters
    ----------
    config : LauncherConfig
        Launcher configuration with url_secret.
    """
    global _config
    _config = config


def _get_secret() -> bytes:
    """Get the URL secret as bytes."""
    if _config is None:
        raise RuntimeError("ID encoder not initialized - call set_config first")
    return _config.url_secret.encode("utf-8")


def _compute_signature(instance_id: str) -> str:
    """Compute HMAC signature for an instance ID.

    Parameters
    ----------
    instance_id : str
        The instance ID to sign.

    Returns
    -------
    str
        Truncated hex signature (8 chars).
    """
    secret = _get_secret()
    sig = hmac.new(secret, instance_id.encode("utf-8"), hashlib.sha256).hexdigest()
    return sig[:8]  # Truncate for shorter URLs


def encode_instance_id(instance_id: str) -> str:
    """Encode an instance ID into a URL-safe token.

    The token format is: base64url(instance_id) + signature
    This ensures:
    - IDs can't be guessed (signature verification required)
    - Original ID is recoverable (for lookup)
    - URL-safe characters only

    Parameters
    ----------
    instance_id : str
        The raw instance ID (e.g., "a1b2c3d4").

    Returns
    -------
    str
        Encoded token safe for use in URLs.

    Examples
    --------
    >>> encode_instance_id("a1b2c3d4")
    'YTFiMmMzZDQ_abc12345'
    """
    # Base64url encode the ID (URL-safe, no padding)
    id_bytes = instance_id.encode("utf-8")
    encoded_id = base64.urlsafe_b64encode(id_bytes).decode("utf-8").rstrip("=")

    # Compute signature
    sig = _compute_signature(instance_id)

    # Combine: encoded_id + underscore + signature
    return f"{encoded_id}_{sig}"


def decode_instance_id(token: str) -> str | None:
    """Decode and verify an encoded instance ID token.

    Parameters
    ----------
    token : str
        The encoded token from encode_instance_id().

    Returns
    -------
    str | None
        The original instance ID if valid, None if invalid/tampered.

    Examples
    --------
    >>> decode_instance_id("YTFiMmMzZDQ_abc12345")
    'a1b2c3d4'
    >>> decode_instance_id("tampered_token")
    None
    """
    try:
        # Split into encoded ID and signature
        if "_" not in token:
            logger.debug("Invalid token format: no underscore separator")
            return None

        parts = token.rsplit("_", 1)
        if len(parts) != 2:
            logger.debug("Invalid token format: wrong number of parts")
            return None

        encoded_id, provided_sig = parts

        # Decode the ID (add padding back for base64)
        padding = 4 - (len(encoded_id) % 4)
        if padding != 4:
            encoded_id += "=" * padding

        try:
            id_bytes = base64.urlsafe_b64decode(encoded_id)
            instance_id = id_bytes.decode("utf-8")
        except Exception as e:
            logger.debug(f"Failed to decode base64: {e}")
            return None

        # Verify signature
        expected_sig = _compute_signature(instance_id)
        if not hmac.compare_digest(provided_sig, expected_sig):
            logger.debug("Signature verification failed")
            return None

        return instance_id

    except Exception as e:
        logger.debug(f"Token decode error: {e}")
        return None


def is_valid_token(token: str) -> bool:
    """Check if a token is valid without returning the ID.

    Parameters
    ----------
    token : str
        The encoded token to verify.

    Returns
    -------
    bool
        True if the token is valid.
    """
    return decode_instance_id(token) is not None
