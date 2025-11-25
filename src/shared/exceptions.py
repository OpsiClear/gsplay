"""
Custom exceptions for the GSPlay.

This module provides domain-specific exceptions for better error handling
and clearer error messages throughout the application.

Clean Architecture Note:
- This file belongs to the Shared layer (cross-cutting concerns)
- Can be imported by any layer (domain, infrastructure, models, viewer)
"""


class GSPlayError(Exception):
    """Base exception for all viewer-related errors."""

    pass


class ModelLoadError(GSPlayError):
    """Raised when a model fails to load."""

    def __init__(self, message: str, model_type: str | None = None, path: str | None = None):
        """
        Initialize ModelLoadError.

        Parameters
        ----------
        message : str
            Error message
        model_type : str | None
            Type of model that failed to load (e.g., 'ply', 'sogs', 'gifstream')
        path : str | None
            Path to the model file/directory that failed to load
        """
        self.model_type = model_type
        self.path = path

        full_message = message
        if model_type:
            full_message = f"[{model_type}] {full_message}"
        if path:
            full_message = f"{full_message} (path: {path})"

        super().__init__(full_message)


class DecompressionError(GSPlayError):
    """Raised when frame decompression fails."""

    def __init__(self, message: str, frame_index: int | None = None):
        """
        Initialize DecompressionError.

        Parameters
        ----------
        message : str
            Error message
        frame_index : int | None
            Index of the frame that failed to decompress
        """
        self.frame_index = frame_index

        full_message = message
        if frame_index is not None:
            full_message = f"{full_message} (frame: {frame_index})"

        super().__init__(full_message)


class RenderError(GSPlayError):
    """Raised when rendering fails."""

    def __init__(self, message: str, frame_index: int | None = None):
        """
        Initialize RenderError.

        Parameters
        ----------
        message : str
            Error message
        frame_index : int | None
            Index of the frame that failed to render
        """
        self.frame_index = frame_index

        full_message = message
        if frame_index is not None:
            full_message = f"{full_message} (frame: {frame_index})"

        super().__init__(full_message)


class InvalidFrameError(GSPlayError):
    """Raised when an invalid frame index is requested."""

    def __init__(
        self,
        message: str,
        frame_index: int,
        total_frames: int | None = None,
    ):
        """
        Initialize InvalidFrameError.

        Parameters
        ----------
        message : str
            Error message
        frame_index : int
            The invalid frame index
        total_frames : int | None
            Total number of available frames
        """
        self.frame_index = frame_index
        self.total_frames = total_frames

        full_message = message
        if total_frames is not None:
            full_message = f"{full_message} (requested: {frame_index}, total: {total_frames})"
        else:
            full_message = f"{full_message} (requested: {frame_index})"

        super().__init__(full_message)


class StreamingError(GSPlayError):
    """Raised when streaming operations fail."""

    def __init__(
        self,
        message: str,
        stream_name: str | None = None,
        url: str | None = None,
    ):
        """
        Initialize StreamingError.

        Parameters
        ----------
        message : str
            Error message
        stream_name : str | None
            Name of the stream that failed
        url : str | None
            URL of the stream source
        """
        self.stream_name = stream_name
        self.url = url

        full_message = message
        if stream_name:
            full_message = f"{full_message} (stream: {stream_name})"
        if url:
            full_message = f"{full_message} (url: {url})"

        super().__init__(full_message)


class ConfigError(GSPlayError):
    """Raised when configuration is invalid or missing."""

    def __init__(
        self,
        message: str,
        config_path: str | None = None,
        field_name: str | None = None,
    ):
        """
        Initialize ConfigError.

        Parameters
        ----------
        message : str
            Error message
        config_path : str | None
            Path to the config file
        field_name : str | None
            Name of the invalid/missing config field
        """
        self.config_path = config_path
        self.field_name = field_name

        full_message = message
        if field_name:
            full_message = f"{full_message} (field: {field_name})"
        if config_path:
            full_message = f"{full_message} (config: {config_path})"

        super().__init__(full_message)


class CacheError(GSPlayError):
    """Raised when cache operations fail."""

    def __init__(
        self,
        message: str,
        cache_path: str | None = None,
        operation: str | None = None,
    ):
        """
        Initialize CacheError.

        Parameters
        ----------
        message : str
            Error message
        cache_path : str | None
            Path to the cache file/directory
        operation : str | None
            Cache operation that failed (e.g., 'read', 'write', 'clear')
        """
        self.cache_path = cache_path
        self.operation = operation

        full_message = message
        if operation:
            full_message = f"{full_message} (operation: {operation})"
        if cache_path:
            full_message = f"{full_message} (path: {cache_path})"

        super().__init__(full_message)


class DataFormatError(GSPlayError):
    """Raised when data is in an unexpected or invalid format."""

    def __init__(
        self,
        message: str,
        expected_format: str | None = None,
        actual_format: str | None = None,
    ):
        """
        Initialize DataFormatError.

        Parameters
        ----------
        message : str
            Error message
        expected_format : str | None
            Expected data format
        actual_format : str | None
            Actual data format encountered
        """
        self.expected_format = expected_format
        self.actual_format = actual_format

        full_message = message
        if expected_format and actual_format:
            full_message = f"{full_message} (expected: {expected_format}, got: {actual_format})"
        elif expected_format:
            full_message = f"{full_message} (expected: {expected_format})"

        super().__init__(full_message)


class GSTensorError(GSPlayError):
    """Raised when Gaussian data is invalid or corrupted."""

    def __init__(
        self,
        message: str,
        field_name: str | None = None,
        expected_shape: tuple | None = None,
        actual_shape: tuple | None = None,
    ):
        """
        Initialize GSTensorError.

        Parameters
        ----------
        message : str
            Error message
        field_name : str | None
            Name of the invalid field (e.g., 'means', 'colors')
        expected_shape : tuple | None
            Expected tensor shape
        actual_shape : tuple | None
            Actual tensor shape
        """
        self.field_name = field_name
        self.expected_shape = expected_shape
        self.actual_shape = actual_shape

        full_message = message
        if field_name:
            full_message = f"{full_message} (field: {field_name})"
        if expected_shape and actual_shape:
            full_message = f"{full_message} (expected shape: {expected_shape}, got: {actual_shape})"

        super().__init__(full_message)
