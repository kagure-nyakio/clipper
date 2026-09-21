class ClipperError(Exception):
    """Base for every error this codebase raises on purpose."""

    status_code = 500


class ProbeError(ClipperError):
    """Raised when ffprobe can't read a video file."""

    status_code = 422  # the file itself is the problem, not the request


class EncodingError(ClipperError):
    """Raised when an ffmpeg encode/transcode operation fails."""

    status_code = 500


class ValidationError(ClipperError):
    """Raised for invalid input parameters (bad timestamps, bad crop size)."""

    status_code = 400


class TranscriptionError(ClipperError):
    """Raised when the ASR engine fails to transcribe an audio file."""

    status_code = 500


class LLMError(ClipperError):
    """Raise when call to LLM fails"""

    status_code = 502
