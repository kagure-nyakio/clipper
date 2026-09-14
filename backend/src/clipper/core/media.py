from fractions import Fraction
from typing import TypedDict

import ffmpeg

from clipper.core.exceptions import (
    EncodingError,
    ProbeError,
    ValidationError,
)
from clipper.core.helpers import validate_range


class VideoMetadata(TypedDict):
    duration_seconds: float
    fps: float | None
    has_audio: bool
    height: int | None
    video_codec: str | None
    width: int | None
    filename: str


def get_video_metadata(video_path: str) -> VideoMetadata:
    """Probe a video file and return its duration, resolution, fps, codec, and audio presence."""
    try:
        probe = ffmpeg.probe(video_path)
        format_info = probe.get("format", {})
        streams = probe.get("streams", [])

        video_stream = next(
            (s for s in streams if s.get("codec_type") == "video"), None
        )
        audio_stream = next(
            (s for s in streams if s.get("codec_type") == "audio"), None
        )

        return {
            "duration_seconds": float(format_info.get("duration", 0)),
            "filename": format_info.get("filename"),
            "fps": float(Fraction(video_stream["r_frame_rate"]))
            if video_stream and video_stream.get("r_frame_rate")
            else None,
            "has_audio": audio_stream is not None,
            "height": int(video_stream.get("height", 0)) if video_stream else None,
            "video_codec": video_stream.get("codec_name") if video_stream else None,
            "width": int(video_stream.get("width", 0)) if video_stream else None,
        }
    except ffmpeg.Error as e:
        stderr = e.stderr.decode("utf-8") if e.stderr else str(e)
        raise ProbeError(f"Could not read '{video_path}': {stderr}") from e


def _run_and_log(stream, output_path: str, label: str) -> str:
    """Run a built ffmpeg stream, capturing output so failures log something useful."""
    try:
        stream.run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
        return output_path
    except ffmpeg.Error as e:
        stderr = e.stderr.decode("utf-8") if e.stderr else str(e)
        raise EncodingError(f"{label} failed for '{output_path}': {stderr}") from e


def extract_audio(video_path: str, output_audio_path: str) -> str:
    """Extract the audio track from a video as uncompressed 16-bit PCM WAV."""
    stream = ffmpeg.input(video_path).output(output_audio_path, acodec="pcm_s16le")
    return _run_and_log(stream, output_audio_path, "Audio extraction")


def trim_silence(
    audio_path: str,
    output_path: str,
    threshold_db: float = -50.0,
    min_silence_duration: float = 1.0,
) -> str:
    """
    Trim leading and trailing silence from an audio file.

    Use AFTER a clip has already been cut and transcribed — not before,
    since trimming leading silence shifts every timestamp measured against it.
    """
    af_filter = (
        f"silenceremove=start_periods=1:start_duration=0:start_threshold={threshold_db}dB:"
        f"stop_periods=1:stop_duration={min_silence_duration}:stop_threshold={threshold_db}dB"
    )
    stream = ffmpeg.input(audio_path).output(output_path, af=af_filter)
    return _run_and_log(stream, output_path, "Silence trim")


def create_clip(
    input_path: str,
    start: str,
    end: str,
    output_path: str,
    precise: bool = False,
) -> str:
    """
    Produce a clip from input_path covering [start, end].

    start/end accept "HH:MM:SS" or "HH:MM:SS.mmm" timestamp strings.

    precise=False (default): fast remux, keyframe-aligned, may drift slightly
    from the requested boundaries.
    precise=True: re-encodes, frame-accurate, slower.
    """
    start_secs, end_secs = validate_range(start, end)
    duration = end_secs - start_secs

    if precise:
        stream = ffmpeg.input(input_path).output(
            output_path, ss=start_secs, t=duration, vcodec="libx264", acodec="aac"
        )
        return _run_and_log(stream, output_path, "Precise re-encode")

    stream = ffmpeg.input(input_path, ss=start_secs, t=duration).output(
        output_path, c="copy"
    )
    return _run_and_log(stream, output_path, "Fast remux")


def resize_video(input_path: str, output_path: str, width: int, height: int) -> str:
    """Scale a video to the given width/height."""
    stream = ffmpeg.input(input_path).output(output_path, vf=f"scale={width}:{height}")
    return _run_and_log(stream, output_path, "Resize")


def add_subtitles(input_path: str, output_path: str, subtitle_source: str) -> str:
    """Burn subtitles from subtitle_source into the video."""
    stream = ffmpeg.input(input_path).output(
        output_path, vf=f"subtitles={subtitle_source}"
    )
    return _run_and_log(stream, output_path, "Subtitle burn-in")


def crop_video(input_path: str, output_path: str, width: int, height: int) -> str:
    """Center-crop a video to the given width/height."""
    metadata = get_video_metadata(input_path)  # raises ProbeError on failure now

    iw, ih = metadata["width"], metadata["height"]
    if iw is None or ih is None:
        raise ProbeError(f"'{input_path}' has no video stream to crop")

    if width > iw or height > ih:
        raise ValidationError(
            f"crop size {width}x{height} exceeds source dimensions {iw}x{ih}"
        )

    x, y = (iw - width) // 2, (ih - height) // 2
    stream = ffmpeg.input(input_path).output(
        output_path, vf=f"crop={width}:{height}:{x}:{y}"
    )
    return _run_and_log(stream, output_path, "Crop")
