import logging
from fractions import Fraction
from typing import TypedDict

import ffmpeg

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class VideoMetadata(TypedDict):
    duration_seconds: float
    fps: float | None
    has_audio: bool
    height: int | None
    video_codec: str | None
    width: int | None
    filename: str


def get_video_metadata(video_path: str) -> VideoMetadata | None:
    try:
        probe = ffmpeg.probe(video_path)

        format_info = probe.get("format", {})

        video_stream = next(
            (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
            None,
        )
        audio_stream = next(
            (stream for stream in probe["streams"] if stream["codec_type"] == "audio"),
            None,
        )

        metadata: VideoMetadata = {
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

        return metadata

    except ffmpeg.Error as e:
        logger.error(f"An error has occured while probing: {e.stderr.decode('utf-8')}")
        return None


def extract_audio(video_path: str, output_audio_path: str) -> str | None:
    try:
        (
            ffmpeg.input(video_path)
            .output(output_audio_path, acodec="pcm_s16le")
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        logger.info(f"Audio extracted to {output_audio_path}")
        return output_audio_path
    except ffmpeg.Error as e:
        logger.error(f"Error: {e.stderr.decode('utf-8')}")
        return None
