from unittest.mock import patch

import ffmpeg
import pytest

from clipper.core.exceptions import EncodingError, ProbeError, ValidationError
from clipper.core.media import (
    add_subtitles,
    create_clip,
    crop_video,
    extract_audio,
    get_video_metadata,
    resize_video,
    trim_silence,
)


def test_get_video_metadata_raises_on_missing_file():
    with pytest.raises(ProbeError, match="Could not read"):
        get_video_metadata("does_not_exist.mp4")


def test_get_video_metadata_returns_video_metadata_if_video_exists(sample_video_path):
    result = get_video_metadata(str(sample_video_path))
    assert result["duration_seconds"] == pytest.approx(2.0, abs=0.1)
    assert result["fps"] == pytest.approx(15.0, abs=0.1)
    assert result["has_audio"]
    assert result["height"] == 240
    assert result["video_codec"] == "h264"
    assert result["width"] == 320


def test_get_video_metadata_raises_on_probe_error():
    sample_error = ffmpeg.Error(
        cmd="ffprobe", stdout=b"", stderr=b"No such file or directory"
    )

    with (
        patch("clipper.core.media.ffmpeg.probe", side_effect=sample_error),
        pytest.raises(ProbeError, match="No such file or directory"),
    ):
        get_video_metadata("missing.mp4")


def test_extract_audio_real_file_produces_valid_audio(sample_video_path, tmp_path):
    output_path = tmp_path / "extracted.wav"

    result = extract_audio(str(sample_video_path), str(output_path))

    assert result == str(output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    probe = ffmpeg.probe(str(output_path))
    audio_streams = [s for s in probe["streams"] if s["codec_type"] == "audio"]
    assert len(audio_streams) == 1


def test_extract_audio_raises_on_missing_input(tmp_path):
    output_path = tmp_path / "should_not_exist.wav"

    with pytest.raises(EncodingError, match="Audio extraction failed"):
        extract_audio("this_file_does_not_exist.mp4", str(output_path))

    assert not output_path.exists()


def test_create_clip_fast_remux_produces_valid_clip(sample_video_path, tmp_path):
    output_path = tmp_path / "clip_fast.mp4"

    result = create_clip(
        str(sample_video_path),
        start="00:00:00",
        end="00:00:01",
        output_path=str(output_path),
    )

    assert result == str(output_path)
    assert output_path.exists()
    probe = ffmpeg.probe(str(output_path))
    assert probe["streams"]


def test_create_clip_precise_reencode_matches_requested_duration(
    sample_video_path, tmp_path
):
    output_path = tmp_path / "clip_precise.mp4"

    result = create_clip(
        str(sample_video_path),
        start="00:00:00",
        end="00:00:01",
        output_path=str(output_path),
        precise=True,
    )

    assert result == str(output_path)
    probe = ffmpeg.probe(str(output_path))
    duration = float(probe["format"]["duration"])
    assert duration == pytest.approx(1.0, abs=0.15)


def test_create_clip_rejects_invalid_timestamp_format(sample_video_path, tmp_path):
    output_path = tmp_path / "should_not_exist.mp4"

    with pytest.raises(ValidationError, match="not a valid HH:MM:SS timestamp"):
        create_clip(
            str(sample_video_path),
            start="bad",
            end="00:00:01",
            output_path=str(output_path),
        )

    assert not output_path.exists()


def test_create_clip_rejects_end_before_start(sample_video_path, tmp_path):
    output_path = tmp_path / "should_not_exist.mp4"

    with pytest.raises(ValidationError, match="must be after start"):
        create_clip(
            str(sample_video_path),
            start="00:00:02",
            end="00:00:01",
            output_path=str(output_path),
        )

    assert not output_path.exists()


def test_resize_video_changes_dimensions(sample_video_path, tmp_path):
    output_path = tmp_path / "resized.mp4"

    result = resize_video(
        str(sample_video_path), str(output_path), width=160, height=120
    )

    assert result == str(output_path)
    metadata = get_video_metadata(str(output_path))
    assert metadata["width"] == 160
    assert metadata["height"] == 120


def test_crop_video_centers_crop_correctly(sample_video_path, tmp_path):
    output_path = tmp_path / "cropped.mp4"

    result = crop_video(str(sample_video_path), str(output_path), width=200, height=150)

    assert result == str(output_path)
    metadata = get_video_metadata(str(output_path))
    assert metadata["width"] == 200
    assert metadata["height"] == 150


def test_crop_video_raises_when_crop_exceeds_source_dimensions(
    sample_video_path, tmp_path
):
    output_path = tmp_path / "cropped_invalid.mp4"

    with pytest.raises(ValidationError, match="exceeds source dimensions"):
        crop_video(str(sample_video_path), str(output_path), width=1000, height=1000)

    assert not output_path.exists()


def test_crop_video_raises_when_no_video_stream(
    sample_video_path, tmp_path, monkeypatch
):
    output_path = tmp_path / "cropped_invalid.mp4"

    monkeypatch.setattr(
        "clipper.core.media.get_video_metadata",
        lambda path: {
            "duration_seconds": 5.0,
            "filename": path,
            "fps": None,
            "has_audio": True,
            "height": None,
            "video_codec": None,
            "width": None,
        },
    )

    with pytest.raises(ProbeError, match="has no video stream to crop"):
        crop_video(str(sample_video_path), str(output_path), width=100, height=100)

    assert not output_path.exists()


def test_crop_video_raises_on_missing_input(tmp_path):
    output_path = tmp_path / "should_not_exist.mp4"

    with pytest.raises(ProbeError):
        crop_video(
            "this_file_does_not_exist.mp4", str(output_path), width=100, height=100
        )

    assert not output_path.exists()


def test_add_subtitles_produces_valid_video(sample_video_path, tmp_path):
    srt_path = tmp_path / "sample.srt"
    srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nTest subtitle\n")
    output_path = tmp_path / "subtitled.mp4"

    result = add_subtitles(str(sample_video_path), str(output_path), str(srt_path))

    assert result == str(output_path)
    assert output_path.exists()
    probe = ffmpeg.probe(str(output_path))
    assert probe["streams"]


def test_trim_silence_produces_valid_audio(sample_video_path, tmp_path):
    audio_path = tmp_path / "audio.wav"
    extract_audio(str(sample_video_path), str(audio_path))
    output_path = tmp_path / "trimmed.wav"

    result = trim_silence(str(audio_path), str(output_path))

    assert result == str(output_path)
    assert output_path.exists()

    probe = ffmpeg.probe(str(output_path))
    assert probe["streams"]


def test_trim_silence_raises_on_missing_input(tmp_path):
    output_path = tmp_path / "should_not_exist.wav"

    with pytest.raises(EncodingError):
        trim_silence("this_file_does_not_exist.wav", str(output_path))

    assert not output_path.exists()
