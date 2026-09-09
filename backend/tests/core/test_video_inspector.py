from unittest.mock import patch

import ffmpeg
import pytest

from clipper.core.video_inspector import extract_audio, get_video_metadata


def test_get_video_metadata_returns_none_on_missing_file():
    result = get_video_metadata("does_not_exist.mp4")
    assert result is None


def test_get_video_metadata_returns_video_metadata_if_video_exists(sample_video_path):
    result = get_video_metadata(str(sample_video_path))
    assert result["duration_seconds"] == pytest.approx(2.0, abs=0.1)
    assert result["fps"] == pytest.approx(15.0, abs=0.1)
    assert result["has_audio"]
    assert result["height"] == 240
    assert result["video_codec"] == "h264"
    assert result["width"] == 320


def test_get_video_metadata_probe_failure(caplog):
    fake_error = ffmpeg.Error(
        cmd="ffprobe", stdout=b"", stderr=b"No such file or directory"
    )

    with patch("clipper.core.video_inspector.ffmpeg.probe", side_effect=fake_error):
        result = get_video_metadata("missing.mp4")

    assert result is None
    assert "error has occured while probing" in caplog.text.lower()
    assert "no such file or directory" in caplog.text.lower()


def test_extract_audio_real_file_produces_valid_audio(sample_video_path, tmp_path):
    output_path = tmp_path / "extracted.wav"

    result = extract_audio(str(sample_video_path), str(output_path))

    assert result == str(output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    probe = ffmpeg.probe(str(output_path))
    audio_streams = [s for s in probe["streams"] if s["codec_type"] == "audio"]
    assert len(audio_streams) == 1


def test_extract_audio_real_missing_input(tmp_path, caplog):
    output_path = tmp_path / "should_not_exist.wav"

    result = extract_audio("this_file_does_not_exist.mp4", str(output_path))

    assert result is None
    assert not output_path.exists()
    assert "error" in caplog.text.lower()
