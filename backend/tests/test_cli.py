from unittest.mock import patch

from clipper.cli import run_pipeline


def test_run_pipeline_returns_empty_list_when_nothing_qualifies(tmp_path):
    with (
        patch("clipper.cli.extract_audio") as extract_audio,
        patch("clipper.cli.transcribe", return_value={"segments": []}),
        patch("clipper.cli.extract_clips_from_video", return_value=[]) as extract_clips,
    ):
        result = run_pipeline("in.mp4", str(tmp_path), 0.85, 3)

    assert result == []
    extract_audio.assert_called_once_with("in.mp4", str(tmp_path / "audio.wav"))
    extract_clips.assert_called_once_with(
        "in.mp4",
        {"segments": []},
        str(tmp_path),
        min_confidence=0.85,
        max_highlights=3,
    )
