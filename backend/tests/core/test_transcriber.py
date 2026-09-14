import re
from unittest.mock import patch

import pytest

from clipper.core.exceptions import TranscriptionError
from clipper.core.transcriber import transcribe


@pytest.mark.slow
def test_transcribe_real_audio_produces_valid_shape(sample_speech_path):
    result = transcribe(str(sample_speech_path))

    assert isinstance(result["text"], str)
    assert len(result["text"]) > 0
    assert len(result["segments"]) > 0

    for segment in result["segments"]:
        assert segment["end"] > segment["start"]
        assert len(segment["words"]) > 0
        for word in segment["words"]:
            assert word["end"] >= word["start"]
            assert isinstance(word["start"], float)
            assert not hasattr(word["start"], "dtype")

    all_starts = [s["start"] for s in result["segments"]]
    assert all_starts == sorted(all_starts)


def _normalize_text(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def test_transcribe_raises_on_engine_failure():
    with patch("clipper.core.transcriber._get_model") as mock_get_model:
        mock_get_model.return_value.transcribe.side_effect = RuntimeError(
            "model crashed"
        )

        with pytest.raises(TranscriptionError, match="Failed to transcribe"):
            transcribe("irrelevant_path.wav")


def test_transcribe_raises_on_missing_file():
    with pytest.raises(TranscriptionError, match="Failed to transcribe"):
        transcribe("this_file_does_not_exist.wav")
