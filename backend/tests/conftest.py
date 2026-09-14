import json
from pathlib import Path

import pytest


@pytest.fixture
def sample_video_path() -> Path:
    return Path(__file__).parent / "fixtures" / "sample.mp4"


@pytest.fixture
def sample_speech_path() -> Path:
    return Path(__file__).parent / "fixtures" / "sample_speech.wav"


@pytest.fixture
def sample_speech_transcript() -> Path:
    path = Path(__file__).parent / "fixtures" / "sample_speech_transcript.json"
    return json.loads(path.read_text())
