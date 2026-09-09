from pathlib import Path

import pytest


@pytest.fixture
def sample_video_path() -> Path:
    return Path(__file__).parent / "fixtures" / "sample.mp4"
