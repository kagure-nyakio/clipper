from unittest.mock import MagicMock, patch

import pytest

from clipper.core.exceptions import LLMError
from clipper.core.highlights import (
    Candidate,
    HighlightVerdict,
    analyze_candidate,
    build_candidate_windows,
    deduplicate_candidates,
    extract_clips_from_video,
    generate_srt_for_candidate,
)


def test_build_candidate_windows_produces_overlapping_windows(sample_speech_transcript):
    windows = build_candidate_windows(
        sample_speech_transcript, window_seconds=10, step_seconds=5
    )

    assert len(windows) > 1
    assert windows[1].start < windows[0].end


def test_build_candidate_windows_respects_target_length(sample_speech_transcript):
    windows = build_candidate_windows(
        sample_speech_transcript, window_seconds=10, step_seconds=5
    )
    for w in windows:
        assert (w.end - w.start) <= 15  # some slack over the target, never wildly over


def test_build_candidate_windows_empty_transcript_returns_empty():
    assert build_candidate_windows({"segments": []}) == []


def test_analyze_candidate_returns_parsed_verdict():
    fake_verdict = HighlightVerdict(
        reasoning="Contains a strong hook and a clear payoff.",
        is_clip_worthy=True,
        confidence=0.85,
        suggested_hook="Wait, WHAT?",
    )
    fake_completion = MagicMock()
    fake_completion.choices[0].finish_reason = "stop"
    fake_completion.choices[0].message.refusal = None
    fake_completion.choices[0].message.parsed = fake_verdict

    fake_client = MagicMock()
    fake_client.chat.completions.parse.return_value = fake_completion

    with patch("clipper.core.highlights._get_client", return_value=fake_client):
        result = analyze_candidate(Candidate(text="...", start=0, end=45))

    assert result.is_clip_worthy is True
    assert result.confidence == 0.85


def test_analyze_candidate_raises_on_missing_parsed():
    fake_completion = MagicMock()
    fake_completion.choices[0].finish_reason = "stop"
    fake_completion.choices[0].message.refusal = None
    fake_completion.choices[0].message.parsed = None

    fake_client = MagicMock()
    fake_client.chat.completions.parse.return_value = fake_completion

    with (
        patch("clipper.core.highlights._get_client", return_value=fake_client),
        pytest.raises(LLMError, match="no parsed content"),
    ):
        analyze_candidate(Candidate(text="...", start=0, end=45))


def test_analyze_candidate_raises_on_refusal():
    fake_completion = MagicMock()
    fake_completion.choices[0].finish_reason = "stop"
    fake_completion.choices[0].message.refusal = "cannot evaluate this content"

    fake_client = MagicMock()
    fake_client.chat.completions.parse.return_value = fake_completion

    with (
        patch("clipper.core.highlights._get_client", return_value=fake_client),
        pytest.raises(LLMError, match="refused"),
    ):
        analyze_candidate(Candidate(text="...", start=0, end=45))


def test_deduplicate_keeps_highest_confidence_and_drops_overlaps():
    c1 = Candidate(text="a", start=0, end=45)
    c2 = Candidate(text="b", start=10, end=55)  # heavily overlaps c1
    c3 = Candidate(text="c", start=100, end=145)  # no overlap
    v1 = HighlightVerdict(
        reasoning="", is_clip_worthy=True, confidence=0.6, suggested_hook=None
    )
    v2 = HighlightVerdict(
        reasoning="", is_clip_worthy=True, confidence=0.9, suggested_hook=None
    )
    v3 = HighlightVerdict(
        reasoning="", is_clip_worthy=True, confidence=0.7, suggested_hook=None
    )

    result = deduplicate_candidates([(c1, v1), (c2, v2), (c3, v3)])

    kept_texts = {c.text for c, _ in result}
    assert kept_texts == {"b", "c"}


def test_generate_srt_retimes_and_clips_segments(tmp_path):
    candidate = Candidate(text="hello there", start=10.0, end=15.0)
    transcript = {
        "segments": [
            {"text": "too early", "start": 0.0, "end": 5.0},
            {"text": "hello there", "start": 10.0, "end": 12.0},
            {"text": "general kenobi", "start": 12.5, "end": 16.0},
        ]
    }
    srt_path = tmp_path / "highlight.srt"

    generate_srt_for_candidate(candidate, transcript, str(srt_path))

    content = srt_path.read_text()
    assert "00:00:00,000 --> 00:00:02,000" in content
    assert "00:00:02,500 --> 00:00:05,000" in content
    assert "hello there" in content
    assert "too early" not in content


def test_extract_clips_uses_precomputed_highlights_without_subtitles():
    candidate = Candidate(text="a strong moment", start=12.5, end=42.75)
    verdict = HighlightVerdict(
        reasoning="Strong hook.",
        is_clip_worthy=True,
        confidence=0.9,
        suggested_hook=None,
    )

    with patch("clipper.core.highlights.create_clip") as create_clip:
        paths = extract_clips_from_video(
            "input.mp4",
            [(candidate, verdict)],
            "clips",
            precise=False,
            burn_subtitles=False,
        )

    assert paths == ["clips/highlight_1.mp4"]
    create_clip.assert_called_once_with(
        "input.mp4",
        "00:00:12.500",
        "00:00:42.750",
        "clips/highlight_1.mp4",
        precise=False,
    )


def test_extract_clips_detects_highlights_and_burns_subtitles(tmp_path):
    candidate = Candidate(text="a strong moment", start=0.0, end=10.0)
    verdict = HighlightVerdict(
        reasoning="Strong hook.",
        is_clip_worthy=True,
        confidence=0.9,
        suggested_hook=None,
    )
    transcript = {"segments": [{"text": "a strong moment", "start": 0.0, "end": 10.0}]}

    with (
        patch(
            "clipper.core.highlights.detect_highlights",
            return_value=[(candidate, verdict)],
        ),
        patch("clipper.core.highlights.create_clip") as create_clip,
        patch("clipper.core.highlights.add_subtitles") as add_subtitles,
    ):
        paths = extract_clips_from_video("input.mp4", transcript, str(tmp_path))

    assert paths == [f"{tmp_path}/highlight_1.mp4"]
    create_clip.assert_called_once()
    add_subtitles.assert_called_once()
