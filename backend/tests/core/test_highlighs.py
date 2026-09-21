from unittest.mock import MagicMock, patch

import pytest

from clipper.core.exceptions import LLMError
from clipper.core.highlights import (
    Candidate,
    HighlightVerdict,
    analyze_candidate,
    build_candidate_windows,
    deduplicate_candidates,
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

    with patch(
        "clipper.core.highlights.client.chat.completions.parse",
        return_value=fake_completion,
    ):
        result = analyze_candidate(Candidate(text="...", start=0, end=45))

    assert result.is_clip_worthy is True
    assert result.confidence == 0.85


def test_analyze_candidate_raises_on_missing_parsed():
    fake_completion = MagicMock()
    fake_completion.choices[0].finish_reason = "stop"
    fake_completion.choices[0].message.refusal = None
    fake_completion.choices[0].message.parsed = None

    with (
        patch(
            "clipper.core.highlights.client.chat.completions.parse",
            return_value=fake_completion,
        ),
        pytest.raises(LLMError, match="no parsed content"),
    ):
        analyze_candidate(Candidate(text="...", start=0, end=45))


def test_analyze_candidate_raises_on_refusal():
    fake_completion = MagicMock()
    fake_completion.choices[0].finish_reason = "stop"
    fake_completion.choices[0].message.refusal = "cannot evaluate this content"

    with (
        patch(
            "clipper.core.highlights.client.chat.completions.parse",
            return_value=fake_completion,
        ),
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
