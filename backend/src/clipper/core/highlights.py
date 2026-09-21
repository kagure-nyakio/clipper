from openai import OpenAI
from pydantic import BaseModel

from clipper.config import OPENAI_API_KEY
from clipper.core.exceptions import LLMError
from clipper.core.media import create_clip

client = OpenAI(api_key=OPENAI_API_KEY)


class Candidate(BaseModel):
    end: float
    start: float
    text: str


class HighlightVerdict(BaseModel):
    reasoning: str
    is_clip_worthy: bool
    confidence: float
    suggested_hook: str | None


def build_candidate_windows(
    transcript: dict, window_seconds: float = 45, step_seconds: float = 20
) -> list[Candidate]:
    segments = transcript["segments"]
    candidates = []
    window_start_idx = 0

    while window_start_idx < len(segments):
        window_segments = []
        window_start_time = segments[window_start_idx]["start"]

        for seg in segments[window_start_idx:]:
            if seg["start"] - window_start_time > window_seconds:
                break
            window_segments.append(seg)

        if window_segments:
            candidates.append(
                Candidate(
                    text=" ".join(s["text"] for s in window_segments),
                    start=window_segments[0]["start"],
                    end=window_segments[-1]["end"],
                )
            )

        # advance to the segment that's ~step_seconds past this window's start
        next_start_time = window_start_time + step_seconds
        window_start_idx = next(
            (i for i, s in enumerate(segments) if s["start"] >= next_start_time),
            len(segments),
        )

    return candidates


def analyze_candidate(candidate: Candidate) -> HighlightVerdict:
    completion = client.chat.completions.parse(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an experienced short-form video editor who finds "
                    "viral-worthy moments in podcast transcripts."
                ),
            },
            {
                "role": "user",
                "content": f"""
Evaluate the transcript segment below for short-form clip potential.

A great clip has at least one of: a surprising claim, an emotional peak,
a clear payoff or punchline, a concise actionable insight — and is fully
self-contained, understandable without earlier context.

<segment>
{candidate.text}
</segment>

First explain your reasoning, then give your verdict.
""",
            },
        ],
        response_format=HighlightVerdict,
    )

    choice = completion.choices[0]
    if choice.finish_reason == "length":
        raise LLMError("Response truncated before completing")
    if choice.message.refusal:
        raise LLMError(f"Model refused: {choice.message.refusal}")

    parsed = choice.message.parsed
    if parsed is None:
        raise LLMError("Model returned no parsed content")

    return parsed


def deduplicate_candidates(
    scored: list[tuple[Candidate, HighlightVerdict]], overlap_threshold: float = 0.3
) -> list[tuple[Candidate, HighlightVerdict]]:
    sorted_scored = sorted(scored, key=lambda x: x[1].confidence, reverse=True)
    kept: list[tuple[Candidate, HighlightVerdict]] = []
    for candidate, verdict in sorted_scored:
        if not any(_overlap_ratio(candidate, k) > overlap_threshold for k, _ in kept):
            kept.append((candidate, verdict))
    return kept


def _overlap_ratio(a: Candidate, b: Candidate) -> float:
    intersection = max(0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return intersection / union if union > 0 else 0


def enforce_duration_bounds(
    candidate: Candidate,
    segments: list[dict],
    min_seconds: float = 30,
    max_seconds: float = 60,
) -> Candidate:
    """Trim to segment boundaries if over max_seconds. Segments are already clean
    sentence-ish chunks from Whisper, so this — not word-level snapping — is the
    scoped-down version of Stage 4 given the deadline: trims at existing clean
    boundaries rather than computing new ones from word timestamps."""
    duration = candidate.end - candidate.start
    if duration <= max_seconds:
        return candidate

    relevant = [
        s
        for s in segments
        if s["start"] >= candidate.start and s["end"] <= candidate.end
    ]
    trimmed, running_start = [], relevant[0]["start"]
    for seg in relevant:
        if seg["end"] - running_start > max_seconds:
            break
        trimmed.append(seg)

    if not trimmed:
        return candidate
    return Candidate(
        text=" ".join(s["text"] for s in trimmed),
        start=trimmed[0]["start"],
        end=trimmed[-1]["end"],
    )


def rank_highlights(
    deduped: list[tuple[Candidate, HighlightVerdict]],
    min_confidence: float = 0.7,
    max_highlights: int = 5,
) -> list[tuple[Candidate, HighlightVerdict]]:
    filtered = [
        (c, v)
        for c, v in deduped
        if v.is_clip_worthy and v.confidence >= min_confidence
    ]
    return sorted(filtered, key=lambda x: x[1].confidence, reverse=True)[
        :max_highlights
    ]


def detect_highlights(
    transcript: dict,
    window_seconds: float = 45,
    step_seconds: float = 20,
    min_confidence: float = 0.7,
    max_highlights: int = 5,
) -> list[tuple[Candidate, HighlightVerdict]]:
    windows = build_candidate_windows(transcript, window_seconds, step_seconds)
    bounded = [enforce_duration_bounds(w, transcript["segments"]) for w in windows]
    scored = [(c, analyze_candidate(c)) for c in bounded]
    deduped = deduplicate_candidates(scored)
    return rank_highlights(deduped, min_confidence, max_highlights)


def _seconds_to_timestamp(seconds: float) -> str:
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def extract_clips_from_video(
    video_path: str, transcript: dict, output_dir: str, precise: bool = True
) -> list[str]:
    output_paths = []
    for i, (candidate, verdict) in enumerate(detect_highlights(transcript)):
        output_path = f"{output_dir}/highlight_{i + 1}.mp4"
        create_clip(
            video_path,
            _seconds_to_timestamp(candidate.start),
            _seconds_to_timestamp(candidate.end),
            output_path,
            precise=precise,
        )
        output_paths.append(output_path)
    return output_paths
