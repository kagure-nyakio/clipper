from typing import TypedDict

from faster_whisper import WhisperModel

from clipper.core.exceptions import TranscriptionError

_model: WhisperModel | None = None


class Word(TypedDict):
    word: str
    start: float
    end: float


class Segment(TypedDict):
    text: str
    start: float
    end: float
    words: list[Word]


class Transcript(TypedDict):
    text: str
    segments: list[Segment]


def _get_model(model_size: str = "base") -> WhisperModel:
    """Load the Whisper model once, reuse across calls (loading is slow, transcribing isn't)."""
    global _model
    if _model is None:
        _model = WhisperModel(model_size, device="cpu", compute_type="int8")
    return _model


def transcribe(audio_path: str, model_size: str = "base") -> Transcript:
    """Transcribe an audio file into segments + word-level timestamps."""
    try:
        model = _get_model(model_size)
        segments_iter, _info = model.transcribe(audio_path, word_timestamps=True)

        segments: list[Segment] = []
        full_text_parts: list[str] = []

        for segment in segments_iter:
            words: list[Word] = [
                {"word": w.word.strip(), "start": float(w.start), "end": float(w.end)}
                for w in (segment.words or [])
            ]
            segments.append(
                {
                    "text": segment.text.strip(),
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "words": words,
                }
            )
            full_text_parts.append(segment.text.strip())

        return {"text": " ".join(full_text_parts), "segments": segments}

    except Exception as e:
        raise TranscriptionError(f"Failed to transcribe '{audio_path}': {e}") from e
