"""Command-line entry point for the video clipping pipeline."""

import argparse
from pathlib import Path

from clipper.core.exceptions import ClipperError
from clipper.core.highlights import extract_clips_from_video
from clipper.core.media import extract_audio
from clipper.core.transcriber import transcribe


def run_pipeline(
    video_path: str,
    output_dir: str,
    min_confidence: float = 0.7,
    max_highlights: int = 5,
) -> list[str]:
    """Extract audio, transcribe it, and write ranked, subtitled clips."""
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    print(f"Extracting audio from {video_path}...")
    audio_path = str(output_dir_path / "audio.wav")
    extract_audio(video_path, audio_path)

    print("Transcribing (this can take a while on CPU)...")
    transcript = transcribe(audio_path)

    print("Finding highlight-worthy moments...")
    return extract_clips_from_video(
        video_path,
        transcript,
        str(output_dir_path),
        min_confidence=min_confidence,
        max_highlights=max_highlights,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract highlight clips from a video, with burnt-in subtitles."
    )
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument(
        "-o",
        "--output-dir",
        default="clips",
        help="Output directory (default: ./clips)",
    )
    parser.add_argument("--min-confidence", type=float, default=0.7)
    parser.add_argument("--max-highlights", type=int, default=5)
    args = parser.parse_args()

    try:
        clips = run_pipeline(
            args.video, args.output_dir, args.min_confidence, args.max_highlights
        )
    except ClipperError as error:
        print(f"Error: {error}")
        raise SystemExit(1) from error

    if not clips:
        print("\nNo moments met the confidence bar for this video.")
        return

    print(f"\n{len(clips)} clip(s) written to {args.output_dir}/:")
    for clip_path in clips:
        print(f"  {clip_path}")


if __name__ == "__main__":
    main()
