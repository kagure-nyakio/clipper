import re

from clipper.core.exceptions import ValidationError

TIMESTAMP_RE = re.compile(r"^(\d+):([0-5]?\d):([0-5]?\d(?:\.\d+)?)$")


def validate_timestamp(ts, field_name="timestamp"):
    """
    Validates 'HH:MM:SS' or 'HH:MM:SS.mmm' format.
    Returns the total seconds as a float if valid.
    Raises ValidationError with a clear message if not.
    """
    if not isinstance(ts, str) or not ts.strip():
        raise ValidationError(f"{field_name} must be a non-empty string, got: {ts!r}")

    match = TIMESTAMP_RE.match(ts.strip())
    if not match:
        raise ValidationError(f"{field_name} '{ts}' is not a valid HH:MM:SS timestamp")

    h, m, s = match.groups()
    h, m, s = int(h), int(m), float(s)

    return h * 3600 + m * 60 + s


def validate_range(start, end):
    """Validates start/end strings and ensures end comes after start."""
    start_secs = validate_timestamp(start, "start")
    end_secs = validate_timestamp(end, "end")

    if end_secs <= start_secs:
        raise ValidationError(f"end ('{end}') must be after start ('{start}')")

    return start_secs, end_secs
