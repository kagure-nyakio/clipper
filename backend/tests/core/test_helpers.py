# tests/core/test_helpers.py
import pytest

from clipper.core.exceptions import ValidationError
from clipper.core.helpers import validate_range, validate_timestamp


@pytest.mark.parametrize(
    "timestamp,expected_seconds",
    [
        ("00:00:01", 1.0),
        ("00:00:01.5", 1.5),
        ("1:2:3", 3723.0),
        ("00:01:00", 60.0),
        ("01:00:00", 3600.0),
        ("25:00:00", 90000.0),
        (" 00:00:02 ", 2.0),
    ],
)
def test_validate_timestamp_parses_valid_formats(timestamp, expected_seconds):
    assert validate_timestamp(timestamp) == pytest.approx(expected_seconds)


@pytest.mark.parametrize(
    "timestamp",
    [
        "bad",
        "1:2",
        "1:2:3:4",
        "00:60:00",
        "00:00:60",
        "-1:00:00",
    ],
)
def test_validate_timestamp_rejects_malformed_strings(timestamp):
    with pytest.raises(ValidationError, match="not a valid HH:MM:SS timestamp"):
        validate_timestamp(timestamp)


@pytest.mark.parametrize("timestamp", ["", "   "])
def test_validate_timestamp_rejects_empty_string(timestamp):
    with pytest.raises(ValidationError, match="must be a non-empty string"):
        validate_timestamp(timestamp)


def test_validate_timestamp_rejects_non_string_input():
    with pytest.raises(ValidationError, match="must be a non-empty string"):
        validate_timestamp(90)


def test_validate_timestamp_error_includes_field_name():
    with pytest.raises(ValidationError, match="start"):
        validate_timestamp("bad", field_name="start")


def test_validate_range_returns_seconds_tuple():
    start_secs, end_secs = validate_range("00:00:10", "00:01:30")
    assert start_secs == pytest.approx(10.0)
    assert end_secs == pytest.approx(90.0)


def test_validate_range_rejects_end_before_start():
    with pytest.raises(ValidationError, match="must be after start"):
        validate_range("00:01:00", "00:00:30")


def test_validate_range_rejects_end_equal_to_start():
    with pytest.raises(ValidationError, match="must be after start"):
        validate_range("00:00:10", "00:00:10")


def test_validate_range_propagates_timestamp_errors():
    with pytest.raises(ValidationError, match="not a valid HH:MM:SS timestamp"):
        validate_range("bad", "00:00:10")
