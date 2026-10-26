"""Tests for the release version comparison."""

from __future__ import annotations

import pytest

from railcam.gui.updates import is_newer, parse_version


@pytest.mark.parametrize(
    ("latest", "current", "expected"),
    [
        ("v1.1.0", "1.0.0", True),
        ("v1.0.10", "1.0.9", True),
        ("v2.0.0", "1.9.9", True),
        ("v1.0.0", "1.0.0", False),
        ("v1.0.0", "1.1.0", False),
        ("v1.2.0-rc.1", "1.0.0", False),
        ("nightly", "1.0.0", False),
    ],
)
def test_is_newer(latest: str, current: str, expected: bool) -> None:
    assert is_newer(latest, current) is expected


def test_parse_version_accepts_tag_prefix() -> None:
    assert parse_version("v1.2.3") == (1, 2, 3)
    assert parse_version("1.2.3") == (1, 2, 3)
