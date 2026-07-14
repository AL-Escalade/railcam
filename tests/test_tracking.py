"""Tests for motion-based climber track building and selection."""

from __future__ import annotations

import pytest

from railcam.pose import (
    ClimberSelector,
    MultiPersonDetectionResult,
    PelvisPosition,
    PersonDetection,
)
from railcam.tracking import build_tracks, select_track, track_to_detections


def person(x: float, y: float) -> PersonDetection:
    return PersonDetection(pelvis=PelvisPosition(x=x, y=y, confidence=0.9))


def frames_of(*frames: list[PersonDetection]) -> list[MultiPersonDetectionResult]:
    return [
        MultiPersonDetectionResult(frame_num=i, persons=list(persons))
        for i, persons in enumerate(frames)
    ]


def climb_y(i: int, count: int = 20) -> float:
    """Vertical trajectory from the bottom (0.9) to the top (0.1)."""
    return 0.9 - 0.8 * i / (count - 1)


STATIC = (0.16, 0.92)


def scene_static_plus_climber(count: int = 20) -> list[MultiPersonDetectionResult]:
    """Reproduces the diagnosed video: a static bystander and one climber."""
    return frames_of(*[[person(*STATIC), person(0.40, climb_y(i, count))] for i in range(count)])


# --- build_tracks --------------------------------------------------------


def test_static_and_climber_form_two_tracks() -> None:
    tracks = build_tracks(scene_static_plus_climber())

    assert len(tracks) == 2
    spans = sorted(track.vertical_span for track in tracks)
    assert spans[0] == pytest.approx(0.0, abs=0.01)
    assert spans[1] == pytest.approx(0.8, abs=0.01)


def test_detection_gap_does_not_split_track() -> None:
    count = 20
    frames = []
    for i in range(count):
        persons = [] if 8 <= i <= 10 else [person(0.40, climb_y(i, count))]
        frames.append(persons)

    tracks = build_tracks(frames_of(*frames))

    assert len(tracks) == 1
    assert len(tracks[0].detections) == count - 3


def test_person_appearing_mid_video_starts_new_track() -> None:
    count = 20
    frames = []
    for i in range(count):
        persons = [person(*STATIC)]
        if i >= 10:
            persons.append(person(0.40, climb_y(i - 10, count)))
        frames.append(persons)

    tracks = build_tracks(frames_of(*frames))

    assert len(tracks) == 2


def test_large_jump_starts_a_new_track_instead_of_teleporting() -> None:
    frames = frames_of(
        [person(0.2, 0.9)],
        [person(0.8, 0.2)],  # 0.9 away: cannot be the same person
    )

    tracks = build_tracks(frames)

    assert len(tracks) == 2


# --- select_track --------------------------------------------------------


def test_static_bystander_is_never_selected_even_when_central() -> None:
    count = 20
    frames = frames_of(
        *[[person(0.50, 0.90), person(0.70, climb_y(i, count))] for i in range(count)]
    )
    tracks = build_tracks(frames)

    selected = select_track(tracks, ClimberSelector.AUTO)

    assert selected is not None
    assert selected.vertical_span > 0.5


def test_left_selector_ignores_static_person_further_left() -> None:
    # Static at x=0.16, climbers at x=0.33 and x=0.66 (the diagnosed video)
    count = 20
    frames = frames_of(
        *[
            [
                person(*STATIC),
                person(0.33, climb_y(i, count)),
                person(0.66, climb_y(i, count)),
            ]
            for i in range(count)
        ]
    )
    tracks = build_tracks(frames)

    left = select_track(tracks, ClimberSelector.LEFT)
    right = select_track(tracks, ClimberSelector.RIGHT)

    assert left is not None and left.mean_x == pytest.approx(0.33, abs=0.02)
    assert right is not None and right.mean_x == pytest.approx(0.66, abs=0.02)


def test_auto_selects_most_central_climbing_track() -> None:
    count = 20
    frames = frames_of(
        *[[person(0.45, climb_y(i, count)), person(0.90, climb_y(i, count))] for i in range(count)]
    )
    tracks = build_tracks(frames)

    selected = select_track(tracks, ClimberSelector.AUTO)

    assert selected is not None
    assert selected.mean_x == pytest.approx(0.45, abs=0.02)


def test_left_selector_ignores_short_fragment_track() -> None:
    # A detection fragment (short vertical span) sits left of the real climber:
    # it must not win the left selection against a full track
    count = 20
    frames = []
    for i in range(count):
        persons = [person(0.40, climb_y(i, count))]
        if i < 4:
            persons.append(person(0.30, 0.60 - 0.04 * i))  # fragment: span 0.12
        frames.append(persons)
    tracks = build_tracks(frames_of(*frames))
    assert len(tracks) == 2

    selected = select_track(tracks, ClimberSelector.LEFT)

    assert selected is not None
    assert selected.mean_x == pytest.approx(0.40, abs=0.02)


def test_fallback_to_all_tracks_when_nothing_climbs() -> None:
    frames = frames_of(*[[person(0.3, 0.5), person(0.7, 0.5)] for _ in range(10)])
    tracks = build_tracks(frames)

    selected = select_track(tracks, ClimberSelector.LEFT)

    assert selected is not None
    assert selected.mean_x == pytest.approx(0.3, abs=0.02)


def test_select_track_with_no_tracks_returns_none() -> None:
    assert select_track([], ClimberSelector.AUTO) is None


# --- track_to_detections -------------------------------------------------


def test_track_converts_to_per_frame_detections_with_gaps() -> None:
    count = 6
    frames = []
    for i in range(count):
        # Realistic climb speed (0.05/frame), gaps at frames 2 and 4
        persons = [] if i in (2, 4) else [person(0.4, 0.9 - 0.05 * i)]
        frames.append(persons)
    tracks = build_tracks(frames_of(*frames))
    frame_nums = list(range(count))

    detections = track_to_detections(tracks[0], frame_nums)

    assert [d.frame_num for d in detections] == frame_nums
    assert [d.position is not None for d in detections] == [
        True,
        True,
        False,
        True,
        False,
        True,
    ]


def test_no_track_yields_only_empty_detections() -> None:
    detections = track_to_detections(None, [0, 1, 2])

    assert all(d.position is None for d in detections)
    assert [d.frame_num for d in detections] == [0, 1, 2]
