"""Tests for pose detection types.

Climber selection is now motion-based: see tests/test_tracking.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from railcam import pose
from railcam.pose import ClimberSelector


class TestClimberSelector:
    """Tests for ClimberSelector enum."""

    def test_left_value(self) -> None:
        assert ClimberSelector.LEFT.value == "left"

    def test_right_value(self) -> None:
        assert ClimberSelector.RIGHT.value == "right"

    def test_auto_value(self) -> None:
        assert ClimberSelector.AUTO.value == "auto"


class _FakeTensor:
    """Mimics the torch tensors ultralytics returns inside Keypoints."""

    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def cpu(self) -> _FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._array


class _FakeKeypoints:
    """One person's keypoints, shaped like ultralytics' Keypoints."""

    def __init__(self, xy: np.ndarray, conf: np.ndarray) -> None:
        self.xy = [_FakeTensor(xy)]
        self.conf = [_FakeTensor(conf)]


class _FakeResult:
    def __init__(self, keypoints: object) -> None:
        self.keypoints = keypoints


def _person(hip_y: float = 80.0, shoulder_y: float = 60.0, x: float = 50.0):
    """Build keypoints for one person on a 100x100 frame."""
    xy = np.zeros((17, 2), dtype=float)
    conf = np.full(17, 0.9)
    for index in (pose.LEFT_SHOULDER, pose.RIGHT_SHOULDER):
        xy[index] = (x, shoulder_y)
    for index in (pose.LEFT_HIP, pose.RIGHT_HIP):
        xy[index] = (x, hip_y)
    return _FakeKeypoints(xy, conf)


def _detector_returning(monkeypatch, results: object) -> pose.PoseDetector:
    """A detector whose model returns the given inference results."""
    monkeypatch.setattr(pose, "YOLO", lambda _path: lambda *a, **k: results)
    return pose.PoseDetector()


FRAME = np.zeros((100, 100, 3), dtype=np.uint8)


class TestDetectAllPersons:
    """Characterizes inference result handling, which had no coverage."""

    def test_returns_every_person_with_visible_hips(self, monkeypatch) -> None:
        detector = _detector_returning(
            monkeypatch, [_FakeResult([_person(x=20.0), _person(x=70.0)])]
        )

        result = detector.detect_all_persons(FRAME, frame_num=3)

        assert result.frame_num == 3
        assert [round(p.pelvis.x, 2) for p in result.persons] == [0.2, 0.7]

    def test_pelvis_is_the_hip_midpoint_normalized(self, monkeypatch) -> None:
        detector = _detector_returning(monkeypatch, [_FakeResult([_person(hip_y=80.0)])])

        person = detector.detect_all_persons(FRAME, frame_num=0).persons[0]

        assert person.pelvis.y == pytest.approx(0.8)

    def test_torso_height_spans_shoulders_to_hips(self, monkeypatch) -> None:
        detector = _detector_returning(
            monkeypatch, [_FakeResult([_person(hip_y=80.0, shoulder_y=60.0)])]
        )

        person = detector.detect_all_persons(FRAME, frame_num=0).persons[0]

        assert person.torso is not None
        assert person.torso.height == pytest.approx(0.2)

    def test_no_results_yields_no_persons(self, monkeypatch) -> None:
        detector = _detector_returning(monkeypatch, [])

        assert detector.detect_all_persons(FRAME, frame_num=0).persons == []

    def test_missing_keypoints_yields_no_persons(self, monkeypatch) -> None:
        detector = _detector_returning(monkeypatch, [_FakeResult(None)])

        assert detector.detect_all_persons(FRAME, frame_num=0).persons == []

    def test_empty_keypoints_yields_no_persons(self, monkeypatch) -> None:
        detector = _detector_returning(monkeypatch, [_FakeResult([])])

        assert detector.detect_all_persons(FRAME, frame_num=0).persons == []


class TestDetectPelvis:
    def test_returns_the_first_person(self, monkeypatch) -> None:
        detector = _detector_returning(
            monkeypatch, [_FakeResult([_person(x=20.0), _person(x=70.0)])]
        )

        result = detector.detect_pelvis(FRAME, frame_num=7)

        assert result.frame_num == 7
        assert result.position is not None
        assert result.position.x == pytest.approx(0.2)

    def test_no_results_yields_no_position(self, monkeypatch) -> None:
        detector = _detector_returning(monkeypatch, [])

        assert detector.detect_pelvis(FRAME, frame_num=0).position is None

    def test_missing_keypoints_yields_no_position(self, monkeypatch) -> None:
        detector = _detector_returning(monkeypatch, [_FakeResult(None)])

        assert detector.detect_pelvis(FRAME, frame_num=0).position is None
