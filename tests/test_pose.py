"""Tests for pose detection types.

Climber selection is now motion-based: see tests/test_tracking.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from railcam import pose
from railcam.pose import ClimberSelector


def _record_loaded_weights(monkeypatch) -> list[str]:
    """Intercept model loading and record the weights path requested.

    Loading real weights would download them; this is the one boundary the
    pose tests cannot cross.
    """
    loaded: list[str] = []

    def fake_yolo(path: str) -> object:
        loaded.append(path)
        return object()

    monkeypatch.setattr(pose, "YOLO", fake_yolo)
    return loaded


class TestModelSize:
    def test_default_loads_small_weights(self, monkeypatch) -> None:
        loaded = _record_loaded_weights(monkeypatch)
        pose.PoseDetector()
        assert loaded[0].endswith("yolov8s-pose.pt")

    def test_explicit_size_loads_matching_weights(self, monkeypatch) -> None:
        loaded = _record_loaded_weights(monkeypatch)
        pose.PoseDetector(model_size="m")
        assert loaded[0].endswith("yolov8m-pose.pt")

    def test_default_constant_is_a_known_size(self) -> None:
        assert pose.DEFAULT_MODEL_SIZE in pose.MODEL_SIZES


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


def _frame(width: int, height: int) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


class TestInferenceResolution:
    """The resolution follows the source: a fixed size loses 4K climbers."""

    def test_scales_with_the_largest_side(self) -> None:
        assert pose.auto_imgsz(2160, 3840) == 1920
        assert pose.auto_imgsz(2560, 1440) == 1280

    def test_stays_within_bounds(self) -> None:
        assert pose.auto_imgsz(320, 240) == pose.MIN_IMGSZ
        assert pose.auto_imgsz(7680, 4320) == pose.MAX_IMGSZ

    def test_is_a_multiple_of_the_stride(self) -> None:
        for side in range(640, 5000, 137):
            assert pose.auto_imgsz(side, side) % pose.IMGSZ_MULTIPLE == 0

    def test_detector_uses_the_auto_size(self, monkeypatch) -> None:
        _record_loaded_weights(monkeypatch)
        detector = pose.PoseDetector()

        assert detector.inference_size(_frame(2160, 3840)) == 1920

    def test_explicit_size_overrides_the_source(self, monkeypatch) -> None:
        _record_loaded_weights(monkeypatch)
        detector = pose.PoseDetector(imgsz=960)

        assert detector.inference_size(_frame(2160, 3840)) == 960

    def test_repair_looks_closer_than_the_main_pass(self, monkeypatch) -> None:
        _record_loaded_weights(monkeypatch)
        detector = pose.PoseDetector()
        frame = _frame(1280, 720)

        assert detector.repair_size(frame) == detector.inference_size(frame) * 2

    def test_repair_is_capped(self, monkeypatch) -> None:
        _record_loaded_weights(monkeypatch)
        detector = pose.PoseDetector()

        assert detector.repair_size(_frame(2160, 3840)) == pose.MAX_REPAIR_IMGSZ

    def test_inference_size_reaches_the_model(self, monkeypatch) -> None:
        _record_loaded_weights(monkeypatch)
        detector = pose.PoseDetector()
        calls: list[int] = []

        def fake_model(frame, imgsz, verbose):
            calls.append(imgsz)
            return []

        monkeypatch.setattr(detector, "_model", fake_model)
        detector.detect_all_persons(_frame(2160, 3840), 0)

        assert calls == [1920]
