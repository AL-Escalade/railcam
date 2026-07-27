"""Tests for pose detection types.

Climber selection is now motion-based: see tests/test_tracking.py.
"""

from __future__ import annotations

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
