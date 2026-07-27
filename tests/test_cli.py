"""Tests for cli argument parsing and wiring."""

from pathlib import Path

import pytest

from railcam import cli
from railcam.cli import create_parser
from railcam.pose import DetectionResult, PelvisPosition, TorsoMeasurement


class _RecordingDetector:
    """Stands in for PoseDetector, recording the model size requested."""

    def __init__(self, model_sizes: list[str], **kwargs: object) -> None:
        model_sizes.append(str(kwargs.get("model_size")))

    def __enter__(self) -> "_RecordingDetector":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def _record_detector_model(monkeypatch) -> list[str]:
    """Intercept detector construction and record the model size it is given."""
    sizes: list[str] = []
    monkeypatch.setattr(cli, "PoseDetector", lambda **kwargs: _RecordingDetector(sizes, **kwargs))
    return sizes


class TestModelOption:
    def test_defaults_to_small(self):
        args = create_parser().parse_args(["video.mp4", "0", "10"])
        assert args.model == "s"

    def test_explicit_model(self):
        args = create_parser().parse_args(["video.mp4", "0", "10", "--model", "n"])
        assert args.model == "n"

    def test_all_sizes_accepted(self):
        for size in ("n", "s", "m", "l", "x"):
            args = create_parser().parse_args(["video.mp4", "0", "10", "--model", size])
            assert args.model == size

    def test_invalid_model_exits(self):
        with pytest.raises(SystemExit):
            create_parser().parse_args(["video.mp4", "0", "10", "--model", "huge"])


class TestModelWiring:
    """The parsed model size must reach the detector."""

    def test_requested_model_reaches_detector(self, monkeypatch, capsys):
        sizes = _record_detector_model(monkeypatch)
        # The run stops at the missing video, after the detector is built.
        assert cli.main(["missing.mp4", "0", "10", "--model", "n"]) == 1
        capsys.readouterr()
        assert sizes == ["n"]

    def test_default_model_reaches_detector(self, monkeypatch, capsys):
        sizes = _record_detector_model(monkeypatch)
        assert cli.main(["missing.mp4", "0", "10"]) == 1
        capsys.readouterr()
        assert sizes == ["s"]


class TestSequentialPlanning:
    """Videos are analyzed one at a time, and no analysis retains frames."""

    def _patch(self, monkeypatch):
        calls: list[str] = []
        analyses: list[cli.VideoAnalysisResult] = []

        def fake_analyze(video_input, detector):
            calls.append(video_input.path.stem)
            analysis = _analysis_for_plan()
            analyses.append(analysis)
            return analysis

        monkeypatch.setattr(cli, "analyze_video", fake_analyze)
        return calls, analyses

    def _inputs(self):
        return [
            cli.VideoInput(path=Path("a.mp4"), start_frame=0, end_frame=10),
            cli.VideoInput(path=Path("b.mp4"), start_frame=0, end_frame=10),
        ]

    def test_videos_analyzed_in_order(self, monkeypatch, capsys):
        calls, _ = self._patch(monkeypatch)

        cli.plan_videos(self._inputs(), detector=object())
        capsys.readouterr()

        assert calls == ["a", "b"]

    def test_analysis_carries_no_frames(self, monkeypatch, capsys):
        _, analyses = self._patch(monkeypatch)

        cli.plan_videos(self._inputs(), detector=object())
        capsys.readouterr()

        for analysis in analyses:
            assert not hasattr(analysis, "frames_cache")

    def test_returns_one_stream_per_video(self, monkeypatch, capsys):
        self._patch(monkeypatch)

        streams = cli.plan_videos(self._inputs(), detector=object())
        capsys.readouterr()

        assert [s.video_input.path.stem for s in streams] == ["a", "b"]

    def test_planning_reads_no_frame(self, monkeypatch, capsys):
        """Nothing is decoded until the crop stream is actually consumed."""
        self._patch(monkeypatch)
        monkeypatch.setattr(
            cli, "extract_frames", lambda *a, **k: pytest.fail("planning decoded a frame")
        )

        cli.plan_videos(self._inputs(), detector=object())
        capsys.readouterr()


def _analysis_for_plan(
    frame_count: int = 3, torso: float = 0.12, width: int = 640, height: int = 480
) -> cli.VideoAnalysisResult:
    """An analysis carrying detections but no frames, as a plan needs."""
    dets = [
        DetectionResult(
            frame_num=i,
            position=PelvisPosition(x=0.5, y=0.5, confidence=0.9),
            torso=TorsoMeasurement(height=torso, shoulder_y=0.4, hip_y=0.5, confidence=0.9),
        )
        for i in range(frame_count)
    ]
    return cli.VideoAnalysisResult(
        detections=dets,
        detections_by_frame={d.frame_num: d for d in dets},
        avg_torso_height=torso,
        fps=30.0,
        video_width=width,
        video_height=height,
        frame_count=frame_count,
    )


class TestCropPlan:
    """The geometry is computable from the analysis alone, with no pixels."""

    def test_plan_needs_no_frames(self) -> None:
        plan = cli.build_crop_plan(_analysis_for_plan(), target_torso_ratio=1 / 6)

        assert plan.output_width > 0
        assert plan.output_height > 0

    def test_scale_factor_reaches_the_target_torso_ratio(self) -> None:
        analysis = _analysis_for_plan(torso=0.12, height=480)
        plan = cli.build_crop_plan(analysis, target_torso_ratio=1 / 6)

        torso_px_in_output = 0.12 * 480 * plan.scale_factor
        assert torso_px_in_output / plan.output_height == pytest.approx(1 / 6, rel=1e-3)

    def test_zero_torso_leaves_scale_untouched(self) -> None:
        plan = cli.build_crop_plan(_analysis_for_plan(torso=0.0), target_torso_ratio=1 / 6)

        assert plan.scale_factor == 1.0

    def test_one_position_per_frame_in_order(self) -> None:
        plan = cli.build_crop_plan(_analysis_for_plan(frame_count=5), target_torso_ratio=1 / 6)

        assert [p.frame_num for p in plan.positions] == [0, 1, 2, 3, 4]

    def test_padding_flagged_when_scaled_frame_is_smaller_than_output(self) -> None:
        # A large torso forces a scale below 1, shrinking the frame under the crop
        plan = cli.build_crop_plan(_analysis_for_plan(torso=0.9), target_torso_ratio=1 / 6)

        assert plan.needs_padding is (
            plan.scaled_width < plan.output_width or plan.scaled_height < plan.output_height
        )
