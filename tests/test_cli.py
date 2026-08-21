"""Tests for cli argument parsing and wiring."""

from pathlib import Path

import numpy as np
import pytest

from railcam import cli
from railcam.cli import create_parser
from railcam.labeling import LABEL_BAND_RATIO, band_height
from railcam.pose import DetectionResult, PelvisPosition, TorsoMeasurement
from railcam.processing import ProcessedPosition


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


def _plan(decoded: int, fps: float, label: str = "", label_band_height: int = 0) -> cli.CropPlan:
    """A crop plan with `decoded` positions."""
    return cli.CropPlan(
        output_width=40,
        output_height=60,
        scaled_width=40,
        scaled_height=60,
        scale_factor=1.0,
        needs_padding=False,
        positions=[
            ProcessedPosition(frame_num=i, x=0.5, y=0.5, interpolated=False) for i in range(decoded)
        ],
        fps=fps,
        label=label,
        label_band_height=label_band_height,
    )


def _stream(plan: cli.CropPlan) -> cli.VideoStream:
    return cli.VideoStream(
        plan=plan,
        video_input=cli.VideoInput(path=Path("v.mp4"), start_frame=0, end_frame=10),
        detections_by_frame={},
    )


class TestOutputDuration:
    """Durations follow the frames that exist; the plan has no other count."""

    def test_duration_uses_the_decoded_frame_count(self, capsys) -> None:
        # 60 frames @30fps = 2.00s, 50 @25fps = 2.00s; LCM(30,25) = 150
        streams = [_stream(_plan(60, 30.0)), _stream(_plan(50, 25.0))]

        output = cli.build_output_stream(streams)
        capsys.readouterr()

        assert output.total_frames == 300

    def test_shorter_video_freezes_rather_than_shortening_the_output(self, capsys) -> None:
        # 60 @30fps = 2.00s vs 30 @30fps = 1.00s; the longer one sets the length
        streams = [_stream(_plan(60, 30.0)), _stream(_plan(30, 30.0))]

        output = cli.build_output_stream(streams)
        capsys.readouterr()

        assert output.total_frames == 60


def _validated(argv: list[str]) -> list[cli.VideoInput]:
    """Parse an argument list the way main does, up to the VideoInput list."""
    return cli.validate_args(create_parser().parse_args(argv))


class TestLabelOption:
    """Labels are paired with the inputs in the order they are given."""

    def test_label_reaches_the_single_video(self) -> None:
        inputs = _validated(["video.mp4", "0", "10", "--label", "Dupont"])

        assert [vi.label for vi in inputs] == ["Dupont"]

    def test_labels_pair_with_inputs_in_order(self) -> None:
        inputs = _validated(
            ["-i", "a.mp4:0:10", "-i", "b.mp4:0:10", "--label", "Dupont", "--label", "Martin"]
        )

        assert [(vi.path.stem, vi.label) for vi in inputs] == [("a", "Dupont"), ("b", "Martin")]

    def test_missing_labels_are_empty(self) -> None:
        inputs = _validated(["-i", "a.mp4:0:10", "-i", "b.mp4:0:10", "--label", "Dupont"])

        assert [vi.label for vi in inputs] == ["Dupont", ""]

    def test_no_label_leaves_every_input_empty(self) -> None:
        inputs = _validated(["-i", "a.mp4:0:10", "-i", "b.mp4:0:10"])

        assert [vi.label for vi in inputs] == ["", ""]

    def test_more_labels_than_inputs_exits_non_zero(self, capsys) -> None:
        with pytest.raises(SystemExit) as exc:
            _validated(["-i", "a.mp4:0:10", "--label", "Dupont", "--label", "Martin"])

        assert exc.value.code != 0
        assert "2" in capsys.readouterr().err

    def test_several_labels_in_positional_mode_exits_non_zero(self, capsys) -> None:
        with pytest.raises(SystemExit) as exc:
            _validated(["video.mp4", "0", "10", "--label", "Dupont", "--label", "Martin"])

        assert exc.value.code != 0
        assert capsys.readouterr().err != ""

    def test_label_starting_with_a_dash_uses_the_attached_form(self) -> None:
        inputs = _validated(["-i", "a.mp4:0:10", "--label=-Dupont"])

        assert inputs[0].label == "-Dupont"

    def test_label_may_contain_colons_and_spaces(self) -> None:
        inputs = _validated(["-i", "a.mp4:0:10", "--label", "Run 2: final"])

        assert inputs[0].label == "Run 2: final"


class TestLabelBandPlanning:
    """The band is a property of the render, shared by every video."""

    def _plan_videos(self, monkeypatch, capsys, video_inputs) -> list[cli.VideoStream]:
        monkeypatch.setattr(cli, "analyze_video", lambda vi, detector: _analysis_for_plan())
        streams = cli.plan_videos(video_inputs, detector=object())
        capsys.readouterr()
        return streams

    def test_unlabeled_render_has_no_band(self, monkeypatch, capsys) -> None:
        streams = self._plan_videos(
            monkeypatch, capsys, [cli.VideoInput(path=Path("a.mp4"), start_frame=0, end_frame=10)]
        )

        plan = streams[0].plan
        assert plan.label_band_height == 0
        assert plan.frame_height == plan.output_height
        assert plan.final_size == (plan.output_width, plan.output_height)

    def test_one_label_gives_every_video_the_same_band_ratio(self, monkeypatch, capsys) -> None:
        streams = self._plan_videos(
            monkeypatch,
            capsys,
            [
                cli.VideoInput(path=Path("a.mp4"), start_frame=0, end_frame=10, label="Dupont"),
                cli.VideoInput(path=Path("b.mp4"), start_frame=0, end_frame=10),
            ],
        )

        ratios = [s.plan.label_band_height / s.plan.output_height for s in streams]
        assert all(h > 0 for h in (s.plan.label_band_height for s in streams))
        assert ratios[0] == pytest.approx(ratios[1])
        assert [s.plan.label for s in streams] == ["Dupont", ""]

    def test_band_matches_the_module_ratio(self, monkeypatch, capsys) -> None:
        streams = self._plan_videos(
            monkeypatch,
            capsys,
            [cli.VideoInput(path=Path("a.mp4"), start_frame=0, end_frame=10, label="Dupont")],
        )

        plan = streams[0].plan
        assert plan.label_band_height == band_height(plan.output_height, LABEL_BAND_RATIO)


class TestLabelBandRendering:
    """The band is appended to the image, and every size measurement sees it."""

    def _frame(self, plan: cli.CropPlan) -> np.ndarray:
        source = np.full((plan.scaled_height, plan.scaled_width, 3), 200, dtype=np.uint8)
        return cli._crop_one_frame(plan, source, plan.positions[0], None)

    def test_cropped_frame_is_taller_by_the_band_height(self) -> None:
        plain = self._frame(_plan(1, 30.0))
        labeled = self._frame(_plan(1, 30.0, label="Dupont", label_band_height=6))

        assert labeled.shape[0] == plain.shape[0] + 6
        assert labeled.shape[1] == plain.shape[1]

    def test_image_above_the_band_is_untouched(self) -> None:
        plan = _plan(1, 30.0, label="Dupont", label_band_height=6)

        labeled = self._frame(plan)

        assert np.array_equal(labeled[: plan.output_height], self._frame(_plan(1, 30.0)))

    def test_band_is_drawn_before_the_output_scaling(self) -> None:
        """The band keeps its proportion of the output height at any size."""
        plan = _plan(1, 30.0, label="Dupont", label_band_height=6)
        plan.target_height = plan.frame_height * 2

        scaled = self._frame(plan)

        assert scaled.shape[0] == plan.frame_height * 2

    def test_single_video_output_size_includes_the_band(self, capsys) -> None:
        plan = _plan(3, 30.0, label="Dupont", label_band_height=6)

        output = cli.build_output_stream([_stream(plan)])
        capsys.readouterr()

        assert output.height == plan.frame_height

    def test_composition_row_height_follows_the_frame_height(self, capsys) -> None:
        plans = [
            _plan(3, 30.0, label="Dupont", label_band_height=6),
            _plan(3, 30.0, label_band_height=6),
        ]

        output = cli.build_output_stream([_stream(p) for p in plans])
        capsys.readouterr()

        assert output.height == max(p.frame_height for p in plans)


class TestImgszOption:
    """The detection resolution is auto by default and overridable."""

    def test_absent_by_default(self) -> None:
        args = create_parser().parse_args(["video.mp4", "0", "10"])

        assert args.imgsz is None

    def test_parsed_value(self) -> None:
        args = create_parser().parse_args(["video.mp4", "0", "10", "--imgsz", "1920"])

        assert args.imgsz == 1920

    def test_below_the_minimum_is_rejected(self, capsys) -> None:
        assert cli.main(["video.mp4", "0", "10", "--imgsz", "320"]) == 1
        assert "--imgsz" in capsys.readouterr().err
