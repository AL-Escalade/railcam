"""Tests for cli argument parsing and wiring."""

from pathlib import Path

import pytest

from railcam import cli
from railcam.cli import create_parser


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


def _analysis(marker: str) -> cli.VideoAnalysisResult:
    """A minimal analysis whose frame cache carries an identifying marker."""
    return cli.VideoAnalysisResult(
        detections=[],
        detections_by_frame={},
        frames_cache={0: marker},
        avg_torso_height=0.1,
        fps=30.0,
        video_width=100,
        video_height=100,
        frame_count=1,
    )


class TestSequentialProcessing:
    """Each video is fully processed before the next one is analyzed."""

    def _patch(self, monkeypatch):
        calls: list[tuple[str, str]] = []
        analyses: list[cli.VideoAnalysisResult] = []

        def fake_analyze(video_input, detector):
            calls.append(("analyze", video_input.path.stem))
            analysis = _analysis(video_input.path.stem)
            analyses.append(analysis)
            return analysis

        def fake_crop(analysis, target_torso, **kwargs):
            calls.append(("crop", analysis.frames_cache[0]))
            return cli.VideoProcessingResult(
                cropped_frames=[], fps=analysis.fps, zoom_factor=1.0, frame_count=1
            )

        monkeypatch.setattr(cli, "analyze_video", fake_analyze)
        monkeypatch.setattr(cli, "crop_video", fake_crop)
        return calls, analyses

    def test_videos_are_interleaved_analyze_then_crop(self, monkeypatch, capsys):
        calls, _ = self._patch(monkeypatch)
        inputs = [
            cli.VideoInput(path=Path("a.mp4"), start_frame=0, end_frame=10),
            cli.VideoInput(path=Path("b.mp4"), start_frame=0, end_frame=10),
        ]

        cli.process_videos(inputs, detector=object(), debug=False)
        capsys.readouterr()

        assert calls == [("analyze", "a"), ("crop", "a"), ("analyze", "b"), ("crop", "b")]

    def test_source_frames_released_after_each_video(self, monkeypatch, capsys):
        _, analyses = self._patch(monkeypatch)
        inputs = [
            cli.VideoInput(path=Path("a.mp4"), start_frame=0, end_frame=10),
            cli.VideoInput(path=Path("b.mp4"), start_frame=0, end_frame=10),
        ]

        cli.process_videos(inputs, detector=object(), debug=False)
        capsys.readouterr()

        assert [a.frames_cache for a in analyses] == [{}, {}]

    def test_returns_one_result_per_video(self, monkeypatch, capsys):
        self._patch(monkeypatch)
        inputs = [
            cli.VideoInput(path=Path("a.mp4"), start_frame=0, end_frame=10),
            cli.VideoInput(path=Path("b.mp4"), start_frame=0, end_frame=10),
        ]

        results = cli.process_videos(inputs, detector=object(), debug=False)
        capsys.readouterr()

        assert len(results) == 2
