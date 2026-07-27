## 1. Shared frame source

- [x] 1.1 Move `FrameSource` from `railcam/gui/frame_source.py` to `railcam/frame_source.py`
- [x] 1.2 Re-export it from `railcam/gui/frame_source.py` so GUI imports and their tests are untouched
- [x] 1.3 Test: existing `tests/test_gui_frame_source.py` still passes; the core module imports nothing from `railcam.gui`

## 2. Streaming encoder

- [x] 2.1 Add `generate_output_stream(frames, output_path, fps, width, height, total_frames, output_format, on_progress)` taking an iterable
- [x] 2.2 Rework `_encode_frames` to consume an iterator with explicit dimensions and count instead of reading them off a list
- [x] 2.3 Raise `OutputGenerationError` when the stream yields fewer frames than `total_frames`
- [x] 2.4 Express `generate_output` in terms of the streaming path, keeping its list signature
- [x] 2.5 Test: streaming and list encoding produce identical commands and piped bytes; a short stream raises; progress is reported per frame

## 3. Crop plan and crop stream

- [x] 3.1 Extract `CropPlan` (output dimensions, scale factor, needs_padding, positions) and `build_crop_plan(analysis, target_torso_ratio, target_width, target_height)` from `crop_video`
- [x] 3.2 Add `crop_frames(plan, video_input, debug, detections_by_frame)` yielding cropped frames from a fresh `extract_frames` pass
- [x] 3.3 Raise `VideoError` when the frame number read does not match the planned position's frame number
- [x] 3.4 Remove `crop_video`: `build_crop_plan` plus `crop_frames` cover every caller, and keeping a list-building wrapper nothing used would have been dead code
- [x] 3.5 Test: plan geometry without any frame (scale factor, output size, padding flag, one position per frame); cropped frames come out in order at the plan's size; a mismatched frame number raises

## 4. Analysis without frame retention

- [x] 4.1 Drop `frames_cache` from `VideoAnalysisResult`
- [x] 4.2 Stream frames in `_detect_poses_with_tracking`, releasing each after detection
- [x] 4.3 Repair gaps through a `FrameSource` opened only when frames are missing
- [x] 4.4 Test: analysis holds no frames; repair reads only the missing frame numbers; no repair opens no source

## 5. Streamed composition

- [x] 5.1 Add `compose_frame_row(frames, target_height)` composing one output frame
- [x] 5.2 Express `compose_frames_horizontal` in terms of it
- [x] 5.3 Add `FrameCursor` wrapping a crop generator, with `advance_to(index)` repeating the current frame when the index does not move and raising on a backwards index
- [x] 5.4 Drive multi-video output by output frame index, pulling each video through its cursor using `time_sync_frame_indices`
- [x] 5.5 Test: cursor repeats without advancing, raises going backwards, freezes past the end; streamed composition matches the list-based result frame for frame

## 6. Wire the pipeline

- [x] 6.1 Rework `process_videos` to return plans and stream factories rather than materialized frames
- [x] 6.2 Drive single-video output straight from `crop_frames` into the streaming encoder
- [x] 6.3 Drive multi-video output through the cursors and `compose_frame_row`
- [x] 6.4 Cropping no longer has a progress stage of its own: it runs as FFmpeg pulls
  frames, so two interleaved bars would break the GUI's stage aggregator. The
  encoding bar covers it, and expected_stage_count drops from 2n+2 to n+2
- [x] 6.5 Test: the GUI progress parser still recognises every stage the CLI emits

## 7. Documentation and validation

- [x] 7.1 Update `CLAUDE.md`: describe the two-pass streaming pipeline and drop the frame cache description
- [x] 7.2 Run `ruff check src tests`, `ruff format src tests`, `mypy src`, `pytest`
- [x] 7.3 Render a clip before and after the change and confirm the output files are identical
- [x] 7.4 Measured on a synthetic 4K clip with a deterministic detector stand-in:
  one video 300 frames, 17443 MB -> 2001 MB peak RSS, 25.8 s -> 29.4 s;
  two videos 200 frames each, 18974 MB -> 3244 MB, 28.0 s -> 28.2 s.
  Interpreter baseline with torch loaded is 242 MB. Numbers recorded in proposal.md
