## 1. Shared frame source

- [ ] 1.1 Move `FrameSource` from `railcam/gui/frame_source.py` to `railcam/frame_source.py`
- [ ] 1.2 Re-export it from `railcam/gui/frame_source.py` so GUI imports and their tests are untouched
- [ ] 1.3 Test: existing `tests/test_gui_frame_source.py` still passes; the core module imports nothing from `railcam.gui`

## 2. Streaming encoder

- [ ] 2.1 Add `generate_output_stream(frames, output_path, fps, width, height, total_frames, output_format, on_progress)` taking an iterable
- [ ] 2.2 Rework `_encode_frames` to consume an iterator with explicit dimensions and count instead of reading them off a list
- [ ] 2.3 Raise `OutputGenerationError` when the stream yields fewer frames than `total_frames`
- [ ] 2.4 Express `generate_output` in terms of the streaming path, keeping its list signature
- [ ] 2.5 Test: streaming and list encoding produce identical commands and piped bytes; a short stream raises; progress is reported per frame

## 3. Crop plan and crop stream

- [ ] 3.1 Extract `CropPlan` (output dimensions, scale factor, needs_padding, positions) and `build_crop_plan(analysis, target_torso_ratio, target_width, target_height)` from `crop_video`
- [ ] 3.2 Add `crop_frames(plan, video_input, debug, detections_by_frame)` yielding cropped frames from a fresh `extract_frames` pass
- [ ] 3.3 Raise `VideoError` when the frame number read does not match the planned position's frame number
- [ ] 3.4 Express the existing `crop_video` behaviour through the plan and the generator
- [ ] 3.5 Test: plan geometry without any frame (scale factor, output size, padding flag, one position per frame); cropped frames come out in order at the plan's size; a mismatched frame number raises

## 4. Analysis without frame retention

- [ ] 4.1 Drop `frames_cache` from `VideoAnalysisResult`
- [ ] 4.2 Stream frames in `_detect_poses_with_tracking`, releasing each after detection
- [ ] 4.3 Repair gaps through a `FrameSource` opened only when frames are missing
- [ ] 4.4 Test: analysis holds no frames; repair reads only the missing frame numbers; no repair opens no source

## 5. Streamed composition

- [ ] 5.1 Add `compose_frame_row(frames, target_height)` composing one output frame
- [ ] 5.2 Express `compose_frames_horizontal` in terms of it
- [ ] 5.3 Add `FrameCursor` wrapping a crop generator, with `advance_to(index)` repeating the current frame when the index does not move and raising on a backwards index
- [ ] 5.4 Drive multi-video output by output frame index, pulling each video through its cursor using `time_sync_frame_indices`
- [ ] 5.5 Test: cursor repeats without advancing, raises going backwards, freezes past the end; streamed composition matches the list-based result frame for frame

## 6. Wire the pipeline

- [ ] 6.1 Rework `process_videos` to return plans and stream factories rather than materialized frames
- [ ] 6.2 Drive single-video output straight from `crop_frames` into the streaming encoder
- [ ] 6.3 Drive multi-video output through the cursors and `compose_frame_row`
- [ ] 6.4 Keep the progress stages reported by the CLI unchanged, so the GUI progress parser keeps working
- [ ] 6.5 Test: the GUI progress parser still recognises every stage the CLI emits

## 7. Documentation and validation

- [ ] 7.1 Update `CLAUDE.md`: describe the two-pass streaming pipeline and drop the frame cache description
- [ ] 7.2 Run `ruff check src tests`, `ruff format src tests`, `mypy src`, `pytest`
- [ ] 7.3 Render a clip before and after the change and confirm the output files are identical
- [ ] 7.4 Measure peak RSS and wall time on a 4K clip, and record the numbers here before archiving
