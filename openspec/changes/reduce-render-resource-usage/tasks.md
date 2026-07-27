## 1. Model selection (CLI)

- [x] 1.1 Change the `PoseDetector` default `model_size` from `m` to `s` (`pose.py:119`)
- [x] 1.2 Add `--model {n,s,m,l,x}` to the argument parser, defaulting to `s`, with help text noting the speed/accuracy trade-off
- [x] 1.3 Pass the parsed value to `PoseDetector(...)` at `cli.py:630`
- [x] 1.4 Test: `--model n` is accepted and reaches the detector; an invalid value exits non-zero

## 2. Model selection (GUI)

- [x] 2.1 Add `model: str = "s"` to `RenderOptions` and to `Project.save` (`project.py:34`, `project.py:65`)
- [x] 2.2 Read it in `_load_render_options` via `data.get("model", defaults.model)`, validating against the accepted sizes and raising `ProjectError` otherwise
- [x] 2.3 Emit `--model` from `build_cli_args` only when it differs from the default (`render.py:44`)
- [x] 2.4 Add the model combo box to the render options row, with the five labelled entries, wired to `optionsChanged` (`render_panel.py:110`)
- [x] 2.5 Read it in `render_options()` and restore it in `apply_options()` (`render_panel.py:208`, `render_panel.py:226`)
- [x] 2.6 Test: round-trip save/load preserves the model; a project file with no model field loads with the default; `build_cli_args` omits the flag at the default and emits it otherwise

## 3. Sequential per-video processing

- [x] 3.1 Replace the analyze-all-then-crop-all loop in `main` with one loop that analyzes and crops each video in turn (`cli.py:631-658`)
- [x] 3.2 Move the per-video padding warning into that loop
- [x] 3.3 Clear the analysis frame cache in the loop after `crop_video` returns
- [x] 3.4 Test: analysis and cropping interleave per video, and each video's frame cache is empty once it is cropped. Cropped output is unchanged by construction — `crop_video` receives the same `(analysis, target_torso)` arguments under either ordering — but comparing rendered output before and after needs real footage, so it is covered by 6.4 rather than by a unit test

## 4. Direct FFmpeg encoding

- [x] 4.1 Replace `_write_frames_to_temp` with a helper that starts FFmpeg reading `rawvideo`/`bgr24` from stdin and writes each frame's bytes to it (`output.py:46`)
- [x] 4.2 Rework `generate_mp4` to use it, deriving `-s WxH` from the first frame
- [x] 4.3 Rework `generate_gif` to the single-invocation `split`/`palettegen`/`paletteuse` filter chain, preserving `stats_mode=diff` and the current dither settings
- [x] 4.4 Raise `OutputGenerationError` with FFmpeg's stderr if the process exits early, without blocking on the pipe
- [x] 4.5 Keep the `on_progress` callback firing per frame written
- [x] 4.6 Test: the constructed FFmpeg command for each format; an empty frame list raises before any process starts; a failing FFmpeg surfaces its stderr

## 5. Downscale interpolation

- [x] 5.1 Use `INTER_AREA` when the scale factor is below 1 in `crop_video` (`cli.py:492`)
- [x] 5.2 Select interpolation from the computed scale in `normalize_frame_height` (`composition.py:290`)
- [x] 5.3 Test: `normalize_frame_height` picks `INTER_AREA` when shrinking and `INTER_LINEAR` when growing

## 6. Documentation and validation

- [x] 6.1 Document `--model` in `README.md` and the GUI option in its usage section
- [x] 6.2 Update `CLAUDE.md`: the output module now pipes frames to FFmpeg (it currently claims this while the code writes PNGs)
- [x] 6.3 Run `ruff check src tests`, `ruff format src tests`, `mypy src`, `pytest` — all clean (mypy was unblocked separately by #6, now enforced in CI)
- [ ] 6.4 Measure peak RSS and wall time on a 4K clip before and after, and record the numbers in the change before archiving. **Needs real footage — not done; no 4K clip is available in this environment**
