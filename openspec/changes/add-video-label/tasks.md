## 1. Label rendering

- [x] 1.1 Add `src/railcam/labeling.py` with `LABEL_BAND_RATIO`, `band_height(image_height)` (even, 0 when the ratio is 0) and `append_label_band(frame, text, band_height)`
- [x] 1.2 Derive the font scale from the band height, shrinking it when the text exceeds the usable width; center the text horizontally and vertically in the band
- [x] 1.3 Test: band height is even and proportional; an empty label yields a uniform band; a long label stays inside the frame width; a zero band height returns the frame unchanged

## 2. Plumbing through the pipeline

- [x] 2.1 Add `label: str = ""` to `VideoInput`
- [x] 2.2 Add `label` and `label_band_height` to `CropPlan`, plus a `frame_height` property, and use it in `final_size`
- [x] 2.3 Accept the band height in `build_crop_plan` and draw the band in `_crop_one_frame` after the debug overlay, before output scaling
- [x] 2.4 In `plan_videos`, use `LABEL_BAND_RATIO` for every video as soon as one input has a non-empty label, and zero otherwise
- [x] 2.5 Use `frame_height` for the composition row height and the size probes in `build_output_stream`
- [x] 2.6 Test: a plan without labels has a zero band and unchanged geometry; one label in a multi-video render gives every plan the same relative band; cropped frames are taller by the band height

## 3. CLI

- [x] 3.1 Add a repeatable `--label TEXT` argument, documented in the epilog examples
- [x] 3.2 Pair labels with inputs in order in `validate_args`; error when more labels than inputs are given, or when more than one is given in positional mode
- [x] 3.3 Test: labels reach the matching `VideoInput`; a missing label is empty; too many labels exit non-zero

## 4. GUI

- [x] 4.1 Add `label: str = ""` to `VideoEntry`, saved and loaded (absent field defaults to empty)
- [x] 4.2 Add a label text field to the player card, wired to `stateChanged`, read by `to_video_entry` and restored by `apply_entry`
- [x] 4.3 Emit `--label` for every video when at least one is labeled, in input order; quote empty arguments in the displayed command
- [x] 4.4 Test: project round-trip preserves labels; a project written before this option loads with empty labels; `build_cli_args` omits `--label` entirely when no video is labeled and emits one per video otherwise

## 6. Unicode label text

- [x] 6.1 Bundle `DejaVuSans-Bold.ttf` (and its license) under `src/railcam/fonts/`, and add `pillow` to the runtime dependencies
- [x] 6.2 Draw the label with Pillow through a coverage mask instead of `cv2.putText`, dropping the ASCII transliteration
- [x] 6.3 Test: accents and non-Latin scripts are drawn rather than folded; the font ships inside the package

## 5. Documentation and validation

- [x] 5.1 Document `--label` in `README.md` and the GUI field in its usage section
- [x] 5.2 Note the label band in the `CLAUDE.md` pipeline description
- [x] 5.3 Run `ruff check src tests`, `ruff format src tests`, `mypy src`, `pytest`
