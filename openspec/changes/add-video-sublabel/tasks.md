## 1. A band of rows

- [x] 1.1 Add `LabelLine(text, height)` to `labeling.py` and make `append_label_band(frame, lines)` draw each row in order, centering its text on its own row
- [x] 1.2 Add `SUBLABEL_BAND_RATIO`, and keep the font of a row derived from that row's height so a smaller row means a smaller font
- [x] 1.3 Test: two rows stack in order, the second row's text is smaller than the first's, a row with empty text stays uniform, no rows leaves the frame untouched

## 2. Plumbing

- [x] 2.1 Add `sublabel: str = ""` to `VideoInput`
- [x] 2.2 Replace `CropPlan.label`/`label_band_height` with the label lines, and sum them in `frame_height`
- [x] 2.3 In `plan_videos`, give every video the same row layout: the first row as soon as any input has a label or a sublabel, the second row as soon as any input has a sublabel
- [x] 2.4 Test: no sublabel anywhere leaves the geometry as it is today; one sublabel gives every plan two rows; the frame grows by the sum of the rows

## 3. CLI

- [x] 3.1 Add a repeatable `--sublabel TEXT` argument, paired with the inputs like `--label`, with the same error paths
- [x] 3.2 Document it in the parser epilog
- [x] 3.3 Test: pairing, empty when absent, too many sublabels exits non-zero, several in positional mode exits non-zero

## 4. GUI

- [x] 4.1 Add `sublabel: str = ""` to `VideoEntry`, saved and loaded (absent field defaults to empty)
- [x] 4.2 Add the second text field to the player card, wired to `stateChanged`, read and restored with the entry
- [x] 4.3 Emit `--sublabel=TEXT` per video when at least one is set, in input order
- [x] 4.4 Test: project round-trip, old project files load, argument emission

## 5. Documentation and validation

- [x] 5.1 Document `--sublabel` and the GUI field in `README.md`
- [x] 5.2 Run `ruff check src tests`, `ruff format src tests`, `mypy src`, `pytest`
