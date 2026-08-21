# Change: Label under each video

## Why

Side-by-side comparisons are anonymous: nothing in the rendered file says which
climber is on the left and which is on the right. Users currently annotate the
output in a video editor afterwards, which defeats the point of a one-command
render.

## What Changes

- Add an optional text label per video, rendered in a solid band appended under
  that video's cropped image, horizontally centered. Default: empty, and an
  empty label changes nothing in the output.
- Add a repeatable `--label TEXT` CLI option, paired with the inputs in the
  order they are given.
- Expose the label as a per-video text field in the GUI, persisted in project
  files and reflected in the displayed CLI command.
- In multi-video mode, give every video a band of the same relative height as
  soon as one label is set (empty band for unlabeled videos), so height
  normalization keeps scaling all images by the same factor and zoom
  normalization is preserved.

## Impact

- Affected specs: `video-label` (new), `cli-interface`, `multi-video`,
  `gui-interface`
- Affected code: `src/railcam/labeling.py` (new), `src/railcam/multi_video.py`,
  `src/railcam/cli.py`, `src/railcam/gui/project.py`,
  `src/railcam/gui/player_widget.py`, `src/railcam/gui/render.py`
- **Output height changes when a label is used**: the crop keeps its 5:3 ratio,
  the band adds height on top of it, so the file is slightly taller than 5:3.
  Renders without labels are byte-identical to before.
