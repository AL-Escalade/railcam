# Change: Second label line under each video

## Why

One label line has to carry both the athlete's name and their time, so users
cram `Zhao 4.704` into a single line sized for a name. The two are read
differently — the name identifies the lane at a glance, the time is a detail —
and they deserve different weight.

## What Changes

- Add an optional second label line per video, drawn under the first one in a
  smaller font, in the same band.
- Add a repeatable `--sublabel TEXT` CLI option, paired with the inputs in the
  order they are given, exactly like `--label`.
- Expose it as a second per-video field in the GUI, persisted in project files
  and reflected in the displayed CLI command.
- Generalize the band to a stack of lines: each line carries its own text and
  size, and the band height follows from those sizes plus an explicit padding,
  the same above the first line as below the last. The second line is 2.1% of
  the image height against 4% for the first, so it reads as clearly secondary.

## Impact

- Affected specs: `video-label`, `cli-interface`, `gui-interface`
- Affected code: `src/railcam/labeling.py`, `src/railcam/multi_video.py`,
  `src/railcam/cli.py`, `src/railcam/gui/project.py`,
  `src/railcam/gui/player_widget.py`, `src/railcam/gui/render.py`
- **The band grows when a second line is used**, so a labeled output is taller
  again. Renders that use no sublabel keep exactly today's geometry, and
  renders with no label at all stay byte-identical.
