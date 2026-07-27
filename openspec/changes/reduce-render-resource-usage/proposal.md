# Change: Reduce render memory and time (stage 1)

## Why

Rendering 4K footage is both slow and memory-hungry. A single 300-frame 4K clip
holds ~7.5 GB of decoded frames in RAM, and multi-video mode multiplies that by
the number of inputs because `cli.py:633-658` analyzes every video before
cropping any of them. Detection runs on `yolov8m-pose` with no way to choose a
lighter model, and `output.py` writes every output frame to disk as PNG instead
of piping to FFmpeg.

This change collects the fixes that need no architectural rework. Removing
`frames_cache` entirely is deferred to a follow-up change.

## What Changes

- Add a `--model {n,s,m,l,x}` CLI flag selecting the YOLOv8-pose model size.
  **BREAKING** (behavior): the default becomes `s` instead of the current
  hard-coded `m`, trading some detection accuracy for a 2-3x faster detection
  pass. `--model m` restores the previous behavior.
- Expose the same choice in the GUI as a labelled render option, included in the
  displayed CLI command and persisted in project files.
- Process each video fully (analyze, then crop) before starting the next one in
  multi-video mode, instead of analyzing all videos first. The two-phase split
  is dead weight: the zoom target is the `TORSO_HEIGHT_RATIO` constant
  (`cli.py:640`) and no longer depends on the other videos' analyses.
- Release a video's decoded source frames as soon as its frames are cropped, so
  they are not held through composition and encoding.
- Pipe raw frames to FFmpeg over stdin instead of writing a numbered PNG
  sequence to a temporary directory, for both MP4 and GIF.
- Use `INTER_AREA` instead of `INTER_LANCZOS4` when downscaling frames.

## Impact

- Affected specs: `cli-interface`, `pose-detection`, `multi-video`,
  `output-format`, `gui-interface`
- Affected code: `src/railcam/cli.py`, `src/railcam/pose.py`,
  `src/railcam/output.py`, `src/railcam/composition.py`,
  `src/railcam/gui/project.py`, `src/railcam/gui/render.py`,
  `src/railcam/gui/render_panel.py`
- Peak memory in multi-video mode drops by roughly the number of inputs; peak
  memory for a single video is unchanged during detection (addressed by the
  follow-up streaming change) but drops before composition.
- Temporary disk usage for output generation drops to zero.
