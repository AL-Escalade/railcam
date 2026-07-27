## Context

Rendering a 4K clip is slow and holds several gigabytes of RAM. Measured cost
drivers, in order:

| Source | Cost |
| --- | --- |
| `frames_cache` (`cli.py:325`) | 24.9 MB per 4K frame; 2.5-7.5 GB for 100-300 frames |
| All videos analyzed before any is cropped (`cli.py:633-658`) | multiplies the above by the number of inputs |
| `cropped_frames` + `composed_frames` | output-sized frames, held simultaneously |
| `yolov8m-pose` inference | 200-500 ms per frame on CPU |
| PNG sequence in a temp dir (`output.py:46`) | full re-encode of every output frame to disk |

This change takes the fixes that do not require restructuring the pipeline. A
follow-up change replaces `frames_cache` with streaming decode.

## Goals / Non-Goals

- Goals: cut peak RAM in multi-video mode, cut detection time, remove temporary
  disk usage, keep the output visually equivalent at the same `--model`.
- Non-Goals: eliminating `frames_cache` for a single video; changing the
  tracking, repair, smoothing or cropping algorithms; GPU support.

## Decisions

### Decision: `--model` defaults to `s`

`PoseDetector.__init__` already takes `model_size` (`pose.py:119`) but `cli.py:630`
calls `PoseDetector()` with no arguments, so `m` is effectively hard-coded.

The default moves to `s`. Rationale: the pipeline already compensates for missed
detections — `repair_track_gaps` re-runs inference at `HIGH_RES_IMGSZ` (1280) on
exactly the frames where the selected track has a hole. A faster model with
targeted high-resolution repair is a better time/accuracy trade than a heavy
model applied uniformly.

Alternatives considered:
- Keep `m` as the default, `--model` opt-in only. Rejected: nobody gets the
  speedup without knowing the flag exists.
- Default to `n`. Rejected: too many gaps to repair; the repair pass at 1280px
  is slower per frame than the detection pass, so a bad first pass can end up
  slower overall.
- Auto-escalate from `n` to `m` when the selected track has too many gaps.
  Rejected for now: unpredictable render time and more code, for a gain that
  `--model` already delivers explicitly. Reconsider once we have measurements.

### Decision: the GUI exposes all five model sizes, labelled

The GUI runs the CLI as a subprocess and builds its arguments from
`RenderOptions` (`render.py:44`), so the option only needs to exist as a field
there to reach the renderer. It is added to `RenderOptions`, to the project file,
and to the render options row as a combo box.

The combo lists all five sizes with labels naming both the trade-off and the CLI
value — `Rapide (n)`, `Équilibré (s)`, `Précis (m)`, `Très précis (l)`,
`Maximal (x)`. Two reasons for all five rather than a curated subset: the panel
also displays the equivalent CLI command, so a project created with `--model x`
must be representable without a special case; and the flag value in parentheses
keeps the copied command readable against the label the user picked.

`PROJECT_VERSION` stays at 1. `_load_render_options` reads every field through
`data.get(key, default)`, so a project written before this change loads with the
new default. The value is validated against the accepted sizes on load, matching
how `climber` is validated in `_load_video_entry`.

### Decision: one video at a time in multi-video mode

`cli.py:640` sets `reference_torso = TORSO_HEIGHT_RATIO`, a constant. Nothing in
phase 2 reads another video's analysis; the loop at `cli.py:646-649` only prints
padding warnings. The phase split is a leftover from when the zoom target was
derived across inputs, so `analyze` and `crop` can be fused into one loop body.

The padding warnings move inside that loop, printed per video as it is analyzed.

### Decision: `frames_cache` is cleared by the caller, not by `crop_video`

`crop_video` receives a `VideoAnalysisResult` it does not own. Clearing the
cache inside it would make the function destructive in a way its signature does
not advertise, and would break any caller that crops the same analysis twice
(the GUI could plausibly want this for a preview). The calling loop in `main`
clears the cache after `crop_video` returns.

### Decision: raw frames over stdin, one FFmpeg invocation

Frames are BGR `uint8` numpy arrays, so they map directly onto FFmpeg's
`rawvideo` input with `-pix_fmt bgr24`. The input dimensions come from the first
frame; all frames are already the same size by the time they reach the encoder.

For GIF, the current two-command palette flow (`palettegen` to a PNG, then
`paletteuse`) requires reading the input twice, which a pipe cannot do. It is
replaced by the single-command form, which keeps `stats_mode=diff`:

```
-filter_complex "[0:v]split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"
```

`generate_output`, `generate_mp4` and `generate_gif` keep their current
signatures (a `list[np.ndarray]`) so this change stays self-contained. Accepting
an iterator is part of the follow-up streaming change.

### Decision: `INTER_AREA` for downscaling only

OpenCV recommends `INTER_AREA` for decimation; it also suppresses aliasing
better than Lanczos when the scale factor is well below 1, which is the common
case here (4K source, ~1080p output). Upscaling keeps `INTER_LINEAR`.
`composition.py:290` always resizes to a common height and can scale either way,
so it selects the interpolation from the computed scale rather than hard-coding
one.

## Risks / Trade-offs

- Detection quality regresses for users who relied on the `m` default →
  documented as breaking in the proposal and in `--help`; `--model m` restores it.
- Piping to FFmpeg loses the ability to inspect intermediate PNGs when
  debugging an encoding problem → the constructed FFmpeg command is logged, and
  stderr is captured and surfaced on failure as it is today.
- A broken pipe (FFmpeg exiting early) must not deadlock the writer → the
  writer checks the process exit status and raises `OutputGenerationError` with
  FFmpeg's stderr instead of blocking on a full pipe.
- `INTER_AREA` produces slightly softer output than Lanczos on some footage →
  visual difference is minor at these scale factors; revisit if it shows.

## Migration Plan

No data migration. The `--model` default change is the only user-visible
behavior change, and is reversible with a flag. No spec capability is removed.

## Open Questions

None blocking. Whether `n` becomes a better default is a measurement question to
revisit after this change ships and real render times are known.
