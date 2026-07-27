## Context

After stage 1, one video's `frames_cache` still holds every decoded source frame
(24.9 MB each at 4K), and `cropped_frames` accumulates the whole output on top.

The pipeline does not need either. Detection reads each frame once in order.
Cropping reads the same range again in the same order. Gap repair is the only
random access, and only for frames where the selected track has no detection.

Measured on a synthetic 4K clip (300 frames):

| Operation | Cost |
| --- | --- |
| Sequential decode | 19.6 ms/frame, ~5.9 s for the range |
| Scattered seek | 74 ms each, ~2.2 s for 30 frames |
| Detection, for scale | 100-500 ms/frame, 30-150 s for the range |

The clip is a `testsrc` pattern and compresses far better than camera footage,
so decode and seek figures are a floor. The ratio is what matters: re-decoding
costs a small fraction of detection.

## Goals / Non-Goals

- Goals: constant memory with respect to clip length; keep output byte-identical
  to the current pipeline; keep the geometry testable without frames.
- Non-Goals: crop geometry, smoothing, climber selection, GPU support, changing
  how many times detection runs.

## Decisions

### Decision: two sequential passes, not one

Cropping needs the smoothed position, smoothing needs every detection, so
cropping cannot happen during the detection pass. Two decode passes are a
consequence of that data dependency, not a design preference. The alternative —
keeping cropped frames from a single pass — is impossible for the same reason.

### Decision: split cropping into a plan and a generator

`crop_video` currently computes output dimensions, scale factor and smoothed
positions, then loops over frames doing the pixel work. The two halves are
separable: everything before the loop depends only on the analysis.

`CropPlan` holds that result. `crop_frames(plan, video_input)` re-decodes the
range and yields cropped frames. This is what makes streaming possible, and it
also makes the geometry testable without fabricating frames — today, asserting a
scale factor requires building a full analysis with a frame cache.

### Decision: the second pass verifies frame alignment

Re-decoding is the one place this change can go wrong silently. If pass 2 sees a
different frame range than pass 1 — a truncated file, a decoder that drops a
frame — positions and pixels desynchronize and the output is subtly mis-framed,
with nothing to indicate it.

Both passes go through the same `extract_frames(path, start, end)`, and
`crop_frames` checks that each frame number it receives matches the position it
is about to apply, raising `VideoError` otherwise. A loud failure is worth more
than a plausible-looking wrong crop.

### Decision: FrameCursor relies on non-decreasing sync indices

`time_sync_frame_indices` already exists and is pure: it maps an output frame
index to a source frame index, repeating indices when the source is slower than
the output or has ended (freeze). It never goes backwards.

`FrameCursor` wraps a crop generator and exposes `advance_to(index)`, pulling
frames forward and repeating the current one when the index does not move. This
is only correct while the indices are non-decreasing, so that property is
asserted in `FrameCursor` and covered by a test: a future change to the sync
model that broke it would otherwise produce a corrupt stream that is very hard
to trace back.

### Decision: FrameSource moves to the core, GUI re-exports

Gap repair needs exactly what `gui/frame_source.py` already provides: access by
frame index, an LRU, and a sequential window that decodes forward rather than
seeking on long-GOP codecs. Duplicating it would leave two implementations of a
subtle decoder behaviour.

It moves to `railcam/frame_source.py`. `railcam/gui/frame_source.py` re-exports
it so GUI imports and their tests are untouched. The core module must not import
anything from `railcam.gui`; the dependency runs the other way.

### Decision: list-based APIs stay, built on the streaming ones

`compose_frames_horizontal` and `generate_output` keep their current signatures
and are expressed in terms of `compose_frame_row` and the streaming encoder.
This is not compatibility for its own sake: their existing tests become the
evidence that the streamed path produces the same result as the list path.

### Decision: durations keep using the requested frame count

Multi-video duration was computed from the requested range while the
synchronization map was built from the frames actually decoded. The two differ
when a range runs past the end of a file, which `validate_frame_range` allows:
it rejects `end > total_frames`, but frame numbers are 0-indexed, so
`end == total_frames` is already one past the last frame.

Making both use the decoded count changed rendered output by one frame. Since
byte-identical output is this change's acceptance criterion, both counts are
kept distinct and the inconsistency preserved. The off-by-one in
`validate_frame_range` is real and worth fixing, but changing rendered durations
belongs in its own change, not in a memory refactor.

## Risks / Trade-offs

- Renders get slower by one sequential decode per video → measured at a small
  fraction of detection time; the memory win is the point.
- A desynchronized second pass would mis-frame silently → guarded by the frame
  number check above.
- Long-GOP sources make repair seeks expensive if many frames are missing →
  `FrameSource` already decodes forward within a 25-frame window rather than
  seeking, and repair only runs on frames the selected track is missing.
- `frames_cache` disappearing changes `VideoAnalysisResult` → it is internal to
  `cli.py`, with no consumer outside it.

## Migration Plan

No data migration, no user-visible interface change. Output must be unchanged;
that is the acceptance criterion, verified by keeping the list-based tests green
and by comparing a rendered clip before and after.

## Open Questions

None blocking.
