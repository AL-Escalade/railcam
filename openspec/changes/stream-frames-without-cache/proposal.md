# Change: Stream frames instead of caching them (stage 2)

## Why

Stage 1 (`reduce-render-resource-usage`) removed the multi-video multiplier but
left the core cost untouched: a single video still holds every decoded source
frame in `VideoAnalysisResult.frames_cache`, about 7.5 GB for 300 frames of 4K.
Cropped output frames then accumulate on top of it.

Nothing needs those frames to be resident. Detection consumes each frame once,
and cropping re-reads them in the same order. The only random access is gap
repair, which touches a few dozen frames at most.

## What Changes

- Detection streams frames and discards them; `VideoAnalysisResult.frames_cache`
  is removed.
- Gap repair reads the frames it needs through `FrameSource`, which moves from
  `railcam/gui/frame_source.py` to `railcam/frame_source.py` so the core and the
  GUI share one implementation.
- Cropping splits into a pure `CropPlan` (output size, scale factor, smoothed
  positions — no pixels) and a `crop_frames` generator that re-decodes the range
  and yields cropped frames one at a time.
- Composition consumes those generators through a `FrameCursor` per video and
  emits one composed frame at a time, instead of building synchronized lists.
- Encoding accepts a stream of frames with explicit dimensions and count.
- `compose_frames_horizontal` and `generate_output` keep their list-based
  signatures, expressed in terms of the streaming primitives, so their existing
  tests keep proving the streamed path produces the same output.

## Impact

- Affected specs: `pose-detection`, `video-cropping`, `multi-video`,
  `output-format`
- Affected code: `src/railcam/cli.py`, `src/railcam/composition.py`,
  `src/railcam/output.py`, `src/railcam/frame_source.py` (moved),
  `src/railcam/gui/frame_source.py` (re-export)
- Peak memory becomes one source frame plus one cropped frame per video, plus
  one composed frame: roughly 50 MB at 4K regardless of clip length, against
  7.5 GB before.
- Each video is decoded twice instead of once. Measured on a synthetic 4K clip:
  19.6 ms per frame sequential, so about 6 s per 300 frames, against 30-150 s
  for the detection pass on the same range.
