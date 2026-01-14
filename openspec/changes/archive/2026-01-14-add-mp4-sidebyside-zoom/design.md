## Context

The railcam CLI currently generates single-video GIFs with pelvis-centered cropping. This change adds three major capabilities:
1. MP4 output (better quality, smaller files)
2. Side-by-side multi-video composition
3. Normalized zoom based on torso height

These features span multiple modules and require architectural decisions for how they interact.

## Goals / Non-Goals

**Goals:**
- Support MP4 as output format with H.264 encoding
- Compose 2+ videos horizontally with independent frame ranges
- Normalize zoom so average torso height = configurable ratio of output height (default 1/3)
- Maintain backward compatibility for single-video GIF workflow

**Non-Goals:**
- Vertical stacking (only horizontal side-by-side)
- Audio support
- Different zoom levels per video (all use the same normalized scale)
- Real-time preview

## Decisions

### 1. Torso Height Calculation

**Decision:** Use the vertical distance between the midpoint of shoulders and the midpoint of hips.

**Landmarks used (YOLOv8-pose COCO format):**
- Left shoulder (index 5), Right shoulder (index 6) → shoulder midpoint
- Left hip (index 11), Right hip (index 12) → hip midpoint (pelvis)

**Formula:**
```
torso_height = abs(hip_midpoint_y - shoulder_midpoint_y)
```

**Alternatives considered:**
- Bounding box: Less accurate, affected by arm positions
- Neck to hips: Neck landmark less reliable in climbing poses

### 2. Zoom Normalization Strategy

**Decision:** Two-pass approach:
1. **Pass 1:** Detect poses on all frames of all videos, compute average torso height per video
2. **Pass 2:** Calculate zoom factor per video such that `average_torso_height * zoom = target_height`

**Target ratio constant:**
```python
TORSO_HEIGHT_RATIO = 1/3  # Torso should be 1/3 of output height
```

**Zoom factor calculation:**
```python
target_torso_pixels = output_height * TORSO_HEIGHT_RATIO
avg_torso_pixels = avg_torso_normalized * video_height
zoom_factor = target_torso_pixels / avg_torso_pixels
```

**Constraints:**
- Minimum zoom: 1.0 (never zoom out beyond original)
- Maximum zoom: 3.0 (avoid extreme pixelation)

### 3. Multi-Video Frame Synchronization

**Decision:** All videos are stretched/compressed to match the same frame count as the longest video.

**Algorithm:**
1. Determine max frame count across all inputs
2. For each video, resample frames to match max count using linear interpolation
3. Compose frames side-by-side at each index

**Example:**
- Video A: 100 frames (100-199)
- Video B: 150 frames (50-199)
- Output: 150 frames, Video A resampled to 150 frames

### 4. Output Format Selection

**Decision:** Use `--format` flag with `mp4` as default.

**FFmpeg encoding settings for MP4:**
```bash
ffmpeg -framerate {fps} -i frame_%05d.png \
  -c:v libx264 -preset medium -crf 23 \
  -pix_fmt yuv420p output.mp4
```

**GIF generation:** Unchanged (palette optimization).

### 5. CLI Input Format

**Decision:** Multiple `--input` arguments with `path:start:end` syntax.

**Examples:**
```bash
# Single video (backward compatible)
railcam video.mp4 100 250 --format mp4

# Side-by-side comparison
railcam --input climber1.mp4:100:250 --input climber2.mp4:50:200 --format mp4
```

When `--input` is used, positional arguments are not allowed.

### 6. Aspect Ratio Change

**Decision:** Change from 4:3 to 5:3 vertical ratio.

**Constants update:**
```python
# Old
ASPECT_WIDTH = 3
ASPECT_HEIGHT = 4

# New
ASPECT_WIDTH = 3
ASPECT_HEIGHT = 5
```

**Rationale:** 5:3 provides more vertical space for the climber, better suited for speed climbing's vertical movement.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         CLI Layer                            │
│  - Parse --input arguments                                   │
│  - Validate inputs                                           │
│  - Select output format                                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Multi-Video Coordinator                   │
│  - For each video: pose detection + torso measurement       │
│  - Calculate normalized zoom factors                         │
│  - Resample frames to match counts                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Frame Processor (per video)               │
│  - Apply zoom factor to crop dimensions                      │
│  - Center on pelvis with zoom                                │
│  - Handle boundaries                                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Composition Layer                         │
│  - Resize all frames to same height                          │
│  - Concatenate horizontally                                  │
│  - Ensure even dimensions                                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Output Layer                              │
│  - GIF: palette generation + dithering                       │
│  - MP4: H.264 encoding                                       │
└─────────────────────────────────────────────────────────────┘
```

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Pose detection slower with torso measurement | Minimal overhead - same inference, different landmark extraction |
| Frame resampling quality | Use OpenCV's optical flow or simple frame duplication/dropping |
| Zoom pixelation | Enforce max zoom factor (3.0x) |
| Memory usage with multiple videos | Process one video at a time, store only positions and zoom factors |

## Migration Plan

1. Update `cropping.py` constants (aspect ratio)
2. Extend `pose.py` to extract shoulder landmarks
3. Add torso measurement and zoom calculation
4. Create `composition.py` for multi-video assembly
5. Rename `gif.py` to `output.py`, add MP4 support
6. Update CLI with new arguments
7. Update tests

**Rollback:** If issues arise, the old single-video GIF mode remains functional.

## Open Questions

- Should frame resampling use frame duplication or interpolation? (Recommended: duplication for simplicity)
- Should we add a `--zoom` override to disable normalization? (Deferred to future change)
