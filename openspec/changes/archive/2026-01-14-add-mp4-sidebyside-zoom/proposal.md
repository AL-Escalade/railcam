# Change: Add MP4 Output, Side-by-Side Videos, and Normalized Zoom

## Why

The current tool generates GIFs, which have quality and size limitations. Users need MP4 output for better quality and smaller files. Additionally, comparing multiple climbers side-by-side requires manually combining videos. Finally, the current tracking centers on the pelvis but doesn't normalize the zoom level, making comparisons inconsistent when climbers are at different distances from the camera.

## What Changes

- **NEW** MP4 output format as an alternative to GIF
- **NEW** Side-by-side video composition from multiple input videos with independent frame ranges
- **NEW** Normalized zoom based on average torso height across all frames (target: 1/3 of output height)
- **BREAKING** Aspect ratio changed from 4:3 to 5:3 (vertical portrait)
- **MODIFIED** CLI interface to support new features

## Impact

- Affected specs:
  - `output-format` (new) - MP4 generation capability
  - `multi-video` (new) - Side-by-side video composition
  - `zoom-normalization` (new) - Torso-based zoom normalization
  - `aspect-ratio` (modified in `video-cropping`) - 4:3 → 5:3
- Affected code:
  - `cli.py` - New command-line arguments
  - `gif.py` → `output.py` - Renamed, add MP4 support
  - `cropping.py` - Updated aspect ratio constants, new zoom calculation
  - `pose.py` - Extract shoulder landmarks for torso measurement
  - New `composition.py` - Side-by-side frame assembly

## Scope

This change extends the existing CLI with:

1. **MP4 output**: Use `--format mp4` or `--format gif` (default: mp4)
2. **Multi-video mode**: Use `--input video.mp4:start:end` multiple times to combine videos horizontally
3. **Normalized zoom**: Each video is zoomed so the average torso height = 1/3 of output height
4. **New aspect ratio**: 5:3 vertical (width:height = 3:5 = 0.6)

## Success Criteria

- MP4 output produces smaller files with better quality than GIF
- Multiple videos are aligned horizontally with synchronized frame counts
- Torso size is visually consistent across different videos
- Existing single-video GIF workflow remains functional
