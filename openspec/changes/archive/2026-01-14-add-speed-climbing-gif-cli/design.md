# Design: Speed Climbing GIF CLI

## Context

Speed climbing videos typically show an athlete climbing a standardized 15m wall. The climber moves vertically at high speed (world records under 5 seconds). Videos are usually shot from a fixed camera position showing the full wall.

**Constraints:**
- Must run locally without cloud dependencies
- Cross-platform support (macOS, Linux, Windows)
- No hard performance requirements, accuracy is priority

## Goals / Non-Goals

**Goals:**
- Accurate pelvis detection and tracking
- Smooth, professional-looking crop motion
- Simple CLI interface with sensible defaults
- Handle edge cases gracefully (climber near borders)

**Non-Goals:**
- Real-time processing
- GUI interface
- Support for multiple climbers in frame
- Support for non-speed-climbing videos

## Decisions

### 1. Programming Language: Python

**Decision:** Use Python as the primary language.

**Rationale:**
- Excellent ecosystem for video processing (OpenCV, FFmpeg bindings)
- MediaPipe has first-class Python support
- Cross-platform without compilation complexity
- Easy to install via pip/pipx

**Alternatives considered:**
- **Node.js**: Weaker ML/video processing ecosystem
- **Rust**: Great performance, but MediaPipe bindings are immature
- **Go**: Limited ML ecosystem

### 2. Pose Estimation: MediaPipe Pose

**Decision:** Use Google's MediaPipe Pose for pelvis detection.

**Rationale:**
- Runs entirely locally (no cloud required)
- Lightweight model with good accuracy
- Provides hip landmarks (LEFT_HIP, RIGHT_HIP) to compute pelvis center
- Cross-platform support
- Active maintenance by Google

**Alternatives considered:**
- **OpenPose**: More accurate but heavier, complex setup
- **YOLO-Pose**: Good but MediaPipe is simpler for single-person tracking
- **Object tracking (CSRT, KCF)**: Requires manual initialization, less accurate

### 3. Video Processing: OpenCV + FFmpeg

**Decision:** Use OpenCV for frame extraction and FFmpeg for GIF generation.

**Rationale:**
- OpenCV: Efficient frame-by-frame processing, integrates well with MediaPipe
- FFmpeg: Industry standard for video/GIF conversion, handles palette optimization

**Alternatives considered:**
- **MoviePy**: Higher-level but adds abstraction we don't need
- **Pillow**: Limited GIF optimization capabilities
- **imageio**: Decent but FFmpeg is more flexible for quality control

### 4. Motion Smoothing: Exponential Moving Average

**Decision:** Apply exponential moving average (EMA) to pelvis positions.

**Rationale:**
- Simple to implement and understand
- Provides natural-looking smoothing
- Single parameter (alpha) to tune responsiveness
- Handles the "camera following" feel well

**Alternatives considered:**
- **Kalman filter**: More complex, overkill for this use case
- **Simple moving average**: Creates lag, less responsive
- **Gaussian smoothing**: Works but EMA is more intuitive

### 5. Interpolation Strategy: Linear Interpolation

**Decision:** Use linear interpolation for frames with failed pose detection.

**Rationale:**
- Speed climbing motion is relatively linear (mostly vertical)
- Simple and predictable
- Works well with EMA smoothing applied afterward

**Alternatives considered:**
- **Spline interpolation**: Overkill for short gaps
- **Hold last position**: Creates jumps when detection resumes

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         CLI Entry                            │
│                    (argument parsing)                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Frame Extraction                         │
│              (OpenCV: read frames in range)                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Pose Detection                           │
│         (MediaPipe: detect pelvis per frame)                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Position Processing                        │
│     (Interpolation → Smoothing → Boundary clamping)         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Frame Cropping                           │
│       (OpenCV: crop 4:3 region around pelvis)               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    GIF Generation                            │
│      (FFmpeg: assemble frames, optimize palette)            │
└─────────────────────────────────────────────────────────────┘
```

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| MediaPipe fails on unusual camera angles | Document supported video orientations; provide debug mode to visualize detections |
| Smoothing makes motion feel laggy | Tunable smoothing parameter; sensible default based on testing |
| Large input videos consume memory | Process frames in streaming fashion, don't load entire video |
| FFmpeg not installed on user system | Provide clear error message with installation instructions; consider bundling |

## Open Questions

1. **Default output resolution**: Should we default to source resolution or a standard size (e.g., 480px width)?
   - *Proposed answer*: Default to automatic based on source, allow override via CLI flag

2. **Smoothing strength**: What EMA alpha value provides the best visual result?
   - *Proposed answer*: Start with alpha=0.3, tune based on testing with real videos

3. **Detection confidence threshold**: Below what confidence should we interpolate?
   - *Proposed answer*: MediaPipe visibility score < 0.5 triggers interpolation
