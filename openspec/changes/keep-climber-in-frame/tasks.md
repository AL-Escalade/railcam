## 1. Measure the reach

- [x] 1.1 Add the climber's reach from the pelvis, per axis, to `VideoAnalysisResult`, measured from the landmarks that passed the confidence threshold
- [x] 1.2 Test: the reach is the largest offset over the clip, and unsure landmarks are ignored

## 2. Cap the zoom

- [x] 2.1 Add `max_zoom_keeping_body_in_frame` to `cropping.py`, with `BODY_MARGIN_RATIO` covering what the keypoints do not
- [x] 2.2 Derive the torso ratio that zoom allows, and take the smallest across the render's videos
- [x] 2.3 Analyze every video before planning any, so the target can be shared
- [x] 2.4 Report the cap when it applies
- [x] 2.5 Test: a close climber lowers the target, a distant one leaves it alone, the cap is shared, and the reach fits the crop at the chosen zoom
