# Change: Never frame the climber out of the crop

## Why

The zoom targets a torso height of 1/6 of the output, whatever the climber then
does with her limbs. On a phone-filmed final where the wall fills the frame, a
high step put her foot outside the crop: the render followed the pelvis while
the leg left the picture. Nothing reported it, and no option could prevent it.

## What Changes

- Measure, during analysis, how far the climber reaches from her pelvis over
  the whole clip, in both axes.
- Cap the zoom so that reach, widened by a margin for what keypoints do not
  cover (feet past the ankles, hands past the wrists), always fits inside the
  crop.
- Share the cap across every video of a render: capping only the video that
  needs it would leave the climbers at different sizes, which is what the zoom
  normalization exists to prevent. Videos are therefore all analyzed before any
  is planned.

## Impact

- Affected specs: `zoom-normalization`
- Affected code: `src/railcam/cropping.py`, `src/railcam/cli.py`
- Renders whose climber already fitted are unchanged. A render that framed too
  close now zooms out, so its climber appears smaller than 1/6 of the height —
  the trade the framing guarantee requires.
