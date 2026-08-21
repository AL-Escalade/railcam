# Design

## Reach, not body height

What has to fit is not the climber's height but her reach from the pelvis,
because the crop is centered there and a climber is not symmetric about it: a
high step puts a foot far below the hips while the hands stay close. Measuring
the reach per axis over the whole clip, and keeping the largest, gives the one
number the zoom has to respect.

The measure comes from the pose landmarks that were already detected, so it
costs nothing extra and needs no second pass.

## The margin

Keypoints stop at the joints. The foot continues past the ankle, the hand past
the wrist, the hair past the head, and the smoothed crop centre trails the
pelvis slightly. `BODY_MARGIN_RATIO` widens the measured reach by a fifth,
which covers all of it with room to spare — on the diagnosed clip the worst
frame then left 19% of the half-height free.

## One target for the whole render

Zoom normalization means every video shows the same climber size, so a cap that
applied to one video only would silently break the comparison it exists to
serve. The target torso ratio is therefore the smallest across the videos, and
`plan_videos` analyzes them all before planning any. Analyses hold detections
and geometry, never frames, so this changes nothing about memory.
