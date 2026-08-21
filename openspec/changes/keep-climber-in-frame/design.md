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

Keypoints stop at the joints, and what continues past them scales with the
climber rather than with how far she happens to be reaching — so the margin is
expressed in torso heights, not as a fraction of the reach. Measured on the
clip this came from, the silhouette hangs about 1.35 torso below the ankle when
the foot points down; a hand reaches roughly half a torso past the wrist, and
hair barely clears the eyes.

Those three differ enough that a single margin is wrong either way: sized for
the foot it zooms out on a climber who is merely reaching high, sized for the
hair it clips her feet. The reach is therefore measured separately upward,
downward and sideways, and each gets its own margin.

## One target for the whole render

Zoom normalization means every video shows the same climber size, so a cap that
applied to one video only would silently break the comparison it exists to
serve. The target torso ratio is therefore the smallest across the videos, and
`plan_videos` analyzes them all before planning any. Analyses hold detections
and geometry, never frames, so this changes nothing about memory.
