## ADDED Requirements

### Requirement: Framing Guarantee

The system SHALL never zoom so far that the climber leaves the crop, and SHALL
lower the target torso height when the zoom the normalization asks for would
put part of her outside the frame.

#### Scenario: Climber reaching beyond the crop

- **WHEN** the climber's reach from her pelvis, at the normalized zoom, would
  fall outside the crop on any frame of the clip
- **THEN** the zoom SHALL be reduced until that reach fits
- **AND** the reduction SHALL be reported

#### Scenario: Climber already inside the crop

- **WHEN** the climber's reach fits at the normalized zoom
- **THEN** the zoom SHALL be the one the normalization asks for

#### Scenario: Margin beyond the keypoints

- **WHEN** the zoom is chosen
- **THEN** it SHALL leave free space beyond the measured keypoints, for the
  parts of the body they do not cover
- **AND** that space SHALL be proportional to the climber's torso height

#### Scenario: Reaching down costs more room than reaching up

- **WHEN** two clips differ only in whether the climber's furthest reach is
  above or below her pelvis
- **THEN** the clip reaching down SHALL be zoomed out further, since the foot
  continues past the ankle where the head barely continues past the eyes

#### Scenario: Cap shared across videos

- **WHEN** several videos are composed and only one of them needs a lower zoom
- **THEN** every video SHALL be zoomed to the same torso height
- **AND** that height SHALL be the lowest the videos require
