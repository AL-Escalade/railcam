## ADDED Requirements

### Requirement: Per-Video Label Band

The system SHALL accept an optional text label for each input video and SHALL
render it, horizontally centered, in a solid band appended under that video's
cropped image. The label SHALL default to empty.

#### Scenario: No label given

- **WHEN** no input has a label
- **THEN** the emitted frames SHALL have the cropped image dimensions
- **AND** no band SHALL be added

#### Scenario: Label rendered under the image

- **WHEN** an input has a non-empty label
- **THEN** the emitted frames SHALL be taller than the cropped image by the band
  height
- **AND** the label text SHALL be centered horizontally within the band
- **AND** the cropped image SHALL keep its 5:3 aspect ratio, unchanged by the band

#### Scenario: Band height proportional to the image

- **WHEN** the band is added
- **THEN** its height SHALL be a fixed fraction of the cropped image height
- **AND** it SHALL be an even number of pixels

#### Scenario: Label wider than the frame

- **WHEN** the label is too long to fit at the nominal font size
- **THEN** the text SHALL be reduced until it fits within the frame width
- **AND** the text SHALL NOT be clipped

#### Scenario: Label with characters outside ASCII

- **WHEN** the label holds accented or non-Latin characters
- **THEN** they SHALL be drawn as typed, using a font bundled with the package
  so the result does not depend on the fonts installed on the machine

#### Scenario: Label with output scaling

- **WHEN** a label is used together with a requested output width or height
- **THEN** the band SHALL be scaled with the image, keeping the same proportion
  of the output height

### Requirement: Uniform Band In Multi-Video Output

When several videos are composed side by side, the system SHALL give every video
a band of the same relative height as soon as any one of them has a label, so
that height normalization scales all images by the same factor.

#### Scenario: One labeled video among several

- **WHEN** two videos are composed and only the first has a label
- **THEN** both videos SHALL receive a band of the same relative height
- **AND** the second video's band SHALL contain no text

#### Scenario: Zoom normalization preserved

- **WHEN** labeled videos are composed side by side
- **THEN** the climber's torso SHALL occupy the same fraction of the image in
  every video, as it does without labels
