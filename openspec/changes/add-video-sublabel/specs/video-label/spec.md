## ADDED Requirements

### Requirement: Second Label Line

The system SHALL accept an optional second label text per video and SHALL draw
it under the first label, in the same band, at a smaller size. It SHALL default
to empty.

#### Scenario: Second line rendered under the first

- **WHEN** an input has both a label and a second label
- **THEN** the second text SHALL be drawn below the first, centered on the same
  width
- **AND** its characters SHALL be smaller than those of the first line

#### Scenario: No second line given

- **WHEN** no input has a second label
- **THEN** the band SHALL keep the height it has for a single line
- **AND** the emitted frames SHALL be identical to those produced before this
  capability existed

#### Scenario: Second line without a first

- **WHEN** an input has a second label but no label
- **THEN** the second text SHALL still be drawn, in its own row
- **AND** the first row SHALL be left empty

## MODIFIED Requirements

### Requirement: Per-Video Label Band

The system SHALL accept an optional text label for each input video and SHALL
render it, horizontally centered, on the first line of a solid band appended
under that video's cropped image. The band SHALL be a stack of lines, each
drawn at its own size, and its height SHALL follow from those sizes. The label
SHALL default to empty.

#### Scenario: No label given

- **WHEN** no input has a label or a second label
- **THEN** the emitted frames SHALL have the cropped image dimensions
- **AND** no band SHALL be added

#### Scenario: Label rendered under the image

- **WHEN** an input has a non-empty label
- **THEN** the emitted frames SHALL be taller than the cropped image by the band
  height
- **AND** the label text SHALL be centered horizontally on its line
- **AND** the cropped image SHALL keep its 5:3 aspect ratio, unchanged by the band

#### Scenario: Band height proportional to the image

- **WHEN** the band is added
- **THEN** each line SHALL be drawn at a fixed fraction of the cropped image
  height
- **AND** the band SHALL leave the same space above its first line as below its
  last one
- **AND** the band SHALL be an even number of pixels

#### Scenario: Band height independent of the words

- **WHEN** two videos carry labels of different lengths
- **THEN** their bands SHALL have the same height

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
the same lines, at the same relative sizes, as soon as any one of them uses a
line, so that height normalization scales all images by the same factor.

#### Scenario: One labeled video among several

- **WHEN** two videos are composed and only the first has a label
- **THEN** both videos SHALL receive a band of the same relative height
- **AND** the second video's band SHALL contain no text

#### Scenario: One video with a second line

- **WHEN** two videos are composed and only the first has a second label
- **THEN** both videos SHALL receive a second row of the same height
- **AND** the second video's second row SHALL contain no text

#### Scenario: Zoom normalization preserved

- **WHEN** labeled videos are composed side by side
- **THEN** the climber's torso SHALL occupy the same fraction of the image in
  every video, as it does without labels
