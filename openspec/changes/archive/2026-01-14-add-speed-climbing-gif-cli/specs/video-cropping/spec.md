## ADDED Requirements

### Requirement: Aspect Ratio

The system SHALL crop frames to a 4:3 vertical aspect ratio (3 width : 4 height).

#### Scenario: Standard crop
- **WHEN** cropping a frame
- **THEN** the crop region SHALL have width:height ratio of 3:4

### Requirement: Pelvis Centering

The system SHALL center the crop region on the climber's pelvis when possible.

#### Scenario: Pelvis can be centered
- **WHEN** the pelvis position allows centering without exceeding video boundaries
- **THEN** the crop region SHALL be centered on the pelvis position both horizontally and vertically

#### Scenario: Pelvis near left edge
- **WHEN** centering horizontally would require pixels outside the left edge of the video
- **THEN** the crop region SHALL be aligned to the left edge while maintaining vertical centering if possible

#### Scenario: Pelvis near right edge
- **WHEN** centering horizontally would require pixels outside the right edge of the video
- **THEN** the crop region SHALL be aligned to the right edge while maintaining vertical centering if possible

#### Scenario: Pelvis near top edge
- **WHEN** centering vertically would require pixels outside the top edge of the video
- **THEN** the crop region SHALL be aligned to the top edge while maintaining horizontal centering if possible

#### Scenario: Pelvis near bottom edge
- **WHEN** centering vertically would require pixels outside the bottom edge of the video
- **THEN** the crop region SHALL be aligned to the bottom edge while maintaining horizontal centering if possible

#### Scenario: Pelvis near corner
- **WHEN** centering would require pixels outside two adjacent edges
- **THEN** the crop region SHALL be aligned to both edges (corner position)

### Requirement: No Black Borders

The system SHALL never add black borders or padding to crop regions.

#### Scenario: Crop always within video bounds
- **WHEN** calculating the crop region for any frame
- **THEN** the crop region SHALL be entirely within the video frame boundaries

### Requirement: Crop Size Calculation

The system SHALL calculate crop dimensions to maximize usable area while respecting aspect ratio.

#### Scenario: Video wider than crop ratio
- **WHEN** the source video has an aspect ratio wider than 3:4
- **THEN** the crop height SHALL match the video height and width SHALL be calculated as height * 3/4

#### Scenario: Video taller than crop ratio
- **WHEN** the source video has an aspect ratio taller than 3:4
- **THEN** the crop width SHALL match the video width and height SHALL be calculated as width * 4/3

### Requirement: Consistent Crop Size

The system SHALL use the same crop dimensions for all frames in a sequence.

#### Scenario: Crop size constant across frames
- **WHEN** processing multiple frames
- **THEN** all frames SHALL be cropped to identical dimensions
