## ADDED Requirements

### Requirement: Vertical Aspect Ratio
The system SHALL crop frames to a 5:3 vertical aspect ratio (width:height = 3:5 = 0.6).

#### Scenario: Calculate crop dimensions
- **WHEN** processing a video frame
- **THEN** the crop region width:height ratio SHALL be 3:5

#### Scenario: Fit within video bounds
- **WHEN** the video is wider than the target ratio
- **THEN** height SHALL be the limiting factor and width SHALL be calculated as `height * 0.6`

#### Scenario: Fit narrow video
- **WHEN** the video is narrower than the target ratio
- **THEN** width SHALL be the limiting factor and height SHALL be calculated as `width / 0.6`

#### Scenario: Even dimensions
- **WHEN** calculating crop dimensions
- **THEN** both width and height SHALL be even numbers (for video encoding compatibility)
