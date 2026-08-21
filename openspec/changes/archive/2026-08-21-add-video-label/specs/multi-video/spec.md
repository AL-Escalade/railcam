## ADDED Requirements

### Requirement: Labeled Video Composition

The composed output SHALL place each video's label under that video's image, so
that a side-by-side comparison names each climber.

#### Scenario: Labels follow their video

- **WHEN** two labeled videos are composed side by side
- **THEN** each label SHALL appear under the video it was given for
- **AND** each label SHALL be centered on the width of its own video, not on the
  composed frame
