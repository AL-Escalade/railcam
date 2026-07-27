## ADDED Requirements

### Requirement: Streaming Frame Encoding

The system SHALL accept frames to encode as a stream, with the frame dimensions
and total count supplied separately, so the whole output need not exist in
memory before encoding starts.

#### Scenario: Frames encoded from a stream

- **WHEN** encoding is given an iterator of frames with dimensions and a count
- **THEN** each frame SHALL be written to FFmpeg as it is produced
- **AND** the encoder SHALL NOT require the frames to be materialized first

#### Scenario: Progress reported while streaming

- **WHEN** frames are encoded from a stream with a progress callback
- **THEN** progress SHALL be reported per frame against the supplied total

#### Scenario: Stream ends early

- **WHEN** the stream yields fewer frames than the supplied count
- **THEN** the system SHALL raise an output generation error naming the shortfall

#### Scenario: List-based encoding still supported

- **WHEN** encoding is given a list of frames
- **THEN** the output SHALL be identical to encoding the same frames as a stream
