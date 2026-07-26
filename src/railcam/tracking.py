"""Motion-based climber track building and selection.

Per-frame detections are grouped into tracks by proximity, then the climber
is chosen among tracks that actually climb (large vertical displacement),
so static bystanders are never selected.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from railcam.pose import (
    ClimberSelector,
    DetectionResult,
    MultiPersonDetectionResult,
    PelvisPosition,
    PersonDetection,
)

# Maximum plausible movement between two consecutive frames (normalized).
MAX_JUMP_PER_FRAME = 0.15
# Cap on gap-scaled jumps: stays below the typical distance between lanes,
# so a track never teleports onto the other climber after a detection gap.
MAX_JUMP_TOTAL = 0.30
# Maximum horizontal deviation from the anchor when repairing a gap. A climber
# stays in a near-fixed vertical lane, so x barely changes even across large
# (vertical) gaps. Unlike the gap-scaled total budget, this stays fixed: it
# rejects the other lane's climber, which the euclidean budget alone lets in
# once the gap widens (see repair_track_gaps).
MAX_LANE_DRIFT = 0.10
# Minimum vertical displacement for a track to qualify as a climber.
MIN_CLIMB_SPAN = 0.10
# A climbing track must also cover at least this fraction of the best track's
# displacement, so detection fragments cannot outrank a full climb.
MIN_RELATIVE_SPAN = 0.5
# During gap repair, a candidate this close to another track's detection on
# the same frame is that other person — never attach it.
OTHER_TRACK_EPSILON = 0.05

FRAME_CENTER_X = 0.5


@dataclass
class Track:
    """Detections of one person across frames."""

    detections: dict[int, PersonDetection] = field(default_factory=dict)

    def add(self, frame_num: int, person: PersonDetection) -> None:
        self.detections[frame_num] = person

    @property
    def last_frame(self) -> int:
        return max(self.detections)

    @property
    def last_position(self) -> PelvisPosition:
        return self.detections[self.last_frame].pelvis

    @property
    def mean_x(self) -> float:
        return sum(p.pelvis.x for p in self.detections.values()) / len(self.detections)

    @property
    def vertical_span(self) -> float:
        ys = [p.pelvis.y for p in self.detections.values()]
        return max(ys) - min(ys)


def _allowed_jump(gap_frames: int) -> float:
    """Maximum association distance after gap_frames without detection."""
    return min(MAX_JUMP_PER_FRAME * max(gap_frames, 1), MAX_JUMP_TOTAL)


def build_tracks(frame_results: list[MultiPersonDetectionResult]) -> list[Track]:
    """Group per-frame detections into tracks by nearest-neighbor association.

    Closest (track, person) pairs are matched first, one-to-one, within the
    gap-scaled jump limit; unmatched persons start new tracks.
    """
    tracks: list[Track] = []
    for result in frame_results:
        candidates = sorted(
            (
                (
                    math.hypot(
                        track.last_position.x - person.pelvis.x,
                        track.last_position.y - person.pelvis.y,
                    ),
                    track_index,
                    person_index,
                )
                for track_index, track in enumerate(tracks)
                for person_index, person in enumerate(result.persons)
            ),
        )
        matched_tracks: set[int] = set()
        matched_persons: set[int] = set()
        for distance, track_index, person_index in candidates:
            if track_index in matched_tracks or person_index in matched_persons:
                continue
            track = tracks[track_index]
            if distance > _allowed_jump(result.frame_num - track.last_frame):
                continue
            track.add(result.frame_num, result.persons[person_index])
            matched_tracks.add(track_index)
            matched_persons.add(person_index)

        for person_index, person in enumerate(result.persons):
            if person_index not in matched_persons:
                new_track = Track()
                new_track.add(result.frame_num, person)
                tracks.append(new_track)
    return tracks


def select_track(tracks: list[Track], selector: ClimberSelector) -> Track | None:
    """Choose the climber's track: selectors apply among climbing tracks.

    Tracks below the climbing displacement threshold (static bystanders) are
    excluded; if none qualifies, all tracks are considered as a fallback.
    """
    if not tracks:
        return None

    climbing = [track for track in tracks if track.vertical_span >= MIN_CLIMB_SPAN]
    if climbing:
        best_span = max(track.vertical_span for track in climbing)
        climbing = [
            track for track in climbing if track.vertical_span >= MIN_RELATIVE_SPAN * best_span
        ]
    candidates = climbing or tracks

    if selector == ClimberSelector.LEFT:
        return min(candidates, key=lambda track: track.mean_x)
    if selector == ClimberSelector.RIGHT:
        return max(candidates, key=lambda track: track.mean_x)
    return min(candidates, key=lambda track: abs(track.mean_x - FRAME_CENTER_X))


def repair_track_gaps(
    track: Track,
    frame_nums: list[int],
    detect: Callable[[int], list[PersonDetection]],
    avoid: list[Track] | None = None,
) -> int:
    """Fill the track's missing frames using a (higher-resolution) detector.

    Gap frames are processed from the nearest known frames outward, so each
    repaired frame anchors the next and repairs chain along the trajectory.
    A recovered person is attached only within the gap-scaled jump distance
    of the nearest known position, and never when it coincides with another
    track's detection on that frame (no stealing the bystander).

    Args:
        track: The selected climber track, modified in place.
        frame_nums: All frame numbers of the analyzed range.
        detect: frame_num -> persons, typically high-resolution inference.
        avoid: Other tracks whose persons must not be attached.

    Returns:
        Number of frames recovered.
    """
    known = set(track.detections)
    if not known:
        return 0
    missing = [f for f in frame_nums if f not in known]

    repaired = 0
    for frame_num in sorted(missing, key=lambda f: min(abs(f - k) for k in known)):
        anchor_frame = min(track.detections, key=lambda k: abs(k - frame_num))
        anchor = track.detections[anchor_frame].pelvis
        allowed = _allowed_jump(abs(frame_num - anchor_frame))

        best: PersonDetection | None = None
        best_distance = allowed
        for candidate in detect(frame_num):
            if _is_other_tracks_person(candidate, frame_num, avoid):
                continue
            # A climber stays in her lane: a large horizontal offset is the
            # other climber, even when the euclidean distance fits the budget.
            if abs(candidate.pelvis.x - anchor.x) > MAX_LANE_DRIFT:
                continue
            distance = math.hypot(candidate.pelvis.x - anchor.x, candidate.pelvis.y - anchor.y)
            if distance <= best_distance:
                best = candidate
                best_distance = distance

        if best is not None:
            track.add(frame_num, best)
            repaired += 1
    return repaired


def _is_other_tracks_person(
    candidate: PersonDetection, frame_num: int, avoid: list[Track] | None
) -> bool:
    """True when the candidate matches another track's detection on this frame."""
    for other in avoid or []:
        detection = other.detections.get(frame_num)
        if detection is not None and (
            math.hypot(
                detection.pelvis.x - candidate.pelvis.x,
                detection.pelvis.y - candidate.pelvis.y,
            )
            < OTHER_TRACK_EPSILON
        ):
            return True
    return False


def track_to_detections(track: Track | None, frame_nums: list[int]) -> list[DetectionResult]:
    """Convert the chosen track to one DetectionResult per frame.

    Frames where the track has no detection yield position None so the
    existing interpolation handles them; other persons are never substituted.
    """
    results: list[DetectionResult] = []
    for frame_num in frame_nums:
        person = track.detections.get(frame_num) if track is not None else None
        if person is None:
            results.append(DetectionResult(frame_num=frame_num, position=None))
        else:
            results.append(
                DetectionResult(
                    frame_num=frame_num,
                    position=person.pelvis,
                    torso=person.torso,
                    landmarks=person.landmarks,
                )
            )
    return results
