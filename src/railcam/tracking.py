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
# Maximum horizontal deviation from the previous position, both when
# associating detections and when repairing a gap. A climber stays in a
# near-fixed vertical lane, so x barely changes even across large (vertical)
# gaps. Unlike the gap-scaled total budget, this stays fixed: it rejects the
# other lane's climber and the belayer at the foot of the wall, which the
# euclidean budget alone lets in once the gap widens.
MAX_LANE_DRIFT = 0.10
# Noise floor: a rise below this is detection jitter, not motion. It only
# decides whether anyone climbs at all -- how much a track must rise to be a
# candidate is set relative to the best track, not by this constant.
MIN_CLIMB_RISE = 0.03
# A climbing track must cover at least this fraction of the best track's rise,
# so detection fragments cannot outrank a full climb.
MIN_RELATIVE_RISE = 0.5
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
    def climb_rise(self) -> float:
        """Largest rise from an earlier frame to a later one.

        Image y grows downward, so climbing means a decreasing y. Measuring
        the rise rather than the unsigned span keeps a bystander swept *down*
        the frame by an upward camera tilt from ranking as a climber -- her
        apparent travel can exceed the climber's, and the relative-span cut
        would then discard the real climber.

        Taking the best earlier-to-later rise, rather than first minus last,
        keeps the measure meaningful when the track starts after the climb
        began or when the climber comes back down (fall, lower-off).
        """
        lowest = 0.0  # largest y seen so far: the lowest point in the frame
        rise = 0.0
        for frame_num in sorted(self.detections):
            y = self.detections[frame_num].pelvis.y
            lowest = max(lowest, y)
            rise = max(rise, lowest - y)
        return rise


def _allowed_jump(gap_frames: int) -> float:
    """Maximum association distance after gap_frames without detection."""
    return min(MAX_JUMP_PER_FRAME * max(gap_frames, 1), MAX_JUMP_TOTAL)


def _dedupe_persons(persons: list[PersonDetection]) -> list[PersonDetection]:
    """Drop detections that land on someone already detected in the frame.

    Pose inference sometimes reports the same person twice, a few pixels
    apart. Only one copy can be matched to her track, so the orphan starts a
    competing track that then steals the following frames and leaves two
    fragments where there was one climb. Distinct people in a speed final are
    lanes apart, well beyond this radius; the most confident copy wins.
    """
    kept: list[PersonDetection] = []
    for person in sorted(persons, key=lambda p: p.pelvis.confidence, reverse=True):
        if any(
            math.hypot(person.pelvis.x - other.pelvis.x, person.pelvis.y - other.pelvis.y)
            < OTHER_TRACK_EPSILON
            for other in kept
        ):
            continue
        kept.append(person)
    return kept


def build_tracks(frame_results: list[MultiPersonDetectionResult]) -> list[Track]:
    """Group per-frame detections into tracks by nearest-neighbor association.

    Closest (track, person) pairs are matched first, one-to-one, within the
    gap-scaled jump limit and the fixed lane width; unmatched persons start
    new tracks.
    """
    tracks: list[Track] = []
    for result in frame_results:
        persons = _dedupe_persons(result.persons)
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
                for person_index, person in enumerate(persons)
            ),
        )
        matched_tracks: set[int] = set()
        matched_persons: set[int] = set()
        for distance, track_index, person_index in candidates:
            if track_index in matched_tracks or person_index in matched_persons:
                continue
            track = tracks[track_index]
            person = persons[person_index]
            # A climber stays in her lane: a large horizontal offset is someone
            # else, however plausible the euclidean distance looks once the
            # gap-scaled budget has grown.
            if abs(track.last_position.x - person.pelvis.x) > MAX_LANE_DRIFT:
                continue
            if distance > _allowed_jump(result.frame_num - track.last_frame):
                continue
            track.add(result.frame_num, person)
            matched_tracks.add(track_index)
            matched_persons.add(person_index)

        for person_index, person in enumerate(persons):
            if person_index not in matched_persons:
                new_track = Track()
                new_track.add(result.frame_num, person)
                tracks.append(new_track)
    return tracks


def select_track(tracks: list[Track], selector: ClimberSelector) -> Track | None:
    """Choose the climber's track: selectors apply among climbing tracks.

    Which tracks climb is decided relative to the one that rises most, with the
    noise floor as the lower bound. An absolute threshold applied *first* would
    let the whole distinction collapse on a short excerpt: the camera follows
    the climber, so her apparent rise stays small -- 0.161 over the 64 frames of
    the Salt Lake City 2023 start -- and a shorter range drops every track below
    any fixed bar at once, leaving the selector to pick the rightmost bystander.

    Only when nobody clears the noise floor -- nothing moved -- do all tracks
    become candidates and position alone decides.

    A camera tilting down lifts every static track together and is not corrected
    for: the rise measured here is apparent, not ground-truth.
    """
    if not tracks:
        return None

    best_rise = max(track.climb_rise for track in tracks)
    if best_rise >= MIN_CLIMB_RISE:
        cut = max(MIN_RELATIVE_RISE * best_rise, MIN_CLIMB_RISE)
        candidates = [track for track in tracks if track.climb_rise >= cut]
    else:
        candidates = tracks

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
