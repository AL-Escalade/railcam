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

# Maximum plausible movement between two consecutive frames (normalized), used
# while a track is too short to have measured its own motion.
MAX_JUMP_PER_FRAME = 0.15
# Cap on gap-scaled jumps: stays below the typical distance between lanes,
# so a track never teleports onto the other climber after a detection gap.
MAX_JUMP_TOTAL = 0.30
# How much faster than its own observed motion a track may plausibly move. It
# is deliberately loose: the repair pass infers at a higher resolution than the
# main one and lands the pelvis a little differently, and that offset must not
# read as a jump. Even so it rejects a candidate a third of a frame away, which
# is what a false positive on a hold looks like.
# The frame rate and how much wall the shot covers are both unknown here, and
# together they span two orders of magnitude, so a fixed budget is either too
# tight for a wide 30 fps shot or -- as it was -- wide enough to swallow a
# third of the frame between two 60 fps frames. A track's own steps carry both
# unknowns already.
STEP_TOLERANCE = 5.0
# Floor on that budget, so a track that has barely moved yet (crouched start,
# a climber setting up) stays repairable.
MIN_STEP_BUDGET = 0.01
# Maximum horizontal deviation from the previous position, both when
# associating detections and when repairing a gap. A climber stays in a
# near-fixed vertical lane, so x barely changes even across large (vertical)
# gaps. Unlike the gap-scaled total budget, this stays fixed: it rejects the
# other lane's climber and the belayer at the foot of the wall, which the
# euclidean budget alone lets in once the gap widens.
MAX_LANE_DRIFT = 0.10
# Minimum upward displacement, jumps aside, for a track to qualify as a climber.
MIN_CLIMB_SPAN = 0.10
# A climbing track must also cover at least this fraction of the best track's
# rise, so detection fragments cannot outrank a full climb.
MIN_RELATIVE_SPAN = 0.5
# ... and be detected on at least this fraction of the frames the best-covered
# climbing track holds. Pose models report the occasional hold as a person, and
# such a detection can drift upward through the frame as the camera pans, which
# the rise alone cannot tell from a climb. It is never found as consistently as
# a person, and the leftmost or rightmost candidate would otherwise be that
# phantom rather than the climber.
MIN_RELATIVE_COVERAGE = 0.5
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
    def typical_step(self) -> float | None:
        """Median per-frame movement of this track, or None if never measured.

        The median ignores the occasional wild detection, which is the point:
        this is the scale a plausible jump is judged against.
        """
        frames = sorted(self.detections)
        steps = []
        for previous, current in zip(frames, frames[1:]):
            before = self.detections[previous].pelvis
            after = self.detections[current].pelvis
            steps.append(math.hypot(after.x - before.x, after.y - before.y) / (current - previous))
        if not steps:
            return None
        return sorted(steps)[len(steps) // 2]

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

    @property
    def _largest_step_up(self) -> float:
        """Largest upward move between two consecutive detections, 0 if none."""
        frames = sorted(self.detections)
        steps = [
            self.detections[before].pelvis.y - self.detections[after].pelvis.y
            for before, after in zip(frames, frames[1:])
        ]
        return max(max(steps, default=0.0), 0.0)

    @property
    def steady_rise(self) -> float:
        """Rise the track achieved without its single largest step.

        A climber gains height gradually, a few thousandths of the frame at a
        time; a track that hops from one hold to the next owes most of its
        rise to one jump. Dropping the largest step costs a climber almost
        nothing and a phantom everything: on the clip this came from, one such
        track took 74% of its rise from a single step, against 4% and 6% for
        the two climbers.
        """
        return max(self.climb_rise - self._largest_step_up, 0.0)


def _allowed_jump(gap_frames: int, track: Track | None = None) -> float:
    """Maximum association distance after gap_frames without detection.

    Args:
        gap_frames: Frames since the track was last seen.
        track: The track being extended; its own motion sets the scale once it
            has been seen at least twice.

    Returns:
        The largest plausible distance, normalized to the frame.
    """
    step = track.typical_step if track is not None else None
    per_frame = MAX_JUMP_PER_FRAME if step is None else max(step * STEP_TOLERANCE, MIN_STEP_BUDGET)
    return min(per_frame * max(gap_frames, 1), MAX_JUMP_TOTAL)


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
            if distance > _allowed_jump(result.frame_num - track.last_frame, track):
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
    """Choose the climber's track: selectors apply among the climbing ones.

    Tracks that barely move next to the best one (static bystanders) are
    excluded, as are those found on far fewer frames than it -- a pose model
    reporting a hold as a person. Both cuts are relative, so they still hold
    on a section where nobody clears the absolute climbing threshold.
    """
    if not tracks:
        return None

    # The absolute threshold only picks the pool: a short section near the top
    # of the wall can leave every track below it, and the run still has a
    # climber. The relative cuts then apply either way, so a bystander or a
    # hold never wins by default.
    candidates = [track for track in tracks if track.steady_rise >= MIN_CLIMB_SPAN] or tracks

    best_rise = max(track.steady_rise for track in candidates)
    candidates = [
        track for track in candidates if track.steady_rise >= MIN_RELATIVE_SPAN * best_rise
    ]
    best_coverage = max(len(track.detections) for track in candidates)
    candidates = [
        track
        for track in candidates
        if len(track.detections) >= MIN_RELATIVE_COVERAGE * best_coverage
    ]

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
        allowed = _allowed_jump(abs(frame_num - anchor_frame), track)

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
