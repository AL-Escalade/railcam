"""Pose detection using YOLOv8-pose for pelvis tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO, settings


class ClimberSelector(Enum):
    """Selector for which climber to track in multi-person videos."""

    LEFT = "left"
    RIGHT = "right"
    AUTO = "auto"  # Single climber or closest to center


# Confidence threshold for hip visibility
CONFIDENCE_THRESHOLD = 0.3

# Normalized X coordinate representing frame center (for AUTO climber selection)
FRAME_CENTER_X = 0.5

# YOLOv8 pose keypoint indices (COCO format)
# 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
# 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
# 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
# 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12

# Pose skeleton connections for visualization (COCO format)
POSE_CONNECTIONS = [
    # Face
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),  # Nose to eyes to ears
    # Arms
    (5, 7),
    (7, 9),  # Left arm
    (6, 8),
    (8, 10),  # Right arm
    # Torso
    (5, 6),  # Shoulders
    (5, 11),
    (6, 12),  # Shoulders to hips
    (11, 12),  # Hips
    # Legs
    (11, 13),
    (13, 15),  # Left leg
    (12, 14),
    (14, 16),  # Right leg
]


@dataclass
class PelvisPosition:
    """Position of the pelvis in a frame."""

    x: float  # Normalized x coordinate (0-1)
    y: float  # Normalized y coordinate (0-1)
    confidence: float


@dataclass
class TorsoMeasurement:
    """Measurement of the torso height in a frame."""

    height: float  # Normalized torso height (0-1, relative to frame height)
    shoulder_y: float  # Normalized y coordinate of shoulder midpoint
    hip_y: float  # Normalized y coordinate of hip midpoint (pelvis)
    confidence: float


@dataclass
class PersonDetection:
    """Detection of a single person with pelvis position."""

    pelvis: PelvisPosition
    torso: TorsoMeasurement | None = None
    # Landmarks: (x, y, conf) normalized
    landmarks: list[tuple[float, float, float]] = field(default_factory=list)


@dataclass
class MultiPersonDetectionResult:
    """Result of multi-person pose detection for a single frame."""

    frame_num: int
    persons: list[PersonDetection]  # All persons with valid pelvis


@dataclass
class DetectionResult:
    """Result of pose detection for a single frame."""

    frame_num: int
    position: PelvisPosition | None  # None if detection failed
    torso: TorsoMeasurement | None = None  # None if torso measurement failed
    # Landmarks: (x, y, conf) normalized
    landmarks: list[tuple[float, float, float]] = field(default_factory=list)


class PoseDetector:
    """Detects pelvis position using YOLOv8-pose."""

    def __init__(
        self,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        model_size: str = "m",  # n, s, m, l, x
    ) -> None:
        self.confidence_threshold = confidence_threshold
        # Load YOLOv8-pose model from stable cache directory (not current working dir)
        model_name = f"yolov8{model_size}-pose.pt"
        weights_dir = settings.get("weights_dir", "")
        model_path = f"{weights_dir}/{model_name}" if weights_dir else model_name
        print(f"Loading YOLOv8-pose model ({model_name})...")
        self._model = YOLO(model_path)
        print("Model loaded.")

    def _extract_person_from_keypoints(
        self,
        keypoints: Any,
        width: int,
        height: int,
    ) -> PersonDetection | None:
        """Extract PersonDetection from YOLO keypoints for a single person.

        Args:
            keypoints: YOLO keypoints object for one person.
            width: Frame width in pixels.
            height: Frame height in pixels.

        Returns:
            PersonDetection if valid pelvis detected, None otherwise.
        """
        # Extract xy coordinates and confidence
        if keypoints.xy is None or len(keypoints.xy) == 0:
            return None

        xy = keypoints.xy[0].cpu().numpy()  # Shape: (17, 2)

        # Get confidence scores if available
        if keypoints.conf is not None and len(keypoints.conf) > 0:
            conf = keypoints.conf[0].cpu().numpy()  # Shape: (17,)
        else:
            conf = np.ones(len(xy))  # Default to 1.0 if no confidence

        # Extract hip positions
        left_hip_xy = xy[LEFT_HIP]
        right_hip_xy = xy[RIGHT_HIP]
        left_hip_conf = conf[LEFT_HIP] if len(conf) > LEFT_HIP else 0.0
        right_hip_conf = conf[RIGHT_HIP] if len(conf) > RIGHT_HIP else 0.0

        # Check hip visibility - at least one hip must be visible
        left_hip_valid = left_hip_conf >= self.confidence_threshold and (
            left_hip_xy[0] > 0 or left_hip_xy[1] > 0
        )
        right_hip_valid = right_hip_conf >= self.confidence_threshold and (
            right_hip_xy[0] > 0 or right_hip_xy[1] > 0
        )

        # No valid hip detected - skip this person
        if not left_hip_valid and not right_hip_valid:
            return None

        # Calculate pelvis position
        if left_hip_valid and right_hip_valid:
            px = (left_hip_xy[0] + right_hip_xy[0]) / 2
            py = (left_hip_xy[1] + right_hip_xy[1]) / 2
            hip_confidence = (left_hip_conf + right_hip_conf) / 2
        elif left_hip_valid:
            px = left_hip_xy[0]
            py = left_hip_xy[1]
            hip_confidence = left_hip_conf
        else:
            px = right_hip_xy[0]
            py = right_hip_xy[1]
            hip_confidence = right_hip_conf

        pelvis = PelvisPosition(x=px / width, y=py / height, confidence=float(hip_confidence))
        hip_y_norm = py / height

        # Build normalized landmarks list
        landmarks: list[tuple[float, float, float]] = []
        for i in range(len(xy)):
            x_norm = xy[i][0] / width
            y_norm = xy[i][1] / height
            c = conf[i] if i < len(conf) else 1.0
            landmarks.append((x_norm, y_norm, float(c)))

        # Calculate torso measurement
        torso_measurement = self._calculate_torso_measurement(
            xy, conf, hip_y_norm, pelvis.confidence, height
        )

        return PersonDetection(
            pelvis=pelvis,
            torso=torso_measurement,
            landmarks=landmarks,
        )

    def _calculate_torso_measurement(
        self,
        xy: np.ndarray,
        conf: np.ndarray,
        hip_y_norm: float,
        pelvis_confidence: float,
        height: int,
    ) -> TorsoMeasurement | None:
        """Calculate torso measurement from keypoints.

        Args:
            xy: Keypoint coordinates array.
            conf: Confidence scores array.
            hip_y_norm: Normalized Y coordinate of hip.
            pelvis_confidence: Confidence of pelvis detection.
            height: Frame height in pixels.

        Returns:
            TorsoMeasurement if shoulders detected, None otherwise.
        """
        left_shoulder_xy = xy[LEFT_SHOULDER]
        right_shoulder_xy = xy[RIGHT_SHOULDER]
        left_shoulder_conf = conf[LEFT_SHOULDER] if len(conf) > LEFT_SHOULDER else 0.0
        right_shoulder_conf = conf[RIGHT_SHOULDER] if len(conf) > RIGHT_SHOULDER else 0.0

        left_shoulder_valid = left_shoulder_conf >= self.confidence_threshold and (
            left_shoulder_xy[0] > 0 or left_shoulder_xy[1] > 0
        )
        right_shoulder_valid = right_shoulder_conf >= self.confidence_threshold and (
            right_shoulder_xy[0] > 0 or right_shoulder_xy[1] > 0
        )

        shoulder_y_norm: float | None = None
        shoulder_confidence: float = 0.0

        if left_shoulder_valid and right_shoulder_valid:
            shoulder_y_norm = (left_shoulder_xy[1] + right_shoulder_xy[1]) / 2 / height
            shoulder_confidence = (left_shoulder_conf + right_shoulder_conf) / 2
        elif left_shoulder_valid:
            shoulder_y_norm = left_shoulder_xy[1] / height
            shoulder_confidence = left_shoulder_conf
        elif right_shoulder_valid:
            shoulder_y_norm = right_shoulder_xy[1] / height
            shoulder_confidence = right_shoulder_conf

        if shoulder_y_norm is None:
            return None

        torso_height = abs(hip_y_norm - shoulder_y_norm)
        combined_conf = (pelvis_confidence + shoulder_confidence) / 2

        return TorsoMeasurement(
            height=torso_height,
            shoulder_y=shoulder_y_norm,
            hip_y=hip_y_norm,
            confidence=float(combined_conf),
        )

    def detect_pelvis(self, frame: np.ndarray, frame_num: int) -> DetectionResult:
        """Detect the pelvis position and torso measurement in a frame.

        The pelvis is computed as the midpoint between left and right hip landmarks.
        The torso height is the distance between shoulder midpoint and hip midpoint.
        Returns the first (most confident) person detected.
        """
        height, width = frame.shape[:2]

        # Run inference
        results = self._model(frame, verbose=False)

        if not results or len(results) == 0:
            return DetectionResult(frame_num=frame_num, position=None, torso=None, landmarks=[])

        result = results[0]

        # Check if keypoints were detected
        if result.keypoints is None or len(result.keypoints) == 0:
            return DetectionResult(frame_num=frame_num, position=None, torso=None, landmarks=[])

        # Get keypoints for the first (most confident) person
        keypoints = result.keypoints[0]

        # Extract person detection using shared helper
        person = self._extract_person_from_keypoints(keypoints, width, height)

        if person is None:
            return DetectionResult(frame_num=frame_num, position=None, torso=None, landmarks=[])

        return DetectionResult(
            frame_num=frame_num,
            position=person.pelvis,
            torso=person.torso,
            landmarks=person.landmarks,
        )

    def detect_all_persons(self, frame: np.ndarray, frame_num: int) -> MultiPersonDetectionResult:
        """Detect all persons with valid pelvis in a frame.

        Only returns persons whose hip landmarks are detected with sufficient confidence.
        This filters out bystanders whose hips are not visible.

        Args:
            frame: The video frame to analyze.
            frame_num: The frame number.

        Returns:
            MultiPersonDetectionResult with all detected persons.
        """
        height, width = frame.shape[:2]

        # Run inference
        results = self._model(frame, verbose=False)

        if not results or len(results) == 0:
            return MultiPersonDetectionResult(frame_num=frame_num, persons=[])

        result = results[0]

        # Check if keypoints were detected
        if result.keypoints is None or len(result.keypoints) == 0:
            return MultiPersonDetectionResult(frame_num=frame_num, persons=[])

        persons: list[PersonDetection] = []

        # Iterate over all detected persons
        for keypoints in result.keypoints:
            person = self._extract_person_from_keypoints(keypoints, width, height)
            if person is not None:
                persons.append(person)

        return MultiPersonDetectionResult(frame_num=frame_num, persons=persons)

    def close(self) -> None:
        """Release resources."""
        pass  # YOLO handles cleanup automatically

    def __enter__(self) -> PoseDetector:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def select_climber(
    persons: list[PersonDetection],
    selector: ClimberSelector,
    previous_position: PelvisPosition | None = None,
) -> PersonDetection | None:
    """Select a climber from detected persons based on selector and tracking.

    Args:
        persons: List of detected persons with valid pelvis.
        selector: Which climber to select (LEFT, RIGHT, AUTO).
        previous_position: Previous pelvis position for proximity tracking.

    Returns:
        The selected PersonDetection, or None if no valid person found.
    """
    if not persons:
        return None

    if len(persons) == 1:
        # Only one person - return them regardless of selector
        return persons[0]

    # Multiple persons detected
    if previous_position is not None:
        # Proximity tracking: select person closest to previous position
        return min(
            persons,
            key=lambda p: (
                (p.pelvis.x - previous_position.x) ** 2 + (p.pelvis.y - previous_position.y) ** 2
            ),
        )

    # Initial selection based on selector
    if selector == ClimberSelector.LEFT:
        # Leftmost person (smallest X)
        return min(persons, key=lambda p: p.pelvis.x)
    elif selector == ClimberSelector.RIGHT:
        # Rightmost person (largest X)
        return max(persons, key=lambda p: p.pelvis.x)
    else:
        # AUTO: closest to center
        return min(persons, key=lambda p: abs(p.pelvis.x - FRAME_CENTER_X))


def person_to_detection_result(person: PersonDetection | None, frame_num: int) -> DetectionResult:
    """Convert a PersonDetection to DetectionResult format.

    Args:
        person: The person detection (may be None).
        frame_num: The frame number.

    Returns:
        DetectionResult compatible with existing code.
    """
    if person is None:
        return DetectionResult(frame_num=frame_num, position=None, torso=None, landmarks=[])

    return DetectionResult(
        frame_num=frame_num,
        position=person.pelvis,
        torso=person.torso,
        landmarks=person.landmarks,
    )


# Fluorescent green color (BGR format)
NEON_GREEN = (57, 255, 20)


def draw_pose_overlay_on_crop(
    cropped_frame: np.ndarray,
    detection: DetectionResult,
    crop_region: Any,  # CropRegion from cropping module
    original_width: int,
    original_height: int,
    color: tuple[int, int, int] = NEON_GREEN,
) -> np.ndarray:
    """Draw full pose skeleton overlay on a cropped frame for debugging.

    Transforms landmark coordinates from original frame space to cropped frame space.
    Draws all detected landmarks as points and skeleton connections as segments.
    """
    overlay = cropped_frame.copy()
    crop_h, crop_w = cropped_frame.shape[:2]

    landmarks = detection.landmarks
    if not landmarks:
        return overlay

    # Convert normalized landmarks to pixel coordinates in cropped frame space
    points: list[tuple[int, int] | None] = []
    for x_norm, y_norm, conf in landmarks:
        if conf >= 0.1 and (x_norm > 0 or y_norm > 0):
            # Convert normalized coords to original frame pixels
            orig_px = x_norm * original_width
            orig_py = y_norm * original_height
            # Transform to cropped frame coordinates
            crop_px = int(orig_px - crop_region.x)
            crop_py = int(orig_py - crop_region.y)
            # Only include if within crop bounds (with some margin for drawing)
            if -50 <= crop_px <= crop_w + 50 and -50 <= crop_py <= crop_h + 50:
                points.append((crop_px, crop_py))
            else:
                points.append(None)
        else:
            points.append(None)

    # Draw connections (segments)
    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx < len(points) and end_idx < len(points):
            start_pt = points[start_idx]
            end_pt = points[end_idx]
            if start_pt is not None and end_pt is not None:
                cv2.line(overlay, start_pt, end_pt, color, 2, cv2.LINE_AA)

    # Draw landmark points (only if within visible area)
    for pt in points:
        if pt is not None and 0 <= pt[0] < crop_w and 0 <= pt[1] < crop_h:
            cv2.circle(overlay, pt, 4, color, -1, cv2.LINE_AA)
            cv2.circle(overlay, pt, 6, color, 1, cv2.LINE_AA)

    # Draw pelvis position with a larger marker
    if detection.position is not None:
        orig_px = detection.position.x * original_width
        orig_py = detection.position.y * original_height
        px = int(orig_px - crop_region.x)
        py = int(orig_py - crop_region.y)
        if 0 <= px < crop_w and 0 <= py < crop_h:
            cv2.circle(overlay, (px, py), 8, (0, 0, 255), -1, cv2.LINE_AA)  # Red center
            cv2.circle(overlay, (px, py), 12, color, 2, cv2.LINE_AA)  # Green ring

    return overlay
