"""Tests for pose detection and climber selection."""

from __future__ import annotations

import pytest

from railcam.pose import (
    ClimberSelector,
    PelvisPosition,
    PersonDetection,
    TorsoMeasurement,
    person_to_detection_result,
    select_climber,
)


class TestClimberSelector:
    """Tests for ClimberSelector enum."""

    def test_left_value(self) -> None:
        assert ClimberSelector.LEFT.value == "left"

    def test_right_value(self) -> None:
        assert ClimberSelector.RIGHT.value == "right"

    def test_auto_value(self) -> None:
        assert ClimberSelector.AUTO.value == "auto"


class TestSelectClimber:
    """Tests for select_climber function."""

    def test_empty_list_returns_none(self) -> None:
        result = select_climber([], ClimberSelector.LEFT)
        assert result is None

    def test_single_person_returns_that_person(self) -> None:
        person = PersonDetection(
            pelvis=PelvisPosition(x=0.3, y=0.5, confidence=0.9)
        )
        result = select_climber([person], ClimberSelector.LEFT)
        assert result is person

    def test_single_person_ignores_selector(self) -> None:
        person = PersonDetection(
            pelvis=PelvisPosition(x=0.3, y=0.5, confidence=0.9)
        )
        # Even with RIGHT selector, single person is returned
        result = select_climber([person], ClimberSelector.RIGHT)
        assert result is person

    def test_left_selector_returns_leftmost(self) -> None:
        left_person = PersonDetection(
            pelvis=PelvisPosition(x=0.2, y=0.5, confidence=0.9)
        )
        right_person = PersonDetection(
            pelvis=PelvisPosition(x=0.8, y=0.5, confidence=0.9)
        )
        result = select_climber([left_person, right_person], ClimberSelector.LEFT)
        assert result is left_person

    def test_right_selector_returns_rightmost(self) -> None:
        left_person = PersonDetection(
            pelvis=PelvisPosition(x=0.2, y=0.5, confidence=0.9)
        )
        right_person = PersonDetection(
            pelvis=PelvisPosition(x=0.8, y=0.5, confidence=0.9)
        )
        result = select_climber([left_person, right_person], ClimberSelector.RIGHT)
        assert result is right_person

    def test_auto_selector_returns_closest_to_center(self) -> None:
        left_person = PersonDetection(
            pelvis=PelvisPosition(x=0.2, y=0.5, confidence=0.9)
        )
        center_person = PersonDetection(
            pelvis=PelvisPosition(x=0.45, y=0.5, confidence=0.9)
        )
        right_person = PersonDetection(
            pelvis=PelvisPosition(x=0.8, y=0.5, confidence=0.9)
        )
        result = select_climber(
            [left_person, center_person, right_person], ClimberSelector.AUTO
        )
        assert result is center_person

    def test_proximity_tracking_overrides_selector(self) -> None:
        left_person = PersonDetection(
            pelvis=PelvisPosition(x=0.2, y=0.5, confidence=0.9)
        )
        right_person = PersonDetection(
            pelvis=PelvisPosition(x=0.8, y=0.5, confidence=0.9)
        )
        # Previous position was on the right
        previous_pos = PelvisPosition(x=0.75, y=0.5, confidence=0.9)

        # Even with LEFT selector, proximity tracking picks the right person
        result = select_climber(
            [left_person, right_person],
            ClimberSelector.LEFT,
            previous_position=previous_pos,
        )
        assert result is right_person

    def test_proximity_tracking_considers_y_coordinate(self) -> None:
        top_person = PersonDetection(
            pelvis=PelvisPosition(x=0.5, y=0.2, confidence=0.9)
        )
        bottom_person = PersonDetection(
            pelvis=PelvisPosition(x=0.5, y=0.8, confidence=0.9)
        )
        # Previous position was at the top
        previous_pos = PelvisPosition(x=0.5, y=0.25, confidence=0.9)

        result = select_climber(
            [top_person, bottom_person],
            ClimberSelector.AUTO,
            previous_position=previous_pos,
        )
        assert result is top_person

    def test_three_persons_left_selector(self) -> None:
        persons = [
            PersonDetection(pelvis=PelvisPosition(x=0.3, y=0.5, confidence=0.9)),
            PersonDetection(pelvis=PelvisPosition(x=0.1, y=0.5, confidence=0.9)),
            PersonDetection(pelvis=PelvisPosition(x=0.7, y=0.5, confidence=0.9)),
        ]
        result = select_climber(persons, ClimberSelector.LEFT)
        assert result is persons[1]  # x=0.1 is leftmost

    def test_identical_x_coordinates_returns_first_match(self) -> None:
        """When persons have identical X coordinates, min/max returns first encountered."""
        person1 = PersonDetection(pelvis=PelvisPosition(x=0.5, y=0.3, confidence=0.9))
        person2 = PersonDetection(pelvis=PelvisPosition(x=0.5, y=0.7, confidence=0.9))

        # Both have same X=0.5, so for LEFT selector, first one should be returned
        result = select_climber([person1, person2], ClimberSelector.LEFT)
        assert result is person1

        # For RIGHT selector, also first one (since it's max of equals)
        result = select_climber([person1, person2], ClimberSelector.RIGHT)
        assert result is person1

        # For AUTO, both are equidistant from center (0.5), first one wins
        result = select_climber([person1, person2], ClimberSelector.AUTO)
        assert result is person1


class TestPersonToDetectionResult:
    """Tests for person_to_detection_result function."""

    def test_none_person_returns_empty_result(self) -> None:
        result = person_to_detection_result(None, frame_num=42)
        assert result.frame_num == 42
        assert result.position is None
        assert result.torso is None
        assert result.landmarks == []

    def test_person_with_pelvis_only(self) -> None:
        person = PersonDetection(
            pelvis=PelvisPosition(x=0.5, y=0.6, confidence=0.85)
        )
        result = person_to_detection_result(person, frame_num=10)
        assert result.frame_num == 10
        assert result.position is not None
        assert result.position.x == 0.5
        assert result.position.y == 0.6
        assert result.position.confidence == 0.85
        assert result.torso is None
        assert result.landmarks == []

    def test_person_with_torso(self) -> None:
        person = PersonDetection(
            pelvis=PelvisPosition(x=0.5, y=0.6, confidence=0.85),
            torso=TorsoMeasurement(height=0.15, shoulder_y=0.45, hip_y=0.6, confidence=0.8),
        )
        result = person_to_detection_result(person, frame_num=5)
        assert result.torso is not None
        assert result.torso.height == 0.15

    def test_person_with_landmarks(self) -> None:
        landmarks = [(0.5, 0.3, 0.9), (0.5, 0.6, 0.85)]
        person = PersonDetection(
            pelvis=PelvisPosition(x=0.5, y=0.6, confidence=0.85),
            landmarks=landmarks,
        )
        result = person_to_detection_result(person, frame_num=1)
        assert result.landmarks == landmarks
