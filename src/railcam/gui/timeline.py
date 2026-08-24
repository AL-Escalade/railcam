"""Scrubbable timeline widget with a highlighted start/end frame range."""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent
from PySide6.QtWidgets import QSizePolicy, QWidget

_MARGIN = 10
_TRACK_HEIGHT = 6

_TRACK_COLOR = QColor("#3a3d42")
_RANGE_COLOR = QColor(255, 122, 26, 110)
_MARKER_COLOR = QColor("#ff7a1a")
_PLAYHEAD_COLOR = QColor("#e8e8e8")


class TimelineWidget(QWidget):
    """Horizontal timeline: scrub with the mouse, shows the selected range."""

    frameScrubbed = Signal(int)

    def __init__(self, frame_count: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._frame_count = max(frame_count, 1)
        self._current = 0
        self._start = 0
        self._end = self._frame_count - 1
        self.setMinimumHeight(30)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_frame_count(self, frame_count: int) -> None:
        """Rescale the timeline to a new video length, clamping what it shows."""
        self._frame_count = max(frame_count, 1)
        self._current = self._clamp(self._current)
        self._start = self._clamp(self._start)
        self._end = self._clamp(self._end)
        self.update()

    def set_current_frame(self, frame: int) -> None:
        self._current = self._clamp(frame)
        self.update()

    def set_range(self, start: int, end: int) -> None:
        self._start = self._clamp(start)
        self._end = self._clamp(end)
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)

        track_top = (self.height() - _TRACK_HEIGHT) / 2
        track = QRectF(_MARGIN, track_top, self.width() - 2 * _MARGIN, _TRACK_HEIGHT)
        painter.setBrush(_TRACK_COLOR)
        painter.drawRoundedRect(track, 3, 3)

        # Selected range highlight
        x_start = self._frame_to_x(self._start)
        x_end = self._frame_to_x(self._end)
        painter.setBrush(_RANGE_COLOR)
        painter.drawRoundedRect(
            QRectF(x_start, track_top, max(x_end - x_start, 2), _TRACK_HEIGHT), 3, 3
        )

        # Start/end markers
        painter.setBrush(_MARKER_COLOR)
        for x in (x_start, x_end):
            painter.drawRoundedRect(QRectF(x - 1.5, track_top - 5, 3, _TRACK_HEIGHT + 10), 1, 1)

        # Playhead
        x_cur = self._frame_to_x(self._current)
        painter.setBrush(_PLAYHEAD_COLOR)
        painter.drawEllipse(QRectF(x_cur - 6, track_top + _TRACK_HEIGHT / 2 - 6, 12, 12))

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._scrub_to(event.position().x())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._scrub_to(event.position().x())

    def _scrub_to(self, x: float) -> None:
        frame = self._x_to_frame(x)
        if frame != self._current:
            self._current = frame
            self.update()
            self.frameScrubbed.emit(frame)

    def _clamp(self, frame: int) -> int:
        return max(0, min(self._frame_count - 1, frame))

    def _usable_width(self) -> float:
        return max(self.width() - 2 * _MARGIN, 1)

    def _frame_to_x(self, frame: int) -> float:
        span = max(self._frame_count - 1, 1)
        return _MARGIN + (frame / span) * self._usable_width()

    def _x_to_frame(self, x: float) -> int:
        span = max(self._frame_count - 1, 1)
        return self._clamp(round((x - _MARGIN) / self._usable_width() * span))
