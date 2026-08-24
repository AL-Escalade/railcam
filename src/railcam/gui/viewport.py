"""Zoom/pan viewport math for the frame display (pure, no Qt).

The viewport is described by a zoom factor (1 = fit the whole frame) and a
normalized view center. `visible_rect` converts that state to the region of
the source frame to display, clamped to the frame bounds.
"""

from __future__ import annotations

ZOOM_PER_NOTCH = 1.2
_NOTCH_DELTA = 120.0  # angleDelta units Qt reports for one mouse-wheel detent


def wheel_zoom_factor(angle_delta: float) -> float:
    """Zoom factor for a wheel event of angle_delta eighths of a degree.

    Following the delta instead of just its sign keeps a trackpad's stream of
    small events as gradual as a single detent of a mouse wheel.
    """
    return float(ZOOM_PER_NOTCH ** (angle_delta / _NOTCH_DELTA))


class Viewport:
    """Zoomable, pannable view over a frame of arbitrary size."""

    MIN_ZOOM = 1.0
    MAX_ZOOM = 8.0

    def __init__(self) -> None:
        self.zoom = 1.0
        self._center_x = 0.5
        self._center_y = 0.5

    @property
    def is_zoomed(self) -> bool:
        return self.zoom > self.MIN_ZOOM

    def visible_rect(self, width: float, height: float) -> tuple[float, float, float, float]:
        """Return the (x, y, w, h) region of a width×height frame to display."""
        view_w = width / self.zoom
        view_h = height / self.zoom
        x = min(max(self._center_x * width - view_w / 2, 0.0), width - view_w)
        y = min(max(self._center_y * height - view_h / 2, 0.0), height - view_h)
        return x, y, view_w, view_h

    def zoom_at(self, factor: float, anchor_x: float, anchor_y: float) -> None:
        """Scale the zoom by factor, keeping the view-relative anchor stationary.

        anchor_x/anchor_y are normalized positions within the currently
        visible region (0..1), e.g. the cursor position over the display.
        """
        # Anchor as a normalized point of the full frame (frame size cancels out)
        x, y, view_w, view_h = self.visible_rect(1.0, 1.0)
        point_x = x + anchor_x * view_w
        point_y = y + anchor_y * view_h

        self.zoom = min(max(self.zoom * factor, self.MIN_ZOOM), self.MAX_ZOOM)

        new_view_w = 1.0 / self.zoom
        new_view_h = 1.0 / self.zoom
        self._center_x = point_x - anchor_x * new_view_w + new_view_w / 2
        self._center_y = point_y - anchor_y * new_view_h + new_view_h / 2

    def pan_view(self, dx: float, dy: float) -> None:
        """Shift the view window by (dx, dy) expressed in view-size fractions."""
        self._center_x += dx / self.zoom
        self._center_y += dy / self.zoom
        # Keep the window inside the frame so panning cannot get lost
        half_w = 1.0 / (2 * self.zoom)
        self._center_x = min(max(self._center_x, half_w), 1.0 - half_w)
        self._center_y = min(max(self._center_y, half_w), 1.0 - half_w)

    def reset(self) -> None:
        self.zoom = 1.0
        self._center_x = 0.5
        self._center_y = 0.5
