"""Tests for the zoom/pan viewport math."""

from __future__ import annotations

import pytest

from railcam.gui.viewport import Viewport

W, H = 400.0, 300.0


def test_initial_view_shows_full_frame() -> None:
    viewport = Viewport()

    assert viewport.visible_rect(W, H) == (0.0, 0.0, W, H)
    assert not viewport.is_zoomed


def test_zoom_at_center_halves_visible_region() -> None:
    viewport = Viewport()

    viewport.zoom_at(2.0, 0.5, 0.5)

    assert viewport.visible_rect(W, H) == (100.0, 75.0, 200.0, 150.0)
    assert viewport.is_zoomed


def test_zoom_keeps_anchor_point_stationary() -> None:
    viewport = Viewport()

    # Anchor at 25% of the view: image point (100, 75)
    viewport.zoom_at(2.0, 0.25, 0.25)

    x, y, rw, rh = viewport.visible_rect(W, H)
    assert x + 0.25 * rw == pytest.approx(100.0)
    assert y + 0.25 * rh == pytest.approx(75.0)


def test_zoom_below_fit_clamps_to_full_frame() -> None:
    viewport = Viewport()

    viewport.zoom_at(0.5, 0.5, 0.5)

    assert viewport.visible_rect(W, H) == (0.0, 0.0, W, H)


def test_zoom_clamps_to_maximum() -> None:
    viewport = Viewport()

    for _ in range(20):
        viewport.zoom_at(2.0, 0.5, 0.5)

    assert viewport.zoom == pytest.approx(Viewport.MAX_ZOOM)


def test_pan_shifts_visible_region() -> None:
    viewport = Viewport()
    viewport.zoom_at(2.0, 0.5, 0.5)  # rect (100, 75, 200, 150)

    viewport.pan_view(0.25, 0.0)  # shift the window right by 25% of its width

    x, y, _, _ = viewport.visible_rect(W, H)
    assert x == pytest.approx(150.0)
    assert y == pytest.approx(75.0)


def test_pan_is_clamped_to_frame_bounds() -> None:
    viewport = Viewport()
    viewport.zoom_at(2.0, 0.5, 0.5)

    viewport.pan_view(10.0, -10.0)  # far beyond the edges

    x, y, rw, rh = viewport.visible_rect(W, H)
    assert x == pytest.approx(W - rw)
    assert y == pytest.approx(0.0)


def test_reset_returns_to_full_frame() -> None:
    viewport = Viewport()
    viewport.zoom_at(4.0, 0.2, 0.8)
    viewport.pan_view(0.1, 0.1)

    viewport.reset()

    assert viewport.visible_rect(W, H) == (0.0, 0.0, W, H)
    assert not viewport.is_zoomed
