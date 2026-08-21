"""Text rows rendered in a solid band appended under a cropped frame."""

from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from importlib import resources

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Row heights as a fraction of the cropped image height
LABEL_BAND_RATIO = 0.08
SUBLABEL_BAND_RATIO = 0.042

# Space left between the drawn pixels of two adjacent rows, as a fraction of
# the upper row's height. Centering each row in its own box put a hole between
# a label and its second line wider than that second line, reading as two
# unrelated captions instead of one two-line caption.
LINE_GAP_RATIO = 0.12

LABEL_COLOR = 255
BAND_COLOR = 0

# Cap height targeted inside the band, as a fraction of its height
CAP_HEIGHT_RATIO = 0.5

# Fraction of the frame width the text may occupy before it is shrunk
USABLE_WIDTH_RATIO = 0.9

# Bundled rather than resolved from the system so a label renders the same on
# every machine, and so every script the font covers is drawn as typed
FONT_FILE = "DejaVuSans-Bold.ttf"


class LabelFontError(Exception):
    """The bundled label font could not be loaded."""


@dataclass(frozen=True)
class LabelLine:
    """One row of the label band.

    Attributes:
        text: Text drawn as typed, centered in the row; empty for a blank row.
        height: Row height in pixels; the font follows it, so a smaller row
            means smaller characters.
    """

    text: str
    height: int


@cache
def _font(size: int) -> ImageFont.FreeTypeFont:
    """Load the bundled font at a given pixel size.

    Args:
        size: Font size in pixels.

    Returns:
        The loaded font.

    Raises:
        LabelFontError: If the bundled font cannot be read.
    """
    try:
        data = resources.files("railcam").joinpath("fonts", FONT_FILE).read_bytes()
        return ImageFont.truetype(io.BytesIO(data), size)
    except OSError as e:
        raise LabelFontError(f"Cannot load the label font {FONT_FILE}: {e}") from e


def band_height(image_height: int, ratio: float = LABEL_BAND_RATIO) -> int:
    """Compute a band row height for an image of the given height.

    Args:
        image_height: Height in pixels of the cropped image the band sits under.
        ratio: Row height as a fraction of the image height.

    Returns:
        An even number of pixels, or 0 when no row is requested (ratio of 0 or
        a non-positive image height). Encoders reject odd dimensions for some
        pixel formats, hence the even rounding.
    """
    if ratio <= 0 or image_height <= 0:
        return 0

    height = int(round(image_height * ratio / 2)) * 2
    # A requested band must stay visible, however small the image is
    return max(height, 2)


def _ink_box(text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int, int, int]:
    """Return the bounding box of the drawn pixels, relative to a top-left origin."""
    left, top, right, bottom = font.getbbox(text, anchor="lt")
    return int(left), int(top), int(right), int(bottom)


def _fitted_font(text: str, row_pixels: int, frame_width: int) -> ImageFont.FreeTypeFont:
    """Pick the font size that fits both the row height and the usable width.

    Args:
        text: Label text to measure.
        row_pixels: Height of the row the text is drawn in.
        frame_width: Width of the frame the band spans.

    Returns:
        The font to draw with.
    """
    font = _font(max(int(row_pixels * CAP_HEIGHT_RATIO), 1))
    left, _, right, _ = _ink_box(text, font)
    text_width = right - left
    usable_width = frame_width * USABLE_WIDTH_RATIO
    if text_width > usable_width:
        font = _font(max(int(font.size * usable_width / text_width), 1))
    return font


def _ink_top_in_row(rows: Sequence[LabelLine], index: int, ink_height: int) -> int:
    """Where a row's drawn pixels start within that row.

    A row on its own is centered in its box, as a single label always was. A
    row that has a neighbour instead hugs the boundary they share, leaving
    only half the gap on that side, so the two texts read as one caption.

    Args:
        rows: All rows of the band, in order.
        index: Index of the row being placed.
        ink_height: Height of that row's drawn pixels.

    Returns:
        Offset in pixels from the top of the row.
    """
    row = rows[index]
    above = rows[index - 1] if index > 0 and rows[index - 1].text.strip() else None
    below = rows[index + 1] if index + 1 < len(rows) and rows[index + 1].text.strip() else None

    if above is not None and below is None:
        return int(above.height * LINE_GAP_RATIO) // 2
    if below is not None and above is None:
        return row.height - ink_height - int(row.height * LINE_GAP_RATIO) // 2
    return (row.height - ink_height) // 2


def _band_mask(rows: Sequence[LabelLine], frame_width: int, band_pixels: int) -> np.ndarray:
    """Render every row's text into one 8-bit coverage mask of the whole band.

    Drawing through a mask keeps the antialiased edges while staying agnostic
    to the frame's channel count. Each text is centered on its drawn pixels
    rather than on the font's line box, so a label with no descender is not
    pushed visually upwards.

    Args:
        rows: Rows to draw, in order.
        frame_width: Band width in pixels.
        band_pixels: Total band height in pixels.

    Returns:
        A (band_pixels, frame_width) array of coverage values in 0..255.
    """
    mask = Image.new("L", (frame_width, band_pixels), 0)
    draw = ImageDraw.Draw(mask)

    row_top = 0
    for index, row in enumerate(rows):
        if row.text.strip():
            font = _fitted_font(row.text, row.height, frame_width)
            left, top, right, bottom = _ink_box(row.text, font)
            x = (frame_width - (right - left)) // 2 - left
            y = row_top + _ink_top_in_row(rows, index, bottom - top) - top
            draw.text((x, y), row.text, font=font, fill=255, anchor="lt")
        row_top += row.height

    return np.asarray(mask)


def _render_band(frame: np.ndarray, rows: Sequence[LabelLine]) -> np.ndarray:
    """Render the whole band, matching the frame's dtype and channel count."""
    frame_width = frame.shape[1]
    band_pixels = sum(row.height for row in rows)
    band = np.full((band_pixels, *frame.shape[1:]), BAND_COLOR, dtype=frame.dtype)

    if frame_width > 0 and any(row.text.strip() for row in rows):
        coverage = _band_mask(rows, frame_width, band_pixels) / 255.0
        if band.ndim == 3:
            coverage = coverage[..., np.newaxis]
        blended = coverage * LABEL_COLOR + (1.0 - coverage) * BAND_COLOR
        band = np.broadcast_to(blended, band.shape).astype(frame.dtype)

    return band


def append_label_band(frame: np.ndarray, lines: Sequence[LabelLine]) -> np.ndarray:
    """Append a black band of text rows under `frame`.

    Rows are stacked top to bottom in the order given, each text centered
    horizontally and sized from its own row's height. A row with a neighbour
    sits close to it, so a label and its second line read as one caption.

    Args:
        frame: Cropped frame, any dtype and channel count; it is left untouched.
        lines: Rows to draw; a row with an empty text yields a uniform row, and
            a row with a non-positive height is skipped.

    Returns:
        A new frame grown by the total height of the rows, or `frame` itself
        when no row has a positive height.

    Raises:
        LabelFontError: If the bundled font cannot be read.
    """
    rows = [line for line in lines if line.height > 0]
    if not rows:
        return frame

    return np.vstack((frame, _render_band(frame, rows)))
