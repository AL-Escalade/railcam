"""Text rows rendered in a solid band appended under a cropped frame."""

from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from importlib import resources

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Text sizes as a fraction of the cropped image height
LABEL_FONT_RATIO = 0.04
SUBLABEL_FONT_RATIO = 0.021

# Space above the first line and below the last, as a fraction of the first
# line's size. The band is measured from its text rather than the other way
# round, so this padding is the same above and below by construction -- when
# lines were centered in fixed boxes instead, the label sat further from the
# image than the last line sat from the bottom edge.
BAND_PAD_RATIO = 0.6

# Space between two lines' drawn pixels, as a fraction of the upper line's
# size. Kept well under the padding so a label and its second line read as one
# caption rather than as two.
LINE_GAP_RATIO = 0.2

LABEL_COLOR = 255
BAND_COLOR = 0

# Measured to size a line's slot: the tallest a line can draw at a given size,
# cap height plus descender, whatever the text actually holds. Sizing a slot
# from its own text would make the band height depend on the words, and two
# videos composed side by side would then get bands of different heights.
REFERENCE_TEXT = "Hg"

# Fraction of the frame width the text may occupy before it is shrunk
USABLE_WIDTH_RATIO = 0.9

# Bundled rather than resolved from the system so a label renders the same on
# every machine, and so every script the font covers is drawn as typed
FONT_FILE = "DejaVuSans-Bold.ttf"


class LabelFontError(Exception):
    """The bundled label font could not be loaded."""


@dataclass(frozen=True)
class LabelLine:
    """One line of the label band.

    Attributes:
        text: Text drawn as typed, centered on the band; empty for a blank line,
            which still takes its place so every video's band matches.
        size: Font size in pixels.
    """

    text: str
    size: int


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


def font_size(image_height: int, ratio: float = LABEL_FONT_RATIO) -> int:
    """Text size for a line drawn under an image of the given height.

    Args:
        image_height: Height in pixels of the cropped image the band sits under.
        ratio: Text size as a fraction of the image height.

    Returns:
        A size in pixels, or 0 when no line is requested (ratio of 0 or a
        non-positive image height).
    """
    if ratio <= 0 or image_height <= 0:
        return 0
    # A requested line must stay legible, however small the image is
    return max(int(round(image_height * ratio)), 1)


@cache
def _slot_height(size: int) -> int:
    """Height reserved for a line of this size, text-independent."""
    _, top, _, bottom = _ink_box(REFERENCE_TEXT, _font(size))
    return bottom - top


def band_height(lines: Sequence[LabelLine]) -> int:
    """Total height of the band drawn under the image.

    Args:
        lines: Lines of the band, top to bottom.

    Returns:
        An even number of pixels, or 0 when no line has a size. Encoders reject
        odd dimensions for some pixel formats, hence the even rounding.
    """
    drawn = [line for line in lines if line.size > 0]
    if not drawn:
        return 0

    padding = 2 * round(drawn[0].size * BAND_PAD_RATIO)
    slots = sum(_slot_height(line.size) for line in drawn)
    gaps = sum(round(line.size * LINE_GAP_RATIO) for line in drawn[:-1])
    total = padding + slots + gaps
    return total + total % 2


def _ink_box(text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int, int, int]:
    """Return the bounding box of the drawn pixels, relative to a top-left origin."""
    left, top, right, bottom = font.getbbox(text, anchor="lt")
    return int(left), int(top), int(right), int(bottom)


def _fitted_font(text: str, size: int, frame_width: int) -> ImageFont.FreeTypeFont:
    """Font for a line, reduced when the text would run past the usable width.

    Args:
        text: Line text to measure.
        size: Nominal font size in pixels.
        frame_width: Width of the frame the band spans.

    Returns:
        The font to draw with.
    """
    font = _font(max(size, 1))
    left, _, right, _ = _ink_box(text, font)
    text_width = right - left
    usable_width = frame_width * USABLE_WIDTH_RATIO
    if text_width > usable_width:
        font = _font(max(int(font.size * usable_width / text_width), 1))
    return font


def _band_mask(lines: Sequence[LabelLine], frame_width: int, band_pixels: int) -> np.ndarray:
    """Render every line of the band into one 8-bit coverage mask.

    Drawing through a mask keeps the antialiased edges while staying agnostic
    to the frame's channel count. Each text is centered on its drawn pixels
    rather than on the font's line box, so a label with no descender is not
    pushed visually upwards.

    Args:
        lines: Lines to draw, top to bottom.
        frame_width: Band width in pixels.
        band_pixels: Total band height in pixels.

    Returns:
        A (band_pixels, frame_width) array of coverage values in 0..255.
    """
    mask = Image.new("L", (frame_width, band_pixels), 0)
    draw = ImageDraw.Draw(mask)

    slot_top = round(lines[0].size * BAND_PAD_RATIO)
    for index, line in enumerate(lines):
        slot = _slot_height(line.size)
        if line.text.strip():
            font = _fitted_font(line.text, line.size, frame_width)
            left, top, right, bottom = _ink_box(line.text, font)
            x = (frame_width - (right - left)) // 2 - left
            y = slot_top + (slot - (bottom - top)) // 2 - top
            draw.text((x, y), line.text, font=font, fill=255, anchor="lt")
        slot_top += slot
        if index + 1 < len(lines):
            slot_top += round(line.size * LINE_GAP_RATIO)

    return np.asarray(mask)


def _render_band(frame: np.ndarray, lines: Sequence[LabelLine], band_pixels: int) -> np.ndarray:
    """Render the whole band, matching the frame's dtype and channel count."""
    frame_width = frame.shape[1]
    band = np.full((band_pixels, *frame.shape[1:]), BAND_COLOR, dtype=frame.dtype)

    if frame_width > 0 and any(line.text.strip() for line in lines):
        coverage = _band_mask(lines, frame_width, band_pixels) / 255.0
        if band.ndim == 3:
            coverage = coverage[..., np.newaxis]
        blended = coverage * LABEL_COLOR + (1.0 - coverage) * BAND_COLOR
        band = np.broadcast_to(blended, band.shape).astype(frame.dtype)

    return band


def append_label_band(frame: np.ndarray, lines: Sequence[LabelLine]) -> np.ndarray:
    """Append a black band of text lines under `frame`.

    Lines are stacked top to bottom in the order given, each centered
    horizontally and drawn at its own size, with the same padding above the
    first line as below the last.

    Args:
        frame: Cropped frame, any dtype and channel count; it is left untouched.
        lines: Lines to draw; an empty text still takes its place, and a line
            with a non-positive size is skipped.

    Returns:
        A new frame grown by the band height, or `frame` itself when no line
        has a positive size.

    Raises:
        LabelFontError: If the bundled font cannot be read.
    """
    drawn = [line for line in lines if line.size > 0]
    band_pixels = band_height(drawn)
    if not band_pixels:
        return frame

    return np.vstack((frame, _render_band(frame, drawn, band_pixels)))
