"""Text label rendered in a solid band appended under a cropped frame."""

from __future__ import annotations

import io
from functools import cache
from importlib import resources

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Band height as a fraction of the cropped image height
LABEL_BAND_RATIO = 0.08

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
    """Compute the label band height for an image of the given height.

    Args:
        image_height: Height in pixels of the cropped image the band sits under.
        ratio: Band height as a fraction of the image height.

    Returns:
        An even number of pixels, or 0 when no band is requested (ratio of 0 or
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


def _fitted_font(text: str, band_pixels: int, frame_width: int) -> ImageFont.FreeTypeFont:
    """Pick the font size that fits both the band height and the usable width.

    Args:
        text: Label text to measure.
        band_pixels: Height of the band the text is drawn in.
        frame_width: Width of the frame the band spans.

    Returns:
        The font to draw with.
    """
    font = _font(max(int(band_pixels * CAP_HEIGHT_RATIO), 1))
    left, _, right, _ = _ink_box(text, font)
    text_width = right - left
    usable_width = frame_width * USABLE_WIDTH_RATIO
    if text_width > usable_width:
        font = _font(max(int(font.size * usable_width / text_width), 1))
    return font


def _text_mask(text: str, band_pixels: int, frame_width: int) -> np.ndarray:
    """Render the label as an 8-bit coverage mask the size of the band.

    Drawing through a mask keeps the antialiased edges while staying agnostic
    to the frame's channel count. The text is centered on its drawn pixels
    rather than on the font's line box, so a label with no descender is not
    pushed visually upwards.

    Args:
        text: Label text.
        band_pixels: Band height in pixels.
        frame_width: Band width in pixels.

    Returns:
        A (band_pixels, frame_width) array of coverage values in 0..255.
    """
    font = _fitted_font(text, band_pixels, frame_width)
    left, top, right, bottom = _ink_box(text, font)
    x = (frame_width - (right - left)) // 2 - left
    y = (band_pixels - (bottom - top)) // 2 - top

    mask = Image.new("L", (frame_width, band_pixels), 0)
    ImageDraw.Draw(mask).text((x, y), text, font=font, fill=255, anchor="lt")
    return np.asarray(mask)


def append_label_band(frame: np.ndarray, text: str, band_height: int) -> np.ndarray:
    """Append a black band under `frame` and draw `text` centered in it.

    Args:
        frame: Cropped frame, any dtype and channel count; it is left untouched.
        text: Label text, drawn as typed; an empty label yields a uniform band.
        band_height: Band height in pixels.

    Returns:
        A new frame of height `frame.shape[0] + band_height`, or `frame` itself
        when the band height is not positive.

    Raises:
        LabelFontError: If the bundled font cannot be read.
    """
    if band_height <= 0:
        return frame

    frame_width = frame.shape[1]
    band = np.full((band_height, *frame.shape[1:]), BAND_COLOR, dtype=frame.dtype)

    if text.strip() and frame_width > 0:
        coverage = _text_mask(text, band_height, frame_width) / 255.0
        if band.ndim == 3:
            coverage = coverage[..., np.newaxis]
        blended = coverage * LABEL_COLOR + (1.0 - coverage) * BAND_COLOR
        band = np.broadcast_to(blended, band.shape).astype(frame.dtype)

    return np.vstack((frame, band))
