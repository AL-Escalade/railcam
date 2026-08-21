"""Tests for the label band rendering module."""

from importlib import resources

import numpy as np

from railcam.labeling import (
    FONT_FILE,
    LABEL_BAND_RATIO,
    SUBLABEL_BAND_RATIO,
    USABLE_WIDTH_RATIO,
    LabelLine,
    append_label_band,
    band_height,
)


def make_frame(height=1000, width=600, value=50):
    return np.full((height, width, 3), value, dtype=np.uint8)


def ink_span(band):
    """(first, last) row index holding drawn pixels."""
    rows = np.where(band.max(axis=(1, 2)) > 0)[0]
    return (int(rows.min()), int(rows.max())) if rows.size else (0, 0)


def ink_height(band):
    rows = np.where(band.max(axis=(1, 2)) > 0)[0]
    return rows.max() - rows.min() + 1 if rows.size else 0


class TestBandHeight:
    def test_proportional_to_image_height(self):
        assert band_height(1000) == 80
        assert band_height(500) == 40

    def test_is_even(self):
        for image_height in range(100, 200):
            assert band_height(image_height) % 2 == 0

    def test_zero_ratio_gives_no_band(self):
        assert band_height(1000, 0) == 0

    def test_non_positive_image_height_gives_no_band(self):
        assert band_height(0) == 0
        assert band_height(-10) == 0

    def test_tiny_image_still_gets_a_visible_band(self):
        assert band_height(10) == 2

    def test_custom_ratio(self):
        assert band_height(1000, 0.2) == 200

    def test_default_ratio_is_the_constant(self):
        assert band_height(1000) == band_height(1000, LABEL_BAND_RATIO)

    def test_sublabel_row_is_shorter_than_the_label_row(self):
        assert 0 < band_height(1000, SUBLABEL_BAND_RATIO) < band_height(1000, LABEL_BAND_RATIO)


class TestAppendLabelBand:
    def test_no_rows_returns_frame_unchanged(self):
        frame = make_frame()

        assert append_label_band(frame, []) is frame

    def test_non_positive_row_heights_return_frame_unchanged(self):
        frame = make_frame()

        assert append_label_band(frame, [LabelLine("Alice", 0)]) is frame
        assert append_label_band(frame, [LabelLine("Alice", -4)]) is frame

    def test_output_height_grows_by_the_band(self):
        frame = make_frame()

        result = append_label_band(frame, [LabelLine("Alice", 80)])

        assert result.shape == (1080, 600, 3)

    def test_output_height_grows_by_the_sum_of_the_rows(self):
        frame = make_frame()

        result = append_label_band(frame, [LabelLine("Alice", 80), LabelLine("4.704", 50)])

        assert result.shape == (1130, 600, 3)

    def test_image_part_is_untouched(self):
        frame = make_frame()

        result = append_label_band(frame, [LabelLine("Alice", 80)])

        assert np.array_equal(result[:1000], frame)
        assert np.array_equal(frame, make_frame())

    def test_empty_label_yields_a_uniform_band(self):
        frame = make_frame()

        result = append_label_band(frame, [LabelLine("", 80)])
        band = result[1000:]

        assert band.shape == (80, 600, 3)
        assert not band.any()

    def test_blank_label_yields_a_uniform_band(self):
        frame = make_frame()

        band = append_label_band(frame, [LabelLine("   ", 80)])[1000:]

        assert not band.any()

    def test_text_is_drawn_and_horizontally_centered(self):
        frame = make_frame()

        band = append_label_band(frame, [LabelLine("Alice", 80)])[1000:]
        columns = np.where(band.max(axis=(0, 2)) > 0)[0]

        assert columns.size > 0
        center = (columns.min() + columns.max()) / 2
        assert abs(center - 600 / 2) <= 6

    def test_text_stays_inside_the_band_vertically(self):
        frame = make_frame()

        # Ascenders and descenders, so a correctly centered render leaves a
        # blank row on both sides instead of being clipped by the band edges
        band = append_label_band(frame, [LabelLine("Alice gjpqy", 80)])[1000:]
        rows = np.where(band.max(axis=(1, 2)) > 0)[0]

        assert rows.min() >= 1
        assert rows.max() <= 78

    def test_long_label_is_shrunk_to_fit_the_width(self):
        frame = make_frame()
        label = "A very long climber label that would never fit at the nominal size"

        band = append_label_band(frame, [LabelLine(label, 80)])[1000:]
        columns = np.where(band.max(axis=(0, 2)) > 0)[0]

        assert columns.max() - columns.min() + 1 <= 600 * USABLE_WIDTH_RATIO + 2

    def test_accents_are_drawn_not_stripped(self):
        frame = make_frame()

        for accented, ascii_text in [
            ("Léa", "Lea"),
            ("Müller", "Muller"),
            ("François", "Francois"),
            ("Anaïs", "Anais"),
        ]:
            band = append_label_band(frame, [LabelLine(accented, 80)])[1000:]

            assert band.max() > 0
            assert not np.array_equal(
                band, append_label_band(frame, [LabelLine(ascii_text, 80)])[1000:]
            )

    def test_non_latin_script_is_drawn(self):
        frame = make_frame()

        band = append_label_band(frame, [LabelLine("Ολυμπία", 80)])[1000:]

        assert band.max() > 0
        assert not np.array_equal(band, append_label_band(frame, [LabelLine("?????", 80)])[1000:])

    def test_typographic_punctuation_is_drawn_as_typed(self):
        frame = make_frame()

        band = append_label_band(frame, [LabelLine("L’Équipe", 80)])[1000:]

        assert band.max() > 0
        assert not np.array_equal(
            band, append_label_band(frame, [LabelLine("L'Equipe", 80)])[1000:]
        )

    def test_grayscale_frame_keeps_its_shape_and_dtype(self):
        frame = np.full((100, 60), 7, dtype=np.uint8)

        result = append_label_band(frame, [LabelLine("Bob", 8)])

        assert result.shape == (108, 60)
        assert result.dtype == np.uint8


class TestRowStack:
    def test_rows_are_drawn_in_the_order_given(self):
        frame = make_frame()

        result = append_label_band(frame, [LabelLine("Alice", 80), LabelLine("", 50)])

        assert result[1000:1080].any()
        assert not result[1080:].any()

    def test_second_row_text_is_smaller_than_the_first(self):
        frame = make_frame()

        stacked = append_label_band(frame, [LabelLine("Alice", 80), LabelLine("Alice", 50)])

        assert ink_height(stacked[1080:]) < ink_height(stacked[1000:1080])

    def test_adjacent_rows_read_as_one_caption(self):
        frame = make_frame()

        band = append_label_band(frame, [LabelLine("Alice", 80), LabelLine("4.704", 50)])[1000:]
        first, second = ink_span(band[:80]), ink_span(band[80:])
        gap = (80 + second[0]) - first[1] - 1

        # Wider than the second line itself would read as two separate captions
        assert 0 < gap < ink_height(band[80:])

    def test_a_single_row_is_centered_in_its_box(self):
        frame = make_frame()

        band = append_label_band(frame, [LabelLine("Alice", 80)])[1000:]
        top, bottom = ink_span(band)

        assert abs(top - (80 - bottom - 1)) <= 1

    def test_empty_first_row_still_leaves_room_for_the_second(self):
        frame = make_frame()

        result = append_label_band(frame, [LabelLine("", 80), LabelLine("4.704", 50)])

        assert not result[1000:1080].any()
        assert result[1080:].any()

    def test_rows_without_height_are_skipped(self):
        frame = make_frame()

        result = append_label_band(frame, [LabelLine("Alice", 80), LabelLine("4.704", 0)])

        assert np.array_equal(result, append_label_band(frame, [LabelLine("Alice", 80)]))


class TestBundledFont:
    def test_font_ships_with_the_package(self):
        assert resources.files("railcam").joinpath("fonts", FONT_FILE).is_file()
