"""Tests for the label band rendering module."""

from importlib import resources

import numpy as np

from railcam.labeling import (
    FONT_FILE,
    LABEL_FONT_RATIO,
    SUBLABEL_FONT_RATIO,
    USABLE_WIDTH_RATIO,
    LabelLine,
    append_label_band,
    band_height,
    font_size,
)

IMAGE_HEIGHT = 1000
BIG, SMALL = 60, 32


def make_frame(height=IMAGE_HEIGHT, width=600, value=50):
    return np.full((height, width, 3), value, dtype=np.uint8)


def band_of(frame, *lines):
    """The band alone, without the image it was appended to."""
    return append_label_band(frame, list(lines))[frame.shape[0] :]


def ink_span(band):
    """(first, last) row index holding drawn pixels."""
    rows = np.where(band.max(axis=(1, 2)) > 0)[0]
    return (int(rows.min()), int(rows.max())) if rows.size else (0, 0)


def ink_height(band):
    rows = np.where(band.max(axis=(1, 2)) > 0)[0]
    return rows.max() - rows.min() + 1 if rows.size else 0


class TestFontSize:
    def test_proportional_to_image_height(self):
        assert font_size(1000) == 40
        assert font_size(500) == 20

    def test_zero_ratio_gives_no_line(self):
        assert font_size(1000, 0) == 0

    def test_non_positive_image_height_gives_no_line(self):
        assert font_size(0) == 0
        assert font_size(-10) == 0

    def test_tiny_image_still_gets_a_visible_line(self):
        assert font_size(10) == 1

    def test_default_ratio_is_the_constant(self):
        assert font_size(1000) == font_size(1000, LABEL_FONT_RATIO)

    def test_sublabel_text_is_smaller_than_the_label(self):
        assert 0 < font_size(1000, SUBLABEL_FONT_RATIO) < font_size(1000, LABEL_FONT_RATIO)


class TestBandHeight:
    def test_no_lines_means_no_band(self):
        assert band_height([]) == 0

    def test_lines_without_a_size_mean_no_band(self):
        assert band_height([LabelLine("Alice", 0)]) == 0

    def test_is_even(self):
        for size in range(8, 80):
            assert band_height([LabelLine("Alice", size)]) % 2 == 0

    def test_grows_with_the_text_size(self):
        assert band_height([LabelLine("Alice", SMALL)]) < band_height([LabelLine("Alice", BIG)])

    def test_second_line_makes_the_band_taller(self):
        one = band_height([LabelLine("Alice", BIG)])
        two = band_height([LabelLine("Alice", BIG), LabelLine("4.704", SMALL)])

        assert two > one

    def test_does_not_depend_on_the_text(self):
        """Videos composed side by side must get bands of the same height."""
        assert band_height([LabelLine("Alice", BIG)]) == band_height([LabelLine("Bob", BIG)])
        assert band_height([LabelLine("", BIG)]) == band_height([LabelLine("Alice", BIG)])


class TestAppendLabelBand:
    def test_no_lines_returns_frame_unchanged(self):
        frame = make_frame()

        assert append_label_band(frame, []) is frame

    def test_non_positive_sizes_return_frame_unchanged(self):
        frame = make_frame()

        assert append_label_band(frame, [LabelLine("Alice", 0)]) is frame
        assert append_label_band(frame, [LabelLine("Alice", -4)]) is frame

    def test_output_height_grows_by_the_band(self):
        frame = make_frame()
        lines = [LabelLine("Alice", BIG)]

        result = append_label_band(frame, lines)

        assert result.shape == (IMAGE_HEIGHT + band_height(lines), 600, 3)

    def test_output_height_grows_with_the_second_line(self):
        frame = make_frame()
        lines = [LabelLine("Alice", BIG), LabelLine("4.704", SMALL)]

        result = append_label_band(frame, lines)

        assert result.shape == (IMAGE_HEIGHT + band_height(lines), 600, 3)

    def test_image_part_is_untouched(self):
        frame = make_frame()

        result = append_label_band(frame, [LabelLine("Alice", BIG)])

        assert np.array_equal(result[:IMAGE_HEIGHT], frame)
        assert np.array_equal(frame, make_frame())

    def test_empty_label_yields_a_uniform_band(self):
        frame = make_frame()
        lines = [LabelLine("", BIG)]

        band = band_of(frame, *lines)

        assert band.shape == (band_height(lines), 600, 3)
        assert not band.any()

    def test_blank_label_yields_a_uniform_band(self):
        frame = make_frame()

        assert not band_of(frame, LabelLine("   ", BIG)).any()

    def test_text_is_drawn_and_horizontally_centered(self):
        frame = make_frame()

        band = band_of(frame, LabelLine("Alice", BIG))
        columns = np.where(band.max(axis=(0, 2)) > 0)[0]

        assert columns.size > 0
        center = (columns.min() + columns.max()) / 2
        assert abs(center - 600 / 2) <= 6

    def test_text_stays_inside_the_band_vertically(self):
        frame = make_frame()

        # Ascenders and descenders, so a correctly placed render leaves a blank
        # row on both sides instead of being clipped by the band edges
        band = band_of(frame, LabelLine("Alice gjpqy", BIG))
        top, bottom = ink_span(band)

        assert top >= 1
        assert bottom <= band.shape[0] - 2

    def test_long_label_is_shrunk_to_fit_the_width(self):
        frame = make_frame()
        label = "A very long climber label that would never fit at the nominal size"

        band = band_of(frame, LabelLine(label, BIG))
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
            band = band_of(frame, LabelLine(accented, BIG))

            assert band.max() > 0
            assert not np.array_equal(band, band_of(frame, LabelLine(ascii_text, BIG)))

    def test_non_latin_script_is_drawn(self):
        frame = make_frame()

        band = band_of(frame, LabelLine("Ολυμπία", BIG))

        assert band.max() > 0
        assert not np.array_equal(band, band_of(frame, LabelLine("?????", BIG)))

    def test_typographic_punctuation_is_drawn_as_typed(self):
        frame = make_frame()

        band = band_of(frame, LabelLine("L’Équipe", BIG))

        assert band.max() > 0
        assert not np.array_equal(band, band_of(frame, LabelLine("L'Equipe", BIG)))

    def test_grayscale_frame_keeps_its_shape_and_dtype(self):
        frame = np.full((100, 60), 7, dtype=np.uint8)
        lines = [LabelLine("Bob", 8)]

        result = append_label_band(frame, lines)

        assert result.shape == (100 + band_height(lines), 60)
        assert result.dtype == np.uint8


class TestLineStack:
    def test_lines_are_drawn_in_the_order_given(self):
        frame = make_frame()

        first = ink_span(band_of(frame, LabelLine("Alice", BIG), LabelLine("", SMALL)))
        second = ink_span(band_of(frame, LabelLine("", BIG), LabelLine("Alice", SMALL)))

        assert first[1] < second[0]

    def test_second_line_text_is_smaller_than_the_first(self):
        frame = make_frame()

        first = band_of(frame, LabelLine("Alice", BIG), LabelLine("", SMALL))
        second = band_of(frame, LabelLine("", BIG), LabelLine("Alice", SMALL))

        assert ink_height(second) < ink_height(first)

    def test_adjacent_lines_read_as_one_caption(self):
        frame = make_frame()

        first_bottom = ink_span(band_of(frame, LabelLine("Alice", BIG), LabelLine("", SMALL)))[1]
        second_alone = band_of(frame, LabelLine("", BIG), LabelLine("4.704", SMALL))
        gap = ink_span(second_alone)[0] - first_bottom - 1

        # A gap wider than the second line itself reads as two captions
        assert 0 < gap < ink_height(second_alone)

    def test_padding_above_the_first_line_matches_the_padding_below_the_last(self):
        frame = make_frame()

        band = band_of(frame, LabelLine("Alice gjpqy", BIG), LabelLine("Alice gjpqy", SMALL))
        top, bottom = ink_span(band)

        assert abs(top - (band.shape[0] - bottom - 1)) <= 2

    def test_empty_first_line_still_leaves_room_for_the_second(self):
        frame = make_frame()

        band = band_of(frame, LabelLine("", BIG), LabelLine("4.704", SMALL))

        assert band.any()
        assert ink_span(band)[0] > band.shape[0] / 3

    def test_lines_without_a_size_are_skipped(self):
        frame = make_frame()

        result = append_label_band(frame, [LabelLine("Alice", BIG), LabelLine("4.704", 0)])

        assert np.array_equal(result, append_label_band(frame, [LabelLine("Alice", BIG)]))


class TestBundledFont:
    def test_font_ships_with_the_package(self):
        assert resources.files("railcam").joinpath("fonts", FONT_FILE).is_file()
