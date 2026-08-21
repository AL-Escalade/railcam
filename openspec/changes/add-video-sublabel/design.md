# Design

## A band of rows, not a band with a text

`append_label_band` took one text and one height. Two lines of different sizes
could be bolted on as a second pair of parameters, but the band is really a
stack: each row has a height, and its font follows that height. Modelling it
that way makes the second line fall out of the existing rendering — the smaller
font is a consequence of the smaller row, not a separate code path — and leaves
a third row costing nothing.

So `LabelLine(text, height)` becomes the unit, and `CropPlan` carries a tuple of
them instead of a text and a band height. `frame_height` sums the rows.

## Why unlabeled videos still get every row

For the same reason the band itself is uniform: `compose_frame_row` normalizes
videos to a common height by scaling, so videos whose bands differ in height
would have their images scaled by different factors, breaking the torso
normalization. The row layout is therefore a property of the render — as soon
as any input has a sublabel, every input gets the second row, empty or not.

## Sizes

`LABEL_FONT_RATIO` (0.04) and `SUBLABEL_FONT_RATIO` (0.021) of the image height
size the two lines, so the second reads as clearly secondary while staying
legible on a phone.

The band is measured from its text rather than the text fitted into a fixed
band: padding above the first line and below the last (`BAND_PAD_RATIO`) and a
gap between lines (`LINE_GAP_RATIO`), both expressed against the line size they
sit next to. That is what makes the space above the label equal to the space
below the last line — with lines centered in fixed boxes it was not, and could
only be made so with opaque constants. The line's slot is measured from a
reference glyph rather than from the text itself, so the band height does not
depend on the words and two composed videos keep matching bands.
