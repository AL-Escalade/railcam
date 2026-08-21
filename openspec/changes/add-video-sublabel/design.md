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

The second row is `SUBLABEL_BAND_RATIO` (0.05) of the image height against 0.08
for the first. Both draw their text at the same fraction of their own row, so
the second line lands at roughly 60% of the first — clearly secondary, still
legible on a phone.
