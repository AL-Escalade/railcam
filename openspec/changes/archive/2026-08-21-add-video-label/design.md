# Design

## Where the band is drawn

The band is appended to the cropped frame inside `_crop_one_frame`, after the
debug overlay and before the optional output scaling. Drawing before scaling
keeps the band a fixed fraction of the image at every output size, and keeps a
single place where a frame's final geometry is decided.

`CropPlan` keeps `output_height` as the *image* height and gains
`label_band_height`, with `frame_height = output_height + label_band_height`
used everywhere the emitted frame size matters (`final_size`, the composition
row height, the size probes). Splitting the two avoids re-deriving the band
height from a total that has already been rounded.

## Why unlabeled videos still get a band

`compose_frame_row` normalizes every video to a common height by scaling. If
one video carried a band and another did not, their images would be scaled by
different factors and the torso normalization the whole tool is built around
would silently break. So the band height is a property of the render, not of
the individual label: when any input has a label, every input gets a band of
`LABEL_BAND_RATIO` of its own image height. Proportional heights survive the
normalization, since image and band scale together.

## Text rendering

Pillow draws the text with `DejaVuSans-Bold.ttf`, bundled under
`railcam/fonts/`: the Hershey fonts `cv2.putText` offers cover ASCII only and
would render `Léa` as `L??a`, which rules them out for climber names. Bundling
the font rather than resolving a system one keeps a render identical on every
machine. The text is drawn into an 8-bit coverage mask that is then blended
onto the band, which keeps the antialiased edges and works whatever the frame's
channel count.

The font size is derived from the band height, then reduced if the text is
wider than the usable width, so a long label shrinks instead of being clipped.
Centering uses the drawn pixels rather than the font's line box, so a label
without descenders is not pushed visually upwards.

The band is drawn once per frame rather than once per render: the cropped frame
is a fresh array per frame anyway, and caching a rendered band would have to be
keyed on a geometry that never varies within a render — no gain worth the state.

## CLI pairing

Labels are a separate repeatable option rather than a fifth field of
`path:start:end:climber`: labels are free text that may contain colons and
spaces, which the spec parser matches on. Pairing is positional, mirroring how
`--input` order already drives the left-to-right layout.
