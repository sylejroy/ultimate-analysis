# Labelling

The Labelling tab builds a dataset from your own videos at full resolution.

1. Pick a video and a frame, or let "Random frame" pick: it favours the videos with few
   labels for their length, so the labels spread evenly over the footage. The boxes the
   current default models find are shown dashed, as suggestions: at most 14 players and
   one disc, the ones the models are surest of, without the observers.
2. Correct them: drag a box to move it, drag a grip to resize it, press Delete to remove
   it. Drag on free space to draw a new one: with the left button a disc, with the right
   button a player (1 and 2 turn the selected box into one or the other). Zoom with the
   mouse wheel, move the view with the middle button. The number of players and discs on
   the frame is shown large: green at 14 and 1, red above.
3. Press Enter to save the frame and move on by the step size, or to another random
   frame. Left and Right step without saving, so frames you skip do not end up in the
   dataset.

Saved frames go to `data/raw/training_data/<dataset name>` as images and YOLO labels,
named after the video and frame number. Each stretch of 300 frames belongs as a whole to
training, validation (one in ten), or testing (one in ten), so near-identical frames never
land on both sides. The dataset is listed in the Model Training tab as soon as it has
frames.

## Labelling the field

Under "Field lines" in the Labelling tab the whole field is drawn over the frame in
perspective, and that drawing is pulled onto the real field:

- drag a corner dot (the four outer corners and where the goal lines meet the sidelines):
  the corner moves. Only corners placed in this frame stay where they were put (drawn
  filled); the rest of the field follows the way the video's camera would see it, so
  with the camera's focal length known three corners are usually enough. From the fourth
  corner on, the three placed last stay
- mouse wheel: zoom, also out past the frame, to reach corners outside it
- drag anywhere else: the picture moves

The drawing always remains a view of a flat field of the right proportions
(USA Ultimate or WFDF, `utils/field_template.py`). A label is right when the drawing lies
on the real lines. The near end of the field may lie behind the camera, as with a drone
above the end zone; it is then not drawn. The two sidelines and the far back line alone do not fix how far the
field reaches towards the camera: a line across the field at a known distance must fit as
well, best the near goal line or a brick mark, since the far goal line is only some 40
pixels from the back line.

A frame starts from the field of the frame just left (moved the way the camera moved),
else from where the field model sees the field ("Start from the field model", see
[Where the field lies](MEASUREMENTS.md#where-the-field-lies)), else from the nearest labelled frame of the video, else from a
first guess. "Place from the field model" puts the drawing there again, and the lines the
field model sees are drawn faintly under it ("Show the field model's lines"). Frames go to
`labelled_field_v1` (`utils/field_label_files.py`): the picture, the lines and corners of
the drawing that lie in or near it, and the mapping from picture to field they give. Each
saved frame is a verified calibration; together they are what a model that finds the field
by itself would be trained on and measured against.

## Labelling discs from a phone

`python scripts/phone_labelling.py` starts a small web server on this PC and prints an
address to open in the phone's browser. The page shows one frame of a game at a time:

- "Is this the disc?" with the disc model's suggestion: Yes, No, or move the box.
- Without a suggestion, or after No: tap the disc on the frame, tap it again on a closer
  view, drag the box onto it. Or "No disc visible", which stores the frame as an example
  of what is not a disc.

Frames come from all games in `data/raw/videos`, favouring those the model is unsure
about. Answers go to the disc-only dataset `labelled_discs_v1`, at full resolution
and with the same file layout as the Labelling tab. The PC has to stay on.

The address contains a key, kept in `data/cache/phone_labelling.key`; requests without
it are refused. On the home network the printed address works as it is (allow Python
through the Windows firewall for private networks when asked). From anywhere else, install
Tailscale on the PC and the phone and sign in with the same account; the script then
prints a second address that works over that private link. Do not forward the port on
your router.
