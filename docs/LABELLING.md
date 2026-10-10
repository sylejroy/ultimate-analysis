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

While a corner is dragged, a magnified piece of the picture is shown beside it. The
painted lines found in the picture are marked in yellow ("Show the painted lines
found"): thin white streaks on the grass, with what lies on a detected player left out,
so a line has a gap where a player stands. A dragged corner snaps so that its line to
a placed corner lies on a long painted line nearby; Shift switches that off.

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

## Same player?

The "Same player?" page of the Labelling tab shows two tracks of a dataset of player
crops (`reid_players_v2`), eight pictures of each, and takes one key per answer:

- `1` same player, `2` different players
- `Space` cannot tell

A picture that shows somebody else than the rest of its track is clicked to take it out
(click again to put it back); the answer is then about the pictures left.

The two tracks are of the same game and the same kind of shirt. Seven pairs in ten are
ones the network that tells players apart is unsure of, the rest are drawn at random;
which it is stands above the pictures. (For the first minute after a dataset is first
opened all are drawn at random: the network is looking at the tracks.) Answers are added to
`pairs.csv` in the dataset named on the page (`labelled_pairs_v1`); a pair is asked once.
Every picture shown is added to `pictures.csv` beside it, as the track's player or as
somebody else.

## Holding the disc?

The "Holding the disc?" page shows a player of a frame of a holder dataset
(`holder_sighted_v1`): on the left the player made larger, on the right where they
stand. `Y` holds the disc, `N` does not, `Space` cannot tell.

Seven frames in ten are ones in which another player stands in the holder's box (the mark
in front of the thrower, a defender at the catch), the rest are drawn at random; which
it is stands above the pictures. The player the dataset takes for the holder is asked
about first. A yes also writes
every other player of that frame down as not holding it, since one disc is in play, and
goes on to another frame; after a no the next player of the frame is asked about.
The last ten answers stand in a row below the pictures, framed green for a yes and red
for a no. A click on one takes that answer back (it is removed from the file, with what
it said of the other players of the frame) and asks about that player again.
Where the box shows two players (the thrower and the mark in one box, mostly), press
T: neither a yes nor a no is true of it. The page then waits for a click on the one who
holds the disc, in the left picture; Space says that neither does, or that it cannot be
told. Such a box is written to `two_in_box.csv` beside the answers, with the place
clicked, and is not learned from. A place clicked makes every player whose box it is
not in a no.

Answers given for wide boxes before this could be said are asked about again first,
the widest first, with what was answered then. A new answer replaces the old one (the
answers as they were are kept once as `holders.csv.before_recheck`); Space leaves it.

A bar at the bottom counts the holders confirmed so far towards a first goal of 1,000.

Above the pictures stands what a holder network makes of the player ("87% sure that this
player holds the disc"): the newest network, or the one chosen beside it; "Off" to answer
without being led. A network taught on these answers knows the players already answered.

### Click the holder

The "Click the holder" page asks about the same frames another way. It plays the second
of the game before the frame, stops on the frame with a box around every player the
player model finds, and takes one click. The whole frame is shown, always: that nobody
holds the disc can only be said of a picture that shows everybody.

- a click on the player who holds the disc: a yes for them and a no for the others
- a right-click on a box that shows two players, one of them with the disc: the box
  opens large in a window of its own, for a click on that one (Space: cannot tell which
  of them; Esc: not two players after all)
- N: nobody holds it; Space: cannot tell; R: play the second again; Backspace: take
  the last answer back

The player a holder network takes for the holder is marked in light blue, with how
sure it is (a thin line under one half: its best guess only): by the newest network, or the one chosen above
the picture; "Off" to answer without being led.

What was stored shows on the frame for a moment before the next one comes: the box
clicked in green as the holder (yellow as a box of two), and in red everybody who does
not hold the disc: all players found where nobody holds it, and all whose box does not
touch the holder's where somebody does. Those become crops to learn from when the
dataset is made, not only the four the source dataset has.

About one frame in three is one in which the disc was found flying: N there, if nobody
holds it. These are looked for beforehand with `scripts/find_flying_disc_frames.py`
(the dataset's own "the disc flies" is right for a third of its frames). Without such frames a network never
learns that there may be no holder, and names one all the same. Tick "Only show frames with a
flying disc" to be asked about nothing else: the frame is then shown at once, without
the second before it. A yellow ring marks where the disc was found flying.

The games are asked about in equal measure: a game is drawn first, then a frame of it.

It writes what the other page writes, to the same files. The dataset knows four players
of a frame at most: a click on a player it does not know makes those four a no, and the
box clicked is kept in `clicked_holders.csv`.

Answers are added to `holders.csv` in the dataset named on the page
(`labelled_holders_v1`). The first picture takes some seconds: the network that enlarges
the players is loaded then.
