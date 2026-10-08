# Measurements

What each model and each stage of the pipeline was measured to do, and the trials
that did not work out. New results go here; the scripts that produce them are listed
in [DEVELOPMENT_GUIDELINES.md](DEVELOPMENT_GUIDELINES.md) under "Testing".

## Detection models

Players and discs are detected by two separate models (the defaults in
`configs/default.yaml`). A dedicated disc model finds clearly more discs than one model
trained on both classes; for players it makes no difference. Selecting the same model in
both roles runs it once per frame instead of twice.

`scripts/benchmark_detectors.py` scores models on the validation and test images of the
default training dataset (141 images with 1,981 players and 98 discs) and times them on
video frames. A detection counts when it overlaps a labelled box of its class with
IoU ≥ 0.5; recall is at the app's confidence threshold (players 0.5, discs 0.3). All
models below were trained on the `roboflow_merged_*_v2` data; times are with TensorRT engines unless
noted.

| Model | Player AP50 | Player recall | Disc AP50 | Disc recall | Time per frame |
| --- | --- | --- | --- | --- | --- |
| YOLO26s at 1280, players only (default player model) | 0.982 | 0.977 | | | 6.5 ms |
| YOLO26s at 1280, discs only, with the frames labelled in the app, trained without mosaic (default disc model) | | | 0.556 | 0.490 | 5.1 ms |
| YOLO26s at 1280, discs only (default disc model before) | | | 0.505 | 0.449 | 5.3 ms |
| YOLO26s at 1280, both classes | 0.980 | 0.976 | 0.384 | 0.296 | 5.4 ms |
| YOLO26s-P2 at 1280, discs only (extra head for small objects) | | | 0.531 | 0.357 | 5.8 ms |
| RT-DETR-L at 960, both classes | 0.977 | 0.970 | 0.280 | 0.316 | 30.5 ms (PyTorch) |

The previous defaults (YOLO11s trained on the older, smaller datasets) scored 0.853 AP50
for players and 0.324 for discs.

A disc is about 11 pixels wide in a 1280×720 frame, so it needs the full image size and
is still the weak spot: the best model finds less than half of the discs.

A trial with the frames labelled in this app: `scripts/build_combined_disc_dataset.py`
adds them to the Roboflow disc data (`combined_discs_v1`: 166 of the 1,467 training
images and 127 of the 982 training discs are own labels), and YOLO26s was trained on it
with the settings of the default disc model.

| Scored on | Discs | Default disc model: AP50 / recall | Trained on the combined data: AP50 / recall |
| --- | --- | --- | --- |
| Old validation and test images (as the table above) | 98 | 0.505 / 0.449 | 0.559 / 0.449 |
| Old test images only | 44 | 0.522 / 0.432 | 0.588 / 0.432 |
| Own labels, validation and test | 49 | 0.629 / 0.551 | 0.640 / 0.551 |
| Own labels, test only | 25 | 0.664 / 0.640 | 0.654 / 0.560 |

AP50 rises a little on the old images, and the number of discs found at the app's
threshold stays the same; with this few discs neither is more than a hint.

With more labels (`combined_discs_v3`: 354 own frames with 260 discs among 1,655
training images) seven ways of training were compared. Each is scored on the old
validation and test images (98 discs) and on the own frames held out for validation and
test (79 discs), as AP50 / recall at the app's threshold, run in PyTorch:

| Trained on `combined_discs_v3` | Old images | Own frames |
| --- | --- | --- |
| Default disc model before (old data only) | 0.505 / 0.45 | 0.556 / 0.49 |
| YOLO26s, the settings used so far | 0.531 / 0.42 | 0.666 / 0.57 |
| YOLO26s without mosaic, size changes of 20% instead of 50% | 0.568 / 0.50 | 0.709 / 0.62 |
| YOLO26s with mixup 0.15 | 0.573 / 0.43 | 0.652 / 0.46 |
| YOLO26m | 0.494 / 0.43 | 0.694 / 0.62 |
| YOLO26s-P2 | 0.492 / 0.36 | 0.671 / 0.59 |
| YOLO11s | 0.521 / 0.43 | 0.636 / 0.57 |

Mosaic puts four shrunken images into one, which makes a disc of 11 pixels smaller still;
without it the model finds more discs on both sets, and it is the default disc model now.
As a TensorRT engine it scores 0.556 / 0.490 on the old images, 0.687 / 0.608 on the own
frames, and 0.611 / 0.542 on both together (177 discs; the model before: 0.524 / 0.469).
The own frames favour models trained on frames labelled the same way, and with this few
discs a difference of 0.03 is noise; that one way of training leads on both sets is what
counts. A larger model, the extra head for small objects, and YOLO11 bring nothing.

Neither does training on tiles of the full-resolution frames (`tiles_discs_v4`), so that
the model sees a disc at its 17 pixels instead of 11, and running it on the whole frame
at 1920: 0.688 / 0.59 on the own frames, the same as the default model at 1280, at about
twice the work per frame. On the old images, which only exist shrunk, it does worse
(0.472 / 0.33).

To train a detector for one class, build a single-class copy of a dataset with
`scripts/build_single_class_dataset.py` and select it in the Model Training tab.

The tab also lists RT-DETR (`rtdetr-l.pt`, `rtdetr-x.pt`), a transformer detector, as an
alternative to YOLO. It stretches the frame to a square and runs in PyTorch only. On this
data it matched YOLO for players, was worse for discs (a quarter of its disc detections
at the app's threshold were right), and took twice as long to train.

## Drone footage and close-ups

Edited games cut away from the drone to a camera at ground level, to title cards, and to
black. The pipeline stops following players there and starts afresh when the drone is
back (`processing/shot_type.py`, `models.shot_type`): a frame counts as drone footage if
the player model finds six players or more, and the kind changes after 15 frames in a row
say so.

- Of 600 random frames of the five edited games, the 526 with six or more players were
  all drone footage (the 36 most doubtful were looked at). Of the 74 with fewer, about 60
  were close-ups, title cards or black; the rest were drone footage with hardly anybody
  in view (a huddle, an empty field).
- Of 600 random frames of the five unedited 2024 games, 9 had fewer than six players.
- On a 50-second stretch around a close-up in each of three edited games, the analysis
  paused once and resumed once, at the cuts.
- A random frame to label is drawn again if it is no drone footage. For the field, that
  is a frame in which the field model sees no line at all: 35 of 40 close-ups, none of
  40 drone frames.

## Engines are loaded with the video

In the app, a TensorRT engine that is first loaded after frames have been analysed
crashes the process with an access violation inside TensorRT. It showed when field
segmentation was switched on only after playing, and when a video began with a close-up
so that the field model was first needed after a seek. Without the window the same
sequence runs. The cause is not known. Ruled out, each by switching it off and still
crashing, or by doing it without the app and not crashing: loading on another thread,
the thread that decodes the next frame ahead, the warm-up of the PyTorch models, the
jersey reader and its background thread, and drawing in a real window (it crashes
offscreen too). Not tried: the three OpenCV packages that share one folder in the
environment. A crash now leaves the stacks of all threads in
`data/cache/crash_traces.log`.
The engines of all three models are therefore loaded one after the other when a video is
opened (`gui/main/pipeline_worker.py`). Choosing another model from a dropdown later
still loads an engine late and has not been tried since.

## Camera motion

The motion of the picture between frames is estimated from background points followed
with optical flow (players masked out) and fitted as a homography. Compared on stretches
of a game where the camera pans, by how well the previous frame moved by the estimate
matches the current one on the background (mean grey difference, lower is better):

| Estimator | After 1 frame | After 30 frames | Time per frame |
| --- | --- | --- | --- |
| None | 9.0 | 15.8 | |
| Phase correlation (shift only) | 10.7 | 19.1 | 10 ms |
| ORB feature matching, homography | 7.7 | 16.5 | 27 ms |
| Optical flow, similarity (shift, zoom, roll) | 7.3 | 18.3 | 13 ms |
| Dense Farneback flow, homography | 3.7 | 12.4 | 22 ms |
| Optical flow, homography, frame to frame | 3.3 | 10.6 | 7 ms |
| Optical flow, homography, same points followed for 30 frames (used) | 3.5 | 8.1 | 4 ms |

Times were taken while a training run used the machine and are only comparable with each
other. `models.tracking.camera_motion_compensation: false` switches the stage off.

## Tracking and player identities

Measured on a 45-second uncut stretch with about 15 players on screen, by how many player
IDs the tracker hands out (fewer is better; players walking into the picture also get
new IDs, so the count never reaches the number of players):

| | Player IDs | Tracks lasting most of the stretch |
| --- | --- | --- |
| Track memory of 30 frames (0.5 s at 60 fps) | 32 | 7 |
| Track memory of 3 s, camera motion given to the tracker | 21 | 12 |

Fast pans still break tracks: a 22-second stretch with the fastest camera motion gave 41
IDs for 14 players.

**Players who cover each other.** The tracker (`models.tracking.backend`) is ByteTrack,
which follows a box by where it was heading, with a rule on top: it learns the two shirt
colours of the game while it runs, gives every track the team its player's shirt mostly
looked like, and never continues a track with a box that clearly wears the other colour
(`processing/team_tracker.py`). In matching a box to a track, where the feet are counts
as much as how far the boxes overlap: two players who cover each other mostly stand at
different depths. (Without that: 13 swaps, 29 changed IDs, 1.39 IDs per player.) A track cut off from the wrong player that way gets its
player back from the identity layer: a new track in the player's kit, where they can
have got to within 1.5 seconds, is that player. DeepSORT, the tracker before
(`backend: deepsort`), matches by looks, and between two players who cover each other
often continues with the wrong one.

Measured on the 21 short clips (7,500 frames, about 13 players on screen). A swap between
opponents is a track whose shirt colour changes from one team's to the other's and stays
there; the events counted for DeepSORT were checked by eye and are real. Swaps between
teammates cannot be seen this way.

| | DeepSORT | ByteTrack with the team rule |
| --- | --- | --- |
| Swaps between opponents | 69 | 10 |
| A player who is clearly the same from one frame to the next gets another ID | 23 | 27 |
| Player IDs per player on screen | 1.30 | 1.34 |

The rule and the count of swaps both rest on shirt colour, so the count favours the rule
somewhat; `data/cache/tracking_old_vs_new.mp4`, if rendered, shows both trackers side by
side. Without the team rule ByteTrack had 48 swaps, with the rule but without giving
players back their IDs 1.7 IDs per player. The other trackers that come with Ultralytics
are no better without the rule: against ByteTrack's 60 swaps and 20 changed IDs (these
with the player model at 960), OC-SORT had 67 and 41, FastTrack 57 and 13, TrackTrack 53
and 28. The measure itself is exactly repeatable.

**Observers.** The people in orange or red shirts on the field are followed like players
but left out of what is shown, read, and counted (`models.tracking.hide_non_players`): a
track whose shirt is orange or red in nearly all its clear sightings, without that being
one of the two team colours. On the 21 clips this hides 8 tracks, all of them observers,
from about half a second after they appear. A first rule without the colour, "in neither
team's colours", also hid players of teams in purple and green.

Recognising a lost player again by looks does not work on this footage. Players are about
90 pixels tall and teammates wear the same kit: a network trained to re-identify people
(OSNet) picked the right one of 16 players in 24% of cases, the tracker's own appearance
vector in 13%. Matching a new track to a missing player by where that player could have
run to (`models.tracking.identity.position_match_seconds`) did nothing for DeepSORT's
lost tracks; it is what reconnects the tracks the team rule cuts. Beyond those 1.5
seconds the jersey number remains: a player whose number is that of a missing player in
the same kit becomes that player again.

Jersey numbers are decided by vote over all readings of a player. A number is shown from
the second reading on, and its certainty grows with agreement; a single reading never
makes a number final.

## Field segmentation

The field lines are fitted to the outline of the predicted field. `scripts/benchmark_segmentation.py`
scores the segmentation model on its validation and test images (48): IoU of the whole
field and of each class, and the average distance between the predicted and the labelled
field outline in pixels of a 1920×1080 frame.

| | Field IoU | Central field IoU | End zone IoU | Outline error |
| --- | --- | --- | --- | --- |
| Default model (YOLO11s-seg) | 0.995 | 0.991 | 0.957 | 2.5 px |

The training images are 16:9 frames stretched to a square, so the app stretches each frame
the same way before segmenting it. Padding the frame to a square instead, as the app did
before, gave 0.983 field IoU, 0.892 for the end zones, and a 14 px outline error.

These labelled images are easy ones from the short clips; the full games are harder. On
200 random frames of the five games, a confidence threshold of 0.6 cut a part of the field
away (mostly the far end zone) in 38 frames, and 0.4 in 17; the scores above are the same
for both. The model has not been trained on the low side-line camera of
`raleigh_vs_portland_2024` and finds no usable field in about a quarter of its frames.

### Retrained on masks drawn from the field labels

The Roboflow outlines are from five short clips. `scripts/build_field_mask_dataset.py`
draws the field and its end zones from the field labels of whole games
(`rendered_field_v2`: 303 such frames and the 461 Roboflow images to train on, 100
frames of one game to validate, 190 frames of three other games to test). Close-ups
from edited games are added as pictures in which nothing is labelled (76 to train on,
113 of the test games to test): without them a model learns that grass is field. No
game is in two of the three parts. On the test part, at the app's settings:

| | Default (YOLO11s-seg, Roboflow) | YOLO11s-seg, labels | YOLO26s-seg, labels | YOLO26s-seg, labels and close-ups |
|---|---:|---:|---:|---:|
| Field IoU, mean | 0.921 | 0.942 | 0.945 | 0.944 |
| End zone IoU, mean | 0.500 | 0.754 | 0.709 | 0.742 |
| Outline error, median / mean | 12.4 / 25.3 px | 30.5 / 33.8 px | 15.7 / 25.7 px | 3.4 / 21.5 px |
| Share of the field not found, mean | 5.8% | 3.6% | 2.3% | 4.5% |
| Grass beside the field taken for field, mean | 2.6% | 3.7% | 3.5% | 3.1% |
| Images with more areas than labelled / fewer | 25 / 84 | 24 / 20 | 8 / 25 | 0 / 40 |
| Close-ups given a field, of 113 | 17 | 65 | 32 | 0 |

And the field estimate made from each model's masks, on the 26 hand-labelled frames of
the three test games (`scripts/benchmark_field_registration.py --games ... --model ...`):

| | Default | YOLO26s-seg, labels | YOLO26s-seg, labels and close-ups |
|---|---:|---:|---:|
| Estimate given | 10 | 12 | 13 |
| Within 5 / 10 / 20 px at the worst labelled corner | 1 / 3 / 5 | 6 / 6 / 7 | 3 / 8 / 10 |
| Worst corner of those given: median, 90% | 22 px, 215 px | 9 px, 160 px | 7 px, 24 px |
| Given and more than 2 yd off | 8 | 6 | 6 |

The model trained with close-ups gives no field on any close-up and never an area too
many, and its estimates are no longer far off (90% under 24 px, from 160). It pays for
that with areas it does not give: fewer than labelled in 40 images, mostly an end zone,
and no estimate at all in 10 of 26 frames. 26 frames and 190 images are few; the
differences between the two YOLO26s models within 10 px are noise. It is not the
default: that needs a TensorRT engine (`scripts/export_tensorrt.py`) and a look at
whole games in the app.

The line fit (RANSAC on the field outline) gives the same lines for the same mask, and
stops when what is left of the outline is shorter than a field line. Measured on 79 masks
from the games:

| | Outline explained | Difference between two runs | Junk lines | Time |
| --- | --- | --- | --- | --- |
| Before (20 random pairs per line, always 4 lines) | 96.2% | 2.0 px (worst tenth 5.5 px) | 6.6% | 4.8 ms |
| Now (100 pairs at once, fixed seed, refined fit) | 96.7% | 0 px | 5.7% | 5.6 ms |

The lines shown are filtered over time (`processing/field_line_filter.py`,
`models.segmentation.contour.ransac.smooth_lines`): between two fits they are moved with
the camera, a new fit is blended in by 40%, a line that was not shown before appears
once two fits in a row have it, and a line one fit misses is kept once. How far a shown
line moves against the field from one frame to the next:

| | Before | Filtered |
| --- | --- | --- |
| Seven short clips: on average / in the worst twentieth of the frames | 0.41 / 1.7 px | 0.10 / 0.2 px |
| The same: frames in which the number of lines changes | 2.4% | 1.0% |
| A stretch of a 60 fps game: on average | 0.61 px | 0.28 px |
| The same: frames in which the number of lines changes | 0.4% | 2.2% |

In about one frame in a hundred a line still jumps by more than 5 pixels, by up to 60:
the fit has then found another line than before, which no blending covers.

## Where the field lies

`processing/field_registration.py` estimates the mapping between a frame and the field
without calibration by hand. The field model marks the central field and the end zones:
the outline gives the sidelines and the far back line, and where an end zone meets the
central field is a goal line. A camera is then placed so that the field's lines lie on
these (`utils/field_camera.py`).

It is a camera and not a free mapping because lines at the far end alone leave a free
mapping open on how far the field reaches towards the viewer. The games are filmed by a
drone that flies along the field about 9 yards up and does not zoom, so the focal length
is the same for a whole video (the labels of a game give it to within 2 to 3 percent:
1691 to 1748 pixels for six frames of one game). With the focal length known, labelled
corners two pixels off put the near goal line some 20 pixels off instead of 70 to 170.
The main tab learns the focal length from the frames as they play; the labelling takes it
from the frames of the video labelled so far.

In the main tab, "Top-down from the field model" (the default) makes the top-down view
from this estimate instead of the calibration (`homography.source`). The lines fitted to
the field's outline are then not drawn: the view is not made from them. The field is estimated each
time the field model runs, moved with the camera in between, and evened out; the lines of
the field are drawn on the view. It costs about 3 ms per frame on average (see the
speed of the whole pipeline).

An estimate is checked against what else the frame shows before it is used
(`implausible`): it is left out if fewer than 90% of the detected players stand on the
field as estimated, or if the field as estimated and the field the model sees share less
than 85% of what either covers. An estimate more than 8 yards from where the field was
followed to must be given twice in a row before it replaces it. The main tab takes the
focal length from the video's field labels where it has any.

Each new estimate goes into the followed field by a fifth (`NEW_ESTIMATE_WEIGHT`); it was
two fifths. Replayed on 40 seconds each of three games (1,220 estimates), how far the
field moves in the view when an estimate comes in, and how far the followed field was
from that estimate, both in yards at the worst of four places in the picture:

| Weight of a new estimate | Moves by: median / 90% | Was off the estimate by: median / 90% |
|---|---:|---:|
| 0.4 (before) | 0.04 / 0.27 | 0.11 / 0.69 |
| 0.3 | 0.03 / 0.23 | 0.12 / 0.77 |
| 0.2 (now) | 0.03 / 0.18 | 0.13 / 0.90 |
| 0.1 | 0.02 / 0.13 | 0.16 / 1.33 |

The field was steady before; the larger moves are a third smaller now. Whether the
followed field or the single estimate is nearer the truth is not known from this.

Against the 63 labelled frames of ten games (`scripts/benchmark_field_registration.py`),
measured only at the corners that were put on the picture by hand, with the focal length
of the game's other labels:

| | Frames |
|---|---:|
| Labelled | 63 |
| No estimate (a sideline or the far end not found, or the lines do not agree) | 22 |
| Left out by the check; all four were more than 2 yd off | 4 |
| Estimate given | 37 |
| ... within 5 / 10 / 20 px at the worst labelled corner | 7 / 18 / 30 |
| ... more than 2 yd off at the worst labelled corner | 21 |

Of the estimates given, the worst corner is off by 10.3 px or 2.2 yd at the median, and
by 38 px or 7.7 yd at the 90th percentile. At the far end, where nearly all labelled
corners are, a pixel is about half a yard along the field, so a few pixels are yards.

What this says:

- The estimate from the masks is a rough one. Under a third of the frames come out within
  10 px. It serves as a start for labelling and for a top-down view that shows where
  players are roughly; it is not a measurement of positions.
- The check removes the estimates that are far off (four, none of them usable) and none
  of the good ones. It cannot tell an estimate that is some yards off: on those, the
  players stand on the field and the masks agree. Neither the lines' misfit nor where
  the camera comes to stand separates them either (tried on these frames).
- An earlier table here gave 0.54 yd at the median. It compared with the label refitted
  as a camera of the same focal length the estimate used, over the whole field, and only
  on frames with an estimate; a focal length that was off cancelled out.
- The thresholds of the check were chosen on these same frames.
- The labels themselves: a camera of the game's focal length fits the corners in a frame
  to under 2 px. What a label says beyond its corners follows from the camera and is not
  measured; two labels 395 frames apart that agree at the far corners differ by tens of
  pixels near the camera (see `docs/REVIEW_2026-10-08.md`).

## Painted lines as a labelling aid

The field labelling shows thin white streaks found in the picture
(`utils/painted_lines.py`). Against the lines of the 76 labelled field frames: the share
of a frame's labelled lines with a found pixel within 4 px, and what is marked more than
8 px from any of them (other fields' lines are among that, so not all of it is clutter).

| Contrast needed / shortest piece | Labelled lines found | Marked on players | Marked elsewhere |
|---|---:|---:|---:|
| 3.5 / 70 px, as before | 40% | 460 px | 13,300 px |
| 3.5 / 70 px, players left out | 40% | 0 | 13,300 px |
| 3.0 / 70 px, players left out (now) | 47% | 0 | 22,100 px |
| 2.7 / 50 px, players left out | 56% | 0 | 40,300 px |
| 2.4 / 45 px, players left out | 64% | 0 | 63,300 px |

Pieces that lie mostly in the boxes of detected players are dropped, and the boxes are
blanked, so a line has a gap where a player stands.

## Jersey number readers

The reader is chosen with "Jersey Number Reader" in the Main Analysis tab or
`models.player_id.method`. Measured on 1,016 hand-labelled crops of 23 players from four
games, plus crops of 14 players with no visible number:

| Reader | Players identified | Wrong | Correct / wrong reads per crop | Time per crop |
| --- | --- | --- | --- | --- |
| `parseq` (default) | 17 of 23 | 0 | 18.9% / 0.3% | 8 ms |
| `florence` | 19 of 23 | 1 | 21.3% / 1.6% | 38 ms |
| `easyocr` | 11 of 23 | 4 | 7.6% / 3.7% | 9 ms |
| `yolo_digits` | not measured: no digit model trained yet | | | |

Most crops show a player from the front or side, so a low per-crop rate is expected; what
counts is that the reads that do come are right. None of the readers reported a number
for the players without one.

- `parseq` runs the PARSeq recognizer on the regions EasyOCR's text detector finds. It is
  downloaded through `torch.hub` on first use.
- `florence` is the Florence-2 vision-language model. It answers for every crop, so reads
  below `models.player_id.florence.min_confidence` (0.7) are discarded.
- `yolo_digits` uses the newest detection run trained on a dataset with "digits" in its
  name (train one in the Model Training tab; `roboflow_digits_v1i` is house numbers, a rough
  starting point). Until one exists the app falls back to EasyOCR, as it does for any
  reader that fails to load.

## Reading only players seen from behind

The number is on the back. A pose model (`yolo11n-pose.pt`) tells which way a player
faces from the order of the shoulders (`processing/facing.py`): of the 1,532 labelled
crops it sees 46% from behind, and those hold 94% of the numbers the reader gets right.
Reading only those (`models.player_id.only_from_behind`) halves the reader's work per
batch of crops, from 35 to 19 ms, but the pose model takes 14 ms a batch itself, most of
it Python that holds up the analysis of the frames: the pipeline ran at 36 instead of 38
frames per second and found the same numbers. So it is off. It would pay with a model
that tells front from back in a millisecond or two, which could be trained on what the
pose model says.

## Jersey reading schedule

Between two readings of a player the app keeps that player's best crop: the sharpest
upper body at the largest size with the least overlap with other players. That crop is
read instead of whatever the frame of the reading happens to show. A player whose crop
could not be read is tried again after two, then four reading intervals; a successful
reading brings back the normal rhythm. Players with a final number are skipped as before.
`models.player_id.crop_selection.enabled: false` gives the previous fixed schedule.

`scripts/benchmark_player_id_scheduling.py` replays the labelled jersey crops through the
voting of the live app (five clips, 23 players with a number, 14 without):

| Schedule | Players right / wrong / unread | Numbers given to players without one | Crops read |
| --- | --- | --- | --- |
| Fixed frame | 12 / 0 / 11 | 0 | 592 |
| Best recent crop, waiting longer after unreadable ones (default) | 12 / 0 / 11 | 0 | 229 |

The same numbers are found with 61% fewer readings. The benchmark replays stored crops, so
it does not show how tracking errors or real overlap between players affect the result.

## Field calibration search

The genetic search of the Field Calibration tab warps the frame once per candidate to see
how much of the top-down view it fills. It now warps a grey image at half size
(`optimization.ga_coverage_scale`; 1.0 is full size). `scripts/benchmark_homography_optimizer.py`,
on 20 frames from five games with 20 candidates each:

| Warped image | Time per generation of 20 | Same best candidate as before |
| --- | --- | --- |
| Colour, full size (before) | 142 ms | |
| Grey, full size | 99 ms | 20 of 20 |
| Grey, half size (default) | 28 ms | 20 of 20 |

## Speed of the whole pipeline

Everything switched on (detection, tracking, jersey numbers, field, both views), 180
frames of the default clip after 30 to warm up, decoding and display not counted:

| | Frames per second | Tracking | Detection |
| --- | --- | --- | --- |
| DeepSORT, player model at 1280 | 23 | 12 ms | 7.5 ms |
| ByteTrack with the team rule, player model at 1280 (default) | 28 | 2.4 ms | 7.8 ms |
| ByteTrack with the team rule, player model at 960 | 30 | 2.3 ms | 6.2 ms |

Those rows were measured with the top-down view from the calibration. With the view from
the field model, now the default, the same run gave 24 frames per second: blacking out
what lies behind the camera tested every pixel of the view (3.4 ms a frame), and the
field fit asked for its misfits element by element (another 3.4 ms). The first is now a
polygon fill and the second works on arrays; the fits agree to a ten-thousandth of a
yard and the tracks of the run are the same.

| Top-down view from the field model | Frames per second | Field estimate | Warping the view |
| --- | --- | --- | --- |
| Before | 24 | 4.8 ms | 8.7 ms |
| Now | 31 | 3.0 ms | 1.8 ms |

The estimate runs on every fifth frame, so it costs about 15 ms there; the slowest
twentieth of the frames takes over 70 ms. What possession, the sideline filter, the
trails and the team colours add is 0.15 ms a frame together.

ByteTrack runs no network of its own. The player model is trained at 1280 like the disc
model, and players are 90 pixels tall and found as well at 960
(`models.player_detection.image_size`): AP50 0.985 and recall 0.973, against 0.982 and
0.977 at 1280, in 4.6 instead of 6.1 ms. It stays at 1280 all the same, because the
tracker does worse with the boxes found at 960: on the 21 clips, before feet were used in
matching, 15 instead of 13 swaps between opponents, and 40 instead of 29 players who are clearly the same from one frame
to the next getting another ID. Whether a player is found says nothing about how steady
the box is. Runs of the speed measurement differ by about one frame per second.

Reading jersey numbers takes 8 ms per frame on average, in bursts of 25 ms and more on
the frames it runs on, most of it in the text detector that finds the number on the crop.
Running that detector on smaller crops loses players (17 of 23 identified at the present
size, 16 at three quarters, 15 at half); one run for all crops of a frame and half
precision gain under a tenth. So the reading itself is as it was, but no frame waits for
it any more (`models.player_id.background_reading`): a second thread reads up to four
crops at a time and the numbers are taken into the vote when they are ready, a frame or
two later. Over the whole default clip (380 frames after 30 to warm up):

| Jersey reading | Frames per second | Slowest twentieth of the frames takes over |
| --- | --- | --- |
| Each frame waits for it | 26 to 27 | 65 to 71 ms |
| In the background (default) | 31 to 34 | 46 to 50 ms |

The same six numbers were found, the first at the same frame, in two runs of three; the
third found five. With the reading in the background a run is no longer exactly
repeatable; set the setting to false to compare two versions of the code frame by frame.

## Measuring the whole pipeline

`scripts/benchmark_pipeline.py` runs detection, tracking, jersey reading, segmentation and
both views on frames decoded beforehand, reports the time per stage, and can store the
tracks, numbers and disc holder of every frame to compare a change against:

```bash
python scripts/benchmark_pipeline.py --output before.json
python scripts/benchmark_pipeline.py --compare before.json
```

## TensorRT engines (optional)

The detection and field segmentation models run about three times faster as TensorRT
engines. An engine is tied to one model, the video frame size, and this GPU and driver.

```bash
python scripts/export_tensorrt.py                 # default models, 1920x1080 video
python scripts/export_tensorrt.py path/to/weights/best.pt
```

Engines are stored next to the weights (`best.384x640.fp16.engine`). On the next start
the app uses an engine when one matches the model and frame size, and PyTorch otherwise;
`models.inference.tensorrt: false` switches engines off. Rebuild after a GPU driver or
TensorRT update.

Setting up a new environment for exporting needs care. The export uses `tensorrt-cu12`,
`onnx`, `onnxslim`, and `nvidia-modelopt`. Installing `nvidia-modelopt` with its
dependencies replaces the CUDA build of PyTorch and upgrades setuptools past the version
DeepSORT works with, after which the app silently falls back to the simple tracker.
Restore both afterwards:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128 --no-deps
python -m pip install setuptools==70.2.0
```

The export script disables Ultralytics' automatic package installation for this reason.

## Profiling

Run `python scripts/profile_app.py`, use the application, and close it; then run
`python scripts/view_profile.py` to open the result in snakeviz. Both use
`profile_output.prof` in the repository root. The profile covers the GUI thread; the
Performance panel in the Main Analysis tab shows the time per pipeline stage.


## Behaviour worth knowing when changing the pipeline

- Redraws of the current frame reuse its processing results. Seeking or switching
  videos clears tracking, OCR identities, and cached segmentation.
- Field segmentation runs every few frames; the mask, contour, and line fit are
  cached until the next run.
- Selecting the same model for players and discs runs it once per frame and splits its
  detections by class. A model only reports the classes it is selected for.
- Possession goes to the player whose box contains the detected disc. The holder only
  changes after the disc has been seen at another player, or at no player, for
  `models.possession.confirm_seconds` without a break (a third of a second), so a disc
  flying past someone does not change it. Frames without a detected disc leave the
  holder as it is.
- The mark stands right in front of the holder and often covers them. A disc in the
  holder's box, or at no player within arm's reach of it, stays the holder's however
  near it is to someone else; a holder who is not found keeps the disc where they were
  last seen (`holder_memory_seconds`); and a player whose box touches the holder's must
  have the disc for `beside_holder_seconds` before they are taken for the holder. A
  player of the team that does not have the disc needs `other_team_seconds`.
- Being sure of a new holder takes a moment, but the change is dated back to the frame
  the disc was first seen there (`FrameResult.possession_since`); the possession bar
  under the video and the strip of the demo are put right back to that frame. The box
  around the holder on the picture cannot be: it appears when the holder is confirmed.
- Live playback skips frames when the analysis is slower than the video. A frame then
  counts for the frames skipped before it in possession, and a trail holds as many
  points as cover `models.tracking.trail_seconds`: with a fixed number of points the
  trails reached back most of a minute at 6 frames per second.
- The colour a team is shown in is its average shirt colour made vivid. It is taken
  from the shirts without the grass that the middle of a box also shows (the colours
  the tracker matches by have the grass in them), and is held once 400 shirts of the
  team have been seen. A holder's jersey number, once read, is written on their
  stretch of the possession bar.
- With the top-down view from the field model, a track whose feet are more than 2 yards
  outside the field as estimated on 85% of its recent sightings is left out
  (`models.tracking.hide_off_field`): the players standing along the sidelines. A
  player who steps out of bounds stays. Not measured against labels: there are none of
  who is in the point.
- A disc that lies still at no player for `ground_seconds` is on the ground: a turnover,
  and the other team is in possession from then on. Something white and round on the
  grass that the disc model takes for a disc (a brick mark) looks the same; it counts
  once at most until a player has the disc again, and that player's team then decides.
- There is no ground truth for possession. Replayed on 80 seconds each of three games
  (what the possession tracker was fed, recorded once), the logic before and now:

  | Stretch | Holder changes | Taken back within 2 s | Holders for under 1 s | Team changes now |
  |---|---:|---:|---:|---:|
  | San Francisco v Colorado, 24:27 | 20 → 11 | 2 → 0 | 4 → 0 | 0 |
  | Chicago v New York, 10:00 | 13 → 10 | 0 → 0 | 0 → 0 | 3 |
  | Portland v San Francisco, 11:00 | 21 → 9 | 0 → 0 | 1 → 1 | 2 |

  A player is named as holder for less of the time (72% → 45%, 51% → 46%, 58% → 38%):
  the disc is found in about half the frames, and where it is seen only now and then
  (between points, when players walk to the line) the logic before named someone from a
  few frames. Fewer changes is what was aimed at; whether each holder is the right one
  has not been checked against labels.
- The disc model is skipped after a stretch with no disc and retried periodically
  (`models.disc_detection.skip_threshold`: 30 frames; `retry_interval`: 5 frames).
