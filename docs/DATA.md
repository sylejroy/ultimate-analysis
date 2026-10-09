# Data and models on disk

Everything under `data/` is local and not tracked by Git.

- `data/models/pretrained/` — base weights; missing YOLO11/YOLO26 weights download
  automatically when selected for training.
- `data/models/detection/`, `data/models/segmentation/` — one folder per training run.
  The model dropdowns list each run's `weights/best.pt`.
- `data/models/reid/` — the networks that tell players apart by their looks, one folder
  per run of `scripts/train_reid.py` (`best.pt`, `training.json`). The one named in
  `models.reid.weights` is used by the app; without it players are not known by looks.
- `data/cache/rosters/` — the players of each video as the app came to know them: looks,
  team, number and what they did, one file per video. Delete a file to start it afresh.
- `data/raw/training_data/` — datasets in YOLO format, named
  `<origin>_<content>_<version>`. The origin says who made the labels and how far the
  folder can be trusted as a source:

  | Folder | What it is | Used for |
  | --- | --- | --- |
  | `labelled_players_discs_v1` | Frames labelled in the Labelling tab, full resolution | Future training |
  | `labelled_discs_v1` | Frames labelled from the phone, discs only, full resolution | Future training |
  | `combined_discs_v4` | `roboflow_merged_discs_v2` plus the disc boxes of both `labelled_` sets, built by `scripts/build_combined_disc_dataset.py`. The default disc model was trained on v3, an earlier state with fewer labels that is no longer kept | Disc models |
  | `combined_players_v1` | `roboflow_merged_players_v2` plus the player boxes of `labelled_players_discs_v1`, built by the same script with `--object player` | Player models |
  | `roboflow_merged_players_v2` | Built from the Roboflow exports: 1,442 images at 1280×720, players only | The default player model, benchmarks |
  | `roboflow_merged_discs_v2` | The same images, discs only | The default disc model, benchmarks |
  | `roboflow_object_detection_v3i` | Roboflow export as downloaded: players and discs, 960×960 | Source of the merged sets |
  | `roboflow_player_disc_detection_v4i` | Roboflow export: players and discs of one game, stretched to 1280×1280 | Source of the merged sets |
  | `roboflow_object_detection_disc_v1i` | Roboflow export: discs only, 1920×1080 | Source of the merged sets |
  | `roboflow_field_finder_v8i` | Roboflow export: field and end zones, stretched to a square | The field segmentation model |
  | `labelled_field_v1` | Frames with the field's corners placed in the Labelling tab ("Field lines") | The field estimate's benchmark, source of the sets below |
  | `propagated_field_v1` | Those labels carried to neighbouring frames with the camera's motion, built by `scripts/build_field_registration_dataset.py` | Source of the rendered sets |
  | `rendered_field_v1` | Field and end zone areas drawn from the field labels, plus the Roboflow outlines, split by game; built by `scripts/build_field_mask_dataset.py` | Field segmentation models |
  | `field_negatives_v1` | Close-ups from edited games in which no field is seen from above, found by `scripts/collect_field_negatives.py` and looked through by hand (`excluded.txt`) | Pictures without a field for the set below |
  | `rendered_field_v2` | `rendered_field_v1` plus the pictures of `field_negatives_v1` with nothing labelled in them (`--negatives`) | Field segmentation models |
  | `reid_players_v2` | Crops of tracked players from 20 stretches of each of the four drone games, with the track and, where read, the jersey number of each; built by `scripts/build_reid_dataset.py`. `pictures_72x144.npy` holds all crops at one size and is made on first use | The re-identification networks |
  | `roboflow_digits_v1i` | Roboflow export: house-number digits | A rough start for a jersey digit detector |

  `labelled_` is labelled with this app, `roboflow_` is a Roboflow export exactly as
  downloaded, and `roboflow_merged_` is built from those exports by
  `scripts/build_merged_detection_dataset.py` (1280×720, 16:9 restored) and
  `scripts/build_single_class_dataset.py` (the labels of one class only). The version
  counts up within one name; Roboflow's own version numbers end in `i`. The folders of
  training runs made before the renaming still end in the old dataset names.
