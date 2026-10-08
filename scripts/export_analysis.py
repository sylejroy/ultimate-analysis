#!/usr/bin/env python3
"""Write what the analysis finds in a stretch of a game to files.

Every frame goes through the analysis pipeline, as in the main tab, without anything
being skipped. Two tables are written, one row per player and frame and one row per
frame for the disc:

    <output>/players.csv   frame, seconds, player, number, team, x1, y1, x2, y2,
                           field_x, field_y, has_disc
    <output>/disc.csv      frame, seconds, state, holder, team_in_possession,
                           flight_seconds, x, y, field_x, field_y

- player is the ID the tracker follows a player by; number their jersey number once it
  has been read; team is 0 or 1 once the tracker knows it.
- x1 ... y2 is the player's box in the frame, x and y the disc's centre, in pixels.
- field_x and field_y are places on the field in its unit (yards for USAU), across and
  along, from where the field model sees the field; empty where that is not known. A
  player's place is that of their feet. A flying disc's place is where it would be one
  yard up (see docs/MEASUREMENTS.md); a disc that is held or lies on the ground is
  placed as the ground under its pixel.
- state is "held", "air" (also: not seen) or "ground".

These are estimates: docs/MEASUREMENTS.md says how far off each is.

Usage:
    python scripts/export_analysis.py VIDEO --start 12:30 --seconds 60 --output data/exports/run1
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)

import cv2  # noqa: E402

from ultimate_analysis.pipeline import AnalysisPipeline, PipelineOptions  # noqa: E402
from ultimate_analysis.processing.player_id import discard_pending_readings  # noqa: E402
from ultimate_analysis.utils.video import seconds_of  # noqa: E402

PLAYER_COLUMNS = "frame seconds player number team x1 y1 x2 y2 field_x field_y has_disc"
DISC_COLUMNS = "frame seconds state holder team_in_possession flight_seconds x y field_x field_y"


def on_field(image_to_field, x: float, y: float) -> tuple:
    """A pixel as a place on the field, rounded; two empty cells if that is not known."""
    if image_to_field is None:
        return "", ""
    mapped = image_to_field @ [x, y, 1.0]
    if abs(mapped[2]) < 1e-12:
        return "", ""
    return round(float(mapped[0] / mapped[2]), 2), round(float(mapped[1] / mapped[2]), 2)


def player_rows(result, index: int, seconds: float) -> list:
    """One row per player followed in a frame."""
    rows = []
    for track in result.tracks:
        if track.class_name != "player":
            continue
        x1, y1, x2, y2 = (round(float(value), 1) for value in track.to_ltrb())
        number = result.player_ids.get(track.track_id, ("", None))[0]
        team = getattr(track, "team", None)
        rows.append(
            [
                index,
                seconds,
                track.track_id,
                number if str(number).isdigit() else "",
                "" if team is None else team,
                x1,
                y1,
                x2,
                y2,
                *on_field(result.image_to_field, (x1 + x2) / 2, y2),
                int(track.track_id == result.holder_id),
            ]
        )
    return rows


def disc_row(result, index: int, seconds: float) -> list:
    """The row of a frame for the disc."""
    found = [d for d in result.detections if d["class_name"] == "disc"]
    x = y = ""
    place = ("", "")
    if found:
        box = max(found, key=lambda d: d.get("confidence", 0.0))["bbox"]
        x, y = round((box[0] + box[2]) / 2, 1), round((box[1] + box[3]) / 2, 1)
        place = on_field(result.image_to_field, x, y)
        if result.disc_place is not None:
            place = tuple(round(value, 2) for value in result.disc_place)
    flight = result.flight_seconds
    return [
        index,
        seconds,
        result.disc_state,
        "" if result.holder_id is None else result.holder_id,
        "" if result.possession_team is None else result.possession_team,
        "" if flight is None else round(flight, 2),
        x,
        y,
        *place,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("--start", default="0", help="Where to start: seconds or m:ss")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--output", type=Path, required=True, help="Folder for the tables")
    parser.add_argument("--no-numbers", action="store_true", help="Do not read jersey numbers")
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        sys.exit(f"Refusing to write into a folder that is not empty: {args.output}")
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        sys.exit(f"Cannot open {args.video}")
    frames_per_second = capture.get(cv2.CAP_PROP_FPS) or 30.0
    first = int(seconds_of(args.start) * frames_per_second)
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)

    pipeline = AnalysisPipeline()
    pipeline.new_video(str(args.video))
    pipeline.set_frame_rate(frames_per_second)
    # The top-down view is only a picture; the places come from the field model anyway
    options = PipelineOptions(player_id=not args.no_numbers, top_down_view=False)

    args.output.mkdir(parents=True, exist_ok=True)
    began, written = time.perf_counter(), 0
    with (
        open(args.output / "players.csv", "w", newline="") as players_file,
        open(args.output / "disc.csv", "w", newline="") as disc_file,
    ):
        players, disc = csv.writer(players_file), csv.writer(disc_file)
        players.writerow(PLAYER_COLUMNS.split())
        disc.writerow(DISC_COLUMNS.split())
        try:
            for index in range(first, first + int(args.seconds * frames_per_second)):
                ok, frame = capture.read()
                if not ok:
                    break
                result = pipeline.process(frame, index, options)
                if not result.wide_shot:
                    continue  # A close-up: nothing is followed
                seconds = round(index / frames_per_second, 3)
                players.writerows(player_rows(result, index, seconds))
                disc.writerow(disc_row(result, index, seconds))
                written += 1
        finally:
            capture.release()
            discard_pending_readings()
    print(
        f"{args.output}: {written} frames in {time.perf_counter() - began:.0f} s "
        f"(players.csv, disc.csv)"
    )


if __name__ == "__main__":
    main()
