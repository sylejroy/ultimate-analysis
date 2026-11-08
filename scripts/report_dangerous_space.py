#!/usr/bin/env python3
"""The dangerous space at every throw of recorded games: where the offence should have
attacked and the defence should have been, and whether they did.

From the throw tables of scripts/build_throw_table.py, the dangerous space at each
release (processing/open_space.py) is compared with where the throw went. Written to a
new folder:

    <output>/throws.csv     a row per throw: its frame (as the app's Copy frame gives
                            it), outcome and gain; the most dangerous place, how
                            dangerous, and how dangerous where the throw went (and the
                            share of the field more dangerous); the defender who left
                            most of the danger and their share
    <output>/<game>.png     the throws of a game that left the most danger unattacked,
                            from above with the offence attacking upwards: the danger
                            shaded, the offence in blue, the defence in red, the thrower
                            ringed in blue, the most dangerous place as a white star,
                            the throw as a white line, the defender who left most of it
                            ringed in red

Places are given as the offence attacks: across from the left sideline, along from the
goal line they defend. Printed per game: how often the throw went to the most dangerous
twentieth of the field, and the completions and gains of throws into dangerous space
against the others.

    python scripts/report_dangerous_space.py data/exports/throws_20261107 \\
        --output data/exports/danger_report_20261108
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
os.chdir(REPO)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import train_throw_targets as training  # noqa: E402

from ultimate_analysis.config.settings import get_setting  # noqa: E402
from ultimate_analysis.processing.field_registration import field_to_canvas  # noqa: E402
from ultimate_analysis.processing.open_space import dangerous_space, load_field_value  # noqa: E402
from ultimate_analysis.processing.throw_targets import (  # noqa: E402
    CENTRES,
    COLUMNS,
    ROWS,
    square_of,
)
from ultimate_analysis.rendering.top_down import draw_field_diagram  # noqa: E402
from ultimate_analysis.utils.field_template import TEMPLATES  # noqa: E402

COLUMNS_OUT = (
    "game frame time outcome gain thrown_x thrown_y top_x top_y top_danger thrown_danger "
    "more_dangerous defender defender_share"
)
TOP_SHARE = 0.05  # The most dangerous twentieth of the field
DANGEROUS_SHARE = 0.2  # Throws into the most dangerous fifth count as attacking it
PICTURES = 6  # Per game
OFFENCE, DEFENCE = (240, 170, 60), (60, 60, 220)


def frame_code(game: str, frame: int, fps: float) -> str:
    seconds = int(frame / fps)
    return f"frame:{game}@{frame} ({seconds // 60}:{seconds % 60:02d})"


def picture(throw, found, text: str) -> np.ndarray:
    """The throw from above, the offence attacking upwards."""
    template = TEMPLATES["usau"]
    size = (360, 900)
    to_view = field_to_canvas(template, size)
    top = int(np.argmax(found.danger))
    defender = int(np.argmax(found.given_up)) if len(found.given_up) else -1
    dots = [{"place": tuple(throw.thrower), "colour": OFFENCE, "label": "", "holder": True}]
    dots += [{"place": (x, y), "colour": OFFENCE, "label": ""} for x, y, *_ in throw.offence]
    dots += [
        {"place": (x, y), "colour": DEFENCE, "label": "", "gives_up": 1.0 if i == defender else 0}
        for i, (x, y, *_) in enumerate(throw.defence)
    ]
    view = draw_field_diagram(
        size,
        template.width,
        template.length,
        template.end_zone,
        to_view,
        dots,
        field_map=found.danger.reshape(ROWS, COLUMNS),
    )

    def at(place):
        mapped = to_view @ [place[0], place[1], 1.0]
        return int(mapped[0] / mapped[2]), int(mapped[1] / mapped[2])

    cv2.line(view, at(throw.thrower), at(throw.caught), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.drawMarker(view, at(CENTRES[top]), (255, 255, 255), cv2.MARKER_STAR, 22, 2)
    for line, words in enumerate(text.split("\n")):
        cv2.putText(
            view, words, (6, 18 + 18 * line), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
        )
    return view


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("exports", type=Path, help="Folder of the games' exports")
    parser.add_argument("--output", type=Path, required=True, help="A new folder")
    args = parser.parse_args()
    if args.output.exists():
        sys.exit(f"Refusing to write into {args.output}: it exists")
    args.output.mkdir(parents=True)
    value = load_field_value(get_setting("models.open_space.value"))

    rows, totals = [], []
    for folder in sorted(p for p in args.exports.iterdir() if (p / "throws.csv").exists()):
        video = cv2.VideoCapture(str(REPO / "data" / "raw" / "videos" / f"{folder.name}.mp4"))
        fps = video.get(cv2.CAP_PROP_FPS) or 60.0
        video.release()
        throws = training.load_game(folder, defaultdict(int))
        found_all = []
        for throw in throws:
            found = dangerous_space(throw, value)
            danger = found.danger
            thrown = float(danger[square_of(throw.caught)])
            top = int(np.argmax(danger))
            defender = int(np.argmax(found.given_up)) if len(found.given_up) else None
            share = (
                float(found.given_up[defender] / found.given_up.sum())
                if defender is not None and found.given_up.sum() > 0
                else 0.0
            )
            more = float((danger > thrown).mean())
            row = [
                folder.name,
                frame_code(folder.name, throw.frame, fps),
                round(throw.frame / fps, 1),
                "complete" if throw.completed else "turnover",
                round(throw.gain, 1),
                *(round(v, 1) for v in throw.caught),
                *(round(float(v), 1) for v in CENTRES[top]),
                round(float(danger[top]), 4),
                round(thrown, 4),
                round(more, 3),
                "" if defender is None else throw.defenders[defender],
                round(share, 2),
            ]
            rows.append(row)
            found_all.append((throw, found, row))
        totals.extend(found_all)

        # The throws that left the most danger unattacked
        missed = sorted(found_all, key=lambda f: f[2][9] - f[2][10], reverse=True)[:PICTURES]
        if missed:
            views = [
                picture(
                    throw,
                    found,
                    f"{row[1].split(' ')[0].split('@')[1]} {row[1].split(' ')[1]}\n"
                    f"{row[3]}, {row[4]:+.0f} yd\nleft {row[9] - row[10]:.3f}",
                )
                for throw, found, row in missed
            ]
            cv2.imwrite(str(args.output / f"{folder.name}.png"), cv2.hconcat(views))
        summarise(folder.name, found_all)
    summarise("All games", totals)

    with open(args.output / "throws.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(COLUMNS_OUT.split())
        writer.writerows(rows)
    print(f"Written to {args.output}")


def summarise(name: str, found_all: list) -> None:
    """How often the offence attacked the dangerous space, and what that brought."""
    if not found_all:
        return
    more = np.array([row[11] for *_, row in found_all])
    completed = np.array([throw.completed for throw, *_ in found_all])
    gain = np.array([throw.gain for throw, *_ in found_all])
    into = more < DANGEROUS_SHARE
    print(
        f"{name}: {len(found_all)} throws; into the most dangerous twentieth "
        f"{(more < TOP_SHARE).mean():.0%}, fifth {into.mean():.0%}. Into it: completed "
        f"{completed[into].mean():.0%}, gain {np.median(gain[into & completed]):+.1f} yd "
        f"(median of completions); elsewhere {completed[~into].mean():.0%}, "
        f"{np.median(gain[~into & completed]):+.1f} yd"
    )


if __name__ == "__main__":
    main()
