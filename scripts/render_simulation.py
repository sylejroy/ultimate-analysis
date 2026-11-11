#!/usr/bin/env python3
"""Render simulated points as a video (simulation/drawing.py): the field from above,
White attacking to the right and Red to the left, the dangerous space for the team with
the disc shaded, the disc in yellow, the wind as an arrow, what happens along the top.

    python scripts/render_simulation.py --points 3 --output simulation.mp4
    python scripts/render_simulation.py --white data/models/movement/<run> --wind 5 ...
"""

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ultimate_analysis.simulation import drawing, game  # noqa: E402
from ultimate_analysis.simulation.drawing import (  # noqa: E402
    call_of,
    caption_of,
    picture_of,
    team_of,
)
from ultimate_analysis.simulation.match import Match  # noqa: E402

FRAMES_PER_STEP = 3  # At 30 frames a second: as fast as the game


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=3)
    parser.add_argument(
        "--white", default="scripted", help='"scripted", "scripted2" or a folder of a team'
    )
    parser.add_argument(
        "--red", default="scripted", help='"scripted", "scripted2" or a folder of a team'
    )
    parser.add_argument("--wind", type=float, default=0.0, help="Strongest wind, yards a second")
    parser.add_argument(
        "--names",
        default="White,Red",
        help="What to call the white and the red team, as WHITE,RED",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    drawing.NAMES = tuple(args.names.split(","))[:2]

    rng = np.random.default_rng(args.seed)
    writer = None
    score = [0, 0]
    for point in range(args.points):
        teams = tuple(
            team_of(kind, args.seed + 100 * side + point, side)
            for side, kind in enumerate((args.white, args.red))
        )
        # Teams take turns to receive the pull
        start = game.pull_start(rng, point % 2, game.a_wind(rng, args.wind))
        match = Match(teams, rng, start)
        caption, shown_for, seen_events = "The pull", 30, 0
        while not match.state.over:
            match.step()
            state = match.state
            for event in state.events[seen_events:]:
                said = caption_of(event)
                if said:
                    caption, shown_for = said, 25
            seen_events = len(state.events)
            call = call_of(state, match.decisions)
            if call and shown_for <= 0:
                caption, shown_for = call, 20
            shown_for -= 1
            if shown_for < -40:
                caption = ""
            picture = picture_of(state, score, caption)
            if writer is None:
                height, width = picture.shape[:2]
                writer = cv2.VideoWriter(
                    str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height)
                )
            for _ in range(FRAMES_PER_STEP):
                writer.write(picture)
        if match.state.scored is not None:
            score[match.state.scored] += 1
        for _ in range(45):  # A moment on the score
            writer.write(picture_of(match.state, score, caption))
    writer.release()
    print(f"{args.points} points to {args.output}")


if __name__ == "__main__":
    main()
