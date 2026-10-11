#!/usr/bin/env python3
"""Score a network that tells who holds the disc on a dataset of marked crops.

For the crops made from hand-labelled discs (scripts/build_holder_ground_truth.py) the
anchors, marked by the label alone, and the possessions around them are scored apart.

    python scripts/benchmark_holder.py data/models/holder/<run>/best.pt train_holders \\
        --without chicago portland

A network that learned from crops of the boxes themselves (generated_holders) is scored
on crops made wider than the boxes with --padded 0.25.
"""

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
os.chdir(REPO)

from train_holder import load_index, pictures_of, score  # noqa: E402

from ultimate_analysis.constants import DEFAULT_PATHS  # noqa: E402
from ultimate_analysis.processing.holder import load_holder_net  # noqa: E402

SHOWN = (
    ("holder_picked", "The one it is surest of is the holder"),
    ("holder_picked_with_speed", "... counting in who stands"),
    ("holder_picked_speed_alone", "... by who stands alone"),
    ("holder_picked_disc_seen", "where a disc is detected on the holder"),
    ("holder_picked_disc_not_seen", "where none is"),
    ("holder_picked_disc_not_seen_with_speed", "where none is, counting in who stands"),
    ("accuracy", "Crops said right (over a half: holding)"),
    ("precision", "Of those it says hold the disc, holders"),
    ("holders_found", "Holders it says hold the disc (over a half)"),
    ("hidden_holders_found", "... of those with no disc detected"),
    ("others_taken_for_holders", "Others it says hold the disc"),
    ("average_precision", "Average precision"),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("weights", type=Path)
    parser.add_argument("dataset", help="Folder in data/raw/training_data")
    parser.add_argument("--videos", nargs="+", default=[], help="Only these (parts of names)")
    parser.add_argument("--without", nargs="+", default=[], help="Not these (parts of names)")
    parser.add_argument(
        "--padded",
        type=float,
        default=0.0,
        help="The dataset's crops are wider than the boxes by this share on every side and "
        "the network learned from the boxes themselves: cut them back",
    )
    args = parser.parse_args()

    rows = load_index(REPO / DEFAULT_PATHS["TRAINING_DATA"] / args.dataset)
    rows = [
        row
        for row in rows
        if (not args.videos or any(part in row["video"] for part in args.videos))
        and not any(part in row["video"] for part in args.without)
    ]
    model = load_holder_net(args.weights)
    for kind in sorted({row.get("kind", "") for row in rows}):
        chosen = [row for row in rows if row.get("kind", "") == kind]
        result = score(model, chosen, pictures_of(chosen, model.size, args.padded))
        print(
            f"{kind or 'crops'}: {len(chosen)} crops, {result['holder_picked_frames']} frames "
            f"with a holder, {result['disc_not_seen_frames']} of them with no disc detected "
            "on the holder"
        )
        for key, title in SHOWN:
            print(f"  {title:<55} {result[key]:>6.1%}")


if __name__ == "__main__":
    main()
