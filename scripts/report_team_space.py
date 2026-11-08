#!/usr/bin/env python3
"""Where each team's defence leaves dangerous space, and whether each offence attacks it:
the dangerous space (processing/open_space.py) at every throw of recorded games, added
up per team.

From the throw tables of scripts/build_throw_table.py. The field around the thrower is
cut into zones, as the offence attacks: deep (20 yd or more ahead), middle (5 to 20 yd
ahead), short (within 5 yd either way), behind (more than 5 yd back); each on the
thrower's side of the field or the far side (when the thrower is 5 yd or more off the
middle). Per team of a game (0 and 1, as the export's teams):

- as defence, the share of the dangerous space it left in each zone, against the
  average of all the games' defences, and which defender left the most of it: the mark
  (within 3 yd of the thrower), the deepest defender, or one underneath;
- as offence, the share of its throws into each zone, against the share of the danger
  there, and how often it threw into the most dangerous fifth of the field.

Written to a new folder: report.md (the tables) and a picture per game of the average
dangerous space each defence left, with where the other team's throws went:

    python scripts/report_team_space.py data/exports/throws_20261107 \\
        --output data/exports/team_space_20261108
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
    WIDTH,
)
from ultimate_analysis.rendering.top_down import draw_field_diagram  # noqa: E402
from ultimate_analysis.utils.field_template import TEMPLATES  # noqa: E402

ZONES = ("deep", "middle", "short", "behind")
SIDES = ("near side", "far side")
MARK_WITHIN = 3.0  # Yards from the thrower
OFF_MIDDLE = 5.0  # Yards: a thrower nearer the middle has no near or far side
DANGEROUS_SHARE = 0.2  # As report_dangerous_space.py
ROLES = ("the mark", "the deepest", "underneath")
OFFENCE = (240, 170, 60)


def zone_of(along: np.ndarray) -> np.ndarray:
    """The zone index of places this far ahead of the thrower (yards)."""
    return np.select([along >= 20, along >= 5, along >= -5], [0, 1, 2], 3)


def side_of(thrower_x: float, across: np.ndarray) -> np.ndarray:
    """0 the thrower's side of the field, 1 the far side, -1 neither (thrower central)."""
    off = thrower_x - WIDTH / 2
    if abs(off) < OFF_MIDDLE:
        return np.full(len(across), -1)
    return np.where(np.sign(across - WIDTH / 2) == np.sign(off), 0, 1)


def cells(thrower, places: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """(4, 2): how much of `weights` falls in each zone and side."""
    found = np.zeros((len(ZONES), len(SIDES)))
    zone = zone_of(places[:, 1] - thrower[1])
    side = side_of(thrower[0], places[:, 0])
    for z in range(len(ZONES)):
        for s in range(len(SIDES)):
            found[z, s] = weights[(zone == z) & (side == s)].sum()
    return found


def role_of(throw, defender: int) -> int:
    """Which defender, by what they were doing at the release."""
    places = throw.defence[:, :2]
    near = np.hypot(*(places - np.array(throw.thrower)).T)
    mark = int(np.argmin(near)) if near.min() <= MARK_WITHIN else -1
    if defender == mark:
        return 0
    others = [k for k in range(len(places)) if k != mark]
    deepest = others[int(np.argmax(places[others, 1]))]
    return 1 if defender == deepest else 2


def teams_of(folder: Path) -> dict:
    """Release frame -> the throwing team."""
    with open(folder / "throws.csv", newline="") as file:
        return {
            int(row["release_frame"]): int(float(row["team"]))
            for row in csv.DictReader(file)
            if row["team"] != ""
        }


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

    # Per (game, team in possession): the sums of what each throw left and did
    teams = defaultdict(
        lambda: {
            "throws": 0,
            "danger": np.zeros((len(ZONES), len(SIDES))),
            "thrown": np.zeros((len(ZONES), len(SIDES))),
            "into": 0,
            "completed": 0,
            "roles": np.zeros(len(ROLES)),
            "map": np.zeros(len(CENTRES)),
            "caught": [],
        }
    )
    for folder in sorted(p for p in args.exports.iterdir() if (p / "throws.csv").exists()):
        thrower_team = teams_of(folder)
        for throw in training.load_game(folder, defaultdict(int)):
            team = thrower_team.get(throw.frame)
            if team is None or len(throw.defence) == 0:
                continue
            found = dangerous_space(throw, value)
            danger = found.danger
            total = danger.sum()
            if total <= 0:
                continue
            kept = teams[(folder.name, team)]
            kept["throws"] += 1
            kept["danger"] += cells(throw.thrower, CENTRES, danger / total)
            caught = np.array([throw.caught])
            kept["thrown"] += cells(throw.thrower, caught, np.ones(1))
            square = int(np.argmin(np.hypot(*(CENTRES - caught[0]).T)))  # The square it went to
            kept["into"] += int((danger > danger[square]).mean() < DANGEROUS_SHARE)
            kept["completed"] += int(throw.completed)
            if found.given_up.sum() > 0:
                for k, share in enumerate(found.given_up / found.given_up.sum()):
                    kept["roles"][role_of(throw, k)] += share
            kept["map"] += danger
            kept["caught"].append(throw.caught)
        print(f"{folder.name}: done", flush=True)

    every = list(teams.values())
    average_danger = sum(t["danger"] for t in every) / sum(t["throws"] for t in every)
    average_roles = sum(t["roles"] for t in every) / sum(t["roles"].sum() for t in every)
    lines = [
        "# Where each defence leaves dangerous space, and whether each offence attacks it",
        "",
        "Shares of the dangerous space at the release, by zone around the thrower (as the",
        "offence attacks; near and far side only when the thrower is 5 yd or more off the",
        "middle), averaged over the throws. In brackets: the average of all the games'",
        "defences. Teams 0 and 1 are the export's.",
        "",
    ]
    games = sorted({game for game, _ in teams})
    for game in games:
        lines += [f"## {game}", ""]
        for defending in (0, 1):
            attacking = 1 - defending
            t = teams.get((game, attacking))
            if not t or not t["throws"]:
                continue
            danger = t["danger"] / t["throws"]
            thrown = t["thrown"] / t["throws"]
            roles = t["roles"] / max(t["roles"].sum(), 1e-9)
            lines += [
                f"**Team {defending} on defence** ({t['throws']} throws of team {attacking}; "
                f"team {attacking} threw into the most dangerous fifth "
                f"{t['into'] / t['throws']:.0%} of the time, completed "
                f"{t['completed'] / t['throws']:.0%})",
                "",
                "| Zone | Danger left, near side | Far side | Team "
                f"{attacking}'s throws there, near | Far |",
                "|---|---:|---:|---:|---:|",
            ]
            for z, zone in enumerate(ZONES):
                lines.append(
                    f"| {zone} | {danger[z, 0]:.0%} ({average_danger[z, 0]:.0%}) | "
                    f"{danger[z, 1]:.0%} ({average_danger[z, 1]:.0%}) | "
                    f"{thrown[z, 0]:.0%} | {thrown[z, 1]:.0%} |"
                )
            central = 1.0 - danger.sum()
            lines += [
                f"| thrower central | {central:.0%} | | {1.0 - thrown.sum():.0%} | |",
                "",
                "Who left it: "
                + ", ".join(
                    f"{role} {roles[r]:.0%} ({average_roles[r]:.0%})"
                    for r, role in enumerate(ROLES)
                ),
                "",
            ]
        _picture(args.output / f"{game}.png", teams, game)
    (args.output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Written to {args.output}")


def _picture(path: Path, teams: dict, game: str) -> None:
    """Side by side, each defence: the average dangerous space it left (the other team
    attacking upwards) and where that team's throws went."""
    template = TEMPLATES["usau"]
    size = (360, 900)
    to_view = field_to_canvas(template, size)
    views = []
    for defending in (0, 1):
        t = teams.get((game, 1 - defending))
        if not t or not t["throws"]:
            continue
        average = (t["map"] / t["throws"]).reshape(ROWS, COLUMNS)
        view = draw_field_diagram(
            size,
            template.width,
            template.length,
            template.end_zone,
            to_view,
            [{"place": tuple(c), "colour": OFFENCE, "label": ""} for c in t["caught"]],
            field_map=average / max(average.max(), 1e-9) * 0.2,
        )
        cv2.putText(
            view,
            f"team {defending} defending",
            (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        views.append(view)
    if views:
        cv2.imwrite(str(path), cv2.hconcat(views))


if __name__ == "__main__":
    main()
