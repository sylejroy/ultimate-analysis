"""The throws read from an export: who threw to whom, from where to where, how far."""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_throw_table as table  # noqa: E402

DISC = "frame seconds restart state holder team_in_possession attack_0 attack_1 game_state"
PLAYERS = "frame seconds player team place_x place_y"


def write(folder: Path, disc_rows, player_rows) -> None:
    for name, columns, rows in (
        ("disc.csv", DISC, disc_rows),
        ("players.csv", PLAYERS, player_rows),
    ):
        with open(folder / name, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(columns.split())
            writer.writerows(rows)


def frame(index, restart, state, holder, team, attack=-1):
    """A disc row, ten frames a second; team 0 attacks as given."""
    return [index, index / 10, restart, state, holder, team, attack, -attack, "live"]


class ThrowTableTest(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        # Player 1 throws to player 2 (team 0), who throws to player 5 of the other team
        states = ["held"] * 3 + ["air"] * 4 + ["held"] * 3 + ["air"] * 2 + ["held"] * 2
        holders = [1] * 3 + [""] * 4 + [2] * 3 + [""] * 2 + [5] * 2
        teams = [0] * 12 + [1] * 2
        disc = [frame(i, 1, states[i], holders[i], teams[i]) for i in range(14)]
        players = []
        for i in range(14):
            seconds = i / 10
            players += [
                [i, seconds, 1, 0, 20.0, 60.0],
                [i, seconds, 2, 0, 25.0, 50.0 - i],  # Runs towards y = 0, which team 0 attacks
                [i, seconds, 5, 1, 10.0, 40.0],
            ]
        write(self.folder, disc, players)

    def test_a_completion_and_a_turnover_are_found(self):
        throws, _ = table.build(self.folder)
        self.assertEqual(
            [(t[8], t[9], t[11]) for t in throws], [(1, 2, "complete"), (2, 5, "turnover")]
        )

    def test_the_gain_is_towards_the_end_zone_attacked(self):
        throws, _ = table.build(self.folder)
        # From y 60 to where player 2 is at the catch (frame 7: y 43): 17 yards gained
        self.assertEqual(throws[0][13:19], [20.0, 60.0, 25.0, 43.0, -1, 17.0])

    def test_the_players_at_the_release_have_sides_and_speeds(self):
        _, released = table.build(self.folder)
        first = {row[1]: row for row in released if row[0] == 0}
        self.assertEqual(first[1][2], "thrower")
        self.assertEqual(first[2][2], "offence")
        self.assertEqual(first[5][2], "defence")
        self.assertEqual(first[2][5:7], [0.0, -10.0])  # Ten yards a second towards y = 0

    def test_a_restart_parts_the_holds(self):
        disc = [frame(0, 1, "held", 1, 0), frame(1, 2, "held", 2, 0)]
        players = [[0, 0.0, 1, 0, 20.0, 60.0], [1, 0.1, 2, 0, 20.0, 70.0]]
        write(self.folder, disc, players)
        self.assertEqual(table.build(self.folder)[0], [])


if __name__ == "__main__":
    sys.exit(unittest.main())
