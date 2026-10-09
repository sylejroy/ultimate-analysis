"""The rows the export writes for a frame."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_analysis.py"


def load_rows():
    """The two functions that make the rows, without the script's imports of the pipeline."""
    source = SCRIPT.read_text()
    start = source.index("def on_field")
    end = source.index("def main()")
    namespace = {}
    exec(compile(source[start:end], str(SCRIPT), "exec"), namespace)
    return namespace


class ExportRowsTest(unittest.TestCase):
    def setUp(self):
        self.rows = load_rows()
        player = SimpleNamespace(
            track_id=7, class_name="player", team=1, to_ltrb=lambda: [100.0, 200.0, 140.0, 290.0]
        )
        disc = SimpleNamespace(track_id=100001, class_name="disc", to_ltrb=lambda: [0, 0, 9, 9])
        self.result = SimpleNamespace(
            tracks=[player, disc],
            player_ids={7: ("23", None)},
            holder_id=7,
            image_to_field=np.diag([0.1, 0.1, 1.0]),
            detections=[
                {"class_name": "disc", "confidence": 0.4, "bbox": [0, 0, 10, 10]},
                {"class_name": "disc", "confidence": 0.9, "bbox": [300, 400, 310, 410]},
                {"class_name": "player", "confidence": 0.9, "bbox": [100, 200, 140, 290]},
            ],
            disc_place=None,
            flight_seconds=None,
            disc_state="held",
            possession_team=1,
        )

    def test_a_player_row_has_the_box_the_feet_on_the_field_and_the_disc(self):
        rows = self.rows["player_rows"](self.result, 12, 0.4)
        self.assertEqual(rows, [[12, 0.4, 7, "23", 1, 100.0, 200.0, 140.0, 290.0, 12.0, 29.0, 1]])

    def test_the_disc_row_takes_the_surest_disc_and_its_place(self):
        row = self.rows["disc_row"](self.result, 12, 0.4)
        self.assertEqual(row, [12, 0.4, "held", 7, 1, "", 305.0, 405.0, 30.5, 40.5])
        # In flight the place worked out for it stands in for the ground under its pixel
        self.result.disc_place, self.result.flight_seconds = (28.0, 37.5), 1.234
        self.result.holder_id, self.result.disc_state = None, "air"
        row = self.rows["disc_row"](self.result, 12, 0.4)
        self.assertEqual(row, [12, 0.4, "air", "", 1, 1.23, 305.0, 405.0, 28.0, 37.5])

    def test_what_is_not_known_is_left_empty(self):
        self.result.image_to_field = None
        self.result.detections = []
        self.result.tracks[0].team = None
        self.result.player_ids = {7: ("Unknown", None)}
        rows = self.rows["player_rows"](self.result, 12, 0.4)
        self.assertEqual(rows[0][3:5] + rows[0][9:11], ["", "", "", ""])
        self.assertEqual(self.rows["disc_row"](self.result, 12, 0.4)[6:], ["", "", "", ""])


if __name__ == "__main__":
    sys.exit(unittest.main())
