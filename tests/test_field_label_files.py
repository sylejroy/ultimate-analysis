"""Frames with labelled field elements on disk."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from support import load_module

VIDEO = "game_2024.mp4"


def camera_pixel(x, y):
    mapped = np.array([[30.0, -4.0, 360.0], [0.0, -3.0, 1000.0], [0.0, 0.011, 0.6]]) @ [x, y, 1.0]
    return (float(mapped[0] / mapped[2]), float(mapped[1] / mapped[2]))


class FieldLabelFileTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module("utils.field_label_files")
        self.template = load_module("utils.field_template").TEMPLATES["usau"]
        self.label_files = load_module("utils.label_files")
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.dataset = Path(self.folder.name) / "labelled_field_test"
        self.frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    def four_lines(self):
        lines = {}
        for name in ("left_sideline", "right_sideline", "far_back_line", "far_goal_line"):
            start, end = (np.array(end) for end in self.template.lines[name])
            lines[name] = [camera_pixel(*(start + s * (end - start))) for s in (0.3, 0.8)]
        return lines

    def test_a_saved_label_comes_back_and_holds_its_mapping(self):
        name = self.label_files.frame_name(VIDEO, 120)
        label = self.module.FieldLabel(self.four_lines(), {"far_brick": camera_pixel(20, 70)})
        self.module.save_frame(self.dataset, name, self.frame, label)

        loaded = self.module.load_label(self.dataset, name)
        self.assertEqual(set(loaded.lines), set(label.lines))
        np.testing.assert_allclose(loaded.points["far_brick"], label.points["far_brick"])
        stored = json.loads((self.dataset / "labels" / f"{name}.json").read_text())
        self.assertEqual(stored["ruleset"], "usau")
        self.assertEqual(stored["image_size"], [1920, 1080])
        # The stored mapping takes the brick mark's pixel to the brick mark
        mapped = np.array(stored["image_to_field"]) @ [*label.points["far_brick"], 1.0]
        np.testing.assert_allclose(mapped[:2] / mapped[2], (20.0, 70.0), atol=1e-3)
        self.assertTrue((self.dataset / "images" / f"{name}.jpg").exists())

    def test_a_label_that_does_not_fix_the_field_is_stored_without_a_mapping(self):
        name = self.label_files.frame_name(VIDEO, 5)
        lines = {"left_sideline": self.four_lines()["left_sideline"]}
        self.module.save_frame(self.dataset, name, self.frame, self.module.FieldLabel(lines))
        stored = json.loads((self.dataset / "labels" / f"{name}.json").read_text())
        self.assertIsNone(stored["image_to_field"])
        self.assertEqual(self.module.summary(self.dataset)["calibrated"], 0)
        self.assertEqual(self.module.summary(self.dataset)["frames"], 1)

    def test_frames_are_listed_split_and_removed(self):
        label = self.module.FieldLabel(self.four_lines())
        names = [self.label_files.frame_name(VIDEO, index) for index in (0, 400, 4000)]
        for name in names:
            self.module.save_frame(self.dataset, name, self.frame, label, "wfdf")
        self.assertEqual(self.module.labelled_frames(self.dataset), sorted(names))
        self.assertEqual(self.module.dataset_ruleset(self.dataset), "wfdf")
        listed = sum(
            len((self.dataset / f"{split}.txt").read_text().split())
            for split in ("train", "val", "test")
        )
        self.assertEqual(listed, 3)

        self.module.remove_frame(self.dataset, names[0])
        self.assertIsNone(self.module.load_label(self.dataset, names[0]))
        self.assertEqual(len(self.module.labelled_frames(self.dataset)), 2)

    def test_a_frame_that_is_not_in_the_dataset_has_no_label(self):
        self.assertIsNone(self.module.load_label(self.dataset, "nothing_frame_000001"))
        self.assertEqual(self.module.labelled_frames(self.dataset), [])
        self.assertEqual(self.module.dataset_ruleset(self.dataset), "usau")


if __name__ == "__main__":
    unittest.main()
