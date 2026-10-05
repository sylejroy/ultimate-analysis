"""Labelled frames on disk: what is saved is read back, and splits are stable."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from support import load_module


class LabelFileTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module("utils.label_files")
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.dataset = Path(folder.name) / "labelled"
        self.frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    def test_saved_boxes_are_read_back_and_unlabelled_frames_are_told_apart(self):
        module = self.module
        name = module.frame_name("videos/game one.mp4", 1234)
        self.assertEqual(name, "game one_frame_001234")
        self.assertEqual(module.frame_index_of(name), 1234)
        self.assertIsNone(module.load_boxes(self.dataset, name, (1920, 1080)))

        boxes = [module.LabelBox(0, 100, 200, 117, 215), module.LabelBox(1, 500.5, 300, 560, 480)]
        module.save_frame(self.dataset, name, self.frame, boxes)
        loaded = module.load_boxes(self.dataset, name, (1920, 1080))
        self.assertEqual([box.class_id for box in loaded], [0, 1])
        np.testing.assert_allclose(
            [(b.x1, b.y1, b.x2, b.y2) for b in loaded],
            [(b.x1, b.y1, b.x2, b.y2) for b in boxes],
            atol=0.01,
        )
        self.assertTrue((self.dataset / "images" / f"{name}.jpg").exists())

        # A frame without objects is a labelled frame too
        empty = module.frame_name("videos/game one.mp4", 1300)
        module.save_frame(self.dataset, empty, self.frame, [])
        self.assertEqual(module.load_boxes(self.dataset, empty, (1920, 1080)), [])

    def test_boxes_are_kept_inside_the_frame_and_slivers_are_dropped(self):
        module = self.module
        name = module.frame_name("game.mp4", 5)
        boxes = [module.LabelBox(1, -50, 900, 100, 1200), module.LabelBox(0, 300, 300, 300.2, 320)]
        module.save_frame(self.dataset, name, self.frame, boxes)
        (loaded,) = module.load_boxes(self.dataset, name, (1920, 1080))
        np.testing.assert_allclose(
            (loaded.x1, loaded.y1, loaded.x2, loaded.y2), (0, 900, 100, 1080), atol=0.01
        )

    def test_splits_keep_neighbouring_frames_together_and_follow_the_saved_frames(self):
        module = self.module
        # Frames of one stretch of play share a split
        stretch = {module.split_of(module.frame_name("a.mp4", i)) for i in range(0, 300, 30)}
        self.assertEqual(len(stretch), 1)
        splits = [module.split_of(module.frame_name("a.mp4", i * 300)) for i in range(400)]
        self.assertGreater(splits.count("train"), 250)
        self.assertGreater(splits.count("val"), 15)
        self.assertGreater(splits.count("test"), 15)

        names = [module.frame_name("a.mp4", i * 300) for i in range(40)]
        for name in names:
            module.save_frame(self.dataset, name, self.frame, [module.LabelBox(0, 10, 10, 30, 30)])
        module.remove_frame(self.dataset, names[0])

        listed = {
            split: (self.dataset / f"{split}.txt").read_text().split() for split in module.SPLITS
        }
        self.assertEqual(sum(len(lines) for lines in listed.values()), 39)
        for split, lines in listed.items():
            for line in lines:
                self.assertEqual(module.split_of(Path(line).stem), split)
                self.assertTrue((self.dataset / line).exists())

        data = yaml.safe_load((self.dataset / "data.yaml").read_text())
        self.assertEqual((data["names"], data["train"]), (["disc", "player"], "train.txt"))
        counts = module.summary(self.dataset)
        self.assertEqual((counts["frames"], counts["disc"], counts["player"]), (39, 39, 0))
        self.assertEqual(module.labelled_frames(self.dataset, "x/a.mp4"), names[1:])
        self.assertEqual(module.labelled_frames(self.dataset, "x/b.mp4"), [])

    def test_a_disc_only_dataset_keeps_its_single_class(self):
        module = self.module
        first = module.frame_name("a.mp4", 10)
        module.save_frame(
            self.dataset, first, self.frame, [module.LabelBox(0, 10, 10, 30, 30)], ["disc"]
        )
        # Saved again without saying so, e.g. from another tool
        module.save_frame(self.dataset, module.frame_name("a.mp4", 900), self.frame, [])

        data = yaml.safe_load((self.dataset / "data.yaml").read_text())
        self.assertEqual((data["nc"], data["names"]), (1, ["disc"]))
        counts = module.summary(self.dataset)
        self.assertEqual((counts["frames"], counts["disc"]), (2, 1))


if __name__ == "__main__":
    unittest.main()
