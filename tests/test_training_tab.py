"""The Model Info box of the training tab says what the files tell, without loading or
downloading a model."""

import tempfile
import unittest
from pathlib import Path

from support import load_module, qt_app


class ModelInfoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = qt_app()
        cls.module = load_module("gui.training.training_tab")

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def test_a_pretrained_weight_not_on_disk_is_said_to_be_not_downloaded(self):
        lines = self.module.model_info_lines(str(self.root / "pretrained" / "yolo26n.pt"))
        self.assertEqual(lines[0], "Path: yolo26n.pt")
        self.assertIn("Not downloaded yet", lines[1])
        self.assertEqual(len(lines), 2)

    def test_a_definition_is_built_as_training_starts(self):
        lines = self.module.model_info_lines("yolo26n-p2.yaml")
        self.assertIn("definition", lines[1])

    def test_a_training_run_is_described_from_its_arguments(self):
        weights = self.root / "run" / "finetune" / "weights" / "best.pt"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"w" * 2 * 1024 * 1024)
        (weights.parent.parent / "args.yaml").write_text(
            "model: data/models/pretrained/yolo26s.pt\n"
            "data: data/raw/training_data/merged_v3/data.yaml\n"
            "imgsz: 1280\nepochs: 60\n",
            encoding="utf-8",
        )
        self.assertEqual(
            self.module.model_info_lines(str(weights)),
            [
                "Path: best.pt",
                "Size: 2.0 MB",
                "Trained from: yolo26s.pt",
                "Trained on: merged_v3",
                "Image size: 1280",
                "Epochs: 60",
            ],
        )

    def test_a_weight_without_a_run_shows_its_size_only(self):
        weights = self.root / "pretrained" / "yolo26s.pt"
        weights.parent.mkdir()
        weights.write_bytes(b"w" * 1024 * 1024)
        self.assertEqual(
            self.module.model_info_lines(str(weights)), ["Path: yolo26s.pt", "Size: 1.0 MB"]
        )


if __name__ == "__main__":
    unittest.main()
