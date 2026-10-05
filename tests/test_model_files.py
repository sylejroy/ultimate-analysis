"""Locating trained models and reading what they were trained with."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from support import load_module


class ModelFilesTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module("utils.model_files")
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def _run(self, task, run, finetune, names=None, imgsz=None):
        """Create a training run folder the way Ultralytics lays it out."""
        weights = self.root / task / run / finetune / "weights" / "best.pt"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"weights")
        if names is not None:
            data = self.root / f"{run}_{finetune}_data.yaml"
            data.write_text(f"names: {names}\n", encoding="utf-8")
            args = f"data: {data.as_posix()}\n" + (f"imgsz: {imgsz}\n" if imgsz else "")
            (weights.parent.parent / "args.yaml").write_text(args, encoding="utf-8")
        return weights

    def test_training_image_size_is_read_from_the_run_folder(self):
        # args.yaml sits next to weights/, not inside it
        weights = self._run("detection", "run", "finetune", names=["disc"], imgsz=1280)
        self.assertEqual(self.module.get_training_image_size(weights), 1280)
        unknown = self._run("detection", "other", "finetune")
        self.assertEqual(self.module.get_training_image_size(unknown), 640)
        self.assertEqual(self.module.get_training_image_size(unknown, default=160), 160)

    def test_detection_models_are_listed_by_the_class_they_detect(self):
        self._run("detection", "players", "f1", names=["disc", "player"])
        self._run("detection", "discs", "f1", names={0: "disc"})
        self._run("detection", "digits", "f1", names=[str(d) for d in range(10)])
        self._run("detection", "unknown", "f1")
        with patch.object(self.module, "models_root", return_value=self.root):
            players = [Path(p).parts[1] for p in self.module.find_detection_models("player")]
            discs = [Path(p).parts[1] for p in self.module.find_detection_models("disc")]
        self.assertEqual(players, ["players", "unknown"])
        self.assertEqual(discs, ["discs", "players", "unknown"])

    def test_display_name_tells_apart_finetunes_of_the_same_run(self):
        single = self._run("segmentation", "field_s", "finetune_1")
        first = self._run("segmentation", "field_x", "finetune_1")
        second = self._run("segmentation", "field_x", "finetune_2")
        self.assertEqual(self.module.model_display_name(single), "field_s")
        self.assertEqual(self.module.model_display_name(first), "field_x/finetune_1")
        self.assertEqual(self.module.model_display_name(second), "field_x/finetune_2")
        with patch.object(self.module, "models_root", return_value=self.root):
            self.assertEqual(len(self.module.find_segmentation_models()), 3)


if __name__ == "__main__":
    unittest.main()
