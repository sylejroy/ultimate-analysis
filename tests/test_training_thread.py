"""The training thread ends its subprocess itself when asked to stop. A training that
prints nothing keeps `run` waiting in a line read, which only returns when the process
ends; without that the window would wait on the thread for as long as the training."""

import subprocess
import sys
import time
import unittest
from unittest.mock import patch

from support import load_module

SILENT = [sys.executable, "-c", "import time; time.sleep(60)"]
REAL_POPEN = subprocess.Popen


class TrainingThreadStopTest(unittest.TestCase):
    def test_stop_ends_a_silent_subprocess(self):
        module = load_module("gui.training.training_thread")
        thread = module.ModelTrainingThread("detection", "m.pt", "d.yaml", 1, 1, 1.0, 0.01, "out")

        def silent(command, **kwargs):
            return REAL_POPEN(SILENT, **kwargs)

        with patch.object(subprocess, "Popen", side_effect=silent):
            thread.start()
            for _ in range(100):  # Until the process has been started
                if thread._process is not None:
                    break
                time.sleep(0.05)
            process = thread._process
            self.assertIsNotNone(process)
            thread.stop_training()
            self.assertTrue(thread.wait(5000), "The thread kept waiting for a line of output")
        self.assertIsNotNone(process.poll())
        self.assertIsNone(thread._process)


if __name__ == "__main__":
    unittest.main()
