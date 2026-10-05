"""Runs a training subprocess off the GUI thread and turns its output into progress."""

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal

from ...utils.logger import get_logger

logger = get_logger("TRAINING")

ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


# Output that marks the switch from dataset preparation to training
TRAINING_START_KEYWORDS = (
    "starting training",
    "train:",
    "optimizer:",
    "lr0=",
    "momentum=",
    "ultralytics yolo",
    "model summary:",
    "freezing",
    "amp:",
    "image sizes",
    "tensorboard:",
)


# Output printed while the dataset is scanned and cached
PREPARATION_KEYWORDS = (
    "scanning",
    "loading",
    "cache",
    "labels",
    "dataset",
    "images",
    "caching",
    "reading",
    "found",
    "missing",
    "empty",
    "checking",
)


class ModelTrainingThread(QThread):
    """Background thread for model training to prevent GUI freezing."""

    progress_update = pyqtSignal(int, str)  # epoch, status message
    raw_output = pyqtSignal(str)  # raw training output line
    training_complete = pyqtSignal(str, bool)  # results_path, success
    error_occurred = pyqtSignal(str)  # error message

    def __init__(
        self,
        task_type: str,
        model_path: str,
        data_path: str,
        epochs: int,
        patience: int,
        batch_size: float,
        lr: float,
        output_dir: str,
        training_params: dict = None,
    ):
        super().__init__()
        self.task_type = task_type
        self.model_path = model_path
        self.data_path = data_path
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.lr = lr
        self.output_dir = output_dir
        self.training_params = training_params or {}
        self.should_stop = False
        self._training_started = False
        self._last_prep_update = 0.0
        self._fallback_sent = False

    def run(self):
        """Run the training process."""

        try:
            logger.info("Starting model training in separate process...")

            # Create training configuration
            training_config = {
                "model_path": self.model_path,
                "data_path": self.data_path,
                "output_dir": self.output_dir,
                "epochs": self.epochs,
                "patience": self.patience,
                "batch_size": self.batch_size,
                "learning_rate": self.lr,
                "device": self.training_params.get("device", 0),
                "workers": self.training_params.get("workers", 0),
                "imgsz": self.training_params.get("imgsz", 640),
                "optimizer": self.training_params.get("optimizer", "SGD"),
                "momentum": self.training_params.get("momentum", 0.937),
                "weight_decay": self.training_params.get("weight_decay", 0.0005),
                "augment": self.training_params.get("augment", True),
                "cosine_lr": self.training_params.get("cosine_lr", False),
                "mosaic": self.training_params.get("mosaic", 1.0),
                "scale": self.training_params.get("scale", 0.5),
                "mixup": self.training_params.get("mixup", 0.0),
                "copy_paste": self.training_params.get("copy_paste", 0.0),
                "hsv_h": self.training_params.get("hsv_h", 0.015),
                "hsv_s": self.training_params.get("hsv_s", 0.7),
                "hsv_v": self.training_params.get("hsv_v", 0.4),
                "results_file": "training_results.txt",
            }

            # Write config to temporary file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(training_config, f, indent=2)
                config_path = f.name

            # The script runs as its own process: ultimate_analysis/training/
            script_path = (
                Path(__file__).resolve().parents[2] / "training" / "train_model_subprocess.py"
            )

            try:
                # Ultralytics prints UTF-8 progress bars; the Windows default pipe
                # encoding (cp1252) cannot decode them and would abort the run.
                process = subprocess.Popen(
                    [sys.executable, str(script_path), "--config", config_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    cwd=os.getcwd(),
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                )

                epoch_count = 0
                total_epochs = self.epochs
                start_time = time.time()

                while True:
                    if self.should_stop:
                        process.terminate()
                        process.wait()
                        break

                    # Read output line by line
                    line = process.stdout.readline()

                    if line == "" and process.poll() is not None:
                        break

                    if line:
                        # Ultralytics prefixes progress lines with terminal control codes
                        line = ANSI_ESCAPE_PATTERN.sub("", line).strip()
                        current_time = time.time()

                        # Emit raw output for display
                        if line:  # Only emit non-empty lines
                            self.raw_output.emit(line)

                        # Simple progress detection
                        epoch_count = self._process_training_line(
                            line, current_time, start_time, epoch_count, total_epochs
                        )

                # Check result
                return_code = process.wait()

                if return_code == 0 and not self.should_stop:
                    # Read results path
                    results_path = self.output_dir
                    if os.path.exists("training_results.txt"):
                        with open("training_results.txt", "r") as f:
                            results_path = f.read().strip()
                        os.remove("training_results.txt")

                    self.training_complete.emit(results_path, True)
                elif self.should_stop:
                    logger.info("Training stopped by user")
                else:
                    self.error_occurred.emit(f"Training failed with return code: {return_code}")

            finally:
                # Cleanup temporary files
                if os.path.exists(config_path):
                    os.remove(config_path)
                if os.path.exists("training_results.txt"):
                    os.remove("training_results.txt")

        except Exception as e:
            logger.error(f"Error: {e}")
            self.error_occurred.emit(str(e))

    def stop_training(self):
        """Request to stop training."""
        self.should_stop = True

    def _process_training_line(
        self, line: str, current_time: float, start_time: float, epoch_count: int, total_epochs: int
    ):
        """Process a single line of training output and emit progress updates."""
        if not line:
            return epoch_count

        lower_line = line.lower()

        if epoch_count == 0:
            # Transition from preparation to training
            if any(keyword in lower_line for keyword in TRAINING_START_KEYWORDS):
                if not self._training_started:
                    self.progress_update.emit(1, "Training starting • Epoch ??/?? • Batch ??/??")
                    self._training_started = True
                return epoch_count

            # During preparation, avoid interpreting numbers as epochs/batches
            if not self._training_started and any(
                keyword in lower_line for keyword in PREPARATION_KEYWORDS
            ):
                # Only update every 3 seconds during prep to avoid spam
                if current_time - self._last_prep_update > 3:
                    self.progress_update.emit(0, "Preparing training • Epoch ??/?? • Batch ??/??")
                    self._last_prep_update = current_time
                return epoch_count

        # Training lines begin with the epoch counter: "1/120  5.2G ... 10% ━── 14/134 1.8it/s"
        epoch_match = re.match(r"(\d+)/(\d+)\s", line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            total_epochs = int(epoch_match.group(2)) or total_epochs

            if current_epoch != epoch_count and 1 <= current_epoch <= total_epochs:
                # Epochs completed so far; batch lines fill in the current epoch
                progress_percent = int(((current_epoch - 1) / total_epochs) * 100)
                self.progress_update.emit(
                    progress_percent, f"Training • Epoch {current_epoch}/{total_epochs}"
                )
                self._training_started = True
                return current_epoch

            # Batch progress within the epoch. The validation bar uses the same layout but
            # does not begin with the epoch counter, so it never moves the progress.
            batch_match = re.search(r"(\d+)%.*?(\d+)/(\d+)", line)
            if batch_match and epoch_count > 0:
                batch_percent = int(batch_match.group(1))
                overall_progress = min(
                    100, int((epoch_count - 1 + batch_percent / 100) / total_epochs * 100)
                )

                status = (
                    f"Training • Epoch {epoch_count}/{total_epochs}"
                    f" • Batch {batch_match.group(2)}/{batch_match.group(3)}"
                )
                rate_match = re.search(r"([\d.]+)(it/s|s/it)", line)
                if rate_match:
                    status += f" • {rate_match.group(1)}{rate_match.group(2)}"
                self.progress_update.emit(overall_progress, status)
                return epoch_count

        # Fallback: if training has been running for a while without clear state detection
        if current_time - start_time > 30 and epoch_count == 0 and not self._fallback_sent:
            if self._training_started:
                self.progress_update.emit(2, "Training • Epoch ??/?? • Batch ??/??")
            else:
                self.progress_update.emit(1, "Training starting • Epoch ??/?? • Batch ??/??")
                self._training_started = True
            self._fallback_sent = True

        return epoch_count
