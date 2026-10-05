"""Live plots of a training run's metrics, read from its results.csv."""

from pathlib import Path
from typing import Optional

import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QVBoxLayout, QWidget

from ...utils.logger import get_logger

logger = get_logger("TRAINING")

# Loss columns of results.csv: (name, label, train colour, validation colour).
# YOLO runs report box, cls, and dfl losses; RT-DETR runs report giou, cls, and l1.
LOSS_CURVES = (
    ("box_loss", "Box", "blue", "cyan"),
    ("giou_loss", "GIoU", "blue", "cyan"),
    ("cls_loss", "Cls", "red", "magenta"),
    ("dfl_loss", "DFL", "green", "orange"),
    ("l1_loss", "L1", "green", "orange"),
)


class TrainingResultsWidget(QWidget):
    """Widget for displaying live training results from results.csv"""

    def __init__(self):
        super().__init__()
        self.results_path = None
        self.reference_path: Optional[Path] = None  # Baseline run to compare against
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # Timer for updating plots
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_plots)

    def start_monitoring(self, results_dir: str):
        """Start monitoring a results directory for CSV updates."""
        self.results_dir = Path(results_dir)
        self.results_path = None
        self.update_timer.start(2000)  # Update every 2 seconds

    def stop_monitoring(self):
        """Stop monitoring for updates."""
        self.update_timer.stop()
        self.results_path = None

    def set_reference_path(self, reference_path: str):
        """Set the reference results.csv file for comparison."""
        self.reference_path = Path(reference_path)
        logger.debug(f"Reference path set to: {self.reference_path}")

    def update_plots(self):
        """Update the plots with latest data from results.csv"""
        # First, try to find the results.csv file
        if not self.results_path or not self.results_path.exists():
            self._find_results_csv()

        if not self.results_path or not self.results_path.exists():
            return

        try:
            # Read the current training CSV file
            df = pd.read_csv(self.results_path)

            # Load reference data
            reference_df = None
            if self.reference_path is not None and self.reference_path.exists():
                try:
                    reference_df = pd.read_csv(self.reference_path)
                    logger.debug(
                        f"Loaded reference data with {len(reference_df)} epochs from {self.reference_path}"
                    )
                except Exception as e:
                    logger.error(f"Could not load reference data: {e}")
            else:
                logger.warning(f"Reference file not found: {self.reference_path}")

            if df.empty:
                return

            # Clear the figure
            self.figure.clear()

            # Create subplots
            ax1 = self.figure.add_subplot(2, 2, 1)
            ax2 = self.figure.add_subplot(2, 2, 2)
            ax3 = self.figure.add_subplot(2, 2, 3)
            ax4 = self.figure.add_subplot(2, 2, 4)

            epochs = df.index + 1

            # Plot training and validation losses
            for name, label, train_color, val_color in LOSS_CURVES:
                if f"train/{name}" in df.columns:
                    ax1.plot(
                        epochs,
                        df[f"train/{name}"],
                        label=f"Train {label} Loss",
                        color=train_color,
                        linewidth=2,
                    )
                if f"val/{name}" in df.columns:
                    ax1.plot(
                        epochs,
                        df[f"val/{name}"],
                        label=f"Val {label} Loss",
                        color=val_color,
                        linestyle="--",
                        linewidth=2,
                    )

            # Add reference data to loss plot
            if reference_df is not None:
                ref_epochs = reference_df.index + 1
                if "train/box_loss" in reference_df.columns:
                    ax1.plot(
                        ref_epochs,
                        reference_df["train/box_loss"],
                        label="Ref Train Box",
                        color="lightblue",
                        alpha=0.6,
                        linestyle="--",
                        linewidth=1.5,
                    )
                if "val/box_loss" in reference_df.columns:
                    ax1.plot(
                        ref_epochs,
                        reference_df["val/box_loss"],
                        label="Ref Val Box",
                        color="lightcyan",
                        alpha=0.6,
                        linestyle="--",
                        linewidth=1.5,
                    )

            ax1.set_title("Loss Curves")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.legend()
            ax1.grid(True)

            # Plot mAP metrics
            if "metrics/mAP50(B)" in df.columns:
                ax2.plot(epochs, df["metrics/mAP50(B)"], label="mAP@0.5", color="blue", linewidth=2)
            if "metrics/mAP50-95(B)" in df.columns:
                ax2.plot(
                    epochs,
                    df["metrics/mAP50-95(B)"],
                    label="mAP@0.5:0.95",
                    color="red",
                    linewidth=2,
                )

            # Add reference mAP data
            if reference_df is not None:
                if "metrics/mAP50(B)" in reference_df.columns:
                    ax2.plot(
                        ref_epochs,
                        reference_df["metrics/mAP50(B)"],
                        label="Ref mAP@0.5",
                        color="lightblue",
                        alpha=0.6,
                        linestyle="--",
                        linewidth=1.5,
                    )
                if "metrics/mAP50-95(B)" in reference_df.columns:
                    ax2.plot(
                        ref_epochs,
                        reference_df["metrics/mAP50-95(B)"],
                        label="Ref mAP@0.5:0.95",
                        color="lightcoral",
                        alpha=0.6,
                        linestyle="--",
                        linewidth=1.5,
                    )

            ax2.set_title("mAP Metrics")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("mAP")
            ax2.legend()
            ax2.grid(True)

            # Plot precision and recall
            if "metrics/precision(B)" in df.columns:
                ax3.plot(
                    epochs,
                    df["metrics/precision(B)"],
                    label="Precision",
                    color="green",
                    linewidth=2,
                )
            if "metrics/recall(B)" in df.columns:
                ax3.plot(
                    epochs, df["metrics/recall(B)"], label="Recall", color="orange", linewidth=2
                )

            # Add reference precision/recall data
            if reference_df is not None:
                if "metrics/precision(B)" in reference_df.columns:
                    ax3.plot(
                        ref_epochs,
                        reference_df["metrics/precision(B)"],
                        label="Ref Precision",
                        color="lightgreen",
                        alpha=0.6,
                        linestyle="--",
                        linewidth=1.5,
                    )
                if "metrics/recall(B)" in reference_df.columns:
                    ax3.plot(
                        ref_epochs,
                        reference_df["metrics/recall(B)"],
                        label="Ref Recall",
                        color="wheat",
                        alpha=0.6,
                        linestyle="--",
                        linewidth=1.5,
                    )

            ax3.set_title("Precision & Recall")
            ax3.set_xlabel("Epoch")
            ax3.set_ylabel("Score")
            ax3.legend()
            ax3.grid(True)

            # Plot learning rate and other metrics
            if "lr/pg0" in df.columns:
                ax4.plot(epochs, df["lr/pg0"], label="Learning Rate", color="purple", linewidth=2)
                ax4.set_ylabel("Learning Rate", color="purple")
                ax4.tick_params(axis="y", labelcolor="purple")

                # Add reference learning rate
                if reference_df is not None and "lr/pg0" in reference_df.columns:
                    ax4.plot(
                        ref_epochs,
                        reference_df["lr/pg0"],
                        label="Ref LR",
                        color="plum",
                        alpha=0.6,
                        linestyle="--",
                        linewidth=1.5,
                    )

                # Add a second y-axis for fitness if available
                if "fitness" in df.columns:
                    ax4_twin = ax4.twinx()
                    ax4_twin.plot(epochs, df["fitness"], label="Fitness", color="red", linewidth=2)
                    ax4_twin.set_ylabel("Fitness", color="red")
                    ax4_twin.tick_params(axis="y", labelcolor="red")

                    # Add reference fitness
                    if reference_df is not None and "fitness" in reference_df.columns:
                        ax4_twin.plot(
                            ref_epochs,
                            reference_df["fitness"],
                            label="Ref Fitness",
                            color="lightcoral",
                            alpha=0.6,
                            linestyle="--",
                            linewidth=1.5,
                        )

            ax4.set_title("Learning Rate & Fitness")
            ax4.set_xlabel("Epoch")
            ax4.legend()
            ax4.grid(True)

            # Adjust layout and refresh
            self.figure.tight_layout()
            self.canvas.draw()

        except Exception as e:
            logger.error(f"Error updating plots: {e}")

    def _find_results_csv(self):
        """Find the results.csv file in subdirectories."""
        if not hasattr(self, "results_dir") or not self.results_dir.exists():
            return

        # Look for results.csv in the directory and subdirectories
        for csv_path in self.results_dir.rglob("results.csv"):
            if csv_path.exists():
                self.results_path = csv_path
                logger.debug(f"Found results.csv at: {csv_path}")
                return

        # Also check direct path
        direct_path = self.results_dir / "results.csv"
        if direct_path.exists():
            self.results_path = direct_path
            logger.debug(f"Found results.csv at: {direct_path}")
