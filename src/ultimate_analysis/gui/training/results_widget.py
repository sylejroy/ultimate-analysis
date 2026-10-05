"""Live plots of a training run's metrics, read from its results.csv."""

from pathlib import Path
from typing import Optional

import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget

from ...utils.logger import get_logger

logger = get_logger("TRAINING")

BACKGROUND = "#2d2d2d"
PLOT_BACKGROUND = "#1e1e1e"
TEXT = "#dddddd"
GRID = "#444444"

# Metric columns of results.csv: (column, label, colour)
ACCURACY_CURVES = (
    ("metrics/mAP50(B)", "mAP50", "#42a5f5"),
    ("metrics/mAP50-95(B)", "mAP50-95", "#ffb74d"),
)
DETECTION_CURVES = (
    ("metrics/precision(B)", "Precision", "#66bb6a"),
    ("metrics/recall(B)", "Recall", "#ef5350"),
)
# Loss columns: (name, label, colour). YOLO runs report box, cls, and dfl or l1 losses;
# RT-DETR runs report giou, cls, and l1.
LOSS_CURVES = (
    ("box_loss", "Box", "#42a5f5"),
    ("giou_loss", "GIoU", "#42a5f5"),
    ("cls_loss", "Class", "#ef5350"),
    ("dfl_loss", "DFL", "#66bb6a"),
    ("l1_loss", "L1", "#66bb6a"),
)
# Validation metrics jump from epoch to epoch when the validation set is small; the bold
# line is their average over this many epochs
SMOOTHING_EPOCHS = 5


def best_epoch(results: pd.DataFrame) -> Optional[int]:
    """Row of the epoch Ultralytics keeps as best.pt (highest fitness)."""
    if "metrics/mAP50-95(B)" not in results or "metrics/mAP50(B)" not in results:
        return None
    fitness = 0.1 * results["metrics/mAP50(B)"] + 0.9 * results["metrics/mAP50-95(B)"]
    return int(fitness.idxmax()) if len(fitness) else None


def smooth(values: pd.Series) -> pd.Series:
    return values.rolling(SMOOTHING_EPOCHS, center=True, min_periods=1).mean()


class TrainingResultsWidget(QWidget):
    """Plots of a run's accuracy, precision and recall, and losses, next to another run."""

    def __init__(self):
        super().__init__()
        self.results_path: Optional[Path] = None
        self.results_dir: Optional[Path] = None
        self.reference_path: Optional[Path] = None  # Run to compare against
        self.reference_name = ""
        self._planned_epochs: Optional[int] = None
        self._patience: Optional[int] = None

        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        self.figure = Figure(figsize=(12, 8), facecolor=BACKGROUND)
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout()
        layout.addWidget(self.summary_label)
        layout.addWidget(self.canvas, 1)
        self.setLayout(layout)

        # Timer for updating plots
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_plots)

    def start_monitoring(
        self, results_dir: str, epochs: Optional[int] = None, patience: Optional[int] = None
    ) -> None:
        """Follow the results.csv a running training writes somewhere in a folder.

        Args:
            results_dir: Folder the run is written to
            epochs: Planned number of epochs, for the width of the plots
            patience: Epochs without improvement after which the run stops early
        """
        self.results_dir = Path(results_dir)
        self.results_path = None
        self._planned_epochs = epochs
        self._patience = patience
        self.update_timer.start(2000)  # Update every 2 seconds

    def stop_monitoring(self) -> None:
        """Stop following the run; the plots stay as they are."""
        self.update_timer.stop()

    def set_reference(self, results_path: Optional[str], name: str = "") -> None:
        """Choose the run the current one is compared with (None for no comparison)."""
        self.reference_path = Path(results_path) if results_path else None
        self.reference_name = name
        self.update_plots()

    def show_run(self, results_path: str) -> None:
        """Show a finished run."""
        self.update_timer.stop()
        self.results_path = Path(results_path)
        self._planned_epochs = self._patience = None
        self.update_plots()

    # ------------------------------------------------------------------ reading

    def _read(self, path: Optional[Path]) -> Optional[pd.DataFrame]:
        if path is None or not path.exists():
            return None
        try:
            results = pd.read_csv(path)
        except Exception as e:  # The file is being written while it is read
            logger.debug(f"Could not read {path}: {e}")
            return None
        results.columns = [column.strip() for column in results.columns]
        return results if len(results) else None

    def _find_results_csv(self) -> None:
        if self.results_dir is not None and self.results_dir.exists():
            found = sorted(self.results_dir.rglob("results.csv"))
            if found:
                self.results_path = found[-1]

    # ------------------------------------------------------------------ drawing

    def update_plots(self) -> None:
        """Draw the plots from the current state of the results files."""
        if self.results_path is None or not self.results_path.exists():
            self._find_results_csv()
        results = self._read(self.results_path)
        reference = self._read(self.reference_path)
        if results is None and reference is None:
            return

        self.figure.clear()
        grid = self.figure.add_gridspec(2, 2, width_ratios=(3, 2))
        accuracy = self.figure.add_subplot(grid[:, 0])
        detection = self.figure.add_subplot(grid[0, 1])
        losses = self.figure.add_subplot(grid[1, 1])

        best = best_epoch(results) if results is not None else None
        for axis, curves in ((accuracy, ACCURACY_CURVES), (detection, DETECTION_CURVES)):
            for column, label, color in curves:
                if reference is not None and column in reference:
                    axis.plot(
                        reference.index + 1,
                        smooth(reference[column]),
                        color=color,
                        linestyle="--",
                        linewidth=1.2,
                        alpha=0.55,
                        label=f"{label}, compared run",
                    )
                if results is not None and column in results:
                    epochs = results.index + 1
                    axis.plot(epochs, results[column], color=color, linewidth=0.8, alpha=0.35)
                    axis.plot(
                        epochs, smooth(results[column]), color=color, linewidth=2.2, label=label
                    )
                    if best is not None and axis is accuracy:
                        axis.plot(best + 1, results[column][best], "o", color=color, markersize=8)
        if best is not None:
            accuracy.axvline(best + 1, color=TEXT, linewidth=0.8, linestyle=":")

        if results is not None:
            for name, label, color in LOSS_CURVES:
                if f"train/{name}" in results:
                    losses.plot(
                        results.index + 1,
                        results[f"train/{name}"],
                        color=color,
                        linewidth=1.8,
                        label=label,
                    )
                if f"val/{name}" in results:
                    losses.plot(
                        results.index + 1,
                        results[f"val/{name}"],
                        color=color,
                        linewidth=1.2,
                        linestyle=":",
                    )
            # The first epochs of a new detection head are off the scale of all the others
            settled = results.filter(like="_loss").iloc[min(3, len(results) - 1) :]
            if len(settled) and settled.max().max() > 0:
                losses.set_ylim(0, float(settled.max().max()) * 1.1)

        titles = (
            (accuracy, "Accuracy on the validation set"),
            (detection, "Precision and recall"),
            (losses, "Losses (solid: training, dotted: validation)"),
        )
        for axis, title in titles:
            axis.set_facecolor(PLOT_BACKGROUND)
            axis.set_title(title, color=TEXT, fontsize=10)
            axis.set_xlabel("Epoch", color=TEXT, fontsize=9)
            axis.tick_params(colors=TEXT, labelsize=8)
            axis.grid(True, color=GRID, linewidth=0.6)
            for spine in axis.spines.values():
                spine.set_color(GRID)
            if self._planned_epochs:
                axis.set_xlim(0, self._planned_epochs + 1)
            handles, _ = axis.get_legend_handles_labels()
            if handles:
                legend = axis.legend(fontsize=8, facecolor=PLOT_BACKGROUND, edgecolor=GRID)
                for text in legend.get_texts():
                    text.set_color(TEXT)
        detection.set_ylim(0, 1)
        # Room above the highest curve, not all the way up to a score nobody reaches
        highest = max(
            (max(line.get_ydata()) for line in accuracy.get_lines() if len(line.get_ydata()) > 2),
            default=1.0,
        )
        accuracy.set_ylim(0, min(1.0, max(0.1, float(highest) * 1.2)))

        self.figure.tight_layout()
        self.canvas.draw()
        self.summary_label.setText(self._summary(results, best))

    def _summary(self, results: Optional[pd.DataFrame], best: Optional[int]) -> str:
        """What the plots say in one line: the best epoch and how long until the run stops."""
        if results is None or best is None:
            return f"Compared run: {self.reference_name}" if self.reference_name else ""
        text = (
            f"Best so far: mAP50 {results['metrics/mAP50(B)'][best]:.3f}, "
            f"mAP50-95 {results['metrics/mAP50-95(B)'][best]:.3f} at epoch {best + 1} "
            f"of {len(results)} run."
        )
        if self._patience and self.update_timer.isActive():
            waited = len(results) - 1 - best
            text += f" {waited} epochs without improvement; the run stops at {self._patience}."
        if self.reference_name:
            text += f"  Dashed: {self.reference_name}."
        return text
