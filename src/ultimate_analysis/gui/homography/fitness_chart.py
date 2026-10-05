"""Line chart of the genetic algorithm's best fitness per generation."""

from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import QWidget

from ...utils.logger import get_logger

try:
    from PyQt5.QtChart import QChart, QChartView, QLineSeries, QValueAxis

    CHARTS_AVAILABLE = True
except ImportError:
    CHARTS_AVAILABLE = False

logger = get_logger("HOMOGRAPHY")


class FitnessChart:
    """Best fitness over the generations. Without PyQt5.QtChart there is no chart
    (view is None) and the methods do nothing."""

    def __init__(self):
        self.view: Optional[QWidget] = None
        if not CHARTS_AVAILABLE:
            logger.warning("PyQt5.QtChart not available, fitness chart disabled")
            return

        chart = QChart()
        chart.setTitle("Fitness Progress")
        chart.setAnimationOptions(QChart.SeriesAnimations)
        chart.setTheme(QChart.ChartThemeDark)

        self._series = QLineSeries()
        self._series.setName("Best Fitness")
        chart.addSeries(self._series)

        self._axis_x = QValueAxis()
        self._axis_x.setLabelFormat("%d")
        self._axis_x.setTitleText("Generation")
        chart.addAxis(self._axis_x, Qt.AlignBottom)
        self._series.attachAxis(self._axis_x)

        self._axis_y = QValueAxis()
        self._axis_y.setLabelFormat("%.3f")
        self._axis_y.setTitleText("Fitness")
        chart.addAxis(self._axis_y, Qt.AlignLeft)
        self._series.attachAxis(self._axis_y)
        self._reset_axes()

        self.view = QChartView()
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setChart(chart)

    def _reset_axes(self) -> None:
        self._axis_x.setRange(0, 10)
        self._axis_y.setRange(0, 1)

    def clear(self) -> None:
        """Remove all points."""
        if self.view is not None:
            self._series.clear()
            self._reset_axes()

    def add_point(self, generation: int, fitness: float) -> None:
        """Add the best fitness of a generation and rescale the axes to the data."""
        if self.view is None:
            return

        self._series.append(generation, fitness)
        if generation > self._axis_x.max():
            self._axis_x.setRange(0, max(10, generation * 1.2))

        # Fit the y axis to the data with 10% padding, at least 0.01
        values = [point.y() for point in self._series.pointsVector()]
        padding = max(0.01, (max(values) - min(values)) * 0.1)
        y_min, y_max = max(0, min(values) - padding), max(values) + padding
        if abs(y_min - self._axis_y.min()) > 0.001 or abs(y_max - self._axis_y.max()) > 0.001:
            self._axis_y.setRange(y_min, y_max)
