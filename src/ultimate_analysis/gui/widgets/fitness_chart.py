"""Reusable fitness chart widget for GA progress.

Encapsulates PyQtChart setup and provides a small API:
- widget: returns the chart view (or a QLabel if unavailable)
- add_point(generation: int, fitness: float)
- clear(): clears data and resets axes
"""

from __future__ import annotations

from typing import Any, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import QLabel

try:  # Optional dependency
    from PyQt5.QtChart import QChart, QChartView, QLineSeries, QValueAxis

    _CHARTS_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    _CHARTS_AVAILABLE = False
    QChart = QChartView = QLineSeries = QValueAxis = None  # type: ignore


class FitnessChart:
    """Lightweight wrapper around a line chart for fitness evolution."""

    def __init__(self) -> None:
        self._available = _CHARTS_AVAILABLE
        self._view: Optional[Any] = None
        self._series = None
        self._axis_x = None
        self._axis_y = None

        if not self._available:
            # Fallback label when QtCharts isn't installed
            self._fallback = QLabel("Fitness chart unavailable\n(PyQt5.QtChart not installed)")
            self._fallback.setAlignment(Qt.AlignCenter)
            self._fallback.setStyleSheet("color: #888; font-size: 10px; margin: 10px;")
            return

        # Build chart components
        self._view = QChartView()
        self._view.setRenderHint(QPainter.Antialiasing)

        self._chart = QChart()
        self._chart.setTitle("Fitness Progress")
        self._chart.setAnimationOptions(QChart.SeriesAnimations)
        self._chart.setTheme(QChart.ChartThemeDark)

        self._series = QLineSeries()
        self._series.setName("Best Fitness")
        self._chart.addSeries(self._series)

        self._axis_x = QValueAxis()
        self._axis_x.setLabelFormat("%d")
        self._axis_x.setTitleText("Generation")
        self._axis_x.setRange(0, 10)
        self._chart.addAxis(self._axis_x, Qt.AlignBottom)
        self._series.attachAxis(self._axis_x)

        self._axis_y = QValueAxis()
        self._axis_y.setLabelFormat("%.3f")
        self._axis_y.setTitleText("Fitness")
        self._axis_y.setRange(0, 1)
        self._chart.addAxis(self._axis_y, Qt.AlignLeft)
        self._series.attachAxis(self._axis_y)

        self._view.setChart(self._chart)

    @property
    def widget(self):
        """Return the widget to embed in layouts (QChartView or QLabel fallback)."""
        return self._view if self._available else self._fallback

    def clear(self) -> None:
        if not self._available or self._series is None:
            return
        self._series.clear()
        self._axis_x.setRange(0, 10)
        self._axis_y.setRange(0, 1)

    def add_point(self, generation: int, fitness: float) -> None:
        if not self._available or self._series is None:
            return
        self._series.append(generation, fitness)

        # Auto-scale X
        if generation > self._axis_x.max():  # type: ignore[union-attr]
            new_range = max(10, int(generation * 1.2))
            self._axis_x.setRange(0, new_range)

        # Auto-scale Y based on data
        points = self._series.pointsVector()
        if points:
            ys = [p.y() for p in points]
            y_min = min(ys)
            y_max = max(ys)
            padding = max(0.01, (y_max - y_min) * 0.1)
            self._axis_y.setRange(max(0.0, y_min - padding), y_max + padding)
