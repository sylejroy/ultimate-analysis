"""
Runtime monitoring dialog for the Ultimate Analysis GUI.
"""
import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem, QLabel
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen, QFont
from typing import Dict, List


class RuntimesDialog(QDialog):
    """
    Dialog for displaying processing and visualization runtimes.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Processing & Visualisation Runtimes")
        self.setMinimumWidth(700)
        self.setMinimumHeight(400)
        self.resize(900, 500)
        
        self.runtimes: Dict[str, List[float]] = {}
        self._setup_ui()
        self._setup_timer()

    def _setup_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout()
        
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Step", "Last Runtime (ms)", "Max (ms)", "History"])
        layout.addWidget(self.table)
        
        self.setLayout(layout)

    def _setup_timer(self):
        """Setup the refresh timer."""
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_table)
        self.timer.start(1000)  # Update every 1 second

    def create_sparkline(self, values: List[float], width: int = 60, height: int = 18, color: str = '#6cf') -> QLabel:
        """
        Create a sparkline widget for runtime history visualization.
        
        Args:
            values: List of runtime values
            width: Sparkline width in pixels
            height: Sparkline height in pixels
            color: Line color as hex string
            
        Returns:
            QLabel widget containing the sparkline
        """
        if not values:
            pixmap = QPixmap(width, height)
            pixmap.fill(Qt.transparent)
            label = QLabel()
            label.setPixmap(pixmap)
            label.setFixedSize(width, height)
            label.setStyleSheet("background: transparent;")
            return label
        
        # Process values for sparkline
        arr = np.array(values[-width:])
        arr = arr[-width:] if len(arr) > width else arr
        arr = np.pad(arr, (width - len(arr), 0), 'constant', constant_values=(arr[0] if len(arr) else 0,))
        
        min_v, max_v = float(np.min(arr)), float(np.max(arr))
        if max_v == min_v:
            min_v -= 1
            max_v += 1
        norm = (arr - min_v) / (max_v - min_v)
        
        # Create pixmap and draw sparkline
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        
        # Draw main sparkline
        pen = QPen(QColor(color))
        pen.setWidth(2)
        painter.setPen(pen)
        
        points = [
            (i, height - 2 - int(n * (height - 8)))
            for i, n in enumerate(norm)
        ]
        
        for i in range(1, len(points)):
            painter.drawLine(points[i-1][0], points[i-1][1], points[i][0], points[i][1])
        
        # Draw units label
        painter.setPen(QColor("#aaa"))
        font = QFont()
        font.setPointSize(7)
        painter.setFont(font)
        painter.drawText(width-20, height-2, "ms")
        painter.end()
        
        label = QLabel()
        label.setPixmap(pixmap)
        label.setFixedSize(width, height)
        label.setStyleSheet("background: transparent;")
        return label

    def log_runtime(self, step: str, runtime_ms: float):
        """
        Log a runtime measurement for a processing step.
        
        Args:
            step: Name of the processing step
            runtime_ms: Runtime in milliseconds
        """
        if step not in self.runtimes:
            self.runtimes[step] = []
        
        self.runtimes[step].append(runtime_ms)
        
        # Keep only last 100 measurements
        if len(self.runtimes[step]) > 100:
            self.runtimes[step] = self.runtimes[step][-100:]

    def refresh_table(self):
        """Refresh the runtime table display."""
        self.table.setRowCount(0)
        
        for step, times in self.runtimes.items():
            last = times[-1] if times else 0
            max_v = max(times) if times else 0
            
            row = self.table.rowCount()
            self.table.insertRow(row)
            
            self.table.setItem(row, 0, QTableWidgetItem(step))
            self.table.setItem(row, 1, QTableWidgetItem(f"{last:.1f}"))
            self.table.setItem(row, 2, QTableWidgetItem(f"{max_v:.1f}"))
            
            # Add sparkline widget for history
            sparkline = self.create_sparkline(times)
            self.table.setCellWidget(row, 3, sparkline)
        
        self.table.resizeColumnsToContents()
        self.table.resizeRowsToContents()

    def closeEvent(self, event):
        """Handle dialog close event."""
        self.timer.stop()
        super().closeEvent(event)
