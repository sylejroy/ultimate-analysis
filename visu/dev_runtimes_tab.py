from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QPushButton, QLabel
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen
from PyQt5.QtCore import Qt

def create_sparkline(history, width=60, height=18, color=QColor(100, 200, 255)):
    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.transparent)
    if not history:
        return QLabel(pixmap=pixmap)
    painter = QPainter(pixmap)
    pen = QPen(color, 2)
    painter.setPen(pen)
    n = len(history)
    if n > 1:
        min_val, max_val = min(history), max(history)
        rng = max(max_val - min_val, 1e-6)
        points = [
            (int(i * (width-2) / (n-1) + 1),
             int(height - 2 - (val - min_val) / rng * (height-4)))
            for i, val in enumerate(history)
        ]
        for i in range(n-1):
            painter.drawLine(points[i][0], points[i][1], points[i+1][0], points[i+1][1])
    painter.end()
    label = QLabel()
    label.setPixmap(pixmap)
    label.setStyleSheet("background: transparent;")
    return label

class DevRuntimesTab(QWidget):
    def __init__(self, main_tab=None):
        super().__init__()
        self.main_tab = main_tab
        layout = QVBoxLayout()
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Step", "Last Runtime (ms)", "Average Runtime (ms)", "History"])
        layout.addWidget(self.table)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh)
        layout.addWidget(self.refresh_button)
        self.setLayout(layout)
        self.runtimes = {}  # {step: [list of runtimes]}
        self.refresh()

    def log_runtime(self, step, runtime_ms):
        if step not in self.runtimes:
            self.runtimes[step] = []
        self.runtimes[step].append(runtime_ms)
        if len(self.runtimes[step]) > 100:
            self.runtimes[step] = self.runtimes[step][-100:]
        self.refresh(live=True)

    def refresh(self, live=False):
        self.table.setRowCount(0)
        for step, times in self.runtimes.items():
            last = times[-1] if times else 0
            avg = sum(times) / len(times) if times else 0
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(step))
            self.table.setItem(row, 1, QTableWidgetItem(f"{last:.1f}"))
            self.table.setItem(row, 2, QTableWidgetItem(f"{avg:.1f}"))
            # Add sparkline for history
            sparkline = create_sparkline(times)
            self.table.setCellWidget(row, 3, sparkline)
        if not live:
            self.table.resizeColumnsToContents()