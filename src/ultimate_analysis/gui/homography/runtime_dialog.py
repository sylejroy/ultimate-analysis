"""Window listing how long the homography tab's processing steps take."""

from typing import Dict, List

from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class RuntimeDialog(QDialog):
    """Current, average, and maximum duration of each processing step.

    Measurements are recorded whether or not the window is open.
    """

    PROCESSES = (
        "Field Segmentation",
        "Morphological Ops",
        "Line Extraction",
        "Homography Calculation",
        "Homography Display",
    )
    HISTORY = 10  # Measurements the average and maximum are taken over

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self._times: Dict[str, List[float]] = {process: [] for process in self.PROCESSES}

        self.setWindowTitle("Processing Runtime Performance")
        self.setModal(False)  # The tab stays usable while this is open
        self.resize(500, 350)
        self.setStyleSheet(
            """
            QDialog {
                background-color: #1a1a1a;
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
                font-size: 12px;
            }
            QTableWidget {
                background-color: #000000;
                color: #ffffff;
                gridline-color: #333333;
                border: 1px solid #555555;
                selection-background-color: #2c5aa0;
            }
            QTableWidget::item {
                padding: 4px;
                border-bottom: 1px solid #333333;
            }
            QTableWidget QHeaderView::section {
                background-color: #2a2a2a;
                color: #ffffff;
                padding: 4px;
                border: 1px solid #555555;
                font-weight: bold;
            }
        """
        )

        layout = QVBoxLayout()

        title_label = QLabel("Processing Runtime Performance (ms)")
        title_label.setStyleSheet("font-size: 14px; font-weight: bold; margin: 10px;")
        layout.addWidget(title_label)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Process", "Current", "Average", "Max"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setRowCount(len(self.PROCESSES))
        for row, process in enumerate(self.PROCESSES):
            self.table.setItem(row, 0, QTableWidgetItem(process))
            for column in (1, 2, 3):
                self.table.setItem(row, column, QTableWidgetItem("0.0"))

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Fixed)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.setColumnWidth(1, 80)
        self.table.setColumnWidth(2, 80)
        layout.addWidget(self.table)

        info_label = QLabel(
            "Real-time performance monitoring. Window can be kept open while using the application."
        )
        info_label.setStyleSheet("font-size: 10px; color: #cccccc; margin: 5px;")
        layout.addWidget(info_label)

        close_button = QPushButton("Close")
        close_button.setStyleSheet(
            """
            QPushButton {
                background-color: #555555;
                color: white;
                border: 1px solid #777777;
                border-radius: 3px;
                padding: 6px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #666666;
            }
            QPushButton:pressed {
                background-color: #444444;
            }
        """
        )
        close_button.clicked.connect(self.close)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def add_measurement(self, process: str, duration_ms: float) -> None:
        """Record how long a processing step took and refresh its row."""
        times = self._times[process]
        times.append(duration_ms)
        if len(times) > self.HISTORY:
            times.pop(0)

        row = self.PROCESSES.index(process)
        for column, value in ((1, times[-1]), (2, sum(times) / len(times)), (3, max(times))):
            self.table.setItem(row, column, QTableWidgetItem(f"{value:.1f}"))
