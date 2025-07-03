from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QPushButton

class DevRuntimesTab(QWidget):
    def __init__(self, main_tab=None):
        super().__init__()
        self.main_tab = main_tab
        layout = QVBoxLayout()
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Step", "Last Runtime (ms)", "Average Runtime (ms)"])
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
        if not live:
            self.table.resizeColumnsToContents()