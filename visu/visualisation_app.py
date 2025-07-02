import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PyQt5.QtWidgets import QApplication, QTabWidget
from main_tab import MainTab
from settings_tab import SettingsTab

class VisualizationApp(QTabWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ultimate Analysis Visualization")
        self.showMaximized()
        self.main_tab = MainTab()
        self.settings_tab = SettingsTab()
        self.addTab(self.main_tab, "Main")
        self.addTab(self.settings_tab, "Settings")
        # Add more tabs as needed

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VisualizationApp()
    window.show()
    sys.exit(app.exec_())