import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PyQt5.QtWidgets import QApplication, QTabWidget
from main_tab import MainTab
from settings_tab import SettingsTab
from dev_runtimes_tab import DevRuntimesTab
from dev_video_preprocessing_tab import DevVideoPreprocessingTab
from dev_yolo_training_tab import DevYoloTrainingTab

class VisualizationApp(QTabWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ultimate Analysis Visualization")
        self.showMaximized()
        self.dev_runtimes_tab = DevRuntimesTab()
        self.main_tab = MainTab(dev_runtimes_tab=self.dev_runtimes_tab)
        self.settings_tab = SettingsTab(main_tab=self.main_tab)
        self.addTab(self.main_tab, "Main")
        self.addTab(self.settings_tab, "Settings")
        self.addTab(self.dev_runtimes_tab, "Dev-Runtimes")
        self.dev_video_preprocessing_tab = DevVideoPreprocessingTab()
        self.addTab(self.dev_video_preprocessing_tab, "Dev-Video Preprocessing")
        self.dev_yolo_training_tab = DevYoloTrainingTab()
        self.addTab(self.dev_yolo_training_tab, "Dev-YOLO Training")
        # Add more tabs as needed

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VisualizationApp()
    window.show()
    sys.exit(app.exec_())