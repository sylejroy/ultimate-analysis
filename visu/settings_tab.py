from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QComboBox, QGroupBox, QFormLayout
import os
from processing.tracking import set_tracker_type, reset_tracker
from visu.tracking_visualisation import reset_track_histories
from visu.main_tab import MainTab

def find_models(root_folder, keyword=None):
    models = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        if "best.pt" in filenames:
            if keyword is None or keyword in dirpath:
                models.append(os.path.relpath(os.path.join(dirpath, "best.pt"), root_folder))
    return models if models else ["None found"]

class SettingsTab(QWidget):
    def __init__(self, parent=None, main_tab=None):
        super().__init__(parent)
        self.main_tab = main_tab
        layout = QVBoxLayout()

        # --- General Settings ---
        general_group = QGroupBox("General Settings")
        general_layout = QFormLayout()
        # Add more general settings here
        general_group.setLayout(general_layout)
        layout.addWidget(general_group)

        # --- Tracker Settings ---
        tracker_group = QGroupBox("Tracker Settings")
        tracker_layout = QFormLayout()
        self.tracker_combo = QComboBox()
        self.tracker_combo.addItems(["DeepSort", "Histogram"])
        tracker_layout.addRow("Tracker Type:", self.tracker_combo)
        tracker_group.setLayout(tracker_layout)
        layout.addWidget(tracker_group)

        # --- Detection Settings ---
        detection_group = QGroupBox("Detection Settings")
        detection_layout = QFormLayout()
        self.model_combo = QComboBox()
        # For detection models
        detection_models = find_models("finetune", keyword="object_detection")
        self.model_combo.addItems(detection_models)
        detection_layout.addRow("Inference Model:", self.model_combo)
        detection_group.setLayout(detection_layout)
        layout.addWidget(detection_group)

        # --- Field Segmentation Settings ---
        field_group = QGroupBox("Field Segmentation Settings")
        field_layout = QFormLayout()
        self.field_model_combo = QComboBox()
        # For field segmentation models
        field_models = find_models("finetune", keyword="field_finder")
        self.field_model_combo.addItems(field_models)
        field_layout.addRow("Field Segmentation Model:", self.field_model_combo)
        field_group.setLayout(field_layout)
        layout.addWidget(field_group)

        # Suggest more settings:
        # - Confidence threshold slider
        # - NMS threshold slider
        # - Frame skip (for faster preview)
        # - Output directory selection
        # - Enable/disable saving results

        self.setLayout(layout)

        # Reset tracker and visualisation when switching type
        self.tracker_combo.currentTextChanged.connect(self._on_tracker_changed)

    def _on_tracker_changed(self, t):
        set_tracker_type(t.lower())
        reset_tracker()
        reset_track_histories()
        if self.main_tab is not None:
            self.main_tab.reset_visualisation()

class VisualisationApp(QWidget):
    def __init__(self):
        super().__init__()
        # ... other initializations ...
        self.main_tab = MainTab()
        self.settings_tab = SettingsTab(main_tab=self.main_tab)
        self.addTab(self.main_tab, "Main")
        self.addTab(self.settings_tab, "Settings")
        # ... rest of the code ...