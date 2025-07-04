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
        detection_models = find_models("finetune", keyword="object_detection")
        self.model_combo.addItems(detection_models)
        detection_layout.addRow("Inference Model:", self.model_combo)
        detection_group.setLayout(detection_layout)
        layout.addWidget(detection_group)

        # --- Field Segmentation Settings ---
        field_group = QGroupBox("Field Segmentation Settings")
        field_layout = QFormLayout()
        self.field_model_combo = QComboBox()
        field_models = find_models("finetune", keyword="field_finder")
        self.field_model_combo.addItems(field_models)
        field_layout.addRow("Field Segmentation Model:", self.field_model_combo)
        field_group.setLayout(field_layout)
        layout.addWidget(field_group)

        # --- Player Identification Settings ---
        player_id_group = QGroupBox("Player Identification")
        player_id_layout = QFormLayout()
        self.player_id_method_combo = QComboBox()
        self.player_id_method_combo.addItems(["YOLO", "EasyOCR"])
        player_id_layout.addRow("Player ID Method:", self.player_id_method_combo)

        self.player_id_model_combo = QComboBox()
        player_id_models = find_models("finetune", keyword="digit_detector")
        self.player_id_model_combo.addItems(player_id_models)
        player_id_layout.addRow("Player ID YOLO Model:", self.player_id_model_combo)
        player_id_group.setLayout(player_id_layout)
        layout.addWidget(player_id_group)

        # Suggest more settings:
        # - Confidence threshold slider
        # - NMS threshold slider
        # - Frame skip (for faster preview)
        # - Output directory selection
        # - Enable/disable saving results

        self.setLayout(layout)

        # Set combo boxes to current model if possible
        self._set_initial_model_selection()

        # Reset tracker and visualisation when switching type
        self.tracker_combo.currentTextChanged.connect(self._on_tracker_changed)

        # Connect model selection changes to update models in main_tab
        self.model_combo.currentTextChanged.connect(self._on_detection_model_changed)
        self.field_model_combo.currentTextChanged.connect(self._on_field_model_changed)

        # Connect player ID settings
        self.player_id_method_combo.currentTextChanged.connect(self._on_player_id_method_changed)
        self.player_id_model_combo.currentTextChanged.connect(self._on_player_id_model_changed)

    def _set_initial_model_selection(self):
        # Set detection model combo to current model if available
        try:
            from processing.inference import weights_path as detection_weights_path
            if detection_weights_path:
                rel_path = os.path.relpath(detection_weights_path, "finetune")
                idx = self.model_combo.findText(rel_path)
                if idx >= 0:
                    self.model_combo.setCurrentIndex(idx)
        except Exception as e:
            print(f"[DEBUG] Could not set initial detection model selection: {e}")

        # Set field model combo to current model if available
        try:
            from processing.field_segmentation import field_model_path
            if field_model_path:
                rel_path = os.path.relpath(field_model_path, "finetune")
                idx = self.field_model_combo.findText(rel_path)
                if idx >= 0:
                    self.field_model_combo.setCurrentIndex(idx)
        except Exception as e:
            print(f"[DEBUG] Could not set initial field model selection: {e}")

        # Set tracker combo to current tracker if available
        try:
            from processing.tracking import tracker_type
            tracker_type_cap = tracker_type.capitalize()
            idx = self.tracker_combo.findText(tracker_type_cap)
            if idx >= 0:
                self.tracker_combo.setCurrentIndex(idx)
        except Exception as e:
            print(f"[DEBUG] Could not set initial tracker type: {e}")

        # Set player ID method combo to current method if available
        try:
            from processing.player_id import get_player_id_method
            method_cap = get_player_id_method().capitalize()
            idx = self.player_id_method_combo.findText(method_cap)
            if idx >= 0:
                self.player_id_method_combo.setCurrentIndex(idx)
        except Exception as e:
            print(f"[DEBUG] Could not set initial player ID method: {e}")

        # Set player ID model combo to current model if available
        try:
            from processing.player_id import get_player_id_model_path
            model_path = get_player_id_model_path()
            if model_path:
                rel_path = os.path.relpath(model_path, "finetune")
                idx = self.player_id_model_combo.findText(rel_path)
                if idx >= 0:
                    self.player_id_model_combo.setCurrentIndex(idx)
        except Exception as e:
            print(f"[DEBUG] Could not set initial player ID model selection: {e}")

    def _on_tracker_changed(self, t):
        set_tracker_type(t.lower())
        reset_tracker()
        reset_track_histories()
        if self.main_tab is not None:
            self.main_tab.reset_visualisation()

    def _on_detection_model_changed(self, model_path):
        print(f"[DEBUG] SettingsTab: Detection model changed to: {model_path}")
        if self.main_tab is not None and hasattr(self.main_tab, "set_detection_model"):
            self.main_tab.set_detection_model(os.path.join("finetune", model_path))

    def _on_field_model_changed(self, model_path):
        print(f"[DEBUG] SettingsTab: Field segmentation model changed to: {model_path}")
        if self.main_tab is not None and hasattr(self.main_tab, "set_field_model"):
            self.main_tab.set_field_model(os.path.join("finetune", model_path))

    def _on_player_id_method_changed(self, method):
        print(f"[DEBUG] SettingsTab: Player ID method changed to: {method}")
        from processing.player_id import set_player_id_method, set_easyocr
        if method.lower() == "easyocr":
            set_easyocr()
        else:
            set_player_id_method("yolo")

    def _on_player_id_model_changed(self, model_path):
        print(f"[DEBUG] SettingsTab: Player ID YOLO model changed to: {model_path}")
        from processing.player_id import set_player_id_model
        set_player_id_model(os.path.join("finetune", model_path))