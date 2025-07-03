from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton, QListWidget, QGroupBox, QFormLayout
import os

class DevYoloTrainingTab(QWidget):
    def __init__(self, training_data_folder="training_data"):
        super().__init__()
        self.training_data_folder = training_data_folder
        layout = QVBoxLayout()

        # Training data folders
        data_group = QGroupBox("Available Training Data")
        data_layout = QVBoxLayout()
        self.data_list = QListWidget()
        self.populate_data_list()
        data_layout.addWidget(self.data_list)
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)

        # Model selection
        model_group = QGroupBox("Model Selection")
        model_layout = QFormLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(["YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l", "YOLOv8x"])
        model_layout.addRow("Base Model:", self.model_combo)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Training controls (mockup)
        train_btn = QPushButton("Train Model")
        layout.addWidget(train_btn)
        self.setLayout(layout)

    def populate_data_list(self):
        self.data_list.clear()
        if not os.path.exists(self.training_data_folder):
            return
        for f in os.listdir(self.training_data_folder):
            path = os.path.join(self.training_data_folder, f)
            if os.path.isdir(path):
                self.data_list.addItem(f)