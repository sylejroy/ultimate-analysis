"""
Development YOLO training tab - comprehensive implementation.
"""
import os
import subprocess
import logging
from typing import List, Optional, Dict, Any
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QLabel, QPushButton, 
    QTableWidget, QTableWidgetItem, QFileDialog, QMessageBox, QProgressBar,
    QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QTextEdit, QSplitter, QFrame, QLineEdit, QSlider
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QImage, QFont

logger = logging.getLogger("ultimate_analysis.gui.dev_yolo_training")


class YOLOTrainingWorker(QThread):
    """Worker thread for YOLO training operations."""
    
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    log_updated = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, operation: str, params: Dict[str, Any]):
        super().__init__()
        self.operation = operation
        self.params = params
        self._process = None
        
    def run(self):
        """Run the YOLO training operation."""
        try:
            if self.operation == "train":
                self._train_model()
            elif self.operation == "validate":
                self._validate_model()
            elif self.operation == "export":
                self._export_model()
            elif self.operation == "predict":
                self._predict_model()
            else:
                raise ValueError(f"Unknown operation: {self.operation}")
                
            self.finished.emit(True, "Operation completed successfully")
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            self.finished.emit(False, str(e))
    
    def _train_model(self):
        """Train YOLO model."""
        self.status_updated.emit("Starting YOLO training...")
        
        # Prepare training command
        model = self.params.get("model", "yolo11n.pt")
        data_config = self.params.get("data_config", "configs/dataset.yaml")
        epochs = self.params.get("epochs", 100)
        img_size = self.params.get("img_size", 640)
        batch_size = self.params.get("batch_size", 16)
        device = self.params.get("device", "auto")
        
        cmd = [
            "python", "-m", "ultralytics.yolo.v8.train",
            "model=" + model,
            "data=" + data_config,
            f"epochs={epochs}",
            f"imgsz={img_size}",
            f"batch={batch_size}",
            f"device={device}",
            "project=data/models/training",
            "name=yolo_training"
        ]
        
        self._run_command(cmd)
        
    def _validate_model(self):
        """Validate YOLO model."""
        self.status_updated.emit("Validating YOLO model...")
        
        model_path = self.params.get("model_path", "data/models/training/yolo_training/weights/best.pt")
        data_config = self.params.get("data_config", "configs/dataset.yaml")
        
        cmd = [
            "python", "-m", "ultralytics.yolo.v8.val",
            "model=" + model_path,
            "data=" + data_config
        ]
        
        self._run_command(cmd)
        
    def _export_model(self):
        """Export YOLO model to different formats."""
        self.status_updated.emit("Exporting YOLO model...")
        
        model_path = self.params.get("model_path", "data/models/training/yolo_training/weights/best.pt")
        export_format = self.params.get("format", "onnx")
        
        cmd = [
            "python", "-m", "ultralytics.yolo.v8.export",
            "model=" + model_path,
            f"format={export_format}"
        ]
        
        self._run_command(cmd)
        
    def _predict_model(self):
        """Run prediction with YOLO model."""
        self.status_updated.emit("Running YOLO prediction...")
        
        model_path = self.params.get("model_path", "data/models/training/yolo_training/weights/best.pt")
        source = self.params.get("source", "data/processed/dev_data")
        
        cmd = [
            "python", "-m", "ultralytics.yolo.v8.predict",
            "model=" + model_path,
            "source=" + source,
            "project=data/predictions",
            "name=yolo_predict"
        ]
        
        self._run_command(cmd)
    
    def _run_command(self, cmd: List[str]):
        """Run a command and capture output."""
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            while True:
                output = self._process.stdout.readline()
                if output == '' and self._process.poll() is not None:
                    break
                if output:
                    self.log_updated.emit(output.strip())
                    
                    # Parse progress if possible
                    if "epoch" in output.lower() and "/" in output:
                        try:
                            # Try to extract progress from epoch info
                            parts = output.split()
                            for i, part in enumerate(parts):
                                if "epoch" in part.lower() and i + 1 < len(parts):
                                    epoch_info = parts[i + 1]
                                    if "/" in epoch_info:
                                        current, total = epoch_info.split("/")
                                        progress = int((int(current) / int(total)) * 100)
                                        self.progress_updated.emit(progress)
                                        break
                        except:
                            pass
            
            # Wait for process to complete
            self._process.wait()
            
            if self._process.returncode != 0:
                raise subprocess.CalledProcessError(self._process.returncode, cmd)
                
        except subprocess.CalledProcessError as e:
            raise Exception(f"Command failed with return code {e.returncode}")
        except Exception as e:
            raise Exception(f"Failed to run command: {e}")
    
    def terminate_process(self):
        """Terminate the running process."""
        if self._process and self._process.poll() is None:
            self._process.terminate()


class DevYoloTrainingTab(QWidget):
    """
    Tab for YOLO training development tools.
    """
    
    def __init__(self):
        super().__init__()
        self.training_worker: Optional[YOLOTrainingWorker] = None
        self.available_models: List[str] = []
        self.training_configs: List[str] = []
        self._init_ui()
        self._load_available_models()
        self._load_training_configs()
    
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QHBoxLayout()
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel: Training configuration and controls
        left_panel = self._create_left_panel()
        splitter.addWidget(left_panel)
        
        # Right panel: Training output and monitoring
        right_panel = self._create_right_panel()
        splitter.addWidget(right_panel)
        
        # Set initial sizes
        splitter.setSizes([400, 800])
        
        layout.addWidget(splitter)
        self.setLayout(layout)
    
    def _create_left_panel(self) -> QWidget:
        """Create the left panel with training configuration."""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Model selection
        model_group = self._create_model_group()
        layout.addWidget(model_group)
        
        # Training parameters
        params_group = self._create_parameters_group()
        layout.addWidget(params_group)
        
        # Data configuration
        data_group = self._create_data_group()
        layout.addWidget(data_group)
        
        # Training controls
        controls_group = self._create_controls_group()
        layout.addWidget(controls_group)
        
        # Status and progress
        status_group = self._create_status_group()
        layout.addWidget(status_group)
        
        panel.setLayout(layout)
        return panel
    
    def _create_right_panel(self) -> QWidget:
        """Create the right panel with training output."""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Training log
        log_group = QGroupBox("Training Log")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.log_text)
        
        # Log controls
        log_controls = QHBoxLayout()
        
        self.clear_log_button = QPushButton("Clear Log")
        self.clear_log_button.clicked.connect(self._clear_log)
        log_controls.addWidget(self.clear_log_button)
        
        self.save_log_button = QPushButton("Save Log")
        self.save_log_button.clicked.connect(self._save_log)
        log_controls.addWidget(self.save_log_button)
        
        log_controls.addStretch()
        log_layout.addLayout(log_controls)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        # Training metrics (placeholder for future implementation)
        metrics_group = QGroupBox("Training Metrics")
        metrics_layout = QVBoxLayout()
        
        self.metrics_label = QLabel("Training metrics will be displayed here during training")
        self.metrics_label.setAlignment(Qt.AlignCenter)
        self.metrics_label.setStyleSheet("color: #888; padding: 20px;")
        metrics_layout.addWidget(self.metrics_label)
        
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)
        
        panel.setLayout(layout)
        return panel
    
    def _create_model_group(self) -> QGroupBox:
        """Create the model selection group."""
        model_group = QGroupBox("Model Configuration")
        layout = QFormLayout()
        
        # Base model selection
        self.base_model_combo = QComboBox()
        self.base_model_combo.addItems([
            "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
            "yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt", "yolo11l-seg.pt", "yolo11x-seg.pt"
        ])
        layout.addRow("Base Model:", self.base_model_combo)
        
        # Custom model path
        model_path_layout = QHBoxLayout()
        self.custom_model_path = QLineEdit()
        self.custom_model_path.setPlaceholderText("Or specify custom model path...")
        model_path_layout.addWidget(self.custom_model_path)
        
        self.browse_model_button = QPushButton("Browse")
        self.browse_model_button.clicked.connect(self._browse_model)
        model_path_layout.addWidget(self.browse_model_button)
        
        layout.addRow("Custom Model:", model_path_layout)
        
        model_group.setLayout(layout)
        return model_group
    
    def _create_parameters_group(self) -> QGroupBox:
        """Create the training parameters group."""
        params_group = QGroupBox("Training Parameters")
        layout = QFormLayout()
        
        # Epochs
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(100)
        layout.addRow("Epochs:", self.epochs_spin)
        
        # Batch size
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 128)
        self.batch_size_spin.setValue(16)
        layout.addRow("Batch Size:", self.batch_size_spin)
        
        # Image size
        self.img_size_combo = QComboBox()
        self.img_size_combo.addItems(["320", "416", "512", "640", "800", "960", "1024"])
        self.img_size_combo.setCurrentText("640")
        layout.addRow("Image Size:", self.img_size_combo)
        
        # Learning rate
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.0001, 1.0)
        self.learning_rate_spin.setValue(0.01)
        self.learning_rate_spin.setDecimals(4)
        self.learning_rate_spin.setSingleStep(0.0001)
        layout.addRow("Learning Rate:", self.learning_rate_spin)
        
        # Device
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu", "0", "1", "2", "3"])
        layout.addRow("Device:", self.device_combo)
        
        # Workers
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 16)
        self.workers_spin.setValue(8)
        layout.addRow("Workers:", self.workers_spin)
        
        params_group.setLayout(layout)
        return params_group
    
    def _create_data_group(self) -> QGroupBox:
        """Create the data configuration group."""
        data_group = QGroupBox("Data Configuration")
        layout = QFormLayout()
        
        # Dataset config
        data_config_layout = QHBoxLayout()
        self.data_config_path = QLineEdit()
        self.data_config_path.setPlaceholderText("Path to dataset.yaml...")
        self.data_config_path.setText("configs/dataset.yaml")
        data_config_layout.addWidget(self.data_config_path)
        
        self.browse_data_button = QPushButton("Browse")
        self.browse_data_button.clicked.connect(self._browse_data_config)
        data_config_layout.addWidget(self.browse_data_button)
        
        layout.addRow("Dataset Config:", data_config_layout)
        
        # Project and name
        self.project_path = QLineEdit()
        self.project_path.setText("data/models/training")
        layout.addRow("Project Path:", self.project_path)
        
        self.experiment_name = QLineEdit()
        self.experiment_name.setText("yolo_training")
        layout.addRow("Experiment Name:", self.experiment_name)
        
        data_group.setLayout(layout)
        return data_group
    
    def _create_controls_group(self) -> QGroupBox:
        """Create the training controls group."""
        controls_group = QGroupBox("Training Controls")
        layout = QVBoxLayout()
        
        # Main training buttons
        main_buttons = QHBoxLayout()
        
        self.train_button = QPushButton("Start Training")
        self.train_button.clicked.connect(self._start_training)
        self.train_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        main_buttons.addWidget(self.train_button)
        
        self.stop_button = QPushButton("Stop Training")
        self.stop_button.clicked.connect(self._stop_training)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")
        main_buttons.addWidget(self.stop_button)
        
        layout.addLayout(main_buttons)
        
        # Additional operation buttons
        operation_buttons = QHBoxLayout()
        
        self.validate_button = QPushButton("Validate Model")
        self.validate_button.clicked.connect(self._validate_model)
        operation_buttons.addWidget(self.validate_button)
        
        self.export_button = QPushButton("Export Model")
        self.export_button.clicked.connect(self._export_model)
        operation_buttons.addWidget(self.export_button)
        
        self.predict_button = QPushButton("Run Prediction")
        self.predict_button.clicked.connect(self._run_prediction)
        operation_buttons.addWidget(self.predict_button)
        
        layout.addLayout(operation_buttons)
        
        controls_group.setLayout(layout)
        return controls_group
    
    def _create_status_group(self) -> QGroupBox:
        """Create the status and progress group."""
        status_group = QGroupBox("Status")
        layout = QVBoxLayout()
        
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        status_group.setLayout(layout)
        return status_group
    
    def _load_available_models(self):
        """Load available models."""
        model_dirs = ["data/models/pretrained", "data/models/finetune"]
        
        for model_dir in model_dirs:
            if os.path.exists(model_dir):
                for root, dirs, files in os.walk(model_dir):
                    for file in files:
                        if file.endswith('.pt'):
                            self.available_models.append(os.path.join(root, file))
    
    def _load_training_configs(self):
        """Load available training configurations."""
        config_dir = "configs"
        if os.path.exists(config_dir):
            for file in os.listdir(config_dir):
                if file.endswith('.yaml') or file.endswith('.yml'):
                    self.training_configs.append(os.path.join(config_dir, file))
    
    def _browse_model(self):
        """Browse for model file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "", "PyTorch Models (*.pt);;All Files (*)"
        )
        if file_path:
            self.custom_model_path.setText(file_path)
    
    def _browse_data_config(self):
        """Browse for data configuration file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Dataset Config", "", "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        if file_path:
            self.data_config_path.setText(file_path)
    
    def _get_model_path(self) -> str:
        """Get the selected model path."""
        if self.custom_model_path.text().strip():
            return self.custom_model_path.text().strip()
        else:
            return f"data/models/pretrained/{self.base_model_combo.currentText()}"
    
    def _start_training(self):
        """Start YOLO training."""
        if self.training_worker and self.training_worker.isRunning():
            QMessageBox.warning(self, "Training", "Training is already running")
            return
        
        # Validate inputs
        if not self.data_config_path.text().strip():
            QMessageBox.warning(self, "Training", "Please specify a dataset configuration file")
            return
        
        # Prepare training parameters
        params = {
            "model": self._get_model_path(),
            "data_config": self.data_config_path.text(),
            "epochs": self.epochs_spin.value(),
            "img_size": int(self.img_size_combo.currentText()),
            "batch_size": self.batch_size_spin.value(),
            "device": self.device_combo.currentText(),
            "learning_rate": self.learning_rate_spin.value(),
            "workers": self.workers_spin.value(),
            "project": self.project_path.text(),
            "name": self.experiment_name.text()
        }
        
        self._start_operation("train", params)
    
    def _stop_training(self):
        """Stop training."""
        if self.training_worker:
            self.training_worker.terminate_process()
            self.training_worker.terminate()
            self.training_worker.wait()
            self._on_operation_finished(False, "Training stopped by user")
    
    def _validate_model(self):
        """Validate model."""
        model_path = self._get_trained_model_path()
        if not model_path:
            QMessageBox.warning(self, "Validation", "No trained model found")
            return
        
        params = {
            "model_path": model_path,
            "data_config": self.data_config_path.text()
        }
        
        self._start_operation("validate", params)
    
    def _export_model(self):
        """Export model."""
        model_path = self._get_trained_model_path()
        if not model_path:
            QMessageBox.warning(self, "Export", "No trained model found")
            return
        
        # Ask for export format
        formats = ["onnx", "torchscript", "coreml", "saved_model", "pb", "tflite"]
        format_choice, ok = QInputDialog.getItem(
            self, "Export Format", "Select export format:", formats, 0, False
        )
        
        if not ok:
            return
        
        params = {
            "model_path": model_path,
            "format": format_choice
        }
        
        self._start_operation("export", params)
    
    def _run_prediction(self):
        """Run prediction."""
        model_path = self._get_trained_model_path()
        if not model_path:
            QMessageBox.warning(self, "Prediction", "No trained model found")
            return
        
        # Ask for source
        source = QFileDialog.getExistingDirectory(self, "Select Source Directory")
        if not source:
            return
        
        params = {
            "model_path": model_path,
            "source": source
        }
        
        self._start_operation("predict", params)
    
    def _get_trained_model_path(self) -> Optional[str]:
        """Get path to trained model."""
        project_path = self.project_path.text()
        experiment_name = self.experiment_name.text()
        model_path = os.path.join(project_path, experiment_name, "weights", "best.pt")
        
        if os.path.exists(model_path):
            return model_path
        
        # Look for any .pt files in the training output
        weights_dir = os.path.join(project_path, experiment_name, "weights")
        if os.path.exists(weights_dir):
            for file in os.listdir(weights_dir):
                if file.endswith('.pt'):
                    return os.path.join(weights_dir, file)
        
        return None
    
    def _start_operation(self, operation: str, params: Dict[str, Any]):
        """Start a training operation."""
        self.training_worker = YOLOTrainingWorker(operation, params)
        self.training_worker.progress_updated.connect(self.progress_bar.setValue)
        self.training_worker.status_updated.connect(self._on_status_updated)
        self.training_worker.log_updated.connect(self._on_log_updated)
        self.training_worker.finished.connect(self._on_operation_finished)
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self._set_training_enabled(False)
        
        self.training_worker.start()
    
    def _on_status_updated(self, message: str):
        """Handle status update."""
        self.status_label.setText(message)
        self._log(f"STATUS: {message}")
    
    def _on_log_updated(self, message: str):
        """Handle log update."""
        self.log_text.append(message)
        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def _on_operation_finished(self, success: bool, message: str):
        """Handle operation completion."""
        self.progress_bar.setVisible(False)
        self._set_training_enabled(True)
        
        if success:
            self.status_label.setText("Ready")
            self._log(f"✓ {message}")
            QMessageBox.information(self, "Success", message)
        else:
            self.status_label.setText("Error")
            self._log(f"✗ {message}")
            QMessageBox.critical(self, "Error", message)
    
    def _set_training_enabled(self, enabled: bool):
        """Enable/disable training controls."""
        self.train_button.setEnabled(enabled)
        self.stop_button.setEnabled(not enabled)
        self.validate_button.setEnabled(enabled)
        self.export_button.setEnabled(enabled)
        self.predict_button.setEnabled(enabled)
    
    def _log(self, message: str):
        """Add message to the log."""
        timestamp = self._get_timestamp()
        self.log_text.append(f"[{timestamp}] {message}")
        logger.info(message)
        
        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")
    
    def _clear_log(self):
        """Clear the log."""
        self.log_text.clear()
    
    def _save_log(self):
        """Save the log to file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Log", "training_log.txt", "Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(self.log_text.toPlainText())
                QMessageBox.information(self, "Save Log", f"Log saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Save Log", f"Failed to save log: {e}")
    
    def closeEvent(self, event):
        """Handle tab close event."""
        if self.training_worker and self.training_worker.isRunning():
            reply = QMessageBox.question(
                self, "Close Tab", 
                "Training is still running. Do you want to stop it and close?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.training_worker.terminate_process()
                self.training_worker.terminate()
                self.training_worker.wait()
            else:
                event.ignore()
                return
        
        super().closeEvent(event)
