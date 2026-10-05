"""Model Training tab: train YOLO detection and segmentation models on custom datasets."""

import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ...utils.logger import get_logger
from ...utils.model_files import find_training_runs, model_display_name, run_dataset_name
from ..widgets.panels import PANEL_WIDTH, compact_combo, side_panel
from .results_widget import TrainingResultsWidget
from .training_thread import ModelTrainingThread

logger = get_logger("TRAINING")

# The settings hold long model and dataset paths
SETTINGS_WIDTH = PANEL_WIDTH + 100


class ModelTrainingTab(QWidget):
    """Tab for training YOLO models with custom datasets."""

    def __init__(self):
        super().__init__()

        # State
        self.current_task = "detection"
        self.current_model_path = ""
        self.current_data_path = ""
        self.training_thread: Optional[ModelTrainingThread] = None
        self.training_config = {}
        self.training_start_time = None  # When training subprocess starts (includes preprocessing)
        self.last_status = ""  # Store last status for time updates

        # Timer for updating elapsed/remaining time every second
        self.time_update_timer = QTimer()
        self.time_update_timer.timeout.connect(self._update_time_display)

        # Initialize UI
        self._init_ui()
        self._load_default_config()
        self._update_model_options()
        self._update_data_options()

        self._update_comparison_options()

    def _get_preferred_model_file(self, weights_dir: Path) -> Optional[Path]:
        """Get the preferred model file from a weights directory.

        Prioritizes best.pt over last.pt if both exist.

        Args:
            weights_dir: Path to the weights directory

        Returns:
            Path to the preferred model file, or None if no suitable file found
        """
        if not weights_dir.exists():
            return None

        # Check for best.pt first (highest priority)
        best_pt = weights_dir / "best.pt"
        if best_pt.exists():
            return best_pt

        # If no best.pt, check for last.pt
        last_pt = weights_dir / "last.pt"
        if last_pt.exists():
            return last_pt

        # If neither best.pt nor last.pt exist, look for any .pt file
        pt_files = list(weights_dir.glob("*.pt"))
        if pt_files:
            # Sort to get consistent results and prefer shorter names (often better)
            pt_files.sort(key=lambda x: (len(x.name), x.name))
            return pt_files[0]

        return None

    def _init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout()

        # Create splitter for main content
        splitter = QSplitter(Qt.Horizontal)

        # Left panel - Configuration
        splitter.addWidget(side_panel(self._create_config_panel(), SETTINGS_WIDTH))

        # Right panel - Training and Results
        results_panel = self._create_results_panel()
        splitter.addWidget(results_panel)

        # Set splitter proportions
        # Output and plots are what is watched for hours; they get the room
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def _create_config_panel(self) -> QWidget:
        """Create the configuration panel."""
        panel = QWidget()
        layout = QVBoxLayout()

        # Task selection
        task_group = QGroupBox("Training Task")
        task_layout = QFormLayout()

        self.task_combo = compact_combo(QComboBox())
        self.task_combo.addItems(["detection", "field segmentation"])
        self.task_combo.currentTextChanged.connect(self._on_task_changed)
        self.task_combo.setToolTip(
            "Select the type of computer vision task:\n\n• Detection: Find and classify objects with bounding boxes\n  - Examples: players, disc, referees in Ultimate Frisbee\n  - Output: [class, x, y, width, height, confidence]\n\n• Field Segmentation: Pixel-level classification of field regions\n  - Examples: field boundaries, end zones, out-of-bounds areas\n  - Output: segmentation masks for each region\n\nDifferent tasks use specialized model architectures and datasets."
        )
        task_layout.addRow("Task Type:", self.task_combo)

        task_group.setLayout(task_layout)
        layout.addWidget(task_group)

        # Model selection
        model_group = QGroupBox("Base Model Selection")
        model_layout = QFormLayout()

        self.model_combo = compact_combo(QComboBox())
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        self.model_combo.setToolTip(
            "Select the base YOLO model to start training from.\n\n• Model sizes: n (nano), s (small), m (medium), l (large), x (extra-large)\n• Examples: yolo11n.pt (fast, 2.6M params), yolo11s.pt (6.5M params), yolo11l.pt (accurate, 25.3M params)\n• Pretrained models: learned features from COCO dataset (80 classes)\n• Auto-download: Missing YOLO26/YOLO11 models will be downloaded automatically\n• Trade-offs: Larger models = better accuracy but slower training/inference\n• Custom models: .pt files from previous training runs"
        )
        model_layout.addRow("Base Model:", self.model_combo)

        # Model info display
        self.model_info_text = QTextEdit()
        self.model_info_text.setMaximumHeight(100)
        self.model_info_text.setReadOnly(True)
        self.model_info_text.setToolTip(
            "Displays information about the selected model including file size and type."
        )
        model_layout.addRow("Model Info:", self.model_info_text)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Dataset selection
        data_group = QGroupBox("Training Dataset")
        data_layout = QFormLayout()

        self.data_combo = compact_combo(QComboBox())
        self.data_combo.currentTextChanged.connect(self._on_data_changed)
        self.data_combo.setToolTip(
            "Select the dataset to train on (YOLO format required).\n\n• Format: data.yaml file with train/val paths and class names\n• Examples: coco8.yaml (sample), custom_dataset_v3.yaml\n• Version numbers: v2, v3, v4 (higher = typically improved)\n• Structure: train/images/, train/labels/, valid/images/, valid/labels/\n• More training images = generally better model performance\n• Labels: .txt files with class_id x_center y_center width height"
        )
        data_layout.addRow("Dataset:", self.data_combo)

        # Dataset info display
        self.dataset_info_text = QTextEdit()
        self.dataset_info_text.setMaximumHeight(100)
        self.dataset_info_text.setReadOnly(True)
        self.dataset_info_text.setToolTip(
            "Shows dataset information including number of classes, class names,\nand training/validation image counts."
        )
        data_layout.addRow("Dataset Info:", self.dataset_info_text)

        data_group.setLayout(data_layout)
        layout.addWidget(data_group)

        # Training parameters
        params_group = QGroupBox("Training Parameters")
        params_layout = QFormLayout()

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(100)
        self.epochs_spin.setToolTip(
            "Number of complete passes through the training dataset.\n\n• Default: 100 epochs\n• Range: 1-1000+ epochs\n• Examples: 50 (quick test), 100 (standard), 300 (fine-tuning)\n• More epochs = longer training but potentially better performance\n• Use early stopping (patience) to prevent overfitting"
        )
        params_layout.addRow("Epochs:", self.epochs_spin)

        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 100)
        self.patience_spin.setValue(10)
        self.patience_spin.setToolTip(
            "Early stopping patience: number of epochs to wait for improvement\nbefore stopping training. Prevents overfitting and saves time.\n\n• Default: 100 epochs (detection), 15 (segmentation)\n• Range: 1-100+ epochs\n• Examples: 10 (aggressive), 50 (moderate), 100 (patient)\n• Higher values = more patient, lower values = stop sooner\n• Monitors validation metrics for improvement"
        )
        params_layout.addRow("Patience:", self.patience_spin)

        self.batch_spin = QDoubleSpinBox()
        self.batch_spin.setRange(0.1, 128)
        self.batch_spin.setDecimals(1)
        self.batch_spin.setValue(16)
        self.batch_spin.setToolTip(
            "Batch size: integer (e.g. 16) or fraction for GPU memory (e.g. 0.8).\n\n• Default: 16 (detection), 8 (segmentation)\n• Integer: 1-128+ (exact number of images per batch)\n• Fraction: 0.1-1.0 (percentage of GPU memory to use)\n• Examples: 16 (fixed), 0.6 (60% GPU memory), -1 (auto 60%)\n• Auto modes: -1 (60% GPU memory), 0.7 (70% GPU memory)\n• Larger batches = more stable gradients but need more memory"
        )
        params_layout.addRow("Batch Size:", self.batch_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 1.0)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setValue(0.01)
        self.lr_spin.setToolTip(
            "Initial learning rate (lr0) - controls how big steps the model takes.\n\n• Default: 0.01 (SGD), 0.001 (Adam)\n• Range: 0.0001-1.0 (typically 0.001-0.01)\n• Examples: 0.001 (conservative), 0.01 (standard), 0.1 (aggressive)\n• Higher values = faster learning but risk instability\n• Lower values = more stable but slower convergence\n• Automatically decays during training using schedulers"
        )
        params_layout.addRow("Learning Rate:", self.lr_spin)

        # Image size
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(320, 1920)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)
        self.imgsz_spin.setToolTip(
            "Input image size for training (square images).\n\n• Default: 640 pixels\n• Range: 320-1280 (must be multiple of 32)\n• Examples: 416 (fast), 640 (standard), 832 (detailed), 1024 (high-res)\n• Larger sizes = better detail recognition but slower training\n• Smaller sizes = faster training but less detail\n• All images resized to this dimension before processing"
        )
        params_layout.addRow("Image Size:", self.imgsz_spin)

        # Optimizer
        self.optimizer_combo = compact_combo(QComboBox())
        self.optimizer_combo.addItems(["SGD", "Adam", "AdamW", "RMSProp"])
        self.optimizer_combo.setCurrentText("SGD")
        self.optimizer_combo.setToolTip(
            "Optimization algorithm for training:\n\n• Default: 'auto' (SGD for most cases)\n• SGD: Simple, stable, momentum-based (good for most cases)\n• Adam: Adaptive learning rates, faster convergence\n• AdamW: Adam with better weight decay handling\n• RMSProp: Good for noisy gradients and RNNs\n• NAdam, RAdam: Advanced Adam variants\n\nSGD with momentum (0.937) is proven effective for YOLO models."
        )
        params_layout.addRow("Optimizer:", self.optimizer_combo)

        # Momentum
        self.momentum_spin = QDoubleSpinBox()
        self.momentum_spin.setRange(0.0, 1.0)
        self.momentum_spin.setDecimals(3)
        self.momentum_spin.setSingleStep(0.001)
        self.momentum_spin.setValue(0.937)
        self.momentum_spin.setToolTip(
            "Momentum factor for SGD optimizer (or beta1 for Adam).\n\n• Default: 0.937 (proven optimal for YOLO)\n• Range: 0.0-1.0 (typically 0.8-0.99)\n• Examples: 0.9 (standard), 0.937 (YOLO optimized), 0.95 (high momentum)\n• Helps accelerate training in consistent directions\n• Higher values = smoother convergence, more momentum\n• Lower values = more responsive to gradient changes"
        )
        params_layout.addRow("Momentum:", self.momentum_spin)

        # Weight Decay
        self.weight_decay_spin = QDoubleSpinBox()
        self.weight_decay_spin.setRange(0.0, 0.01)
        self.weight_decay_spin.setDecimals(4)
        self.weight_decay_spin.setSingleStep(0.0001)
        self.weight_decay_spin.setValue(0.0005)
        self.weight_decay_spin.setToolTip(
            "L2 regularization penalty to prevent overfitting.\n\n• Default: 0.0005 (YOLO optimized)\n• Range: 0.0-0.01 (typically 0.0001-0.001)\n• Examples: 0.0001 (light), 0.0005 (standard), 0.001 (strong)\n• Penalizes large weights to keep the model simple\n• Higher values = stronger regularization, lower overfitting risk\n• Too high = underfitting, too low = overfitting risk"
        )
        params_layout.addRow("Weight Decay:", self.weight_decay_spin)

        # Cosine Learning Rate Scheduler
        self.cosine_lr_check = QCheckBox()
        self.cosine_lr_check.setChecked(False)
        self.cosine_lr_check.setToolTip(
            "Enable cosine learning rate scheduler.\n\n• Default: False (linear decay)\n• Cosine scheduler: Learning rate follows a cosine curve over epochs\n• Benefits: Smoother convergence, better final performance, avoids sharp drops\n• Best for: Long training runs, fine-tuning, when you want gradual learning rate decay\n• Alternative: Linear decay (default) or step-wise schedulers\n• Works well with warmup epochs for stable training start"
        )
        params_layout.addRow("Cosine LR Scheduler:", self.cosine_lr_check)

        # Workers
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 16)
        self.workers_spin.setValue(0)  # Use 0 for Windows compatibility
        self.workers_spin.setToolTip(
            "Number of CPU threads for data loading (per GPU if multi-GPU).\n\n• Default: 8 (Linux/Mac), 0 (Windows recommended)\n• Range: 0-16+ (depends on CPU cores)\n• Examples: 0 (single-thread), 4 (quad-core), 8 (standard)\n• Higher values = faster data loading but more CPU usage\n• Set to 0 for Windows to avoid multiprocessing errors\n• Use 2x CPU cores for optimal performance on Linux/Mac"
        )
        params_layout.addRow("Workers:", self.workers_spin)

        # Augmentation checkbox
        self.augment_check = QCheckBox()
        self.augment_check.setChecked(True)
        self.augment_check.setToolTip(
            "Enable data augmentation during training.\n\n• Default: True (recommended for most cases)\n• Includes: rotation, scaling, flipping, color changes, mosaic\n• Benefits: Improves robustness, prevents overfitting, increases dataset variety\n• Examples: HSV shifts, geometric transforms, mixup, copy-paste\n• Disable only for: perfect datasets, specific requirements\n• Automatically disabled in final epochs (close_mosaic=10)"
        )
        params_layout.addRow("Data Augmentation:", self.augment_check)

        # Additional augmentation parameters
        aug_group = QGroupBox("Augmentation Settings")
        aug_layout = QFormLayout()

        # Mosaic probability
        self.mosaic_spin = QDoubleSpinBox()
        self.mosaic_spin.setRange(0.0, 1.0)
        self.mosaic_spin.setDecimals(2)
        self.mosaic_spin.setValue(1.0)
        self.mosaic_spin.setToolTip(
            "Probability of mosaic augmentation (combines 4 images).\n• Default: 1.0 (always on)\n• Range: 0.0-1.0\n• Highly effective for scene understanding"
        )
        aug_layout.addRow("Mosaic:", self.mosaic_spin)

        # Random zoom
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.0, 0.9)
        self.scale_spin.setDecimals(2)
        self.scale_spin.setSingleStep(0.05)
        self.scale_spin.setValue(0.5)
        self.scale_spin.setToolTip(
            "Random zoom: images are shown at between (1 - value) and (1 + value) times "
            "their size.\n• Default: 0.5\n• Use about 0.2 for small objects such as the "
            "disc: at 0.5 a disc is often shrunk to half its size, too small to learn from"
        )
        aug_layout.addRow("Scale:", self.scale_spin)

        # Mixup probability
        self.mixup_spin = QDoubleSpinBox()
        self.mixup_spin.setRange(0.0, 1.0)
        self.mixup_spin.setDecimals(2)
        self.mixup_spin.setValue(0.0)
        self.mixup_spin.setToolTip(
            "Probability of mixup augmentation (blends images).\n• Default: 0.0\n• Range: 0.0-1.0\n• Enhances generalization"
        )
        aug_layout.addRow("Mixup:", self.mixup_spin)

        # Copy-paste probability
        self.copy_paste_spin = QDoubleSpinBox()
        self.copy_paste_spin.setRange(0.0, 1.0)
        self.copy_paste_spin.setDecimals(2)
        self.copy_paste_spin.setValue(0.0)
        self.copy_paste_spin.setToolTip(
            "Copy-paste augmentation (segmentation only).\n• Default: 0.0\n• Range: 0.0-1.0\n• Copies objects between images"
        )
        aug_layout.addRow("Copy-Paste:", self.copy_paste_spin)

        # HSV-H (Hue)
        self.hsv_h_spin = QDoubleSpinBox()
        self.hsv_h_spin.setRange(0.0, 1.0)
        self.hsv_h_spin.setDecimals(3)
        self.hsv_h_spin.setValue(0.015)
        self.hsv_h_spin.setToolTip(
            "HSV Hue augmentation range.\n• Default: 0.015\n• Range: 0.0-1.0\n• Adjusts color hue for lighting variety"
        )
        aug_layout.addRow("HSV-H (Hue):", self.hsv_h_spin)

        # HSV-S (Saturation)
        self.hsv_s_spin = QDoubleSpinBox()
        self.hsv_s_spin.setRange(0.0, 1.0)
        self.hsv_s_spin.setDecimals(2)
        self.hsv_s_spin.setValue(0.7)
        self.hsv_s_spin.setToolTip(
            "HSV Saturation augmentation range.\n• Default: 0.7\n• Range: 0.0-1.0\n• Adjusts color intensity"
        )
        aug_layout.addRow("HSV-S (Saturation):", self.hsv_s_spin)

        # HSV-V (Value/Brightness)
        self.hsv_v_spin = QDoubleSpinBox()
        self.hsv_v_spin.setRange(0.0, 1.0)
        self.hsv_v_spin.setDecimals(2)
        self.hsv_v_spin.setValue(0.4)
        self.hsv_v_spin.setToolTip(
            "HSV Value (brightness) augmentation range.\n• Default: 0.4\n• Range: 0.0-1.0\n• Adjusts brightness for lighting conditions"
        )
        aug_layout.addRow("HSV-V (Brightness):", self.hsv_v_spin)

        aug_group.setLayout(aug_layout)
        layout.addWidget(aug_group)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Config buttons
        config_buttons_layout = QHBoxLayout()

        self.load_config_btn = QPushButton("Load Config")
        self.load_config_btn.clicked.connect(self._load_config)
        self.load_config_btn.setToolTip(
            "Load training parameters from a saved YAML configuration file.\nThis will update all parameter values in the interface.\nUseful for reusing proven parameter combinations."
        )
        config_buttons_layout.addWidget(self.load_config_btn)

        self.save_config_btn = QPushButton("Save Config")
        self.save_config_btn.clicked.connect(self._save_config)
        self.save_config_btn.setToolTip(
            "Save current training parameters to a YAML configuration file.\nAllows you to reuse these settings later or share with others.\nConfigurations are saved per task type (detection/segmentation)."
        )
        config_buttons_layout.addWidget(self.save_config_btn)

        layout.addLayout(config_buttons_layout)

        # Training controls
        controls_group = QGroupBox("Training Controls")
        controls_layout = QVBoxLayout()

        self.start_training_btn = QPushButton("Start Training")
        self.start_training_btn.clicked.connect(self._start_training)
        self.start_training_btn.setToolTip(
            "Begin training the selected model on the chosen dataset.\nEnsure you have selected both a base model and dataset.\nTraining runs in background and shows live progress graphs."
        )
        controls_layout.addWidget(self.start_training_btn)

        self.stop_training_btn = QPushButton("Stop Training")
        self.stop_training_btn.clicked.connect(self._stop_training)
        self.stop_training_btn.setEnabled(False)
        self.stop_training_btn.setToolTip(
            "Stop the current training process.\nThis will terminate training gracefully and save progress.\nThe model will be saved in its current state."
        )
        controls_layout.addWidget(self.stop_training_btn)

        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)

        layout.addStretch()
        panel.setLayout(layout)
        return panel

    def _create_results_panel(self) -> QWidget:
        """Create the training results panel."""
        panel = QWidget()
        layout = QVBoxLayout()

        # Progress section
        progress_group = QGroupBox("Training Progress")
        progress_group.setMaximumHeight(300)
        progress_layout = QVBoxLayout()
        progress_layout.setSpacing(1)  # Minimal spacing
        progress_layout.setContentsMargins(8, 3, 8, 3)  # Smaller margins

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumHeight(18)  # Smaller progress bar
        progress_layout.addWidget(self.progress_bar)

        # Single line for progress info and elapsed time
        self.progress_info_label = QLabel("Ready to start training")
        self.progress_info_label.setMaximumHeight(16)  # Compact text
        progress_layout.addWidget(self.progress_info_label)

        # Raw output display
        self.raw_output_text = QTextEdit()
        self.raw_output_text.setMaximumHeight(220)
        self.raw_output_text.setReadOnly(True)
        self.raw_output_text.setFont(QFont("Consolas", 8))  # Small monospace font
        self.raw_output_text.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.raw_output_text.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        progress_layout.addWidget(self.raw_output_text)

        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        # Results visualization
        results_group = QGroupBox("Training Results")
        results_layout = QVBoxLayout()

        # Earlier run whose curves are drawn dashed behind the current one
        comparison_row = QHBoxLayout()
        comparison_row.addWidget(QLabel("Compare with:"))
        self.comparison_combo = compact_combo(QComboBox())
        self.comparison_combo.setToolTip(
            "An earlier run to draw behind the current one. The latest run on the selected "
            "dataset is chosen automatically."
        )
        self.comparison_combo.currentIndexChanged.connect(self._on_comparison_changed)
        comparison_row.addWidget(self.comparison_combo, 1)
        results_layout.addLayout(comparison_row)

        # Training results widget
        self.results_widget = TrainingResultsWidget()
        results_layout.addWidget(self.results_widget)

        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        panel.setLayout(layout)
        return panel

    def _load_default_config(self):
        """Load default training configuration."""
        try:
            config_path = Path("configs/training.yaml")
            if config_path.exists():
                with open(config_path, "r") as f:
                    self.training_config = yaml.safe_load(f)
                self._apply_config_to_ui()
            else:
                # Create default config
                self.training_config = {
                    "detection": {
                        "epochs": 100,
                        "patience": 10,
                        "batch_size": 16,
                        "learning_rate": 0.01,
                        "cosine_lr": False,
                    },
                    "segmentation": {
                        "epochs": 150,
                        "patience": 15,
                        "batch_size": 8,
                        "learning_rate": 0.01,
                        "cosine_lr": False,
                    },
                }
        except Exception as e:
            logger.error(f"Error loading config: {e}")

    def _apply_config_to_ui(self):
        """Apply loaded configuration to UI elements."""
        task_config = self.training_config.get(self.current_task, {})

        if "epochs" in task_config:
            self.epochs_spin.setValue(task_config["epochs"])
        if "patience" in task_config:
            self.patience_spin.setValue(task_config["patience"])
        if "batch_size" in task_config:
            self.batch_spin.setValue(task_config["batch_size"])
        if "learning_rate" in task_config:
            self.lr_spin.setValue(task_config["learning_rate"])
        if "imgsz" in task_config:
            self.imgsz_spin.setValue(task_config["imgsz"])
        if "optimizer" in task_config:
            self.optimizer_combo.setCurrentText(task_config["optimizer"])
        if "momentum" in task_config:
            self.momentum_spin.setValue(task_config["momentum"])
        if "weight_decay" in task_config:
            self.weight_decay_spin.setValue(task_config["weight_decay"])
        if "workers" in task_config:
            self.workers_spin.setValue(task_config["workers"])
        if "augment" in task_config:
            self.augment_check.setChecked(task_config["augment"])
        if "cosine_lr" in task_config:
            self.cosine_lr_check.setChecked(task_config["cosine_lr"])
        if "mosaic" in task_config:
            self.mosaic_spin.setValue(task_config["mosaic"])
        if "scale" in task_config:
            self.scale_spin.setValue(task_config["scale"])
        if "mixup" in task_config:
            self.mixup_spin.setValue(task_config["mixup"])
        if "copy_paste" in task_config:
            self.copy_paste_spin.setValue(task_config["copy_paste"])
        if "hsv_h" in task_config:
            self.hsv_h_spin.setValue(task_config["hsv_h"])
        if "hsv_s" in task_config:
            self.hsv_s_spin.setValue(task_config["hsv_s"])
        if "hsv_v" in task_config:
            self.hsv_v_spin.setValue(task_config["hsv_v"])

    def _on_task_changed(self, task: str):
        """Handle task type change."""
        self.current_task = "detection" if task == "detection" else "segmentation"
        self._update_model_options()
        self._update_data_options()
        self._update_comparison_options()
        self._apply_config_to_ui()

    def _update_model_options(self):
        """Update available model options based on task."""
        self.model_combo.clear()

        models_path = Path("data/models")
        pretrained_path = models_path / "pretrained"

        available_models = []

        if self.current_task == "detection":
            # Add common detection models (Ultralytics will auto-download if missing).
            # RT-DETR is a transformer detector: it stretches frames to a square and needs
            # several times the compute of YOLO and a lower learning rate.
            common_detection_models = [
                "yolo26n.pt",
                "yolo26s.pt",
                "yolo26m.pt",
                "yolo26l.pt",
                "yolo26x.pt",
                "yolo11n.pt",
                "yolo11s.pt",
                "yolo11m.pt",
                "yolo11l.pt",
                "yolo11x.pt",
                "rtdetr-l.pt",
                "rtdetr-x.pt",
            ]

            # Add pretrained detection models (non-seg)
            if pretrained_path.exists():
                for model_file in pretrained_path.glob("*.pt"):
                    if "-seg" not in model_file.name and "-pose" not in model_file.name:
                        available_models.append(str(model_file))

            # Add common models that aren't already in the list
            for model_name in common_detection_models:
                model_path = pretrained_path / model_name
                if str(model_path) not in available_models:
                    available_models.append(str(model_path))

            # Variants with an extra detection head for small objects such as the disc.
            # They are built from their definition and start from the base model's weights.
            available_models.extend(["yolo26n-p2.yaml", "yolo26s-p2.yaml", "yolo26m-p2.yaml"])

            # Add existing detection models for further tuning (including finetune directories)
            detection_path = models_path / "detection"
            if detection_path.exists():
                for model_dir in detection_path.iterdir():
                    if model_dir.is_dir():
                        # Check for direct weights directory
                        weights_dir = model_dir / "weights"
                        preferred_model = self._get_preferred_model_file(weights_dir)
                        if preferred_model:
                            available_models.append(str(preferred_model))

                        # Check for finetune subdirectories
                        for subdir in model_dir.iterdir():
                            if subdir.is_dir() and subdir.name.startswith("finetune"):
                                finetune_weights_dir = subdir / "weights"
                                preferred_finetune_model = self._get_preferred_model_file(
                                    finetune_weights_dir
                                )
                                if preferred_finetune_model:
                                    available_models.append(str(preferred_finetune_model))

        else:  # segmentation
            # Add common YOLO segmentation models (Ultralytics will auto-download if missing)
            common_seg_models = [
                "yolo26n-seg.pt",
                "yolo26s-seg.pt",
                "yolo26m-seg.pt",
                "yolo26l-seg.pt",
                "yolo26x-seg.pt",
                "yolo11n-seg.pt",
                "yolo11s-seg.pt",
                "yolo11m-seg.pt",
                "yolo11l-seg.pt",
                "yolo11x-seg.pt",
            ]

            # Add pretrained segmentation models
            if pretrained_path.exists():
                for model_file in pretrained_path.glob("*-seg.pt"):
                    available_models.append(str(model_file))

            # Add common seg models that aren't already in the list
            for model_name in common_seg_models:
                model_path = pretrained_path / model_name
                if str(model_path) not in available_models:
                    available_models.append(str(model_path))

            # Add existing segmentation models for further tuning (including finetune directories)
            seg_path = models_path / "segmentation"
            if seg_path.exists():
                for model_dir in seg_path.iterdir():
                    if model_dir.is_dir():
                        # Check for direct weights directory
                        weights_dir = model_dir / "weights"
                        preferred_model = self._get_preferred_model_file(weights_dir)
                        if preferred_model:
                            available_models.append(str(preferred_model))

                        # Check for finetune subdirectories
                        for subdir in model_dir.iterdir():
                            if subdir.is_dir() and subdir.name.startswith("finetune"):
                                finetune_weights_dir = subdir / "weights"
                                preferred_finetune_model = self._get_preferred_model_file(
                                    finetune_weights_dir
                                )
                                if preferred_finetune_model:
                                    available_models.append(str(preferred_finetune_model))

        self.model_combo.addItems(available_models)

        # Preselect the configured default model (full path or file name)
        default_model = self.training_config.get(self.current_task, {}).get("default_model")
        if default_model:
            for index, model in enumerate(available_models):
                if model == default_model or Path(model).name == default_model:
                    self.model_combo.setCurrentIndex(index)
                    break

    def _update_data_options(self):
        """Update available dataset options based on task."""
        self.data_combo.clear()

        training_data_path = Path("data/raw/training_data")
        if not training_data_path.exists():
            return

        available_datasets = []

        if self.current_task == "detection":
            # Look for object detection datasets
            for dataset_dir in training_data_path.iterdir():
                if dataset_dir.is_dir():
                    name_lower = dataset_dir.name.lower()
                    if "field" not in name_lower and any(
                        keyword in name_lower
                        for keyword in ["object_detection", "player", "disc", "detection", "digits"]
                    ):
                        yaml_files = list(dataset_dir.glob("*.yaml"))
                        if yaml_files:
                            available_datasets.append(str(yaml_files[0]))
        else:  # segmentation
            # Look for field segmentation datasets
            for dataset_dir in training_data_path.iterdir():
                if dataset_dir.is_dir():
                    name_lower = dataset_dir.name.lower()
                    if "field" in name_lower:
                        yaml_files = list(dataset_dir.glob("*.yaml"))
                        if yaml_files:
                            available_datasets.append(str(yaml_files[0]))

        # By name: the names start with where a dataset comes from (labelled_, roboflow_)
        available_datasets.sort()

        self.data_combo.addItems(available_datasets)

        # Select the configured default dataset, otherwise the first one
        if available_datasets:
            default_index = 0
            default_dataset = self.training_config.get(self.current_task, {}).get("default_dataset")
            for index, dataset in enumerate(available_datasets):
                if Path(dataset).parent.name == default_dataset:
                    default_index = index
                    break
            self.data_combo.setCurrentIndex(default_index)
            self._on_data_changed(available_datasets[default_index])

    def _update_comparison_options(self) -> None:
        """List the earlier runs of the current task, newest first."""
        if not hasattr(self, "comparison_combo"):
            return
        self.comparison_combo.blockSignals(True)
        self.comparison_combo.clear()
        self.comparison_combo.addItem("Nothing", None)
        for results in find_training_runs(self.current_task):
            name = model_display_name(results.parent / "weights" / "best.pt")
            self.comparison_combo.addItem(name, str(results))
        self.comparison_combo.blockSignals(False)
        self._select_comparison_for_dataset()

    def _select_comparison_for_dataset(self) -> None:
        """Compare with the latest run on the selected dataset, else with the latest run."""
        if not hasattr(self, "comparison_combo") or self.comparison_combo.count() < 2:
            return
        dataset = Path(self.current_data_path).parent.name if self.current_data_path else ""
        index = 1
        for candidate in range(1, self.comparison_combo.count()):
            if dataset and run_dataset_name(self.comparison_combo.itemData(candidate)) == dataset:
                index = candidate
                break
        self.comparison_combo.setCurrentIndex(index)
        self._on_comparison_changed(index)

    def _on_comparison_changed(self, index: int) -> None:
        self.results_widget.set_reference(
            self.comparison_combo.itemData(index), self.comparison_combo.itemText(index)
        )

    def _on_model_changed(self, model_path: str):
        """Handle model selection change."""
        self.current_model_path = model_path
        self._update_model_info()

    def _on_data_changed(self, data_path: str):
        """Handle dataset selection change."""
        self.current_data_path = data_path
        self._select_comparison_for_dataset()
        self._update_dataset_info()

    def _update_model_info(self):
        """Update model information display."""
        if not self.current_model_path:
            self.model_info_text.clear()
            return

        info_lines = []
        model_path = Path(self.current_model_path)

        info_lines.append(f"Path: {model_path.name}")
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            info_lines.append(f"Size: {size_mb:.1f} MB")

        # Try to extract more info from the model
        try:
            from ultralytics import YOLO

            model = YOLO(str(model_path))
            if hasattr(model, "model"):
                info_lines.append(f"Type: {type(model.model).__name__}")
        except Exception as e:
            info_lines.append(f"Info: Could not load model details ({str(e)[:50]}...)")

        self.model_info_text.setPlainText("\n".join(info_lines))

    def _update_dataset_info(self):
        """Update dataset information display."""
        if not self.current_data_path:
            self.dataset_info_text.clear()
            return

        info_lines = []
        data_path = Path(self.current_data_path)

        info_lines.append(f"Config: {data_path.name}")

        try:
            with open(data_path, "r") as f:
                data_config = yaml.safe_load(f)

            if "names" in data_config:
                info_lines.append(f"Classes: {len(data_config['names'])}")
                info_lines.append(f"Labels: {', '.join(data_config['names'])}")

            # Count images if possible
            dataset_dir = data_path.parent
            train_dir = dataset_dir / "train" / "images"
            val_dir = dataset_dir / "valid" / "images"

            if train_dir.exists():
                train_count = len(list(train_dir.glob("*")))
                info_lines.append(f"Train images: {train_count}")

            if val_dir.exists():
                val_count = len(list(val_dir.glob("*")))
                info_lines.append(f"Validation images: {val_count}")

        except Exception as e:
            info_lines.append(f"Error reading dataset: {str(e)[:50]}...")

        self.dataset_info_text.setPlainText("\n".join(info_lines))

    def _load_config(self):
        """Load configuration from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Training Configuration", "configs/", "YAML files (*.yaml *.yml)"
        )

        if file_path:
            try:
                with open(file_path, "r") as f:
                    self.training_config = yaml.safe_load(f)
                self._apply_config_to_ui()
                QMessageBox.information(self, "Success", "Configuration loaded successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load configuration: {e}")

    def _save_config(self):
        """Save current configuration to file."""
        # Update config with current UI values
        task_config = {
            "epochs": self.epochs_spin.value(),
            "patience": self.patience_spin.value(),
            "batch_size": self.batch_spin.value(),
            "learning_rate": self.lr_spin.value(),
            "imgsz": self.imgsz_spin.value(),
            "optimizer": self.optimizer_combo.currentText(),
            "momentum": self.momentum_spin.value(),
            "weight_decay": self.weight_decay_spin.value(),
            "workers": self.workers_spin.value(),
            "augment": self.augment_check.isChecked(),
            "cosine_lr": self.cosine_lr_check.isChecked(),
            "mosaic": self.mosaic_spin.value(),
            "scale": self.scale_spin.value(),
            "mixup": self.mixup_spin.value(),
            "copy_paste": self.copy_paste_spin.value(),
            "hsv_h": self.hsv_h_spin.value(),
            "hsv_s": self.hsv_s_spin.value(),
            "hsv_v": self.hsv_v_spin.value(),
        }
        # The selected model and dataset become the defaults for this task
        if self.current_model_path:
            model_path = Path(self.current_model_path)
            task_config["default_model"] = (
                model_path.name if model_path.parent.name == "pretrained" else str(model_path)
            )
        if self.current_data_path:
            task_config["default_dataset"] = Path(self.current_data_path).parent.name

        self.training_config[self.current_task] = task_config

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Training Configuration",
            "configs/training.yaml",
            "YAML files (*.yaml *.yml)",
        )

        if file_path:
            try:
                with open(file_path, "w") as f:
                    yaml.dump(self.training_config, f, default_flow_style=False)
                QMessageBox.information(self, "Success", "Configuration saved successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save configuration: {e}")

    def _start_training(self):
        """Start model training."""
        if not self.current_model_path or not self.current_data_path:
            QMessageBox.warning(self, "Warning", "Please select both a model and dataset.")
            return

        # Create descriptive output directory name

        # Extract model name from path
        model_name = Path(self.current_model_path).stem
        if model_name.endswith(".pt"):
            model_name = model_name[:-3]

        # Extract dataset name
        dataset_name = Path(self.current_data_path).parent.name

        # Create date prefix and find unique number
        date_prefix = datetime.now().strftime("%Y%m%d")
        base_name = f"{self.current_task}_{model_name}_{dataset_name}"

        # Determine output directory
        models_base = Path("data/models")
        if self.current_task == "detection":
            base_dir = models_base / "detection"
        else:
            base_dir = models_base / "segmentation"

        # Find unique directory name by incrementing number
        counter = 1
        while True:
            dir_name = f"{date_prefix}_{counter}_{base_name}"
            output_dir = base_dir / dir_name
            if not output_dir.exists():
                break
            counter += 1

        output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created output directory: {output_dir}")

        # Collect all training parameters from UI
        training_params = {
            "device": 0,  # Use first GPU if available, otherwise CPU
            "workers": self.workers_spin.value(),
            "imgsz": self.imgsz_spin.value(),
            "optimizer": self.optimizer_combo.currentText(),
            "momentum": self.momentum_spin.value(),
            "weight_decay": self.weight_decay_spin.value(),
            "augment": self.augment_check.isChecked(),
            "cosine_lr": self.cosine_lr_check.isChecked(),
            "mosaic": self.mosaic_spin.value(),
            "scale": self.scale_spin.value(),
            "mixup": self.mixup_spin.value(),
            "copy_paste": self.copy_paste_spin.value(),
            "hsv_h": self.hsv_h_spin.value(),
            "hsv_s": self.hsv_s_spin.value(),
            "hsv_v": self.hsv_v_spin.value(),
        }

        # Create training thread
        self.training_thread = ModelTrainingThread(
            task_type=self.current_task,
            model_path=self.current_model_path,
            data_path=self.current_data_path,
            epochs=self.epochs_spin.value(),
            patience=self.patience_spin.value(),
            batch_size=self.batch_spin.value(),
            lr=self.lr_spin.value(),
            output_dir=str(output_dir),
            training_params=training_params,
        )

        # Connect signals
        self.training_thread.progress_update.connect(self._on_training_progress)
        self.training_thread.raw_output.connect(self._on_raw_output)
        self.training_thread.training_complete.connect(self._on_training_complete)
        self.training_thread.error_occurred.connect(self._on_training_error)

        # Update UI
        self.start_training_btn.setEnabled(False)
        self.stop_training_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        # Set maximum to 100 for percentage-based progress
        self.progress_bar.setMaximum(100)
        self.progress_info_label.setText("Starting training...")
        self.raw_output_text.clear()

        # Start training
        self.training_thread.start()

        # Initialize timing for progress estimation
        self.training_start_time = time.time()

        # Start timer for updating time display every second
        self.time_update_timer.start(1000)  # Update every 1000ms (1 second)

        # Live monitoring finds the timestamped run folder that training creates in here
        self.results_widget.start_monitoring(
            str(output_dir), self.epochs_spin.value(), self.patience_spin.value()
        )

    def _stop_training(self):
        """Stop current training."""
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.stop_training()
            self.training_thread.wait()

        # Stop results monitoring
        self.results_widget.stop_monitoring()

        self._reset_training_ui()
        self.progress_info_label.setText("Training stopped by user")

    def _on_training_progress(self, progress_value: int, status: str):
        """Handle training progress update."""
        # Store the status for time updates
        self.last_status = status

        # Update progress display with time calculations
        self._update_progress_display(progress_value, status)

    def _update_progress_display(self, progress_value: int, status: str):
        """Update progress display with current time calculations."""
        # Set progress bar value (0-100)
        self.progress_bar.setValue(progress_value)

        # Update combined progress info with elapsed time on one line
        if self.training_start_time:
            elapsed_time = time.time() - self.training_start_time
            elapsed_str = self._format_elapsed_time(elapsed_time)

            # Extract epoch info for time estimation
            current_epoch = 0
            total_epochs = 0
            iterations_per_sec = 0

            # Try to extract epoch numbers
            if "Epoch " in status and "/" in status:
                try:
                    epoch_part = status.split("Epoch ")[1].split()[0]  # Get "5/100"
                    if "/" in epoch_part and epoch_part != "??/??":
                        current_epoch = int(epoch_part.split("/")[0])
                        total_epochs = int(epoch_part.split("/")[1])
                except (ValueError, IndexError):
                    pass

            # Try to extract iterations per second if available
            if "it/s" in status:
                try:
                    it_match = re.search(r"([\d.]+)it/s", status)
                    if it_match:
                        iterations_per_sec = float(it_match.group(1))
                except (ValueError, IndexError):
                    pass

            # Estimate the remaining time
            remaining_str = ""

            # Use iterations per second when the status carries it
            if iterations_per_sec > 0 and current_epoch > 0 and total_epochs > 0:
                # Extract batch info if available
                if "Batch " in status:
                    try:
                        batch_part = status.split("Batch ")[1].split()[0]  # Get "25/50"
                        if "/" in batch_part:
                            current_batch = int(batch_part.split("/")[0])
                            total_batches = int(batch_part.split("/")[1])

                            # Estimate remaining batches in current epoch
                            remaining_batches_epoch = total_batches - current_batch
                            # Estimate remaining batches in all remaining epochs
                            remaining_epochs = total_epochs - current_epoch
                            total_remaining_batches = remaining_batches_epoch + (
                                remaining_epochs * total_batches
                            )

                            # Estimate time based on iteration speed
                            estimated_remaining = total_remaining_batches / iterations_per_sec
                            remaining_str = self._format_elapsed_time(estimated_remaining)
                    except (ValueError, IndexError):
                        pass

            # Otherwise fall back to the overall progress percentage
            elif progress_value > 5:  # Only if we have meaningful progress
                time_per_percent = elapsed_time / progress_value
                remaining_percent = 100 - progress_value
                estimated_remaining = time_per_percent * remaining_percent
                remaining_str = self._format_elapsed_time(estimated_remaining)

            # Build the display string
            if remaining_str:
                self.progress_info_label.setText(
                    f"{status} • {elapsed_str} elapsed • {remaining_str} remaining"
                )
            else:
                self.progress_info_label.setText(f"{status} • {elapsed_str} elapsed")
        else:
            # No timing info available yet
            self.progress_info_label.setText(status)

    def _on_raw_output(self, line: str):
        """Handle raw training output."""
        # Ultralytics re-prints its progress bar as a new line for every update. Overwrite
        # the previous bar in place, as a terminal would, instead of flooding the display.
        is_progress = re.search(r"\d+% [━╸─]", line) is not None
        cursor = self.raw_output_text.textCursor()
        cursor.movePosition(cursor.End)
        if is_progress and getattr(self, "_last_output_was_progress", False):
            cursor.select(cursor.BlockUnderCursor)
            cursor.removeSelectedText()
        self._last_output_was_progress = is_progress and "100% " not in line

        # Append new line to the output display
        self.raw_output_text.append(line)

        # Auto-scroll to bottom to show latest output
        cursor = self.raw_output_text.textCursor()
        cursor.movePosition(cursor.End)
        self.raw_output_text.setTextCursor(cursor)

    def _on_training_complete(self, results_path: str, success: bool):
        """Handle training completion."""
        self._reset_training_ui()

        # Stop monitoring since training is complete
        self.results_widget.stop_monitoring()

        if success:
            self.progress_info_label.setText("Training completed successfully!")
            # The widget should show the final results already
        else:
            self.progress_info_label.setText("Training completed with issues")

    def _on_training_error(self, error_msg: str):
        """Handle training error."""
        self._reset_training_ui()

        # Stop monitoring on error
        self.results_widget.stop_monitoring()

        self.progress_info_label.setText(f"Training failed: {error_msg}")
        QMessageBox.critical(self, "Training Error", f"Training failed:\n{error_msg}")

    def _format_elapsed_time(self, elapsed_seconds: float) -> str:
        """Format elapsed time in a compact, readable format."""
        if elapsed_seconds > 3600:  # More than 1 hour
            hours = int(elapsed_seconds // 3600)
            minutes = int((elapsed_seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
        elif elapsed_seconds > 60:  # More than 1 minute
            minutes = int(elapsed_seconds // 60)
            seconds = int(elapsed_seconds % 60)
            return f"{minutes}m {seconds}s"
        else:
            seconds = int(elapsed_seconds)
            return f"{seconds}s"

    def _update_time_display(self):
        """Update the time display every second during training."""
        if not self.training_start_time or not self.last_status:
            return

        # Recalculate time information using the last known status
        self._update_progress_display(self.progress_bar.value(), self.last_status)

    def _reset_training_ui(self):
        """Reset training UI elements."""
        self.start_training_btn.setEnabled(True)
        self.stop_training_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_info_label.setText("Ready to start training")
        # The output stays until the next run starts: it holds the final results, or the
        # error a failed run ended with

        # Reset timing variables
        self.training_start_time = None
        self.last_status = ""

        # Stop time update timer
        self.time_update_timer.stop()

    def closeEvent(self, event):
        """Stop a running training with the application, so it does not continue unseen."""
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.stop_training()
            self.training_thread.wait()
        self.results_widget.stop_monitoring()
        super().closeEvent(event)
