"""Model Tuning tab for training YOLO models.

This module provides a GUI interface for training both detection and segmentation
models using Ultralytics YOLO framework with custom datasets.
"""

import os
import sys
import yaml
import subprocess
import glob
import re
import time
import json
import tempfile
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, QTextEdit, QProgressBar,
    QSplitter, QScrollArea, QFrame, QLineEdit, QFileDialog, QMessageBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QImage, QFont


class TrainingResultsWidget(QWidget):
    """Widget for displaying live training results from results.csv"""
    
    def __init__(self):
        super().__init__()
        self.results_path = None
        self.reference_path = Path("data/models/detection/object_detection_yolo11l/finetune3/results.csv")
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        # Timer for updating plots
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_plots)
        
    def start_monitoring(self, results_dir: str):
        """Start monitoring a results directory for CSV updates."""
        self.results_dir = Path(results_dir)
        self.results_path = None
        self.update_timer.start(2000)  # Update every 2 seconds
        
    def stop_monitoring(self):
        """Stop monitoring for updates."""
        self.update_timer.stop()
        self.results_path = None
        
    def set_reference_path(self, reference_path: str):
        """Set the reference results.csv file for comparison."""
        self.reference_path = Path(reference_path)
        print(f"[TRAINING_RESULTS] Reference path set to: {self.reference_path}")
        
    def update_plots(self):
        """Update the plots with latest data from results.csv"""
        # First, try to find the results.csv file
        if not self.results_path or not self.results_path.exists():
            self._find_results_csv()
            
        if not self.results_path or not self.results_path.exists():
            return
            
        try:
            # Read the current training CSV file
            df = pd.read_csv(self.results_path)
            
            # Load reference data
            reference_df = None
            if self.reference_path.exists():
                try:
                    reference_df = pd.read_csv(self.reference_path)
                    print(f"[TRAINING_RESULTS] Loaded reference data with {len(reference_df)} epochs from {self.reference_path}")
                except Exception as e:
                    print(f"[TRAINING_RESULTS] Could not load reference data: {e}")
            else:
                print(f"[TRAINING_RESULTS] Reference file not found: {self.reference_path}")
            
            if df.empty:
                return
                
            # Clear the figure
            self.figure.clear()
            
            # Create subplots
            ax1 = self.figure.add_subplot(2, 2, 1)
            ax2 = self.figure.add_subplot(2, 2, 2)
            ax3 = self.figure.add_subplot(2, 2, 3)
            ax4 = self.figure.add_subplot(2, 2, 4)
            
            epochs = df.index + 1
            
            # Plot training and validation losses
            if 'train/box_loss' in df.columns:
                ax1.plot(epochs, df['train/box_loss'], label='Train Box Loss', color='blue', linewidth=2)
            if 'train/cls_loss' in df.columns:
                ax1.plot(epochs, df['train/cls_loss'], label='Train Cls Loss', color='red', linewidth=2)
            if 'train/dfl_loss' in df.columns:
                ax1.plot(epochs, df['train/dfl_loss'], label='Train DFL Loss', color='green', linewidth=2)
            if 'val/box_loss' in df.columns:
                ax1.plot(epochs, df['val/box_loss'], label='Val Box Loss', color='cyan', linestyle='--', linewidth=2)
            if 'val/cls_loss' in df.columns:
                ax1.plot(epochs, df['val/cls_loss'], label='Val Cls Loss', color='magenta', linestyle='--', linewidth=2)
            if 'val/dfl_loss' in df.columns:
                ax1.plot(epochs, df['val/dfl_loss'], label='Val DFL Loss', color='orange', linestyle='--', linewidth=2)
                
            # Add reference data to loss plot
            if reference_df is not None:
                ref_epochs = reference_df.index + 1
                if 'train/box_loss' in reference_df.columns:
                    ax1.plot(ref_epochs, reference_df['train/box_loss'], label='Ref Train Box', color='lightblue', alpha=0.6, linestyle='--', linewidth=1.5)
                if 'val/box_loss' in reference_df.columns:
                    ax1.plot(ref_epochs, reference_df['val/box_loss'], label='Ref Val Box', color='lightcyan', alpha=0.6, linestyle='--', linewidth=1.5)
                
            ax1.set_title('Loss Curves')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Plot mAP metrics
            if 'metrics/mAP50(B)' in df.columns:
                ax2.plot(epochs, df['metrics/mAP50(B)'], label='mAP@0.5', color='blue', linewidth=2)
            if 'metrics/mAP50-95(B)' in df.columns:
                ax2.plot(epochs, df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', color='red', linewidth=2)
                
            # Add reference mAP data
            if reference_df is not None:
                if 'metrics/mAP50(B)' in reference_df.columns:
                    ax2.plot(ref_epochs, reference_df['metrics/mAP50(B)'], label='Ref mAP@0.5', color='lightblue', alpha=0.6, linestyle='--', linewidth=1.5)
                if 'metrics/mAP50-95(B)' in reference_df.columns:
                    ax2.plot(ref_epochs, reference_df['metrics/mAP50-95(B)'], label='Ref mAP@0.5:0.95', color='lightcoral', alpha=0.6, linestyle='--', linewidth=1.5)
                
            ax2.set_title('mAP Metrics')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('mAP')
            ax2.legend()
            ax2.grid(True)
            
            # Plot precision and recall
            if 'metrics/precision(B)' in df.columns:
                ax3.plot(epochs, df['metrics/precision(B)'], label='Precision', color='green', linewidth=2)
            if 'metrics/recall(B)' in df.columns:
                ax3.plot(epochs, df['metrics/recall(B)'], label='Recall', color='orange', linewidth=2)
                
            # Add reference precision/recall data
            if reference_df is not None:
                if 'metrics/precision(B)' in reference_df.columns:
                    ax3.plot(ref_epochs, reference_df['metrics/precision(B)'], label='Ref Precision', color='lightgreen', alpha=0.6, linestyle='--', linewidth=1.5)
                if 'metrics/recall(B)' in reference_df.columns:
                    ax3.plot(ref_epochs, reference_df['metrics/recall(B)'], label='Ref Recall', color='wheat', alpha=0.6, linestyle='--', linewidth=1.5)
                
            ax3.set_title('Precision & Recall')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Score')
            ax3.legend()
            ax3.grid(True)
            
            # Plot learning rate and other metrics
            if 'lr/pg0' in df.columns:
                ax4.plot(epochs, df['lr/pg0'], label='Learning Rate', color='purple', linewidth=2)
                ax4.set_ylabel('Learning Rate', color='purple')
                ax4.tick_params(axis='y', labelcolor='purple')
                
                # Add reference learning rate
                if reference_df is not None and 'lr/pg0' in reference_df.columns:
                    ax4.plot(ref_epochs, reference_df['lr/pg0'], label='Ref LR', color='plum', alpha=0.6, linestyle='--', linewidth=1.5)
                
                # Add a second y-axis for fitness if available
                if 'fitness' in df.columns:
                    ax4_twin = ax4.twinx()
                    ax4_twin.plot(epochs, df['fitness'], label='Fitness', color='red', linewidth=2)
                    ax4_twin.set_ylabel('Fitness', color='red')
                    ax4_twin.tick_params(axis='y', labelcolor='red')
                    
                    # Add reference fitness
                    if reference_df is not None and 'fitness' in reference_df.columns:
                        ax4_twin.plot(ref_epochs, reference_df['fitness'], label='Ref Fitness', color='lightcoral', alpha=0.6, linestyle='--', linewidth=1.5)
                        
            ax4.set_title('Learning Rate & Fitness')
            ax4.set_xlabel('Epoch')
            ax4.legend()
            ax4.grid(True)
            
            # Adjust layout and refresh
            self.figure.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            print(f"[TRAINING_RESULTS] Error updating plots: {e}")
            
    def _find_results_csv(self):
        """Find the results.csv file in subdirectories."""
        if not hasattr(self, 'results_dir') or not self.results_dir.exists():
            return
            
        # Look for results.csv in the directory and subdirectories
        for csv_path in self.results_dir.rglob("results.csv"):
            if csv_path.exists():
                self.results_path = csv_path
                print(f"[TRAINING_RESULTS] Found results.csv at: {csv_path}")
                return
                
        # Also check direct path
        direct_path = self.results_dir / "results.csv"
        if direct_path.exists():
            self.results_path = direct_path
            print(f"[TRAINING_RESULTS] Found results.csv at: {direct_path}")
            
    def clear_plots(self):
        """Clear all plots."""
        self.figure.clear()
        self.canvas.draw()


class ModelTrainingThread(QThread):
    """Background thread for model training to prevent GUI freezing."""
    
    progress_update = pyqtSignal(int, str)  # epoch, status message
    raw_output = pyqtSignal(str)  # raw training output line
    training_complete = pyqtSignal(str, bool)  # results_path, success
    error_occurred = pyqtSignal(str)  # error message
    
    def __init__(self, task_type: str, model_path: str, data_path: str, 
                 epochs: int, patience: int, batch_size: float, lr: float,
                 output_dir: str, training_params: dict = None):
        super().__init__()
        self.task_type = task_type
        self.model_path = model_path
        self.data_path = data_path
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.lr = lr
        self.output_dir = output_dir
        self.training_params = training_params or {}
        self.should_stop = False
        
    def run(self):
        """Run the training process."""
        import subprocess
        import json
        import tempfile
        
        try:
            print("[TRAINING] Starting model training in separate process...")
            
            # Create training configuration
            training_config = {
                'model_path': self.model_path,
                'data_path': self.data_path,
                'output_dir': self.output_dir,
                'epochs': self.epochs,
                'patience': self.patience,
                'batch_size': self.batch_size,
                'learning_rate': self.lr,
                'device': self.training_params.get('device', 0),
                'workers': self.training_params.get('workers', 0),
                'imgsz': self.training_params.get('imgsz', 640),
                'optimizer': self.training_params.get('optimizer', 'SGD'),
                'momentum': self.training_params.get('momentum', 0.937),
                'weight_decay': self.training_params.get('weight_decay', 0.0005),
                'augment': self.training_params.get('augment', True),
                'mosaic': self.training_params.get('mosaic', 1.0),
                'mixup': self.training_params.get('mixup', 0.0),
                'copy_paste': self.training_params.get('copy_paste', 0.0),
                'hsv_h': self.training_params.get('hsv_h', 0.015),
                'hsv_s': self.training_params.get('hsv_s', 0.7),
                'hsv_v': self.training_params.get('hsv_v', 0.4),
                'results_file': 'training_results.txt'
            }
            
            # Write config to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(training_config, f, indent=2)
                config_path = f.name
            
            # Path to training script
            script_path = Path(__file__).parent.parent / "training" / "train_model_subprocess.py"
            
            try:
                # Reset training state flags
                if hasattr(self, '_training_started'):
                    delattr(self, '_training_started')
                if hasattr(self, '_last_prep_update'):
                    delattr(self, '_last_prep_update')
                if hasattr(self, '_fallback_sent'):
                    delattr(self, '_fallback_sent')
                
                # Start training process
                import subprocess
                process = subprocess.Popen(
                    [sys.executable, str(script_path), '--config', config_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    universal_newlines=True,
                    cwd=os.getcwd()
                )
                
                epoch_count = 0
                total_epochs = self.epochs
                last_update_time = 0
                start_time = time.time()
                
                while True:
                    if self.should_stop:
                        process.terminate()
                        process.wait()
                        break
                        
                    # Read output line by line
                    line = process.stdout.readline()
                    
                    if line == '' and process.poll() is not None:
                        break
                        
                    if line:
                        line = line.strip()
                        current_time = time.time()
                        
                        # Emit raw output for display
                        if line:  # Only emit non-empty lines
                            self.raw_output.emit(line)
                        
                        # Simple progress detection
                        epoch_count = self._process_training_line(line, current_time, start_time, epoch_count, total_epochs)
                
                # Check result
                return_code = process.wait()
                
                if return_code == 0 and not self.should_stop:
                    # Read results path
                    results_path = self.output_dir
                    if os.path.exists("training_results.txt"):
                        with open("training_results.txt", "r") as f:
                            results_path = f.read().strip()
                        os.remove("training_results.txt")
                    
                    self.training_complete.emit(results_path, True)
                elif self.should_stop:
                    print("[TRAINING] Training stopped by user")
                else:
                    self.error_occurred.emit(f"Training failed with return code: {return_code}")
                    
            finally:
                # Cleanup temporary files
                if os.path.exists(config_path):
                    os.remove(config_path)
                if os.path.exists("training_results.txt"):
                    os.remove("training_results.txt")
                    
        except Exception as e:
            print(f"[TRAINING] Error: {e}")
            self.error_occurred.emit(str(e))
    
    def stop_training(self):
        """Request to stop training."""
        self.should_stop = True
    
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
    
    def _process_training_line(self, line: str, current_time: float, start_time: float, epoch_count: int, total_epochs: int):
        """Process a single line of training output and emit progress updates."""
        # Skip empty lines
        if not line.strip():
            return epoch_count
            
        # Check for training start indicators (transition from preparation to training)
        training_start_keywords = [
            "starting training",
            "train:",
            "starting epoch",
            "beginning training", 
            "training started",
            "optimizer:",
            "lr0=",  # Learning rate initialization
            "momentum=",  # Training hyperparameters
            "ultralytics yolo",  # Ultralytics banner
            "python train.py",  # Training command
            "model summary:",  # Model summary output
            "freezing",  # Layer freezing
            "amp:",  # Mixed precision
            "epochs:",  # Epoch configuration
            "image sizes",  # Image size configuration
            "starting training for",  # Common ultralytics phrase
            "training parameters:",  # Parameter listing
            "wandb:",  # Weights & Biases integration
            "tensorboard:",  # TensorBoard logging
        ]
        
        # Check for preparation phase - look for common pre-training keywords
        prep_keywords = [
            "scanning", "loading", "cache", "labels", "dataset", "images",
            "caching", "reading", "found", "missing", "empty", "checking"
        ]
        is_preparing = any(keyword in line.lower() for keyword in prep_keywords)
        is_training_starting = any(keyword in line.lower() for keyword in training_start_keywords)
        
        # Debug logging to help identify patterns
        if is_training_starting:
            print(f"[TRAINING_DEBUG] Training start detected: {line}")
        elif is_preparing:
            print(f"[TRAINING_DEBUG] Preparation detected: {line}")
        else:
            # Check for any number patterns that might be epochs
            if re.search(r'\b(\d+)/(\d+)\b', line) and epoch_count == 0:
                print(f"[TRAINING_DEBUG] Unclassified number pattern: {line}")
        
        # Detect transition from preparation to training
        if is_training_starting and epoch_count == 0:
            # Mark that training has started but no epochs detected yet
            if not hasattr(self, '_training_started'):
                print(f"[TRAINING_DEBUG] Setting training started flag")
                self.progress_update.emit(1, "Training starting • Epoch ??/?? • Batch ??/??")
                self._training_started = True
            return epoch_count
        
        # During preparation, avoid interpreting numbers as epochs/batches
        if is_preparing and epoch_count == 0 and not hasattr(self, '_training_started'):
            # Only update every 3 seconds during prep to avoid spam
            if not hasattr(self, '_last_prep_update') or current_time - self._last_prep_update > 3:
                self.progress_update.emit(0, "Preparing training • Epoch ??/?? • Batch ??/??")
                self._last_prep_update = current_time
            return epoch_count
        
        # Look for actual epoch training patterns (usually after preparation)
        # More specific patterns to avoid false positives during preparation
        epoch_patterns = [
            r'Epoch\s+(\d+)/(\d+)',  # "Epoch 1/100"
            r'(\d+)/(\d+).*?epochs?',  # "1/100 epochs"
            r'^(\d+)/(\d+)\s+[\d.]+[GM]?',  # Ultralytics format: "1/10      4.96G" (start of line)
            r'^(\d+)/(\d+)\s+',  # Simpler start of line pattern: "1/10 " 
        ]
        
        for pattern in epoch_patterns:
            epoch_match = re.search(pattern, line, re.IGNORECASE)
            if epoch_match:
                print(f"[TRAINING_DEBUG] Epoch pattern matched: '{pattern}' -> groups: {epoch_match.groups()} from line: {line[:100]}...")
                try:
                    current_epoch = int(epoch_match.group(1))
                    detected_total = int(epoch_match.group(2))
                    
                    # Update total epochs if we detect a different value
                    if detected_total != total_epochs and detected_total > 0:
                        total_epochs = detected_total
                    
                    # Only update if epoch changed and is reasonable
                    if current_epoch != epoch_count and 1 <= current_epoch <= total_epochs:
                        # Set actual training start time on first epoch detection (excluding preprocessing)
                        if not hasattr(self, 'actual_training_start_time') or self.actual_training_start_time is None:
                            self.actual_training_start_time = time.time()
                            print(f"[TRAINING_DEBUG] Actual training start time set at epoch {current_epoch}")
                        
                        epoch_count = current_epoch
                        progress_percent = int((current_epoch / total_epochs) * 100)
                        status = f"Training • Epoch {current_epoch}/{total_epochs}"
                        print(f"[TRAINING_DEBUG] Epoch detected: {current_epoch}/{total_epochs}")
                        self.progress_update.emit(progress_percent, status)
                        # Mark that actual training has started
                        self._training_started = True
                        return epoch_count
                except (ValueError, ZeroDivisionError):
                    pass
        
        # Alternative detection: Look for progress bars or percentage indicators
        # This catches cases where ultralytics uses different output formats
        progress_indicators = [
            r'(\d+)%.*?(\d+)/(\d+)',  # Progress percentage with counts
            r'\|.*?\|.*?(\d+)/(\d+)',  # Progress bar format
            r'(\d+)/(\d+).*?\[.*?\]',  # Count with time brackets
        ]
        
        for pattern in progress_indicators:
            progress_match = re.search(pattern, line)
            if progress_match and not is_preparing:
                # If we see progress indicators and we're not in preparation,
                # we're likely in training even without explicit epoch markers
                if not hasattr(self, '_training_started') and epoch_count == 0:
                    print(f"[TRAINING_DEBUG] Progress indicator detected, marking training as started: {line}")
                    self.progress_update.emit(2, "Training • Epoch ??/?? • Batch ??/??")
                    self._training_started = True
                return epoch_count
        
        # Look for batch/iteration progress within epochs (with iterations/sec info)
        # Pattern like: "50%|███████ | 25/50 [00:30<00:15, 1.67it/s]"
        batch_match = re.search(r'(\d+)%.*?(\d+)/(\d+).*?\[([\d:]+)<([\d:]+),\s*([\d.]+)it/s\]', line)
        if batch_match and epoch_count > 0:
            try:
                batch_percent = int(batch_match.group(1))
                current_batch = int(batch_match.group(2))
                total_batches = int(batch_match.group(3))
                iterations_per_sec = float(batch_match.group(6))
                
                # Calculate overall progress: completed epochs + current epoch progress
                epoch_progress = ((epoch_count - 1) / total_epochs) * 100
                current_epoch_progress = (batch_percent / 100) * (1 / total_epochs) * 100
                overall_progress = min(100, int(epoch_progress + current_epoch_progress))
                
                status = f"Training • Epoch {epoch_count}/{total_epochs} • Batch {current_batch}/{total_batches} • {iterations_per_sec:.1f}it/s"
                self.progress_update.emit(overall_progress, status)
                return epoch_count
            except (ValueError, ZeroDivisionError):
                pass
        
        # Simpler batch progress without timing info
        elif re.search(r'(\d+)%.*?(\d+)/(\d+)', line) and epoch_count > 0:
            batch_match = re.search(r'(\d+)%.*?(\d+)/(\d+)', line)
            try:
                batch_percent = int(batch_match.group(1))
                current_batch = int(batch_match.group(2))
                total_batches = int(batch_match.group(3))
                
                # Calculate overall progress
                epoch_progress = ((epoch_count - 1) / total_epochs) * 100
                current_epoch_progress = (batch_percent / 100) * (1 / total_epochs) * 100
                overall_progress = min(100, int(epoch_progress + current_epoch_progress))
                
                status = f"Training • Epoch {epoch_count}/{total_epochs} • Batch {current_batch}/{total_batches}"
                self.progress_update.emit(overall_progress, status)
                return epoch_count
            except (ValueError, ZeroDivisionError):
                pass
        
        # Fallback: if training has been running for a while without clear state detection
        if current_time - start_time > 30 and epoch_count == 0:
            if not hasattr(self, '_fallback_sent'):
                if hasattr(self, '_training_started'):
                    self.progress_update.emit(2, "Training • Epoch ??/?? • Batch ??/??")
                else:
                    self.progress_update.emit(1, "Training starting • Epoch ??/?? • Batch ??/??")
                    self._training_started = True
                self._fallback_sent = True
        
        return epoch_count


class ModelTuningTab(QWidget):
    """Tab for tuning YOLO models with custom datasets."""
    
    def __init__(self):
        super().__init__()
        
        # State
        self.current_task = "detection"
        self.current_model_path = ""
        self.current_data_path = ""
        self.training_thread: Optional[ModelTrainingThread] = None
        self.training_config = {}
        self.current_results_dir = None
        self.training_start_time = None  # When training subprocess starts (includes preprocessing)
        self.actual_training_start_time = None  # When actual epoch training begins
        self.last_epoch = 0
        
        # Initialize UI
        self._init_ui()
        self._load_default_config()
        self._update_model_options()
        self._update_data_options()
        
    def _init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout()
        
        # Create splitter for main content
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Configuration
        config_panel = self._create_config_panel()
        splitter.addWidget(config_panel)
        
        # Right panel - Training and Results
        results_panel = self._create_results_panel()
        splitter.addWidget(results_panel)
        
        # Set splitter proportions
        splitter.setStretchFactor(0, 1)  # Config panel
        splitter.setStretchFactor(1, 2)  # Results panel
        
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)
        
    def _create_config_panel(self) -> QWidget:
        """Create the configuration panel."""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Task selection
        task_group = QGroupBox("Training Task")
        task_layout = QFormLayout()
        
        self.task_combo = QComboBox()
        self.task_combo.addItems(["detection", "field segmentation"])
        self.task_combo.currentTextChanged.connect(self._on_task_changed)
        self.task_combo.setToolTip("Select the type of computer vision task:\n\n• Detection: Find and classify objects with bounding boxes\n  - Examples: players, disc, referees in Ultimate Frisbee\n  - Output: [class, x, y, width, height, confidence]\n\n• Field Segmentation: Pixel-level classification of field regions\n  - Examples: field boundaries, end zones, out-of-bounds areas\n  - Output: segmentation masks for each region\n\nDifferent tasks use specialized model architectures and datasets.")
        task_layout.addRow("Task Type:", self.task_combo)
        
        task_group.setLayout(task_layout)
        layout.addWidget(task_group)
        
        # Model selection
        model_group = QGroupBox("Base Model Selection")
        model_layout = QFormLayout()
        
        self.model_combo = QComboBox()
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        self.model_combo.setToolTip("Select the base YOLO model to start training from.\n\n• Model sizes: n (nano), s (small), m (medium), l (large), x (extra-large)\n• Examples: yolo11n.pt (fast, 2.6M params), yolo11l.pt (accurate, 25.3M params)\n• Pretrained models: learned features from COCO dataset (80 classes)\n• Trade-offs: Larger models = better accuracy but slower training/inference\n• Custom models: .pt files from previous training runs")
        model_layout.addRow("Base Model:", self.model_combo)
        
        # Model info display
        self.model_info_text = QTextEdit()
        self.model_info_text.setMaximumHeight(100)
        self.model_info_text.setReadOnly(True)
        self.model_info_text.setToolTip("Displays information about the selected model including file size and type.")
        model_layout.addRow("Model Info:", self.model_info_text)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Dataset selection
        data_group = QGroupBox("Training Dataset")
        data_layout = QFormLayout()
        
        self.data_combo = QComboBox()
        self.data_combo.currentTextChanged.connect(self._on_data_changed)
        self.data_combo.setToolTip("Select the dataset to train on (YOLO format required).\n\n• Format: data.yaml file with train/val paths and class names\n• Examples: coco8.yaml (sample), custom_dataset_v3.yaml\n• Version numbers: v2, v3, v4 (higher = typically improved)\n• Structure: train/images/, train/labels/, valid/images/, valid/labels/\n• More training images = generally better model performance\n• Labels: .txt files with class_id x_center y_center width height")
        data_layout.addRow("Dataset:", self.data_combo)
        
        # Dataset info display
        self.dataset_info_text = QTextEdit()
        self.dataset_info_text.setMaximumHeight(100)
        self.dataset_info_text.setReadOnly(True)
        self.dataset_info_text.setToolTip("Shows dataset information including number of classes, class names,\nand training/validation image counts.")
        data_layout.addRow("Dataset Info:", self.dataset_info_text)
        
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        
        # Training parameters
        params_group = QGroupBox("Training Parameters")
        params_layout = QFormLayout()
        
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(100)
        self.epochs_spin.setToolTip("Number of complete passes through the training dataset.\n\n• Default: 100 epochs\n• Range: 1-1000+ epochs\n• Examples: 50 (quick test), 100 (standard), 300 (fine-tuning)\n• More epochs = longer training but potentially better performance\n• Use early stopping (patience) to prevent overfitting")
        params_layout.addRow("Epochs:", self.epochs_spin)
        
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 100)
        self.patience_spin.setValue(10)
        self.patience_spin.setToolTip("Early stopping patience: number of epochs to wait for improvement\nbefore stopping training. Prevents overfitting and saves time.\n\n• Default: 100 epochs (detection), 15 (segmentation)\n• Range: 1-100+ epochs\n• Examples: 10 (aggressive), 50 (moderate), 100 (patient)\n• Higher values = more patient, lower values = stop sooner\n• Monitors validation metrics for improvement")
        params_layout.addRow("Patience:", self.patience_spin)
        
        self.batch_spin = QDoubleSpinBox()
        self.batch_spin.setRange(0.1, 128)
        self.batch_spin.setDecimals(1)
        self.batch_spin.setValue(16)
        self.batch_spin.setToolTip("Batch size: integer (e.g. 16) or fraction for GPU memory (e.g. 0.8).\n\n• Default: 16 (detection), 8 (segmentation)\n• Integer: 1-128+ (exact number of images per batch)\n• Fraction: 0.1-1.0 (percentage of GPU memory to use)\n• Examples: 16 (fixed), 0.6 (60% GPU memory), -1 (auto 60%)\n• Auto modes: -1 (60% GPU memory), 0.7 (70% GPU memory)\n• Larger batches = more stable gradients but need more memory")
        params_layout.addRow("Batch Size:", self.batch_spin)
        
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 1.0)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setValue(0.01)
        self.lr_spin.setToolTip("Initial learning rate (lr0) - controls how big steps the model takes.\n\n• Default: 0.01 (SGD), 0.001 (Adam)\n• Range: 0.0001-1.0 (typically 0.001-0.01)\n• Examples: 0.001 (conservative), 0.01 (standard), 0.1 (aggressive)\n• Higher values = faster learning but risk instability\n• Lower values = more stable but slower convergence\n• Automatically decays during training using schedulers")
        params_layout.addRow("Learning Rate:", self.lr_spin)
        
        # Image size
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(320, 1280)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)
        self.imgsz_spin.setToolTip("Input image size for training (square images).\n\n• Default: 640 pixels\n• Range: 320-1280 (must be multiple of 32)\n• Examples: 416 (fast), 640 (standard), 832 (detailed), 1024 (high-res)\n• Larger sizes = better detail recognition but slower training\n• Smaller sizes = faster training but less detail\n• All images resized to this dimension before processing")
        params_layout.addRow("Image Size:", self.imgsz_spin)
        
        # Optimizer
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(['SGD', 'Adam', 'AdamW', 'RMSProp'])
        self.optimizer_combo.setCurrentText('SGD')
        self.optimizer_combo.setToolTip("Optimization algorithm for training:\n\n• Default: 'auto' (SGD for most cases)\n• SGD: Simple, stable, momentum-based (good for most cases)\n• Adam: Adaptive learning rates, faster convergence\n• AdamW: Adam with better weight decay handling\n• RMSProp: Good for noisy gradients and RNNs\n• NAdam, RAdam: Advanced Adam variants\n\nSGD with momentum (0.937) is proven effective for YOLO models.")
        params_layout.addRow("Optimizer:", self.optimizer_combo)
        
        # Momentum
        self.momentum_spin = QDoubleSpinBox()
        self.momentum_spin.setRange(0.0, 1.0)
        self.momentum_spin.setDecimals(3)
        self.momentum_spin.setSingleStep(0.001)
        self.momentum_spin.setValue(0.937)
        self.momentum_spin.setToolTip("Momentum factor for SGD optimizer (or beta1 for Adam).\n\n• Default: 0.937 (proven optimal for YOLO)\n• Range: 0.0-1.0 (typically 0.8-0.99)\n• Examples: 0.9 (standard), 0.937 (YOLO optimized), 0.95 (high momentum)\n• Helps accelerate training in consistent directions\n• Higher values = smoother convergence, more momentum\n• Lower values = more responsive to gradient changes")
        params_layout.addRow("Momentum:", self.momentum_spin)
        
        # Weight Decay
        self.weight_decay_spin = QDoubleSpinBox()
        self.weight_decay_spin.setRange(0.0, 0.01)
        self.weight_decay_spin.setDecimals(4)
        self.weight_decay_spin.setSingleStep(0.0001)
        self.weight_decay_spin.setValue(0.0005)
        self.weight_decay_spin.setToolTip("L2 regularization penalty to prevent overfitting.\n\n• Default: 0.0005 (YOLO optimized)\n• Range: 0.0-0.01 (typically 0.0001-0.001)\n• Examples: 0.0001 (light), 0.0005 (standard), 0.001 (strong)\n• Penalizes large weights to keep the model simple\n• Higher values = stronger regularization, lower overfitting risk\n• Too high = underfitting, too low = overfitting risk")
        params_layout.addRow("Weight Decay:", self.weight_decay_spin)
        
        # Workers
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 16)
        self.workers_spin.setValue(0)  # Use 0 for Windows compatibility
        self.workers_spin.setToolTip("Number of CPU threads for data loading (per GPU if multi-GPU).\n\n• Default: 8 (Linux/Mac), 0 (Windows recommended)\n• Range: 0-16+ (depends on CPU cores)\n• Examples: 0 (single-thread), 4 (quad-core), 8 (standard)\n• Higher values = faster data loading but more CPU usage\n• Set to 0 for Windows to avoid multiprocessing errors\n• Use 2x CPU cores for optimal performance on Linux/Mac")
        params_layout.addRow("Workers:", self.workers_spin)
        
        # Augmentation checkbox
        self.augment_check = QCheckBox()
        self.augment_check.setChecked(True)
        self.augment_check.setToolTip("Enable data augmentation during training.\n\n• Default: True (recommended for most cases)\n• Includes: rotation, scaling, flipping, color changes, mosaic\n• Benefits: Improves robustness, prevents overfitting, increases dataset variety\n• Examples: HSV shifts, geometric transforms, mixup, copy-paste\n• Disable only for: perfect datasets, specific requirements\n• Automatically disabled in final epochs (close_mosaic=10)")
        params_layout.addRow("Data Augmentation:", self.augment_check)
        
        # Additional augmentation parameters
        aug_group = QGroupBox("Augmentation Settings")
        aug_layout = QFormLayout()
        
        # Mosaic probability
        self.mosaic_spin = QDoubleSpinBox()
        self.mosaic_spin.setRange(0.0, 1.0)
        self.mosaic_spin.setDecimals(2)
        self.mosaic_spin.setValue(1.0)
        self.mosaic_spin.setToolTip("Probability of mosaic augmentation (combines 4 images).\n• Default: 1.0 (always on)\n• Range: 0.0-1.0\n• Highly effective for scene understanding")
        aug_layout.addRow("Mosaic:", self.mosaic_spin)
        
        # Mixup probability
        self.mixup_spin = QDoubleSpinBox()
        self.mixup_spin.setRange(0.0, 1.0)
        self.mixup_spin.setDecimals(2)
        self.mixup_spin.setValue(0.0)
        self.mixup_spin.setToolTip("Probability of mixup augmentation (blends images).\n• Default: 0.0\n• Range: 0.0-1.0\n• Enhances generalization")
        aug_layout.addRow("Mixup:", self.mixup_spin)
        
        # Copy-paste probability
        self.copy_paste_spin = QDoubleSpinBox()
        self.copy_paste_spin.setRange(0.0, 1.0)
        self.copy_paste_spin.setDecimals(2)
        self.copy_paste_spin.setValue(0.0)
        self.copy_paste_spin.setToolTip("Copy-paste augmentation (segmentation only).\n• Default: 0.0\n• Range: 0.0-1.0\n• Copies objects between images")
        aug_layout.addRow("Copy-Paste:", self.copy_paste_spin)
        
        # HSV-H (Hue)
        self.hsv_h_spin = QDoubleSpinBox()
        self.hsv_h_spin.setRange(0.0, 1.0)
        self.hsv_h_spin.setDecimals(3)
        self.hsv_h_spin.setValue(0.015)
        self.hsv_h_spin.setToolTip("HSV Hue augmentation range.\n• Default: 0.015\n• Range: 0.0-1.0\n• Adjusts color hue for lighting variety")
        aug_layout.addRow("HSV-H (Hue):", self.hsv_h_spin)
        
        # HSV-S (Saturation)
        self.hsv_s_spin = QDoubleSpinBox()
        self.hsv_s_spin.setRange(0.0, 1.0)
        self.hsv_s_spin.setDecimals(2)
        self.hsv_s_spin.setValue(0.7)
        self.hsv_s_spin.setToolTip("HSV Saturation augmentation range.\n• Default: 0.7\n• Range: 0.0-1.0\n• Adjusts color intensity")
        aug_layout.addRow("HSV-S (Saturation):", self.hsv_s_spin)
        
        # HSV-V (Value/Brightness)
        self.hsv_v_spin = QDoubleSpinBox()
        self.hsv_v_spin.setRange(0.0, 1.0)
        self.hsv_v_spin.setDecimals(2)
        self.hsv_v_spin.setValue(0.4)
        self.hsv_v_spin.setToolTip("HSV Value (brightness) augmentation range.\n• Default: 0.4\n• Range: 0.0-1.0\n• Adjusts brightness for lighting conditions")
        aug_layout.addRow("HSV-V (Brightness):", self.hsv_v_spin)
        
        aug_group.setLayout(aug_layout)
        layout.addWidget(aug_group)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Config buttons
        config_buttons_layout = QHBoxLayout()
        
        self.load_config_btn = QPushButton("Load Config")      
        self.load_config_btn.clicked.connect(self._load_config)
        self.load_config_btn.setToolTip("Load training parameters from a saved YAML configuration file.\nThis will update all parameter values in the interface.\nUseful for reusing proven parameter combinations.")
        config_buttons_layout.addWidget(self.load_config_btn)
        
        self.save_config_btn = QPushButton("Save Config")
        self.save_config_btn.clicked.connect(self._save_config)
        self.save_config_btn.setToolTip("Save current training parameters to a YAML configuration file.\nAllows you to reuse these settings later or share with others.\nConfigurations are saved per task type (detection/segmentation).")
        config_buttons_layout.addWidget(self.save_config_btn)
        
        layout.addLayout(config_buttons_layout)
        
        # Training controls
        controls_group = QGroupBox("Training Controls")
        controls_layout = QVBoxLayout()
        
        self.start_training_btn = QPushButton("Start Training")
        self.start_training_btn.clicked.connect(self._start_training)
        self.start_training_btn.setToolTip("Begin training the selected model on the chosen dataset.\nEnsure you have selected both a base model and dataset.\nTraining runs in background and shows live progress graphs.")
        controls_layout.addWidget(self.start_training_btn)
        
        self.stop_training_btn = QPushButton("Stop Training")
        self.stop_training_btn.clicked.connect(self._stop_training)
        self.stop_training_btn.setEnabled(False)
        self.stop_training_btn.setToolTip("Stop the current training process.\nThis will terminate training gracefully and save progress.\nThe model will be saved in its current state.")
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
        progress_group.setMaximumHeight(200)  # Make section smaller
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
        self.raw_output_text.setMaximumHeight(120)  # Compact but readable
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
                with open(config_path, 'r') as f:
                    self.training_config = yaml.safe_load(f)
                self._apply_config_to_ui()
            else:
                # Create default config
                self.training_config = {
                    'detection': {
                        'epochs': 100,
                        'patience': 10,
                        'batch_size': 16,
                        'learning_rate': 0.01
                    },
                    'segmentation': {
                        'epochs': 150,
                        'patience': 15,
                        'batch_size': 8,
                        'learning_rate': 0.01
                    }
                }
        except Exception as e:
            print(f"[MODEL_TUNING] Error loading config: {e}")
            
    def _apply_config_to_ui(self):
        """Apply loaded configuration to UI elements."""
        task_config = self.training_config.get(self.current_task, {})
        
        if 'epochs' in task_config:
            self.epochs_spin.setValue(task_config['epochs'])
        if 'patience' in task_config:
            self.patience_spin.setValue(task_config['patience'])
        if 'batch_size' in task_config:
            self.batch_spin.setValue(task_config['batch_size'])
        if 'learning_rate' in task_config:
            self.lr_spin.setValue(task_config['learning_rate'])
        if 'imgsz' in task_config:
            self.imgsz_spin.setValue(task_config['imgsz'])
        if 'optimizer' in task_config:
            self.optimizer_combo.setCurrentText(task_config['optimizer'])
        if 'momentum' in task_config:
            self.momentum_spin.setValue(task_config['momentum'])
        if 'weight_decay' in task_config:
            self.weight_decay_spin.setValue(task_config['weight_decay'])
        if 'workers' in task_config:
            self.workers_spin.setValue(task_config['workers'])
        if 'augment' in task_config:
            self.augment_check.setChecked(task_config['augment'])
        if 'mosaic' in task_config:
            self.mosaic_spin.setValue(task_config['mosaic'])
        if 'mixup' in task_config:
            self.mixup_spin.setValue(task_config['mixup'])
        if 'copy_paste' in task_config:
            self.copy_paste_spin.setValue(task_config['copy_paste'])
        if 'hsv_h' in task_config:
            self.hsv_h_spin.setValue(task_config['hsv_h'])
        if 'hsv_s' in task_config:
            self.hsv_s_spin.setValue(task_config['hsv_s'])
        if 'hsv_v' in task_config:
            self.hsv_v_spin.setValue(task_config['hsv_v'])
            
    def _on_task_changed(self, task: str):
        """Handle task type change."""
        if task == "detection":
            self.current_task = "detection"
        else:  # field segmentation
            self.current_task = "segmentation"
            
        self._update_model_options()
        self._update_data_options()
        self._apply_config_to_ui()
        
    def _update_model_options(self):
        """Update available model options based on task."""
        self.model_combo.clear()
        
        models_path = Path("data/models")
        pretrained_path = models_path / "pretrained"
        
        available_models = []
        
        if self.current_task == "detection":
            # Add pretrained detection models (non-seg)
            if pretrained_path.exists():
                for model_file in pretrained_path.glob("*.pt"):
                    if "-seg" not in model_file.name and "-pose" not in model_file.name:
                        available_models.append(str(model_file))
            
            # Add existing detection models for further tuning (including finetune directories)
            detection_path = models_path / "detection"
            if detection_path.exists():
                for model_dir in detection_path.iterdir():
                    if model_dir.is_dir():
                        # Check for direct weights/best.pt
                        weights_path = model_dir / "weights" / "best.pt"
                        if weights_path.exists():
                            available_models.append(str(weights_path))
                        
                        # Check for finetune subdirectories
                        for subdir in model_dir.iterdir():
                            if subdir.is_dir() and subdir.name.startswith("finetune"):
                                finetune_weights = subdir / "weights" / "best.pt"
                                if finetune_weights.exists():
                                    available_models.append(str(finetune_weights))
                            
        else:  # segmentation
            # Add pretrained segmentation models
            if pretrained_path.exists():
                for model_file in pretrained_path.glob("*-seg.pt"):
                    available_models.append(str(model_file))
            
            # Add existing segmentation models for further tuning (including finetune directories)
            seg_path = models_path / "segmentation"
            if seg_path.exists():
                for model_dir in seg_path.iterdir():
                    if model_dir.is_dir():
                        # Check for direct weights/best.pt
                        weights_path = model_dir / "weights" / "best.pt"
                        if weights_path.exists():
                            available_models.append(str(weights_path))
                        
                        # Check for finetune subdirectories
                        for subdir in model_dir.iterdir():
                            if subdir.is_dir() and subdir.name.startswith("finetune"):
                                finetune_weights = subdir / "weights" / "best.pt"
                                if finetune_weights.exists():
                                    available_models.append(str(finetune_weights))
        
        self.model_combo.addItems(available_models)
        
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
                    if any(keyword in name_lower for keyword in ["object_detection", "player", "disc", "detection"]):
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
        
        # Sort datasets to prefer v3i specifically, then newer versions
        def extract_version(path_str):
            """Extract version number from dataset path for sorting, preferring v3i."""
            # Look for patterns like v3i, v4i, etc.
            version_match = re.search(r'\.v(\d+)i', path_str)
            if version_match:
                version = int(version_match.group(1))
                # Give v3i highest priority
                if version == 3:
                    return 1000  # High priority for v3i
                return version
            return 0  # Default for paths without version
        
        # Sort by version number (v3i first, then descending order)
        available_datasets.sort(key=extract_version, reverse=True)
        
        self.data_combo.addItems(available_datasets)
        
        # Select the first (newest) dataset by default if available
        if available_datasets:
            self.data_combo.setCurrentIndex(0)
            self._on_data_changed(available_datasets[0])
        
    def _on_model_changed(self, model_path: str):
        """Handle model selection change."""
        self.current_model_path = model_path
        self._update_model_info()
        
    def _on_data_changed(self, data_path: str):
        """Handle dataset selection change."""
        self.current_data_path = data_path
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
            if hasattr(model, 'model'):
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
            with open(data_path, 'r') as f:
                data_config = yaml.safe_load(f)
                
            if 'names' in data_config:
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
                with open(file_path, 'r') as f:
                    self.training_config = yaml.safe_load(f)
                self._apply_config_to_ui()
                QMessageBox.information(self, "Success", "Configuration loaded successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load configuration: {e}")
                
    def _save_config(self):
        """Save current configuration to file."""
        # Update config with current UI values
        task_config = {
            'epochs': self.epochs_spin.value(),
            'patience': self.patience_spin.value(),
            'batch_size': self.batch_spin.value(),
            'learning_rate': self.lr_spin.value(),
            'imgsz': self.imgsz_spin.value(),
            'optimizer': self.optimizer_combo.currentText(),
            'momentum': self.momentum_spin.value(),
            'weight_decay': self.weight_decay_spin.value(),
            'workers': self.workers_spin.value(),
            'augment': self.augment_check.isChecked(),
            'mosaic': self.mosaic_spin.value(),
            'mixup': self.mixup_spin.value(),
            'copy_paste': self.copy_paste_spin.value(),
            'hsv_h': self.hsv_h_spin.value(),
            'hsv_s': self.hsv_s_spin.value(),
            'hsv_v': self.hsv_v_spin.value()
        }
        
        self.training_config[self.current_task] = task_config
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Training Configuration", "configs/training.yaml", "YAML files (*.yaml *.yml)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
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
        if model_name.endswith('.pt'):
            model_name = model_name[:-3]
            
        # Extract dataset name
        dataset_name = Path(self.current_data_path).parent.name
        
        # Create date prefix and find unique number
        date_prefix = datetime.now().strftime("%Y%m%d")
        base_name = f"{date_prefix}_{self.current_task}_{model_name}_{dataset_name}"
        
        # Determine output directory
        models_base = Path("data/models")
        if self.current_task == "detection":
            base_dir = models_base / "detection"
        else:
            base_dir = models_base / "segmentation"
            
        # Find unique directory name by incrementing number
        counter = 1
        while True:
            dir_name = f"{base_name}_{counter}"
            output_dir = base_dir / dir_name
            if not output_dir.exists():
                break
            counter += 1
        
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[TRAINING] Created output directory: {output_dir}")
        
        # Collect all training parameters from UI
        training_params = {
            'device': 0,  # Use first GPU if available, otherwise CPU
            'workers': self.workers_spin.value(),
            'imgsz': self.imgsz_spin.value(),
            'optimizer': self.optimizer_combo.currentText(),
            'momentum': self.momentum_spin.value(),
            'weight_decay': self.weight_decay_spin.value(),
            'augment': self.augment_check.isChecked(),
            'mosaic': self.mosaic_spin.value(),
            'mixup': self.mixup_spin.value(),
            'copy_paste': self.copy_paste_spin.value(),
            'hsv_h': self.hsv_h_spin.value(),
            'hsv_s': self.hsv_s_spin.value(),
            'hsv_v': self.hsv_v_spin.value()
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
            training_params=training_params
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
        
        # Start training
        self.training_thread.start()
        
        # Initialize timing for progress estimation
        self.training_start_time = time.time()
        self.last_epoch = 0
        
        # Set results directory for live monitoring
        # The training will create a timestamped subdirectory, so we need to find the latest one
        self.current_results_dir = output_dir
        self.results_widget.start_monitoring(str(output_dir))
        
    def _stop_training(self):
        """Stop current training."""
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.stop_training()
            self.training_thread.wait()
            
        # Stop results monitoring
        self.results_widget.stop_monitoring()
        self.current_results_dir = None
            
        self._reset_training_ui()
        self.progress_info_label.setText("Training stopped by user")
        
    def _on_training_progress(self, progress_value: int, status: str):
        """Handle training progress update."""
        # Set progress bar value (0-100)
        self.progress_bar.setValue(progress_value)
        
        # Update combined progress info with elapsed time on one line
        if self.training_start_time:
            elapsed_time = time.time() - self.training_start_time
            elapsed_str = self._format_elapsed_time(elapsed_time)
            
            # For time remaining estimates, use actual training time (excluding preprocessing)
            actual_training_elapsed = 0
            if hasattr(self, 'actual_training_start_time') and self.actual_training_start_time:
                actual_training_elapsed = time.time() - self.actual_training_start_time
            
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
                    it_match = re.search(r'([\d.]+)it/s', status)
                    if it_match:
                        iterations_per_sec = float(it_match.group(1))
                except (ValueError, IndexError):
                    pass
            
            # Calculate remaining time using multiple methods
            remaining_str = ""
            
            # Method 1: Use epoch-based estimation if we have valid epoch info and actual training time
            if current_epoch > 0 and total_epochs > 0 and actual_training_elapsed > 0 and current_epoch != self.last_epoch:
                avg_time_per_epoch = actual_training_elapsed / current_epoch
                remaining_epochs = total_epochs - current_epoch
                estimated_remaining = avg_time_per_epoch * remaining_epochs
                
                if estimated_remaining > 0:
                    remaining_str = self._format_elapsed_time(estimated_remaining)
                    self.last_epoch = current_epoch
            
            # Method 2: Use iterations per second for more accurate short-term estimates
            elif iterations_per_sec > 0 and current_epoch > 0 and total_epochs > 0:
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
                            total_remaining_batches = remaining_batches_epoch + (remaining_epochs * total_batches)
                            
                            # Estimate time based on iteration speed
                            estimated_remaining = total_remaining_batches / iterations_per_sec
                            remaining_str = self._format_elapsed_time(estimated_remaining)
                    except (ValueError, IndexError):
                        pass
            
            # Method 3: Use overall progress percentage as fallback
            elif progress_value > 5:  # Only if we have meaningful progress
                time_per_percent = elapsed_time / progress_value
                remaining_percent = 100 - progress_value
                estimated_remaining = time_per_percent * remaining_percent
                remaining_str = self._format_elapsed_time(estimated_remaining)
            
            # Build the display string
            if remaining_str:
                self.progress_info_label.setText(f"{status} • {elapsed_str} elapsed • {remaining_str} remaining")
            else:
                self.progress_info_label.setText(f"{status} • {elapsed_str} elapsed")
        else:
            # No timing info available yet
            self.progress_info_label.setText(status)
    
    def _on_raw_output(self, line: str):
        """Handle raw training output."""
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
        self.current_results_dir = None
        
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
        
    def _reset_training_ui(self):
        """Reset training UI elements."""
        self.start_training_btn.setEnabled(True)
        self.stop_training_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_info_label.setText("Ready to start training")
        self.raw_output_text.clear()
        
        # Reset timing variables
        self.training_start_time = None
        self.actual_training_start_time = None
        self.last_epoch = 0
        
        