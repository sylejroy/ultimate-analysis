#!/usr/bin/env python3
"""
Standalone training script for YOLO models.
This script is called by the GUI as a subprocess to train models without conflicts.
"""
import argparse
import json
import multiprocessing
import os
import sys
from datetime import datetime
from pathlib import Path

# Required for Windows multiprocessing
if __name__ == "__main__":
    multiprocessing.freeze_support()

    # Add src to path
    current_dir = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(current_dir))

    # Set environment variables to reduce conflicts
    os.environ["PYTHONPATH"] = str(current_dir)

    from ultralytics import YOLO

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train YOLO model")
    parser.add_argument("--config", required=True, help="Path to training config JSON file")
    args = parser.parse_args()

    # Load training configuration
    with open(args.config, "r") as f:
        config = json.load(f)

    # Extract configuration
    model_path = config["model_path"]
    data_path = config["data_path"]
    output_dir = config["output_dir"]

    # Load model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # Set up training arguments
    train_args = {
        "data": data_path,
        "epochs": config["epochs"],
        "patience": config["patience"],
        "batch": config["batch_size"],
        "lr0": config["learning_rate"],
        "project": output_dir,
        "name": f"finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "exist_ok": True,
        "verbose": True,
        "plots": True,
        "save": True,
        "device": config.get("device", 0),  # Use GPU if available
        "workers": config.get("workers", 0),  # Set to 0 to avoid multiprocessing issues on Windows
        "imgsz": config.get("imgsz", 640),
        "optimizer": config.get("optimizer", "SGD"),
        "momentum": config.get("momentum", 0.937),
        "weight_decay": config.get("weight_decay", 0.0005),
        "augment": config.get("augment", True),
        "cos_lr": config.get("cosine_lr", False),
        "mosaic": config.get("mosaic", 1.0),
        "mixup": config.get("mixup", 0.0),
        "copy_paste": config.get("copy_paste", 0.0),
        "hsv_h": config.get("hsv_h", 0.015),
        "hsv_s": config.get("hsv_s", 0.7),
        "hsv_v": config.get("hsv_v", 0.4),
    }

    print(f"Training arguments: {train_args}")

    # Start training
    results = model.train(**train_args)
    print(f"Training completed. Results directory: {results.save_dir}")

    # Write results path to file for the GUI to read
    results_file = config.get("results_file", "training_results.txt")
    with open(results_file, "w") as f:
        f.write(str(results.save_dir))
