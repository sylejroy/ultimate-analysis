"""Trained model files: where they are and what they were trained with.

Training runs are stored as data/models/<task>/<run>/<finetune>/weights/best.pt, with the
arguments Ultralytics trained them with in <finetune>/args.yaml.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from ..config.settings import get_setting
from ..constants import DEFAULT_PATHS, FALLBACK_DEFAULTS

PathLike = Union[str, Path]


def models_root() -> Path:
    """Folder that holds all model files."""
    return Path(get_setting("models.base_path", DEFAULT_PATHS["MODELS"]))


def default_model_path(kind: str) -> str:
    """Configured default weights for "player_detection", "disc_detection", or "segmentation"."""
    return get_setting(f"models.{kind}.default_model", FALLBACK_DEFAULTS[f"model_{kind}"])


def get_training_args(weights: PathLike) -> Dict[str, Any]:
    """Arguments the run these weights belong to was trained with ({} if unknown)."""
    weights = Path(weights)
    # Ultralytics writes args.yaml to the run folder, one level above weights/
    for folder in (weights.parent.parent, weights.parent):
        args_file = folder / "args.yaml"
        if args_file.exists():
            try:
                with open(args_file, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except (OSError, yaml.YAMLError):
                return {}
    return {}


def get_training_image_size(weights: PathLike, default: int = 640) -> int:
    """Image size the model was trained at, which is the size it should run at."""
    return int(get_training_args(weights).get("imgsz", default))


def get_class_names(weights: PathLike) -> Optional[List[str]]:
    """Class names of the dataset a model was trained on.

    Returns None when they cannot be determined (e.g. the dataset was removed).
    """
    try:
        with open(get_training_args(weights)["data"], "r", encoding="utf-8") as f:
            names = yaml.safe_load(f)["names"]
        return [str(name) for name in (names.values() if isinstance(names, dict) else names)]
    except Exception:
        return None


def find_detection_models(target_class: str) -> List[str]:
    """Finished detection runs that can detect a class, as paths relative to models_root().

    Runs whose classes cannot be determined are included.
    """
    root = models_root()
    models = []
    for weights in (root / "detection").rglob("best.pt"):
        class_names = get_class_names(weights)
        if class_names is None or target_class in class_names:
            models.append(str(weights.relative_to(root)))
    return sorted(models)


def find_training_runs(task: str) -> List[Path]:
    """results.csv of every training run of a task ("detection" or "segmentation"), newest first.

    Runs that were stopped early are included; their curves are worth comparing with too.
    """
    results = (models_root() / task).rglob("results.csv")
    return sorted(results, key=lambda path: path.stat().st_mtime, reverse=True)


def run_dataset_name(results: PathLike) -> str:
    """Name of the dataset a run was trained on ("" if unknown)."""
    data = get_training_args(Path(results).parent / "weights" / "best.pt").get("data")
    return Path(str(data)).parent.name if data else ""


def find_segmentation_models() -> List[str]:
    """Paths of the finished field segmentation runs."""
    return sorted(str(weights) for weights in (models_root() / "segmentation").rglob("best.pt"))


def model_display_name(weights: PathLike) -> str:
    """Short name of a training run for model dropdowns: "<run>" or "<run>/<finetune>"."""
    weights = Path(weights)
    finetune = weights.parent.parent
    run = finetune.parent
    # Most runs have a single finetune folder; name it only when there are several
    siblings = [d for d in run.iterdir() if d.is_dir()] if run.is_dir() else []
    return f"{run.name}/{finetune.name}" if len(siblings) > 1 else run.name
