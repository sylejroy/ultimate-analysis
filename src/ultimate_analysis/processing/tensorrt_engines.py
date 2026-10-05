"""TensorRT engines for the YOLO models.

A PyTorch YOLO forward pass spends most of its time on per-layer overhead rather than on
the GPU. A TensorRT engine runs the same weights as one compiled graph. An engine is built
for one model, one network input size, and one precision, on this GPU and driver, so
engines are built explicitly (`scripts/export_tensorrt.py`) and stored next to the weights.
At runtime a model uses its engine when a matching one exists and PyTorch otherwise.
"""

import math
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..config.settings import get_setting
from ..utils.logger import get_logger

logger = get_logger("TENSORRT")

# (weights file, network input shape, half precision) -> loaded engine, or None when there
# is no usable engine. Looked up once per combination; restart to pick up a new engine.
_engines: Dict[Tuple[str, Tuple[int, int], bool], Optional[Any]] = {}


def network_input_shape(
    frame_shape: Tuple[int, int], imgsz: int, stride: int = 32
) -> Tuple[int, int]:
    """Network input (height, width) Ultralytics uses for a frame with a PyTorch model.

    The frame is scaled so its longer side is `imgsz` and padded up to a multiple of the
    stride. An engine built for this shape receives exactly the same input.
    """
    frame_h, frame_w = frame_shape[:2]
    scale = min(imgsz / frame_h, imgsz / frame_w)
    scaled_h, scaled_w = round(frame_h * scale), round(frame_w * scale)
    return math.ceil(scaled_h / stride) * stride, math.ceil(scaled_w / stride) * stride


def _runs_as_engine(model: Any) -> bool:
    """Whether engines apply to a model.

    Engines are built for YOLO's letterboxed input and its output format. RT-DETR stretches
    frames to a square and decodes its output differently, so it keeps running in PyTorch.
    """
    return type(model).__name__ != "RTDETR"


def engine_path(weights: Path, input_shape: Tuple[int, int], half: bool) -> Path:
    """Where the engine for these weights, input shape, and precision is stored."""
    precision = "fp16" if half else "fp32"
    name = f"{weights.stem}.{input_shape[0]}x{input_shape[1]}.{precision}.engine"
    return weights.with_name(name)


def get_engine(
    model: Any, frame_shape: Tuple[int, ...], imgsz: int, half: bool
) -> Optional[Tuple[Any, Tuple[int, int]]]:
    """The engine to run instead of `model` on frames of this shape, if there is one.

    Args:
        model: Loaded PyTorch YOLO model
        frame_shape: Shape of the frames passed to predict
        imgsz: Image size the model runs at
        half: Whether the model runs in half precision

    Returns:
        (engine model, network input shape to pass as imgsz), or None to use `model`
    """
    if not get_setting("models.inference.tensorrt", True) or not _runs_as_engine(model):
        return None

    weights = getattr(model, "ckpt_path", None)
    if not isinstance(weights, (str, Path)):
        return None

    input_shape = network_input_shape(frame_shape[:2], imgsz)
    key = (str(weights), input_shape, half)
    if key not in _engines:
        _engines[key] = _load_engine(Path(weights), input_shape, half, getattr(model, "task", None))

    engine = _engines[key]
    return (engine, input_shape) if engine is not None else None


def _load_engine(
    weights: Path, input_shape: Tuple[int, int], half: bool, task: Optional[str]
) -> Optional[Any]:
    """Load the stored engine, or None if it is missing, outdated, or does not load."""
    path = engine_path(weights, input_shape, half)

    try:
        if not path.exists():
            return None
        if path.stat().st_mtime < weights.stat().st_mtime:
            logger.warning(f"Ignoring engine older than its weights: {path}")
            return None

        import numpy as np
        from ultralytics import YOLO

        engine = YOLO(str(path), task=task)
        # The first call deserializes the engine; do it now and prove that it runs
        engine.predict(
            np.zeros((*input_shape, 3), dtype=np.uint8), imgsz=input_shape, verbose=False
        )
        logger.info(f"Using TensorRT engine {path.name}")
        return engine

    except Exception as e:
        # Engines are tied to the GPU, driver, and TensorRT version they were built with
        logger.warning(f"TensorRT engine {path} is not usable, running PyTorch instead: {e}")
        return None


def export_engine(weights: Path, input_shape: Tuple[int, int], half: bool) -> Path:
    """Build the engine for these weights and store it next to them.

    The export runs on a temporary copy of the weights, so the intermediate ONNX file does
    not end up in the model folder.

    Raises:
        ValueError: The weights are not a YOLO model
    """
    from ultralytics import YOLO

    weights = Path(weights)
    target = engine_path(weights, input_shape, half)

    with tempfile.TemporaryDirectory() as work_dir:
        work_weights = Path(work_dir) / weights.name
        shutil.copy2(weights, work_weights)
        model = YOLO(str(work_weights))
        if not _runs_as_engine(model):
            raise ValueError(f"TensorRT engines are built for YOLO models only: {weights}")
        exported = model.export(
            format="engine", imgsz=list(input_shape), half=half, device=0, dynamic=False
        )
        shutil.move(str(exported), str(target))

    return target
