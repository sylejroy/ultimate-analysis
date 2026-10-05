"""The camera-to-top-down homography: its eight parameters, files, and output canvas."""

import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import yaml

from ..config.settings import get_setting
from ..utils.logger import get_logger

logger = get_logger("HOMOGRAPHY")

# The 3x3 matrix row by row; H22 is fixed at 1
PARAMETER_NAMES = ("H00", "H01", "H02", "H10", "H11", "H12", "H20", "H21")
IDENTITY_PARAMETERS: Dict[str, float] = {
    "H00": 1.0,
    "H01": 0.0,
    "H02": 0.0,
    "H10": 0.0,
    "H11": 1.0,
    "H12": 0.0,
    "H20": 0.0,
    "H21": 0.0,
}


def default_parameters_file() -> Path:
    """File the default homography is loaded from at startup."""
    return Path(get_setting("homography.default_params_file", "configs/homography_params.yaml"))


def parameter_range(name: str) -> Tuple[float, float]:
    """Range the sliders offer for a parameter."""
    if name in ("H00", "H01", "H10", "H11"):  # scale and skew
        setting, fallback = "homography.slider_range_main", (-50.0, 50.0)
    elif name in ("H20", "H21"):  # perspective
        setting, fallback = "homography.slider_range_perspective", (-0.2, 0.2)
    else:  # translation
        return -10000.0, 10000.0

    configured = get_setting(setting, list(fallback))
    if isinstance(configured, list) and len(configured) == 2:
        return float(configured[0]), float(configured[1])
    return fallback


def parameters_to_matrix(parameters: Dict[str, float]) -> np.ndarray:
    """The 3x3 homography matrix for a set of parameters."""
    return np.array(
        [
            [parameters["H00"], parameters["H01"], parameters["H02"]],
            [parameters["H10"], parameters["H11"], parameters["H12"]],
            [parameters["H20"], parameters["H21"], 1.0],
        ],
        dtype=np.float32,
    )


def load_parameters(path: Union[str, Path]) -> Dict[str, float]:
    """Read homography parameters from a YAML file.

    Accepts files written by save_parameters and plain {H00: ..., ...} mappings.
    Parameters missing from the file are left out of the result.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    stored = data.get("homography_parameters", data)
    return {name: float(stored[name]) for name in PARAMETER_NAMES if name in stored}


def save_parameters(
    path: Union[str, Path],
    parameters: Dict[str, float],
    video_file: Optional[str] = None,
    frame_index: int = 0,
    description: Optional[str] = None,
) -> None:
    """Write homography parameters and where they were calibrated to a YAML file."""
    metadata = {"created_at": datetime.datetime.now().isoformat()}
    if description:
        metadata["description"] = description
    metadata.update(
        {
            "video_file": video_file,
            "frame_index": frame_index,
            "application": "Ultimate Analysis",
            "version": "1.0",
        }
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(
            {"homography_parameters": dict(parameters), "metadata": metadata},
            f,
            default_flow_style=False,
            sort_keys=False,
        )


def load_default_matrix() -> Optional[np.ndarray]:
    """The default homography matrix, or None if there is no complete default file."""
    path = default_parameters_file()
    if not path.exists():
        logger.info(f"No default homography file: {path}")
        return None
    try:
        return parameters_to_matrix(load_parameters(path))
    except Exception as e:
        logger.error(f"Error loading homography parameters from {path}: {e}")
        return None


def output_canvas_size(frame_width: int, frame_height: int) -> Tuple[int, int]:
    """Size (width, height) of the top-down canvas for a frame size.

    The canvas has homography.output_aspect_ratio (height:width) and
    homography.buffer_factor times the area of the frame.
    """
    buffer_factor = get_setting("homography.buffer_factor", 2.5)
    aspect_ratio = get_setting("homography.output_aspect_ratio", 3.0)
    target_area = int(frame_width * frame_height * buffer_factor)

    if aspect_ratio >= 1.0:
        # Taller than wide: height = aspect_ratio * width
        width = int(np.sqrt(target_area / aspect_ratio))
        return width, int(width * aspect_ratio)

    height = int(np.sqrt(target_area * aspect_ratio))
    return int(height / aspect_ratio), height
