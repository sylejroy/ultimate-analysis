"""Which way a player faces, from where a pose model finds their shoulders.

Jersey numbers are on the back. Seen from behind, a player's left shoulder is on the left
of the picture; seen from the front it is on the right; seen from the side the two
shoulders are nearly on top of each other. On the labelled jersey crops 42% show a player
from behind, and those hold 92% of the numbers the reader gets right, so the reader need
not look at the rest.
"""

from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from ..config.settings import get_setting
from ..utils.logger import get_logger

logger = get_logger("FACING")

BACK, FRONT, SIDE = "back", "front", "side"

INPUT_SIZE = (96, 160)  # Width and height every crop is brought to; small crops need no more
MIN_PERSON_CONFIDENCE = 0.1
MIN_KEYPOINT_CONFIDENCE = 0.3
# Shoulders closer together than this share of the upper body's height: seen from the side
MIN_SHOULDER_WIDTH = 0.25
LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP = 5, 6, 11, 12

_pose_model: Any = None
_pose_model_failed = False


def _load_pose_model() -> Any:
    """The pose model, or None if it cannot be loaded (asked for once only)."""
    global _pose_model, _pose_model_failed
    if _pose_model is None and not _pose_model_failed:
        path = (
            Path(get_setting("models.base_path", "data/models")) / "pretrained" / "yolo11n-pose.pt"
        )
        try:
            from ultralytics import YOLO

            if not path.exists():
                raise FileNotFoundError(path)
            _pose_model = YOLO(str(path))
        except Exception as e:
            _pose_model_failed = True
            logger.warning(f"No pose model, so players are read whichever way they face: {e}")
    return _pose_model


def facing_of(keypoints: np.ndarray, confidences: np.ndarray, crop_height: int) -> Optional[str]:
    """BACK, FRONT or SIDE from the keypoints of one person (COCO order), None if unclear."""
    if min(confidences[LEFT_SHOULDER], confidences[RIGHT_SHOULDER]) < MIN_KEYPOINT_CONFIDENCE:
        return None
    left, right = keypoints[LEFT_SHOULDER], keypoints[RIGHT_SHOULDER]
    if min(confidences[LEFT_HIP], confidences[RIGHT_HIP]) >= MIN_KEYPOINT_CONFIDENCE:
        hips = (keypoints[LEFT_HIP] + keypoints[RIGHT_HIP]) / 2
        upper_body = float(np.linalg.norm((left + right) / 2 - hips))
    else:
        upper_body = 0.3 * crop_height
    if abs(left[0] - right[0]) < MIN_SHOULDER_WIDTH * max(1.0, upper_body):
        return SIDE
    return BACK if left[0] < right[0] else FRONT


def facings(crops: List[np.ndarray]) -> List[Optional[str]]:
    """For each player crop BACK, FRONT, SIDE, or None where no person is made out.

    The network is run directly on the crops, all brought to one small size: going through
    Ultralytics' predict costs 11 ms a call whatever the number of crops, several times
    what the network itself takes. Each crop shows one player, so the keypoints of the
    place the network is surest of are that player's; no sorting out of double boxes.
    """
    model = _load_pose_model()
    if model is None or not crops:
        return [None] * len(crops)
    import cv2
    import torch

    width, height = INPUT_SIZE
    batch = np.stack(
        [cv2.resize(crop, INPUT_SIZE, interpolation=cv2.INTER_LINEAR) for crop in crops]
    )
    network = model.model
    parameter = next(network.parameters())
    if not parameter.is_cuda and torch.cuda.is_available():
        network.cuda().eval()
        parameter = next(network.parameters())
    with torch.inference_mode():
        # BGR pictures (N, H, W, 3) -> RGB (N, 3, H, W) between 0 and 1
        tensor = torch.from_numpy(batch).to(parameter.device).flip(-1).permute(0, 3, 1, 2)
        output = network(tensor.to(parameter.dtype) / 255.0)
        output = output[0] if isinstance(output, (tuple, list)) else output
        # (N, 4 box + 1 score + 17 * 3 keypoints, places)
        best = output[:, 4, :].argmax(dim=1)
        chosen = output[torch.arange(len(crops)), :, best].float().cpu().numpy()

    found: List[Optional[str]] = []
    for row in chosen:
        if row[4] < MIN_PERSON_CONFIDENCE:
            found.append(None)
            continue
        keypoints = row[5:].reshape(17, 3)
        found.append(facing_of(keypoints[:, :2], keypoints[:, 2], height))
    return found


def available() -> bool:
    """Whether the facing of players can be told at all (the pose model loads)."""
    return _load_pose_model() is not None
