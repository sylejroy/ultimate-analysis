"""The disc as the disc detector and the heat map see it together.

Two networks look for the disc: the detector (processing/inference.py), which gives
boxes, and the heat map (processing/disc_heatmap.py), which says for every pixel how
likely a disc's middle is there. Each finds things that are no disc, but rarely the
same things: where both say disc, it is one far more surely than either says alone.

So a disc the detector is not sure enough of counts if the heat map has it too, and a
spot the heat map is very sure of counts where the detector has nothing. A disc the
detector is sure of counts as before, and for more where the heat map agrees: of
several, the one both see is taken.

The third to know of the disc is the holder network (processing/holder.py), which sees
it in a player's hands. It says nothing of where the disc is, and stands in for it in
frames where neither of these two finds it (models.holder.sure, in possession.py).
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..config.settings import get_setting

# A spot of the heat map is at a disc box if it is in the box widened by this many
# times its size and by this many pixels: a disc is a few pixels, and so is the
# distance between where the two put its middle
NEAR_BOX = 0.5
NEAR_PIXELS = 4.0
# The box of a disc that only the heat map found: this share of a player's height
# across, and no smaller than this (pixels)
HEAT_DISC_SIZE = 0.12
HEAT_DISC_PIXELS = 8.0

Peak = Tuple[float, float, float]  # x, y in the frame, and how sure (0 to 1)


def is_enabled() -> bool:
    """Whether the detector and the heat map are put together (models.disc_fusion.enabled)."""
    return bool(get_setting("models.disc_fusion.enabled", False))


def weak_discs_from() -> float:
    """The confidence from which the detector's discs are looked at, though it is not
    sure of them (0: only those it is sure of)."""
    return float(get_setting("models.disc_fusion.weak_from", 0.1)) if is_enabled() else 0.0


def _peak_at(box: Sequence[float], peaks: Sequence[Peak]) -> Optional[Peak]:
    """The surest spot of the heat map at a disc box, if there is one."""
    x1, y1, x2, y2 = box
    pad_x = NEAR_BOX * (x2 - x1) + NEAR_PIXELS
    pad_y = NEAR_BOX * (y2 - y1) + NEAR_PIXELS
    at = [p for p in peaks if x1 - pad_x <= p[0] <= x2 + pad_x and y1 - pad_y <= p[1] <= y2 + pad_y]
    return max(at, key=lambda peak: peak[2], default=None)


def fuse_discs(
    discs: Sequence[Dict[str, Any]],
    weak: Sequence[Dict[str, Any]],
    peaks: Sequence[Peak],
    height: float,
) -> List[Dict[str, Any]]:
    """The discs of a frame, from what the detector and the heat map say.

    Args:
        discs: The discs the detector is sure of
        weak: Those it is less sure of (from models.disc_fusion.weak_from on)
        peaks: The spots the heat map is surest of
        height: How tall a player is in the frame, in pixels

    Returns:
        The discs to go by, as detections. One that both saw is the surer for it; one
        that the heat map alone found has "source": "heat map".
    """
    agree = float(get_setting("models.disc_fusion.agree_heat", 0.1))
    alone = float(get_setting("models.disc_fusion.heat_alone", 0.5))
    fused: List[Dict[str, Any]] = []
    for detection, sure in [(d, True) for d in discs] + [(d, False) for d in weak]:
        peak = _peak_at(detection["bbox"], peaks)
        seen_too = peak is not None and peak[2] >= agree
        if not sure and not seen_too:
            continue
        taken = {key: value for key, value in detection.items() if key != "weak"}
        if seen_too:
            # Two that err apart: wrong only if both are
            taken["confidence"] = 1.0 - (1.0 - float(detection["confidence"])) * (1.0 - peak[2])
            taken["source"] = "detector and heat map"
        fused.append(taken)
    half = max(HEAT_DISC_PIXELS, HEAT_DISC_SIZE * height) / 2.0
    for x, y, sure in peaks:
        if alone <= 0 or sure < alone or any(_peak_at(d["bbox"], [(x, y, sure)]) for d in fused):
            continue
        fused.append(
            {
                "bbox": [int(x - half), int(y - half), int(x + half), int(y + half)],
                "confidence": float(sure),
                "class_id": 0,
                "class_name": "disc",
                "model_type": "disc_model",
                "source": "heat map",
            }
        )
    return fused
