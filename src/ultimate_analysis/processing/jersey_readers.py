"""A jersey number reader that can replace EasyOCR's recognizer.

Every reader takes the upper-body crops of the players and returns, per crop, a list of
(box, text, confidence) in the format EasyOCR produces, so the rest of the player ID
pipeline (digit filtering, validation, probabilistic tracking) is the same for all of them.

- parseq: PARSeq recognizer on regions found by EasyOCR's text detector. On 1,016
  labelled crops of 23 players it identified 17 players and none wrongly; EasyOCR's own
  recognizer 11, four of them wrongly. EasyOCR stays as the fallback where PARSeq cannot
  be loaded (it comes through torch.hub). Florence-2 and a YOLO digit detector were
  tried as well and are gone.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..utils.logger import get_logger

logger = get_logger("PLAYER_ID")

# (box as four [x, y] corners in crop pixels, text, confidence)
Reading = Tuple[List[List[float]], str, float]

# Reader name -> label shown in the GUI
READER_LABELS = {
    "easyocr": "EasyOCR",
    "parseq": "PARSeq + text detector",
}


def _box(x1: float, y1: float, x2: float, y2: float) -> List[List[float]]:
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


class ParseqReader:
    """PARSeq scene-text recognizer on the text regions EasyOCR's detector finds.

    PARSeq reads one line of text and has no detector of its own; given a whole upper
    body it reads nothing useful. EasyOCR's detector (CRAFT) supplies the number's region.
    """

    UPSCALE = 2.0

    def __init__(self, text_detector: Any):
        import torch
        from torchvision import transforms

        self._torch = torch
        self._text_detector = text_detector
        self._model = (
            torch.hub.load(
                "baudm/parseq", "parseq", pretrained=True, trust_repo=True, skip_validation=True
            )
            .eval()
            .cuda()
        )
        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(
                    self._model.hparams.img_size, transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5),
            ]
        )

    def read(self, crops: List[np.ndarray]) -> List[List[Reading]]:
        regions: List[np.ndarray] = []
        owners: List[Tuple[int, List[List[float]]]] = []  # (crop index, box in crop pixels)

        for index, crop in enumerate(crops):
            enlarged = cv2.resize(
                crop, None, fx=self.UPSCALE, fy=self.UPSCALE, interpolation=cv2.INTER_CUBIC
            )
            horizontal, _ = self._text_detector.detect(
                enlarged,
                text_threshold=0.4,
                low_text=0.3,
                link_threshold=0.2,
                canvas_size=1280,
                mag_ratio=2.0,
            )
            for x1, x2, y1, y2 in horizontal[0] if horizontal else []:
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = int(x2), int(y2)
                if x2 - x1 >= 6 and y2 - y1 >= 6:
                    regions.append(enlarged[y1:y2, x1:x2])
                    scale = self.UPSCALE
                    owners.append((index, _box(x1 / scale, y1 / scale, x2 / scale, y2 / scale)))

        readings: List[List[Reading]] = [[] for _ in crops]
        if not regions:
            return readings

        torch = self._torch
        batch = torch.stack(
            [self._transform(np.ascontiguousarray(region[..., ::-1])) for region in regions]
        ).cuda()
        with torch.inference_mode():
            texts, confidences = self._model.tokenizer.decode(self._model(batch).softmax(-1))

        for (index, box), text, confidence in zip(owners, texts, confidences):
            # Team names and sponsor text are detected as well; only pure numbers count
            if text.isdigit():
                readings[index].append((box, text, float(confidence.prod())))
        return readings


# Loaded readers by name; None marks a reader that could not be loaded
_readers: Dict[str, Optional[Any]] = {}


def get_reader(name: str, text_detector_factory: Callable[[], Any]) -> Optional[Any]:
    """The named reader, loaded on first use; None if it is not available.

    Args:
        name: "parseq"
        text_detector_factory: Returns the initialized EasyOCR reader (PARSeq uses its
            text detector)
    """
    if name not in _readers:
        try:
            if name == "parseq":
                detector = text_detector_factory()
                if detector is None:
                    raise RuntimeError("EasyOCR's text detector is not available")
                _readers[name] = ParseqReader(detector)
            else:
                raise ValueError(f"unknown jersey reader '{name}'")
            logger.info(f"Jersey reader loaded: {READER_LABELS.get(name, name)}")
        except Exception as e:
            logger.warning(f"Jersey reader '{name}' is not available, using EasyOCR instead: {e}")
            _readers[name] = None
    return _readers[name]
