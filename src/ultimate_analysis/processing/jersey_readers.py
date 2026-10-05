"""Jersey number readers that can replace EasyOCR's recognizer.

Every reader takes the upper-body crops of the players and returns, per crop, a list of
(box, text, confidence) in the format EasyOCR produces, so the rest of the player ID
pipeline (digit filtering, validation, probabilistic tracking) is the same for all of them.

Measured on 1,016 labelled crops of 23 players from four games (see README):
- parseq:      PARSeq recognizer on regions found by EasyOCR's text detector
- florence:    Florence-2 vision-language model reading the whole crop
- yolo_digits: a YOLO model trained to detect the digits 0-9
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..config.settings import get_setting
from ..utils.logger import get_logger
from ..utils.model_files import get_training_image_size

logger = get_logger("PLAYER_ID")

# (box as four [x, y] corners in crop pixels, text, confidence)
Reading = Tuple[List[List[float]], str, float]

# Reader name -> label shown in the GUI
READER_LABELS = {
    "easyocr": "EasyOCR",
    "parseq": "PARSeq + text detector",
    "florence": "Florence-2 (vision-language model)",
    "yolo_digits": "YOLO digit detector",
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


class FlorenceReader:
    """Florence-2 vision-language model asked to read the text in the crop."""

    MODEL_NAME = "florence-community/Florence-2-base"
    BATCH_SIZE = 8

    def __init__(self):
        import torch
        from transformers import AutoProcessor, Florence2ForConditionalGeneration

        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(self.MODEL_NAME)
        self._model = (
            Florence2ForConditionalGeneration.from_pretrained(self.MODEL_NAME, dtype=torch.float16)
            .eval()
            .cuda()
        )
        self._special_tokens = torch.tensor(self._processor.tokenizer.all_special_ids).cuda()

    def read(self, crops: List[np.ndarray]) -> List[List[Reading]]:
        from PIL import Image

        torch = self._torch
        # The model answers for every crop, also when no number is visible; its own
        # certainty separates real reads from guesses.
        min_confidence = get_setting("models.player_id.florence.min_confidence", 0.7)

        readings: List[List[Reading]] = []
        for start in range(0, len(crops), self.BATCH_SIZE):
            chunk = crops[start : start + self.BATCH_SIZE]
            images = [Image.fromarray(np.ascontiguousarray(crop[..., ::-1])) for crop in chunk]
            inputs = self._processor(
                text=["<OCR>"] * len(images), images=images, return_tensors="pt"
            ).to("cuda", torch.float16)

            with torch.inference_mode():
                generated = self._model.generate(
                    **inputs,
                    max_new_tokens=6,
                    num_beams=1,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
                token_probabilities = (
                    self._model.compute_transition_scores(
                        generated.sequences, generated.scores, normalize_logits=True
                    )
                    .float()
                    .exp()
                )

            texts = self._processor.batch_decode(generated.sequences, skip_special_tokens=True)
            new_tokens = generated.sequences[:, -token_probabilities.shape[1] :]
            for crop, text, probabilities, tokens in zip(
                chunk, texts, token_probabilities, new_tokens
            ):
                is_text = ~torch.isin(tokens, self._special_tokens)
                confidence = float(probabilities[is_text].prod()) if is_text.any() else 0.0
                text = text.strip()
                if text.isdigit() and confidence >= min_confidence:
                    height, width = crop.shape[:2]
                    readings.append([(_box(0, 0, width, height), text, confidence)])
                else:
                    readings.append([])
        return readings


class YoloDigitReader:
    """YOLO model trained to detect the digits 0-9; the digits are read left to right."""

    MAX_DIGITS = 2

    def __init__(self, weights: Path):
        from ultralytics import YOLO

        self._model = YOLO(str(weights))
        self._imgsz = get_training_image_size(weights, default=160)

    def read(self, crops: List[np.ndarray]) -> List[List[Reading]]:
        min_confidence = get_setting("models.player_id.yolo_digits.min_confidence", 0.5)
        results = self._model.predict(
            crops, imgsz=self._imgsz, conf=min_confidence, agnostic_nms=True, verbose=False
        )

        readings: List[List[Reading]] = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            digits = [result.names[int(cls)] for cls in result.boxes.cls.cpu().numpy()]

            # Most confident digits, then left to right
            order = sorted(np.argsort(-confidences)[: self.MAX_DIGITS], key=lambda i: boxes[i][0])
            if not order:
                readings.append([])
                continue

            x1, y1 = boxes[order, 0].min(), boxes[order, 1].min()
            x2, y2 = boxes[order, 2].max(), boxes[order, 3].max()
            text = "".join(digits[i] for i in order)
            readings.append([(_box(x1, y1, x2, y2), text, float(confidences[order].min()))])
        return readings


def find_digit_model() -> Optional[Path]:
    """Weights of the digit detector: the configured file, else the newest digits run."""
    configured = get_setting("models.player_id.yolo_digits.model", "")
    if configured:
        path = Path(configured)
        return path if path.exists() else None

    models_path = Path(get_setting("models.base_path", "data/models")) / "detection"
    candidates = list(models_path.glob("*digits*/*/weights/best.pt"))
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


# Loaded readers by name; None marks a reader that could not be loaded
_readers: Dict[str, Optional[Any]] = {}


def get_reader(name: str, text_detector_factory: Callable[[], Any]) -> Optional[Any]:
    """The named reader, loaded on first use; None if it is not available.

    Args:
        name: "parseq", "florence", or "yolo_digits"
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
            elif name == "florence":
                _readers[name] = FlorenceReader()
            elif name == "yolo_digits":
                weights = find_digit_model()
                if weights is None:
                    raise RuntimeError(
                        "no digit detector has been trained (dataset digits.v1i.yolov8)"
                    )
                _readers[name] = YoloDigitReader(weights)
            else:
                raise ValueError(f"unknown jersey reader '{name}'")
            logger.info(f"Jersey reader loaded: {READER_LABELS.get(name, name)}")
        except Exception as e:
            logger.warning(f"Jersey reader '{name}' is not available, using EasyOCR instead: {e}")
            _readers[name] = None
    return _readers[name]
