#!/usr/bin/env python3
"""Score detection models on labelled images and time them on video frames.

For every model and each class it shares with the dataset, reports AP50 and the precision
and recall at the confidence threshold the app uses for that class. A detection counts as
correct when it overlaps a labelled box of its class with IoU >= 0.5. Models run the way
the app runs them: at their training image size, in half precision if configured, and
through their TensorRT engine when one exists.

    python scripts/benchmark_detectors.py                       # configured default models
    python scripts/benchmark_detectors.py path/to/best.pt ...   # these models
    python scripts/benchmark_detectors.py --splits test         # one split only
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from ultimate_analysis.config.settings import get_setting  # noqa: E402
from ultimate_analysis.processing.inference import (  # noqa: E402
    FP16_KWARGS,
    load_detection_model,
)
from ultimate_analysis.processing.tensorrt_engines import get_engine  # noqa: E402
from ultimate_analysis.utils.model_files import (  # noqa: E402
    default_model_path,
    model_display_name,
)
from ultimate_analysis.utils.video import find_video_files  # noqa: E402

TRAINING_DATA = REPO / "data" / "raw" / "training_data"
# Confidence threshold the app applies per class
CLASS_SETTINGS = {"player": "models.player_detection", "disc": "models.disc_detection"}
IOU_MATCH = 0.5
MIN_CONFIDENCE = 0.01  # AP needs the low-confidence detections as well
WARMUP_FRAMES = 10

Sample = Tuple[np.ndarray, Dict[str, np.ndarray]]


def load_samples(dataset: Path, splits: List[str]) -> List[Sample]:
    """[(image, {class name: labelled boxes as x1, y1, x2, y2 pixels})] of the splits."""
    names = yaml.safe_load((dataset / "data.yaml").read_text())["names"]
    names = list(names.values()) if isinstance(names, dict) else names

    samples = []
    for split in splits:
        for image_path in sorted((dataset / split / "images").glob("*")):
            image = cv2.imread(str(image_path))
            height, width = image.shape[:2]
            boxes: Dict[str, list] = {name: [] for name in names}
            label_path = dataset / split / "labels" / f"{image_path.stem}.txt"
            for line in label_path.read_text().splitlines() if label_path.exists() else []:
                class_id, x, y, w, h = (float(value) for value in line.split())
                boxes[names[int(class_id)]].append(
                    [
                        (x - w / 2) * width,
                        (y - h / 2) * height,
                        (x + w / 2) * width,
                        (y + h / 2) * height,
                    ]
                )
            samples.append((image, {name: np.array(b).reshape(-1, 4) for name, b in boxes.items()}))
    return samples


def predict(model: Any, imgsz: int, image: np.ndarray, confidence: float) -> Any:
    """Detections of one image, run the way the app runs the model."""
    precision = FP16_KWARGS if get_setting("models.inference.half_precision", False) else {}
    runtime_model, kwargs = model, {"imgsz": imgsz, **precision}
    engine = get_engine(model, image.shape, imgsz, half=bool(precision))
    if engine is not None:
        runtime_model, kwargs = engine[0], {"imgsz": engine[1]}
    iou = get_setting("models.player_detection.nms_threshold", 0.45)
    return runtime_model.predict(image, conf=confidence, iou=iou, verbose=False, **kwargs)[0].boxes


def match_detections(detections: np.ndarray, labelled: np.ndarray) -> np.ndarray:
    """Whether each detection (sorted by confidence) hits a labelled box not taken before."""
    correct = np.zeros(len(detections), dtype=bool)
    taken = np.zeros(len(labelled), dtype=bool)
    for index, (x1, y1, x2, y2) in enumerate(detections):
        if taken.all():
            break
        overlap_w = np.clip(
            np.minimum(x2, labelled[:, 2]) - np.maximum(x1, labelled[:, 0]), 0, None
        )
        overlap_h = np.clip(
            np.minimum(y2, labelled[:, 3]) - np.maximum(y1, labelled[:, 1]), 0, None
        )
        overlap = overlap_w * overlap_h
        areas = (labelled[:, 2] - labelled[:, 0]) * (labelled[:, 3] - labelled[:, 1])
        iou = overlap / ((x2 - x1) * (y2 - y1) + areas - overlap)
        iou[taken] = 0.0
        best = int(iou.argmax())
        if iou[best] >= IOU_MATCH:
            correct[index] = taken[best] = True
    return correct


def average_precision(confidences: np.ndarray, correct: np.ndarray, labelled_count: int) -> float:
    """Mean precision over 101 recall levels (the COCO measure) at one IoU threshold."""
    order = np.argsort(-confidences)
    hits = np.cumsum(correct[order])
    recall = np.concatenate(([0.0], hits / max(labelled_count, 1), [1.0]))
    precision = np.concatenate(([1.0], hits / np.arange(1, len(hits) + 1), [0.0]))
    # Precision at a recall level is the best precision reachable at that recall or beyond
    precision = np.flip(np.maximum.accumulate(np.flip(precision)))
    return float(np.interp(np.linspace(0, 1, 101), recall, precision).mean())


def score_model(model: Any, imgsz: int, samples: List[Sample]) -> Dict[str, Dict[str, float]]:
    """{class name: AP50, precision, recall, labelled count} for the classes the model shares."""
    class_names = {i: name for i, name in dict(model.names).items() if name in CLASS_SETTINGS}
    confidences: Dict[str, list] = {name: [] for name in class_names.values()}
    correct: Dict[str, list] = {name: [] for name in class_names.values()}
    labelled_count = {name: 0 for name in class_names.values()}

    for image, labelled in samples:
        boxes = predict(model, imgsz, image, MIN_CONFIDENCE)
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        for class_id, name in class_names.items():
            if name not in labelled:
                continue
            order = np.argsort(-conf[cls == class_id])
            confidences[name].append(conf[cls == class_id][order])
            correct[name].append(match_detections(xyxy[cls == class_id][order], labelled[name]))
            labelled_count[name] += len(labelled[name])

    scores = {}
    for name in class_names.values():
        if not labelled_count[name]:
            continue
        conf = np.concatenate(confidences[name])
        hit = np.concatenate(correct[name])
        kept = conf >= get_setting(f"{CLASS_SETTINGS[name]}.confidence_threshold", 0.5)
        scores[name] = {
            "ap50": average_precision(conf, hit, labelled_count[name]),
            "precision": hit[kept].sum() / max(kept.sum(), 1),
            "recall": hit[kept].sum() / labelled_count[name],
            "labelled": labelled_count[name],
        }
    return scores


def read_frames(video: Path, count: int) -> List[np.ndarray]:
    """Frames spread evenly over a video."""
    capture = cv2.VideoCapture(str(video))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for position in np.linspace(0, max(total - 1, 0), count).astype(int):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
        ok, frame = capture.read()
        if ok:
            frames.append(frame)
    capture.release()
    return frames


def time_per_frame(model: Any, imgsz: int, frames: List[np.ndarray]) -> float:
    """Milliseconds one frame takes, including pre- and postprocessing."""
    for frame in frames[:WARMUP_FRAMES]:
        predict(model, imgsz, frame, 0.3)
    start = time.perf_counter()
    for frame in frames:
        predict(model, imgsz, frame, 0.3)
    return (time.perf_counter() - start) / len(frames) * 1000


def default_video() -> Path:
    """The video the app opens at startup, otherwise the first one it lists."""
    videos = [Path(video) for video in find_video_files()]
    if not videos:
        sys.exit("No video found for timing; pass --video or --frames 0")
    wanted = get_setting("video.default_video", "")
    return next((video for video in videos if video.name == wanted), videos[0])


def main() -> None:
    training = yaml.safe_load((REPO / "configs" / "training.yaml").read_text())
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("weights", nargs="*", type=Path, help="default: the configured models")
    parser.add_argument("--dataset", default=training["detection"]["default_dataset"])
    parser.add_argument("--splits", nargs="+", default=["valid", "test"])
    parser.add_argument("--video", type=Path, help="default: the app's default video")
    parser.add_argument("--frames", type=int, default=100, help="frames to time; 0 = skip")
    args = parser.parse_args()

    weights = args.weights or list(
        dict.fromkeys(
            Path(default_model_path(kind)) for kind in ("player_detection", "disc_detection")
        )
    )
    samples = load_samples(TRAINING_DATA / args.dataset, args.splits)
    frames = read_frames(args.video or default_video(), args.frames) if args.frames else []
    print(f"{args.dataset} ({', '.join(args.splits)}): {len(samples)} images")
    if frames:
        print(f"Timing on {len(frames)} frames of {frames[0].shape[1]}x{frames[0].shape[0]} video")

    header = f"{'class':<8}{'labelled':>9}{'AP50':>8}{'precision':>11}{'recall':>8}"
    for path in weights:
        loaded = load_detection_model(str(path))
        if loaded is None:
            print(f"\n{path}: could not be loaded")
            continue
        model, imgsz = loaded
        title = f"{model_display_name(path)} (image size {imgsz})"
        if frames:
            title += f": {time_per_frame(model, imgsz, frames):.1f} ms per frame"
        print(f"\n{title}\n{header}")
        for name, score in score_model(model, imgsz, samples).items():
            print(
                f"{name:<8}{score['labelled']:>9}{score['ap50']:>8.3f}"
                f"{score['precision']:>11.3f}{score['recall']:>8.3f}"
            )


if __name__ == "__main__":
    main()
