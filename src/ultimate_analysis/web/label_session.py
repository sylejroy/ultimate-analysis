"""Disc labelling in small steps, as the phone page asks for them.

A task is one frame of a game video. The disc model suggests where the disc is; the person
confirms the suggestion, points at the disc, or says that none can be seen. Each answer
stores the frame in a disc-only dataset: with the disc's box, or without any box, which
teaches the model what is not a disc.

There is no web or model code here: frames and suggestions come from two functions the
caller supplies.
"""

import random
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..utils import label_files
from ..utils.label_files import LabelBox

CLASS_NAMES = ["disc"]
# Suggestions the model is sure of are mostly right and teach it little; most of them are
# passed over in favour of frames it is unsure about or finds nothing in
CONFIDENT = 0.6
KEEP_CONFIDENT = 0.3
REMEMBERED_TASKS = 8
MAX_TRIES = 30

# Fitting a box to the disc at a tapped spot
FIT_REACH = 32  # The disc is looked for this many pixels around the tap
FIT_MIN_CONTRAST = 25.0  # How much lighter than its surroundings (grey levels) a disc is
FIT_MAX_SIZE = 48  # Anything larger is not a disc but a shirt or a line
DEFAULT_DISC_SIZE = (18.0, 14.0)  # Used when no disc stands out at the tap

Box = Tuple[float, float, float, float]  # x1, y1, x2, y2 in frame pixels
# (video path, frame index) -> frame, or None if it cannot be read
ReadFrame = Callable[[str, int], Optional[np.ndarray]]
# frame -> [(box, confidence)], best first
FindDiscs = Callable[[np.ndarray], List[Tuple[Box, float]]]


class LabelSession:
    """Hands out frames to label and stores the answers."""

    def __init__(
        self,
        dataset_dir: Path,
        videos: Dict[str, int],
        read_frame: ReadFrame,
        find_discs: FindDiscs,
        seed: Optional[int] = None,
    ):
        """
        Args:
            dataset_dir: Dataset folder the labelled frames go to
            videos: Video path -> number of frames
            read_frame: Reads a frame of a video
            find_discs: Suggests discs on a frame
            seed: Fixes the order of the frames (for tests)
        """
        self.dataset_dir = Path(dataset_dir)
        self._videos = videos
        self._read_frame = read_frame
        self._find_discs = find_discs
        self._random = random.Random(seed)
        self._tasks: "OrderedDict[str, Tuple[str, int, np.ndarray]]" = OrderedDict()
        self._next_id = 0
        self._saved: List[str] = []  # Frame names stored in this session, for undo
        self._passed_over = set()  # Frames skipped in this session

    # ------------------------------------------------------------------ tasks

    def next_task(self) -> Optional[dict]:
        """A frame to label, or None if none could be found.

        Returns:
            {"task", "video", "frame", "width", "height", "box" (suggested, or None),
            "confidence", "labelled" (frames in the dataset)}
        """
        for _ in range(MAX_TRIES):
            video = self._random.choice(list(self._videos))
            index = self._random.randrange(max(1, self._videos[video]))
            name = label_files.frame_name(video, index)
            if name in self._passed_over or self._is_labelled(name):
                continue
            frame = self._read_frame(video, index)
            if frame is None:
                continue

            suggestions = self._find_discs(frame)
            box, confidence = suggestions[0] if suggestions else (None, 0.0)
            if confidence >= CONFIDENT and self._random.random() > KEEP_CONFIDENT:
                continue

            task_id = str(self._next_id)
            self._next_id += 1
            self._tasks[task_id] = (video, index, frame)
            while len(self._tasks) > REMEMBERED_TASKS:
                self._tasks.popitem(last=False)
            return {
                "task": task_id,
                "video": Path(video).stem,
                "frame": index,
                "width": frame.shape[1],
                "height": frame.shape[0],
                "box": [round(float(value), 1) for value in box] if box is not None else None,
                "confidence": round(float(confidence), 2),
                "labelled": self.labelled_count(),
            }
        return None

    def _is_labelled(self, name: str) -> bool:
        return (self.dataset_dir / "labels" / f"{name}.txt").exists()

    def labelled_count(self) -> int:
        return len(label_files.labelled_frames(self.dataset_dir))

    # ------------------------------------------------------------------ pictures

    def picture(
        self, task_id: str, x: float, y: float, width: float, height: float, output_width: int
    ) -> Optional[bytes]:
        """JPEG of a part of a task's frame, scaled to a width in pixels.

        The part is moved to lie inside the frame; `view` gives where it ends up.
        """
        if task_id not in self._tasks:
            return None
        frame = self._tasks[task_id][2]
        x, y, width, height = self.view(frame.shape[1], frame.shape[0], x, y, width, height)
        part = frame[y : y + height, x : x + width]
        output_height = max(1, round(output_width * height / width))
        # Enlarged pixels stay sharp: the disc is only a few of them
        interpolation = cv2.INTER_NEAREST if output_width > width else cv2.INTER_AREA
        part = cv2.resize(part, (output_width, output_height), interpolation=interpolation)
        ok, encoded = cv2.imencode(".jpg", part, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return encoded.tobytes() if ok else None

    @staticmethod
    def view(
        frame_width: int, frame_height: int, x: float, y: float, width: float, height: float
    ) -> Tuple[int, int, int, int]:
        """A wanted part of a frame (x, y, width, height), moved and cut to lie inside it."""
        width = int(min(max(width, 8), frame_width))
        height = int(min(max(height, 8), frame_height))
        x = int(min(max(x, 0), frame_width - width))
        y = int(min(max(y, 0), frame_height - height))
        return x, y, width, height

    def fit_box(self, task_id: str, x: float, y: float) -> Optional[List[float]]:
        """A box around the disc at a tapped spot of a task's frame.

        The disc is lighter than the grass around it; the box is put around the light
        patch at the tap. Seen from the side a disc is a thin line and from above a
        circle, so a box of one fixed shape rarely fits. Where no such patch is found (a disc
        in front of a white shirt), a box of typical size is returned instead.
        """
        if task_id not in self._tasks:
            return None
        frame = self._tasks[task_id][2]
        side = 2 * FIT_REACH
        left, top, width, height = self.view(
            frame.shape[1], frame.shape[0], x - FIT_REACH, y - FIT_REACH, side, side
        )
        patch = cv2.cvtColor(frame[top : top + height, left : left + width], cv2.COLOR_BGR2GRAY)
        patch = patch.astype(np.float32)

        # A disc is lighter than the grass around it; the surroundings are what the rim of
        # the patch shows. Dark things next to it (a shirt, a shadow) are thereby left out.
        rim = np.concatenate(
            [part.ravel() for part in (patch[:4], patch[-4:], patch[:, :4], patch[:, -4:])]
        )
        contrast = patch - float(np.median(rim))
        threshold = max(FIT_MIN_CONTRAST, 0.5 * float(contrast.max()))
        standing_out = (contrast > threshold).astype(np.uint8)

        count, labels, stats, _ = cv2.connectedComponentsWithStats(standing_out, connectivity=8)
        tap_x, tap_y = int(x - left), int(y - top)
        best, best_distance = None, 8.0  # A tap may miss the disc by a few pixels
        for label in range(1, count):
            ys, xs = np.nonzero(labels == label)
            distance = float(np.hypot(xs - tap_x, ys - tap_y).min())
            if distance < best_distance:
                best, best_distance = label, distance

        if best is not None:
            box_x, box_y, box_w, box_h, area = stats[best]
            if area >= 6 and box_w <= FIT_MAX_SIZE and box_h <= FIT_MAX_SIZE:
                # One pixel of margin: the edge of a disc is blurred into the grass
                return [
                    float(left + box_x - 1),
                    float(top + box_y - 1),
                    float(left + box_x + box_w + 1),
                    float(top + box_y + box_h + 1),
                ]
        half_w, half_h = DEFAULT_DISC_SIZE[0] / 2, DEFAULT_DISC_SIZE[1] / 2
        return [x - half_w, y - half_h, x + half_w, y + half_h]

    # ------------------------------------------------------------------ answers

    def save(self, task_id: str, box: Optional[Sequence[float]]) -> bool:
        """Store a task's frame with the disc's box, or with none (no disc can be seen)."""
        if task_id not in self._tasks:
            return False
        video, index, frame = self._tasks.pop(task_id)
        name = label_files.frame_name(video, index)
        boxes = [LabelBox(0, *map(float, box))] if box is not None else []
        label_files.save_frame(self.dataset_dir, name, frame, boxes, CLASS_NAMES)
        self._saved.append(name)
        return True

    def skip(self, task_id: str) -> bool:
        """Leave a task's frame out, e.g. when it cannot be told whether there is a disc."""
        if task_id not in self._tasks:
            return False
        video, index, _ = self._tasks.pop(task_id)
        self._passed_over.add(label_files.frame_name(video, index))
        return True

    def undo(self) -> Optional[str]:
        """Take the frame stored last in this session out of the dataset again."""
        if not self._saved:
            return None
        name = self._saved.pop()
        label_files.remove_frame(self.dataset_dir, name)
        self._passed_over.add(name)
        return name
