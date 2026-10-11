#!/usr/bin/env python3
"""Label the tracks of a stretch by hand, starting from the tracker's output.

Every player box carries the tracker's ID. The work is to say which player each ID is,
and where an ID changes player. A player is named by a short label of your choice, the
same for the same player throughout the stretch: the jersey number with the team's
letter is handiest ("w7", "d23"); "x" is for people who are not players of the point
(beside the field, observers).

    python scripts/label_tracks.py hand_tracks

Keys:
    Right / Left        a frame on, a frame back        Shift: ten frames
    Space               play / pause
    click a box         select it (its tracker ID is shown)
    type a label, Enter name the selected ID from this frame on (what was named later
                        for the same ID stays, so a swap is named at the frame it happens)
    A                   apply the typed label to the whole ID instead
    x, Enter            the box is nobody's (left out of the truth)
    Delete              take back the naming of the selected ID at this frame
    Tab                 jump to the next frame where an ID begins
    S                   save (also saved on close)

The result, per stretch: labels.json  {"<tracker id>": [[from frame, "label"], ...]}
Boxes of an ID without a label are left out of the truth. Frames are the stretch's own,
counted from 0.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "raw" / "training_data"
SHOWN_WIDTH = 1600


def colour_of(label: str) -> Tuple[int, int, int]:
    if not label:
        return (160, 160, 160)
    if label == "x":
        return (60, 60, 60)
    seed = sum(ord(c) * (i + 1) for i, c in enumerate(label))
    hue = (seed * 47) % 180
    return tuple(
        int(v) for v in cv2.cvtColor(np.uint8([[[hue, 200, 240]]]), cv2.COLOR_HSV2BGR)[0, 0]
    )


class Stretch:
    """One stretch's frames, tracker boxes and labels."""

    def __init__(self, folder: Path):
        self.folder = folder
        data = json.loads((folder / "tracks.json").read_text())
        self.video, self.start, self.count = data["video"], data["start"], data["frames"]
        self.boxes: Dict[int, list] = {int(k): v for k, v in data["boxes"].items()}
        self.labels: Dict[int, List[Tuple[int, str]]] = {}
        if (folder / "labels.json").exists():
            self.labels = {
                int(k): [tuple(x) for x in v]
                for k, v in json.loads((folder / "labels.json").read_text()).items()
            }
        self.capture = cv2.VideoCapture(str(REPO / "data" / "raw" / "videos" / f"{self.video}.mp4"))
        self._cache: Dict[int, np.ndarray] = {}
        self._at = -1

    def frame(self, index: int) -> Optional[np.ndarray]:
        if index in self._cache:
            return self._cache[index]
        if self._at != index - 1:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.start + index)
        ok, frame = self.capture.read()
        self._at = index
        if not ok:
            return None
        if len(self._cache) > 400:
            self._cache.pop(next(iter(self._cache)))
        self._cache[index] = frame
        return frame

    def label_of(self, track_id: int, index: int) -> str:
        label = ""
        for since, name in sorted(self.labels.get(track_id, [])):
            if since <= index:
                label = name
        return label

    def name(self, track_id: int, index: int, label: str, whole: bool) -> None:
        if whole:
            self.labels[track_id] = [(0, label)]
            return
        kept = [(s, n) for s, n in self.labels.get(track_id, []) if s != index]
        self.labels[track_id] = sorted(kept + [(index, label)])

    def unname(self, track_id: int, index: int) -> None:
        kept = [(s, n) for s, n in self.labels.get(track_id, []) if s != index]
        if kept:
            self.labels[track_id] = kept
        else:
            self.labels.pop(track_id, None)

    def first_frames(self) -> Dict[int, int]:
        first = {}
        for index in sorted(self.boxes):
            for box in self.boxes[index]:
                first.setdefault(box[0], index)
        return first

    def unlabelled(self) -> List[int]:
        return [t for t in self.first_frames() if t not in self.labels]

    def save(self) -> None:
        (self.folder / "labels.json").write_text(
            json.dumps({str(k): v for k, v in sorted(self.labels.items())}, indent=0)
        )


class Window(QMainWindow):
    def __init__(self, folders: List[Path]):
        super().__init__()
        self.folders = folders
        self.stretch: Optional[Stretch] = None
        self.index = 0
        self.selected: Optional[int] = None
        self.scale = 1.0
        self.setWindowTitle("Label tracks")
        self.picture = QLabel()
        self.picture.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.picture.mousePressEvent = self._clicked
        self.entry = QLineEdit()
        self.entry.setPlaceholderText("label, then Enter (A: whole track)")
        self.entry.returnPressed.connect(lambda: self._name(whole=False))
        self.status = QLabel()
        self.players = QListWidget()
        self.players.setFixedWidth(230)
        self.stretches = QListWidget()
        self.stretches.setFixedWidth(230)
        for folder in folders:
            self.stretches.addItem(folder.name)
        self.stretches.currentRowChanged.connect(self._open)
        side = QVBoxLayout()
        side.addWidget(QLabel("Stretches"))
        side.addWidget(self.stretches)
        side.addWidget(QLabel("Players named (label: tracker IDs)"))
        side.addWidget(self.players)
        save = QPushButton("Save (S)")
        save.clicked.connect(self._save)
        side.addWidget(save)
        middle = QVBoxLayout()
        middle.addWidget(self.picture)
        middle.addWidget(self.entry)
        middle.addWidget(self.status)
        layout = QHBoxLayout()
        layout.addLayout(middle)
        layout.addLayout(side)
        root = QWidget()
        root.setLayout(layout)
        self.setCentralWidget(root)
        self.timer = QTimer()
        self.timer.timeout.connect(lambda: self._go(2))
        self.stretches.setCurrentRow(0)

    # ------------------------------------------------------------------ stretches

    def _open(self, row: int) -> None:
        if self.stretch is not None:
            self.stretch.save()
        self.stretch = Stretch(self.folders[row])
        self.index, self.selected = 0, None
        self._show()

    def _save(self) -> None:
        if self.stretch is not None:
            self.stretch.save()
            self.status.setText(f"Saved {self.stretch.folder / 'labels.json'}")

    def closeEvent(self, event) -> None:
        self._save()
        event.accept()

    # ------------------------------------------------------------------ showing

    def _show(self) -> None:
        stretch = self.stretch
        frame = stretch.frame(self.index)
        if frame is None:
            return
        drawn = frame.copy()
        for track_id, x1, y1, x2, y2, number, team in stretch.boxes.get(self.index, []):
            label = stretch.label_of(track_id, self.index)
            colour = colour_of(label)
            thick = 3 if track_id == self.selected else 1
            cv2.rectangle(drawn, (int(x1), int(y1)), (int(x2), int(y2)), colour, thick)
            text = f"{label or '?'} [{track_id}]" + (f" #{number}" if number.isdigit() else "")
            cv2.putText(
                drawn, text, (int(x1), int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3
            )
            cv2.putText(
                drawn, text, (int(x1), int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1
            )
        self.scale = SHOWN_WIDTH / drawn.shape[1]
        shown = cv2.resize(drawn, (SHOWN_WIDTH, int(drawn.shape[0] * self.scale)))
        shown = np.ascontiguousarray(shown)
        image = QImage(
            shown.data, shown.shape[1], shown.shape[0], 3 * shown.shape[1], QImage.Format_BGR888
        )
        self.picture.setPixmap(QPixmap.fromImage(image))
        unlabelled = stretch.unlabelled()
        self.status.setText(
            f"{stretch.folder.name}  frame {self.index} of {stretch.count}  "
            f"selected ID {self.selected if self.selected is not None else '-'}  "
            f"IDs without a label: {len(unlabelled)}"
            + (
                f" (next begins at frame {min(stretch.first_frames()[t] for t in unlabelled)})"
                if unlabelled
                else ""
            )
        )
        by_label: Dict[str, List[int]] = {}
        for track_id, names in stretch.labels.items():
            for _, name in names:
                by_label.setdefault(name, []).append(track_id)
        self.players.clear()
        for name in sorted(by_label):
            self.players.addItem(f"{name}: {sorted(set(by_label[name]))}")

    def _go(self, step: int) -> None:
        if self.stretch is None:
            return
        self.index = max(0, min(self.stretch.count - 1, self.index + step))
        self._show()

    # ------------------------------------------------------------------ labelling

    def _clicked(self, event) -> None:
        if self.stretch is None:
            return
        x, y = event.pos().x() / self.scale, event.pos().y() / self.scale
        hits = [
            (abs((x1 + x2) / 2 - x) + abs(y2 - y), track_id)
            for track_id, x1, y1, x2, y2, *_ in self.stretch.boxes.get(self.index, [])
            if x1 <= x <= x2 and y1 <= y <= y2
        ]
        if hits:
            self.selected = min(hits)[1]
            self.entry.setFocus()
        self._show()

    def _name(self, whole: bool) -> None:
        label = self.entry.text().strip()
        if self.stretch is None or self.selected is None or not label:
            return
        self.stretch.name(self.selected, self.index, label, whole)
        self.entry.clear()
        self._show()

    def keyPressEvent(self, event) -> None:
        key, shift = event.key(), bool(event.modifiers() & Qt.ShiftModifier)
        if key == Qt.Key_Right:
            self._go(10 if shift else 1)
        elif key == Qt.Key_Left:
            self._go(-10 if shift else -1)
        elif key == Qt.Key_Space:
            self.timer.stop() if self.timer.isActive() else self.timer.start(33)
        elif key == Qt.Key_Tab:
            if self.stretch is not None:
                later = [f for f in self.stretch.first_frames().values() if f > self.index]
                if later:
                    self.index = min(later)
                    self._show()
        elif key == Qt.Key_Delete:
            if self.stretch is not None and self.selected is not None:
                self.stretch.unname(self.selected, self.index)
                self._show()
        elif key == Qt.Key_A and self.entry.text().strip() and not self.entry.hasFocus():
            self._name(whole=True)
        elif key == Qt.Key_S and not self.entry.hasFocus():
            self._save()
        else:
            super().keyPressEvent(event)
            return
        event.accept()


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    root = DATA / sys.argv[1]
    folders = sorted(p for p in root.iterdir() if (p / "tracks.json").exists())
    if not folders:
        sys.exit(f"No prepared stretches in {root}: run scripts/prepare_track_labels.py first")
    app = QApplication(sys.argv)
    window = Window(folders)
    window.resize(SHOWN_WIDTH + 260, 1000)
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()
