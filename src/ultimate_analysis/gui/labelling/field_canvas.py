"""A frame with a drawing of the field on it that is pulled into place with the mouse."""

from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QPolygonF
from PyQt5.QtWidgets import QWidget

from ...utils.field_camera import focal_of, move_camera
from ...utils.field_label_files import FieldLabel
from ...utils.field_template import FieldFit, FieldTemplate, field_segment_in_image
from ..widgets.images import frame_to_pixmap

FIELD_DRAWING = QColor(0, 200, 255)
MODEL_LINES = QColor(255, 255, 255, 110)  # What the field model sees, faint
PAINTED_LINES = (255, 235, 60, 150)  # The painted lines found in the picture: R, G, B, alpha
SAVED = QColor(0, 220, 90)
HOVERED = QColor(255, 220, 0)
HANDLE_RADIUS = 6  # Of the dots on the corners, in screen pixels
REACH = 12  # How close the mouse must be to a corner dot to take it
MAX_ZOOM = 16.0
# Zoomed out further than the whole frame, the corners that lie outside the picture show
MIN_ZOOM = 0.3
# A place counts as seen up to this many frame sizes from the picture: just in front of the
# camera it lies so far out that it is of no use to hold on to
MAX_REACH = 50.0
# A label holds the lines up to this many frame sizes outside the picture
LABEL_MARGIN = 2.0
# The dots that can be dragged: the outer corners and where the goal lines meet the sidelines
HANDLES = (
    "far_back_left",
    "far_back_right",
    "far_goal_left",
    "far_goal_right",
    "near_goal_left",
    "near_goal_right",
    "near_back_left",
    "near_back_right",
)


def homography(places: np.ndarray, pixels: np.ndarray) -> Optional[np.ndarray]:
    """The mapping that takes four places on the field to four pixels, or None."""
    rows = []
    for (x, y), (u, v) in zip(places, pixels):
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, -u])
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, -v])
    try:
        _, sizes, rest = np.linalg.svd(np.array(rows, dtype=np.float64))
    except np.linalg.LinAlgError:
        return None
    if sizes[-1] < 1e-12 * sizes[0]:
        return None  # Three of the four in a row
    return rest[-1].reshape(3, 3)


class FieldCanvas(QWidget):
    """Shows a frame with the field drawn over it; the drawing is pulled onto the real field.

    The drawing is the whole field in perspective. It is one object: whatever is done to
    one part, the rest follows so that it stays a view of a flat field. The near end of the
    field may lie behind the camera (a drone above the end zone); it is then not drawn.

    - Drag a corner dot: move that corner. Only corners placed in this frame stay where
      they were put (drawn filled); the rest of the field follows as a camera would see
      it. From the fourth corner on, the three placed last stay
    - Mouse wheel: zoom at the mouse, also out past the frame to reach corners outside it
    - Drag anywhere else: move the picture
    """

    label_changed = pyqtSignal()
    hovered_changed = pyqtSignal(str)  # Name of the line under the mouse, or ""

    def __init__(self, template: FieldTemplate):
        super().__init__()
        self.template = template
        # Field place -> frame pixel, scaled so that what lies in front of the camera has a
        # positive third coordinate
        self.mapping: Optional[np.ndarray] = None
        self.unconfirmed = False  # The drawing has not been saved for this frame
        self.hovered = ""

        self._pixmap: Optional[QPixmap] = None
        self._frame_size = (1, 1)  # width, height
        self._zoom = 1.0
        self._view_center: Optional[Tuple[float, float]] = None
        self._drag: Optional[str] = None  # The corner dot being dragged
        # Corner dots in the order they were last dragged, most recent first
        self._recent: List[str] = []
        # Where each of them was put, in frame pixels
        self._placed: Dict[str, Tuple[float, float]] = {}
        # Focal length of the video's camera in pixels, if known: the field then follows
        # a dragged corner the way that camera would see it
        self.focal: Optional[float] = None
        # Lines the field model sees in the frame, each two pixels; shown faintly
        self.model_lines: List[np.ndarray] = []
        self._frame_focal: Optional[float] = None
        self._painted: Optional[QPixmap] = None  # The painted lines found, as an overlay
        self._pan_from: Optional[QPointF] = None

        self.setMinimumSize(640, 360)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

    # ------------------------------------------------------------------ content

    def set_frame(
        self, frame: np.ndarray, mapping: Optional[np.ndarray], unconfirmed: bool
    ) -> None:
        """Show a frame with the field drawn by the given mapping (None: a first guess)."""
        size = (frame.shape[1], frame.shape[0])
        if size != self._frame_size:
            self._frame_size = size
            self.reset_view()
        self._pixmap = frame_to_pixmap(frame)
        self.mapping = None
        if mapping is not None:
            self.mapping = np.asarray(mapping, dtype=np.float64)
        else:
            self.place(self._outer_corners(), self.first_guess())
        self.unconfirmed = unconfirmed
        self._drag = None
        # No corner has been put on its place in this frame yet
        self._recent = []
        self._placed = {}
        # The video's focal length, else the one the field as first shown implies
        self._frame_focal = self.focal or focal_of(self.mapping, self._frame_size)
        self.update()

    def set_painted_lines(self, mask: Optional[np.ndarray]) -> None:
        """Show where painted lines were found in the frame (a mask of its size), or not."""
        if mask is None:
            self._painted = None
        else:
            height, width = mask.shape
            overlay = np.zeros((height, width, 4), dtype=np.uint8)
            overlay[mask > 0] = PAINTED_LINES
            image = QImage(overlay.data, width, height, 4 * width, QImage.Format_RGBA8888)
            self._painted = QPixmap.fromImage(image.copy())
        self.update()

    def first_guess(self) -> np.ndarray:
        """Where the outer corners are in a typical view from behind an end zone."""
        width, height = self._frame_size
        return np.array(
            [
                [-0.55 * width, 1.45 * height],
                [1.55 * width, 1.45 * height],
                [0.64 * width, 0.17 * height],
                [0.36 * width, 0.17 * height],
            ]
        )

    def _held(self, name: str, seen: Dict[str, Tuple[float, float]]) -> List[str]:
        """The corners placed in this frame that stay where they are while this one is dragged.

        At most the three placed last, and never three in a row with the dragged one or
        each other: three corners on one sideline say nothing about the other side.
        """
        places = self.template.points

        def in_a_row(first: str, second: str, third: str) -> bool:
            (x1, y1), (x2, y2), (x3, y3) = places[first], places[second], places[third]
            return abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)) < 1e-9

        held: List[str] = []
        for candidate in self._recent:
            if candidate == name or candidate in held or candidate not in seen:
                continue
            chosen = [name, *held]
            if any(
                in_a_row(chosen[i], chosen[j], candidate)
                for i in range(len(chosen))
                for j in range(i + 1, len(chosen))
            ):
                continue
            held.append(candidate)
            if len(held) == 3:
                break
        return held

    def _handle_pixels(self, fit: FieldFit) -> Dict[str, np.ndarray]:
        """Name -> frame pixel of each corner dot that is in front of the camera."""
        found = {}
        for name in HANDLES:
            pixel = self._in_image(fit, np.array([self.template.points[name]]))
            if pixel is not None:
                found[name] = pixel[0]
        return found

    def _drag_corner(self, name: str, pixel: Tuple[float, float]) -> None:
        fit = self.fit()
        if fit is None:
            return
        self._placed[name] = (float(pixel[0]), float(pixel[1]))
        held = self._held(name, self._placed)
        if len(held) == 3:
            # Four corners fix the view by themselves
            names = [name, *held]
            self.place(
                np.array([self.template.points[other] for other in names]),
                np.array([self._placed[other] for other in names]),
            )
            return
        # Fewer leave it open: the field follows as the camera would see it
        placed = {other: self._placed[other] for other in self._recent if other in self._placed}
        mapping = move_camera(
            self.template, self.mapping, placed, self._frame_size, self._frame_focal
        )
        if mapping is not None:
            self.mapping = mapping
            self.update()
            self.label_changed.emit()

    def reset_view(self) -> None:
        """Show the whole frame."""
        self._zoom = 1.0
        self._view_center = None
        self.update()

    def label(self) -> FieldLabel:
        """The drawing as a label: what of the field lies in or near the picture.

        Each line by two pixels on it, as far apart as possible, and each corner dot.
        """
        fit = self.fit()
        width, height = self._frame_size

        def near_picture(pixel: np.ndarray) -> bool:
            return bool(
                -LABEL_MARGIN * width <= pixel[0] <= (1 + LABEL_MARGIN) * width
                and -LABEL_MARGIN * height <= pixel[1] <= (1 + LABEL_MARGIN) * height
            )

        label = FieldLabel()
        if fit is None:
            return label
        for name, (start, end) in self.template.lines.items():
            seen = [
                pixel
                for part in field_segment_in_image(fit, start, end)
                for pixel in part
                if near_picture(pixel)
            ]
            if len(seen) >= 2:
                label.lines[name] = [tuple(map(float, seen[0])), tuple(map(float, seen[-1]))]
        for name, pixel in self._handle_pixels(fit).items():
            if near_picture(pixel):
                label.points[name] = tuple(map(float, pixel))
        return label

    # ------------------------------------------------------------------ the field

    def _outer_corners(self) -> np.ndarray:
        """The outer corners in field coordinates: near left, near right, far right, far left."""
        w, length = self.template.width, self.template.length
        return np.array([[0.0, 0.0], [w, 0.0], [w, length], [0.0, length]])

    def fit(self) -> Optional[FieldFit]:
        """The mapping between field and picture as the drawing has it."""
        if self.mapping is None:
            return None
        return FieldFit(
            image_to_field=np.linalg.inv(self.mapping),
            field_to_image=self.mapping,
            statements=8,
            error=0.0,
            worst=("", 0.0),
            front_sign=1.0,
        )

    def _in_image(self, fit: FieldFit, places: np.ndarray) -> Optional[np.ndarray]:
        """Field places -> frame pixels, or None if one of them is not in front of the camera."""
        mapped = np.column_stack([places, np.ones(len(places))]) @ fit.field_to_image.T
        scale = np.abs(fit.field_to_image[2]).max()
        if np.any(mapped[:, 2] <= 1e-12 * scale):
            return None
        pixels = mapped[:, :2] / mapped[:, 2:3]
        if np.abs(pixels).max() > MAX_REACH * max(self._frame_size):
            return None
        return pixels

    def place(self, field_places: np.ndarray, pixels: np.ndarray) -> bool:
        """Lay the field so that four of its places lie on four pixels; False if that is no view."""
        mapping = homography(np.asarray(field_places, float), np.asarray(pixels, float))
        if mapping is None:
            return False
        depth = np.column_stack([field_places, np.ones(4)]) @ mapping[2]
        # All four in front of the camera
        if np.any(depth == 0) or len(set(np.sign(depth))) != 1:
            return False
        mapping = mapping * np.sign(depth[0])
        turn = np.linalg.det(mapping)
        if turn == 0 or not np.all(np.isfinite(mapping)):
            return False
        # Not the field seen from below
        if self.mapping is not None and np.sign(turn) != np.sign(np.linalg.det(self.mapping)):
            return False
        self.mapping = mapping
        self.update()
        self.label_changed.emit()
        return True

    # ------------------------------------------------------------------ view

    def _scale(self) -> float:
        """Screen pixels per frame pixel."""
        width, height = self._frame_size
        return min(self.width() / width, self.height() / height) * self._zoom

    def _origin(self) -> Tuple[float, float]:
        """Screen position of the frame's top-left corner."""
        width, height = self._frame_size
        center_x, center_y = self._view_center or (width / 2, height / 2)
        scale = self._scale()
        return self.width() / 2 - center_x * scale, self.height() / 2 - center_y * scale

    def to_frame(self, position: QPointF) -> Tuple[float, float]:
        origin_x, origin_y = self._origin()
        scale = self._scale()
        return (position.x() - origin_x) / scale, (position.y() - origin_y) / scale

    def to_screen(self, x: float, y: float) -> QPointF:
        origin_x, origin_y = self._origin()
        scale = self._scale()
        return QPointF(origin_x + x * scale, origin_y + y * scale)

    # ------------------------------------------------------------------ mouse

    def _lines_on_screen(self, fit: FieldFit) -> Dict[str, List[np.ndarray]]:
        """Name -> the line as drawn, in frame pixels (cut where it leaves the camera's view)."""
        return {
            name: field_segment_in_image(fit, start, end)
            for name, (start, end) in self.template.lines.items()
        }

    def _under_mouse(self, position: QPointF) -> Tuple[str, str]:
        """What is at a screen position: ("corner" | "line" | "", name)."""
        fit = self.fit()
        if fit is None:
            return "", ""
        near = [
            ((self.to_screen(*pixel) - position).manhattanLength(), name)
            for name, pixel in self._handle_pixels(fit).items()
        ]
        if near and min(near)[0] <= REACH:
            return "corner", min(near)[1]
        x, y = self.to_frame(position)
        best = (REACH / self._scale(), "")
        for name, parts in self._lines_on_screen(fit).items():
            for part in parts:
                start, end = part[:-1], part[1:]
                along = end - start
                share = np.clip(
                    ((x - start[:, 0]) * along[:, 0] + (y - start[:, 1]) * along[:, 1])
                    / np.maximum((along**2).sum(axis=1), 1e-12),
                    0.0,
                    1.0,
                )
                nearest = start + share[:, None] * along
                distance = float(np.hypot(nearest[:, 0] - x, nearest[:, 1] - y).min())
                if distance < best[0]:
                    best = (distance, name)
        return ("line", best[1]) if best[1] else ("", "")

    def _set_hovered(self, kind: str, name: str) -> None:
        """Name the line under the mouse; a hand shows over a dot that can be dragged."""
        hovered = name if kind == "line" else ""
        self.setCursor(Qt.PointingHandCursor if kind == "corner" else Qt.ArrowCursor)
        if hovered != self.hovered:
            self.hovered = hovered
            self.hovered_changed.emit(hovered)
            self.update()

    def wheelEvent(self, event):
        before = self.to_frame(event.pos())
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self._zoom = min(MAX_ZOOM, max(MIN_ZOOM, self._zoom * factor))
        if abs(self._zoom - 1.0) < 1e-6:
            self._zoom = 1.0
            self._view_center = None
        else:
            # Keep the frame point under the mouse where it is
            scale = self._scale()
            self._view_center = (
                before[0] - (event.pos().x() - self.width() / 2) / scale,
                before[1] - (event.pos().y() - self.height() / 2) / scale,
            )
        self.update()

    def mousePressEvent(self, event):
        self.setFocus()
        position = QPointF(event.pos())
        kind, name = self._under_mouse(position)
        if event.button() == Qt.LeftButton and kind == "corner":
            self._drag = name
            # Held from now on, when another corner is dragged
            self._recent = [name, *(other for other in self._recent if other != name)][:4]
            self.update()
        else:
            self._pan_from = position
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        position = QPointF(event.pos())
        if self._pan_from is not None:
            scale = self._scale()
            width, height = self._frame_size
            center_x, center_y = self._view_center or (width / 2, height / 2)
            delta = position - self._pan_from
            self._view_center = (center_x - delta.x() / scale, center_y - delta.y() / scale)
            self._pan_from = position
            self.update()
        elif self._drag is not None:
            self._drag_corner(self._drag, self.to_frame(position))
        else:
            self._set_hovered(*self._under_mouse(position))

    def mouseReleaseEvent(self, event):
        self._pan_from = None
        self._drag = None
        self._set_hovered(*self._under_mouse(QPointF(event.pos())))

    # ------------------------------------------------------------------ drawing

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(26, 26, 26))
        if self._pixmap is None:
            return

        width, height = self._frame_size
        scale = self._scale()
        origin_x, origin_y = self._origin()
        painter.setRenderHint(QPainter.SmoothPixmapTransform, scale < 2)
        painter.drawPixmap(
            QRectF(origin_x, origin_y, width * scale, height * scale),
            self._pixmap,
            QRectF(0, 0, width, height),
        )
        if self._painted is not None:
            painter.drawPixmap(
                QRectF(origin_x, origin_y, width * scale, height * scale),
                self._painted,
                QRectF(0, 0, width, height),
            )
        fit = self.fit()
        if fit is None:
            return
        painter.setRenderHint(QPainter.Antialiasing)

        painter.setPen(QPen(MODEL_LINES, 1.5))
        for line in self.model_lines:
            painter.drawLine(self.to_screen(*line[0]), self.to_screen(*line[1]))

        colour = FIELD_DRAWING if self.unconfirmed else SAVED
        for name, parts in self._lines_on_screen(fit).items():
            pen = QPen(
                HOVERED if name == self.hovered else colour, 3 if name == self.hovered else 2
            )
            if self.unconfirmed and name != self.hovered:
                pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            for part in parts:
                painter.drawPolyline(QPolygonF([self.to_screen(x, y) for x, y in part]))
        # The marks: a small cross each, lying flat on the field
        painter.setPen(QPen(colour, 1.5))
        size = self.template.width / 40.0
        for x, y in self.template.points.values():
            for start, end in (((x - size, y), (x + size, y)), ((x, y - size), (x, y + size))):
                for part in field_segment_in_image(fit, start, end, pieces=4):
                    painter.drawPolyline(QPolygonF([self.to_screen(u, v) for u, v in part]))
        # The corner dots: filled where they were placed in this frame, and so stay put
        dots = self._handle_pixels(fit)
        staying = set(self._recent)
        for name, pixel in dots.items():
            painter.setPen(QPen(colour, 2))
            painter.setBrush(colour if name in staying else Qt.NoBrush)
            radius = HANDLE_RADIUS if name in staying else HANDLE_RADIUS - 1
            painter.drawEllipse(self.to_screen(*pixel), radius, radius)
        painter.setBrush(Qt.NoBrush)
