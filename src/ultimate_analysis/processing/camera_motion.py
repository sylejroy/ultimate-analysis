"""Camera motion: how the picture moved from one frame to the next.

The camera pans and zooms while the field stays where it is. The motion of the background
between two frames is a homography, since the field is flat. It is estimated by following
points on the background with optical flow; the players are masked out, because they move
on their own.

The same points are followed over many frames and the motion since their first frame (the
key frame) is fitted directly. Fitting every pair of consecutive frames separately and
multiplying the results lets small errors add up twice as fast, and costs twice the time.
"""

from typing import Optional, Sequence

import cv2
import numpy as np

from ..config.settings import get_setting

WORK_SIZE = (640, 360)  # Motion is estimated on a reduced grey image
MAX_CORNERS = 300
MIN_POINTS = 12
KEY_FRAME_MAX_AGE = 30  # Frames the same points are followed before new ones are picked
KEY_FRAME_MIN_KEPT = 0.5  # ... or earlier, once this share of them is lost
LK_PARAMETERS = {"winSize": (21, 21), "maxLevel": 3}
RANSAC_THRESHOLD_PX = 2.0
MIN_AGREEING = 0.5  # Share of the followed points that must fit one motion


class CameraMotionEstimator:
    """Estimates the background motion between consecutive frames of a video."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Forget the previous frame (new video, seek)."""
        self._previous: Optional[np.ndarray] = None
        self._previous_mask: Optional[np.ndarray] = None
        self._key_points: Optional[np.ndarray] = (
            None  # Where the followed points were in the key frame
        )
        self._points: Optional[np.ndarray] = None  # Where they are in the previous frame
        self._key_count = 0
        self._age = 0
        self._since_key = np.eye(3)  # Motion from the key frame to the previous frame

    def update(
        self, frame: np.ndarray, player_boxes: Sequence[Sequence[float]]
    ) -> Optional[np.ndarray]:
        """Motion from the previous frame to this one.

        Args:
            frame: Video frame (BGR)
            player_boxes: Boxes (x1, y1, x2, y2) of the moving objects in the frame

        Returns:
            3x3 homography that maps a pixel position in the previous frame to its position
            in this frame, or None when it cannot be told (first frame, a cut, too little
            texture). Positions carried over from earlier frames are then unrelated.
        """
        frame_h, frame_w = frame.shape[:2]
        gray = cv2.cvtColor(
            cv2.resize(frame, WORK_SIZE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY
        )
        mask = self._background_mask(player_boxes, frame_w, frame_h)
        previous, previous_mask = self._previous, self._previous_mask
        self._previous, self._previous_mask = gray, mask
        if previous is None:
            return None

        if self._key_points is None:
            corners = cv2.goodFeaturesToTrack(previous, MAX_CORNERS, 0.01, 8, mask=previous_mask)
            if corners is None or len(corners) < MIN_POINTS:
                return None
            self._key_points = self._points = corners
            self._key_count = len(corners)
            self._age = 0
            self._since_key = np.eye(3)

        # Points that were followed wrongly are sorted out by the fit below; checking each
        # one by following it back as well costs as much again and changes nothing
        moved, found, _ = cv2.calcOpticalFlowPyrLK(
            previous, gray, self._points, None, **LK_PARAMETERS
        )
        kept = found.ravel() == 1
        key_points, points = self._key_points[kept], moved[kept]

        key_to_current = None
        if len(points) >= MIN_POINTS:
            key_to_current, inliers = cv2.findHomography(
                key_points, points, cv2.RANSAC, RANSAC_THRESHOLD_PX
            )
        # After a cut the points land anywhere, and a few of them always agree by chance
        agreeing = 0 if key_to_current is None else int(inliers.sum())
        if agreeing < max(MIN_POINTS, MIN_AGREEING * len(points)):
            self._key_points = None  # Start again from the next frame
            return None

        # Points that do not move with the background are players that were not detected
        on_background = inliers.ravel() == 1
        self._key_points, self._points = key_points[on_background], points[on_background]
        previous_to_current = key_to_current @ np.linalg.inv(self._since_key)
        self._since_key = key_to_current
        self._age += 1
        if (
            self._age >= KEY_FRAME_MAX_AGE
            or len(self._points) < KEY_FRAME_MIN_KEPT * self._key_count
        ):
            self._key_points = None

        # From the reduced image back to frame pixels
        scale = np.diag([frame_w / WORK_SIZE[0], frame_h / WORK_SIZE[1], 1.0])
        return scale @ previous_to_current @ np.linalg.inv(scale)

    @staticmethod
    def _background_mask(
        boxes: Sequence[Sequence[float]], frame_w: int, frame_h: int
    ) -> np.ndarray:
        """255 where points may be picked: everywhere but on the players and their shadows."""
        mask = np.full((WORK_SIZE[1], WORK_SIZE[0]), 255, dtype=np.uint8)
        scale_x, scale_y = WORK_SIZE[0] / frame_w, WORK_SIZE[1] / frame_h
        for x1, y1, x2, y2 in boxes:
            pad = 0.25 * (y2 - y1) * scale_y + 3
            cv2.rectangle(
                mask,
                (int(x1 * scale_x - pad), int(y1 * scale_y - pad)),
                (int(x2 * scale_x + pad), int(y2 * scale_y + 2 * pad)),
                0,
                -1,
            )
        return mask


def is_enabled() -> bool:
    return bool(get_setting("models.tracking.camera_motion_compensation", True))
