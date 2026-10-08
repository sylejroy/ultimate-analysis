"""Video files: discovery, metadata, and frame-by-frame reading with OpenCV."""

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..config.settings import get_setting
from ..constants import DEFAULT_PATHS, MAX_FPS, MIN_FPS, SUPPORTED_VIDEO_EXTENSIONS
from .logger import get_logger

logger = get_logger("VIDEO")


def seconds_of(text: str) -> float:
    """Seconds from "90", "1:30" or "0:01:30"."""
    seconds = 0.0
    for part in str(text).split(":"):
        seconds = seconds * 60.0 + float(part)
    return seconds


def find_video_files() -> List[str]:
    """Paths of the videos in the video folder, sorted."""
    folder = Path(DEFAULT_PATHS["RAW_VIDEOS"])
    if not folder.exists():
        return []
    return sorted(
        str(path)
        for path in folder.glob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    )


def get_video_duration(video_path: str) -> str:
    """Get video duration as formatted string.

    Args:
        video_path: Path to video file

    Returns:
        Duration string in format "MM:SS" or "Unknown"
    """
    info = get_video_info(video_path)
    return info["duration_formatted"] if info is not None else "Unknown"


def get_video_info(video_path: str) -> Optional[dict]:
    """Get comprehensive video information.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with video properties or None if failed
    """
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration_seconds = frame_count / fps if fps > 0 else 0
            minutes = int(duration_seconds // 60)
            seconds = int(duration_seconds % 60)

            return {
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "duration_seconds": duration_seconds,
                "duration_formatted": f"{minutes:02d}:{seconds:02d}" if fps > 0 else "Unknown",
            }

    except Exception as e:
        logger.error(f"Error getting video info for {video_path}: {e}")
    finally:
        if cap is not None:
            cap.release()

    return None


class VideoPlayer:
    """Simple video player using OpenCV for Ultimate Analysis application."""

    def __init__(self):
        """Initialize the video player."""
        self.cap: Optional[cv2.VideoCapture] = None
        self.current_video_path: Optional[str] = None
        self.total_frames: int = 0
        self.fps: float = 25.0
        self.current_frame_idx: int = 0
        self.frame_width: int = 0
        self.frame_height: int = 0

        # During playback the frame after the one being processed is decoded here, so
        # decoding overlaps with processing. While a decode is pending, the capture must
        # not be touched from the main thread.
        self._decode_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="decode")
        self._decode_ahead: Optional[Future] = None

    def _take_decode_ahead(self) -> Optional[Tuple[bool, Optional[np.ndarray]]]:
        """Wait for the pending decode, if any, and return its (ok, frame) result."""
        pending, self._decode_ahead = self._decode_ahead, None
        return pending.result() if pending is not None else None

    def load_video(self, video_path: str) -> bool:
        """Load a video file for playback.

        Args:
            video_path: Path to the video file

        Returns:
            True if video loaded successfully, False otherwise
        """
        logger.info(f"Loading video: {video_path}")

        # Validate file path
        if not Path(video_path).exists():
            logger.warning(f"Video file not found: {video_path}")
            return False

        # Check file extension
        if not video_path.lower().endswith(SUPPORTED_VIDEO_EXTENSIONS):
            logger.debug(f"Unsupported video format: {video_path}")
            return False

        # Close existing video if open
        self.close_video()

        try:
            # Open video with OpenCV
            self.cap = cv2.VideoCapture(video_path)

            if not self.cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                self.close_video()
                return False

            # Get video properties
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Validate and clamp FPS
            if not np.isfinite(self.fps) or self.fps < MIN_FPS or self.fps > MAX_FPS:
                logger.warning(f"Invalid FPS {self.fps}, using default")
                self.fps = get_setting("video.default_fps", 25.0)

            self.current_video_path = video_path
            self.current_frame_idx = 0

            logger.info(
                f"Video loaded: {video_path} ({self.total_frames} frames, {self.fps:.2f} fps, "
                f"{self.total_frames / self.fps:.1f}s)"
            )

            return True

        except Exception as e:
            logger.error(f"Error loading video {video_path}: {e}")
            self.close_video()
            return False

    def get_next_frame(self) -> Optional[np.ndarray]:
        """Get the next frame from the video.

        Returns:
            Frame as numpy array (H, W, C) in BGR format, or None if no more frames
        """
        if self.cap is None or not self.cap.isOpened():
            return None

        decoded = self._take_decode_ahead()
        ret, frame = decoded if decoded is not None else self.cap.read()

        if not ret:
            logger.debug("End of video reached")
            return None

        self.current_frame_idx += 1
        # Decode the following frame while this one is processed
        self._decode_ahead = self._decode_executor.submit(self.cap.read)
        return frame

    def seek_to_frame(self, frame_idx: int) -> bool:
        """Seek to a specific frame in the video.

        Args:
            frame_idx: Frame index to seek to (0-based)

        Returns:
            True if seek successful, False otherwise
        """
        if self.cap is None or not self.cap.isOpened():
            return False

        # A frame decoded ahead belongs to the old position
        self._take_decode_ahead()

        # Clamp frame index to valid range
        frame_idx = max(0, min(frame_idx, self.total_frames - 1))

        try:
            if not self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx):
                return False
            self.current_frame_idx = frame_idx
            logger.debug(f"Seeked to frame {frame_idx}")
            return True

        except Exception as e:
            logger.error(f"Error seeking to frame {frame_idx}: {e}")
            return False

    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the current frame without advancing.

        Returns:
            Current frame as numpy array, or None if not available
        """
        if self.cap is None or not self.cap.isOpened():
            return None

        # The frame decoded ahead is the one at the current position
        if self._decode_ahead is not None:
            ret, frame = self._decode_ahead.result()
            return frame.copy() if ret else None

        # Save current position
        current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)

        # Read frame
        ret, frame = self.cap.read()

        if ret:
            # Restore position
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
            return frame

        return None

    def get_video_info(self) -> dict:
        """Get information about the currently loaded video.

        Returns:
            Dictionary with video information
        """
        if self.cap is None:
            return {
                "loaded": False,
                "path": None,
                "total_frames": 0,
                "fps": 0.0,
                "duration": 0.0,
                "current_frame": 0,
                "width": 0,
                "height": 0,
            }

        width = self.frame_width
        height = self.frame_height
        duration = self.total_frames / self.fps if self.fps > 0 else 0.0

        return {
            "loaded": True,
            "path": self.current_video_path,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "duration": duration,
            "current_frame": self.current_frame_idx,
            "width": width,
            "height": height,
        }

    def close_video(self) -> None:
        """Close the currently loaded video and release resources."""
        self._take_decode_ahead()
        if self.cap is not None:
            logger.info(f"Closing video: {self.current_video_path}")
            self.cap.release()
            self.cap = None

        self.current_video_path = None
        self.total_frames = 0
        self.fps = 25.0
        self.current_frame_idx = 0
        self.frame_width = 0
        self.frame_height = 0

    def is_loaded(self) -> bool:
        """Check if a video is currently loaded.

        Returns:
            True if video is loaded and ready for playback
        """
        return self.cap is not None and self.cap.isOpened()

    def __del__(self):
        """Cleanup when VideoPlayer is destroyed."""
        self.close_video()
