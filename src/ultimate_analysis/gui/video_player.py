"""Video player component for handling video playback.

This module provides a simple video player using OpenCV for frame extraction
and basic playback controls.
"""

from pathlib import Path
from queue import Queue
from threading import Event, Lock, Thread
from typing import Optional

import cv2
import numpy as np

from ..config.settings import get_setting
from ..constants import MAX_FPS, MIN_FPS, SUPPORTED_VIDEO_EXTENSIONS


class VideoPlayer:
    """Simple video player using OpenCV for Ultimate Analysis application."""

    def __init__(self):
        """Initialize the video player."""
        self.cap: Optional[cv2.VideoCapture] = None
        self.current_video_path: Optional[str] = None
        self.total_frames: int = 0
        self.fps: float = 25.0
        self.current_frame_idx: int = 0

        # Async frame buffer
        self.frame_buffer: Queue = Queue(maxsize=10)  # Buffer up to 10 frames
        self.buffer_thread: Optional[Thread] = None
        self.buffer_stop_event: Event = Event()
        self.buffer_enabled: bool = (
            False  # Disable async buffering due to OpenCV thread safety issues
        )
        self.cap_lock: Lock = Lock()  # Synchronize access to VideoCapture

    def _start_frame_buffer(self):
        """Start the background frame buffering thread."""
        if not self.buffer_enabled or self.cap is None:
            return

        self._stop_frame_buffer()  # Stop any existing thread

        self.buffer_stop_event.clear()
        self.buffer_thread = Thread(target=self._frame_buffer_worker, daemon=True)
        self.buffer_thread.start()
        print("[VIDEO_PLAYER] Started async frame buffering")

    def _stop_frame_buffer(self):
        """Stop the background frame buffering thread."""
        if self.buffer_thread is not None:
            self.buffer_stop_event.set()
            self.buffer_thread.join(timeout=1.0)
            self.buffer_thread = None

        # Clear buffer
        while not self.frame_buffer.empty():
            try:
                self.frame_buffer.get_nowait()
            except Exception:
                break

    def _frame_buffer_worker(self):
        """Background worker thread for frame buffering."""
        while not self.buffer_stop_event.is_set() and self.cap is not None:
            try:
                # Check if we need to buffer more frames
                if self.frame_buffer.qsize() < self.frame_buffer.maxsize:
                    # Use lock to safely access VideoCapture
                    with self.cap_lock:
                        if self.cap is None or not self.cap.isOpened():
                            break

                        # Try to read next frame
                        current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                        ret, frame = self.cap.read()

                        if ret:
                            # Store frame with its index
                            frame_data = (int(current_pos), frame)
                            try:
                                self.frame_buffer.put(frame_data, timeout=0.1)
                            except Exception:
                                # Buffer full, skip this frame
                                pass
                        else:
                            # End of video, wait a bit before checking again
                            self.buffer_stop_event.wait(0.1)
                else:
                    # Buffer full, wait before checking again
                    self.buffer_stop_event.wait(0.05)

            except Exception as e:
                print(f"[VIDEO_PLAYER] Frame buffer error: {e}")
                break

        print("[VIDEO_PLAYER] Frame buffer worker stopped")

    def load_video(self, video_path: str) -> bool:
        """Load a video file for playback.

        Args:
            video_path: Path to the video file

        Returns:
            True if video loaded successfully, False otherwise
        """
        print(f"[VIDEO_PLAYER] Loading video: {video_path}")

        # Validate file path
        if not Path(video_path).exists():
            print(f"[VIDEO_PLAYER] Video file not found: {video_path}")
            return False

        # Check file extension
        if not video_path.lower().endswith(SUPPORTED_VIDEO_EXTENSIONS):
            print(f"[VIDEO_PLAYER] Unsupported video format: {video_path}")
            return False

        # Close existing video if open
        self.close_video()

        try:
            # Open video with OpenCV
            self.cap = cv2.VideoCapture(video_path)

            if not self.cap.isOpened():
                print(f"[VIDEO_PLAYER] Failed to open video: {video_path}")
                self.cap = None
                return False

            # Get video properties
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)

            # Validate and clamp FPS
            if self.fps < MIN_FPS or self.fps > MAX_FPS:
                print(f"[VIDEO_PLAYER] Invalid FPS {self.fps}, using default")
                self.fps = get_setting("video.default_fps", 25.0)

            self.current_video_path = video_path
            self.current_frame_idx = 0

            # Start async frame buffering (only if enabled)
            if self.buffer_enabled:
                self._start_frame_buffer()

            print("[VIDEO_PLAYER] Video loaded successfully:")
            print(f"  - Path: {video_path}")
            print(f"  - Frames: {self.total_frames}")
            print(f"  - FPS: {self.fps}")
            print(f"  - Duration: {self.total_frames / self.fps:.1f}s")

            return True

        except Exception as e:
            print(f"[VIDEO_PLAYER] Error loading video {video_path}: {e}")
            self.cap = None
            return False

    def get_next_frame(self) -> Optional[np.ndarray]:
        """Get the next frame from the video.

        Returns:
            Frame as numpy array (H, W, C) in BGR format, or None if no more frames
        """
        if self.cap is None or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()

        if not ret:
            print("[VIDEO_PLAYER] End of video reached")
            return None

        self.current_frame_idx += 1
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

        # Clamp frame index to valid range
        frame_idx = max(0, min(frame_idx, self.total_frames - 1))

        try:
            with self.cap_lock:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                self.current_frame_idx = frame_idx

            # Clear and restart frame buffer after seeking (only if enabled)
            if self.buffer_enabled:
                self._stop_frame_buffer()
                self._start_frame_buffer()

            print(f"[VIDEO_PLAYER] Seeked to frame {frame_idx}")
            return True

        except Exception as e:
            print(f"[VIDEO_PLAYER] Error seeking to frame {frame_idx}: {e}")
            return False

    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the current frame without advancing.

        Uses async frame buffer for improved performance.

        Returns:
            Current frame as numpy array, or None if not available
        """
        if self.cap is None or not self.cap.isOpened():
            return None

        # Try to get frame from buffer first
        if self.buffer_enabled and not self.frame_buffer.empty():
            try:
                frame_idx, frame = self.frame_buffer.get_nowait()
                # Check if this is the frame we want
                if frame_idx == self.current_frame_idx:
                    return frame.copy()
                else:
                    # Wrong frame, put it back and fall through to direct read
                    try:
                        self.frame_buffer.put_nowait((frame_idx, frame))
                    except Exception:
                        pass  # Buffer might be full
            except Exception:
                pass  # Buffer empty or error

        # Fall back to direct reading (synchronous) with lock
        with self.cap_lock:
            if self.cap is None or not self.cap.isOpened():
                return None

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

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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
        # Stop frame buffering first
        self._stop_frame_buffer()

        if self.cap is not None:
            print(f"[VIDEO_PLAYER] Closing video: {self.current_video_path}")
            self.cap.release()
            self.cap = None

        self.current_video_path = None
        self.total_frames = 0
        self.fps = 25.0
        self.current_frame_idx = 0

    def is_loaded(self) -> bool:
        """Check if a video is currently loaded.

        Returns:
            True if video is loaded and ready for playback
        """
        return self.cap is not None and self.cap.isOpened()

    def __del__(self):
        """Cleanup when VideoPlayer is destroyed."""
        self.close_video()
