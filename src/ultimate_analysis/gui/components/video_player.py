"""
Video player component for the Ultimate Analysis GUI.
"""
import cv2
import numpy as np
from typing import Optional


class VideoPlayer:
    """
    A video player that handles video loading, frame navigation, and buffering.
    """
    
    def __init__(self):
        self.cap = None
        self._frame_buffer = None
        self._buffer_size = 0
        self._last_frame_index = -1

    def load_video(self, path: str) -> bool:
        """
        Load a video file for playback.
        
        Args:
            path: Path to the video file
            
        Returns:
            True if video was loaded successfully, False otherwise
        """
        if self.cap:
            self.cap.release()
        
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            self.cap = None
            return False
            
        # Clear buffer when loading new video
        self._frame_buffer = None
        self._buffer_size = 0
        self._last_frame_index = -1
        return True

    def get_next_frame(self) -> Optional[np.ndarray]:
        """
        Get the next frame from the video.
        
        Returns:
            The next frame as a numpy array, or None if no more frames
        """
        if not self.cap:
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            self.cap = None
            return None
        
        # Store last frame for potential reuse
        self._frame_buffer = frame
        current_index = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self._last_frame_index = current_index
        
        return frame
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """
        Get the current frame without advancing.
        
        Returns:
            The current frame as a numpy array, or None if no frame available
        """
        return self._frame_buffer if self._frame_buffer is not None else None
    
    def skip_frames(self, count: int) -> None:
        """
        Skip the specified number of frames for faster navigation.
        
        Args:
            count: Number of frames to skip
        """
        if not self.cap:
            return
            
        current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        new_pos = current_pos + count
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
    
    def get_fps(self) -> float:
        """
        Get video FPS.
        
        Returns:
            Video FPS, or 30.0 if no video is loaded
        """
        if not self.cap:
            return 30.0
        return self.cap.get(cv2.CAP_PROP_FPS)
    
    def get_frame_count(self) -> int:
        """
        Get total frame count.
        
        Returns:
            Total number of frames in the video, or 0 if no video is loaded
        """
        if not self.cap:
            return 0
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def get_current_position(self) -> int:
        """
        Get the current frame position.
        
        Returns:
            Current frame index, or -1 if no video is loaded
        """
        if not self.cap:
            return -1
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
    
    def set_position(self, frame_index: int) -> bool:
        """
        Set the current frame position.
        
        Args:
            frame_index: The frame index to seek to
            
        Returns:
            True if seek was successful, False otherwise
        """
        if not self.cap:
            return False
            
        success = self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        if success:
            self._last_frame_index = frame_index
        return success
    
    def release(self) -> None:
        """Release the video capture resource."""
        if self.cap:
            self.cap.release()
            self.cap = None
        self._frame_buffer = None
        self._buffer_size = 0
        self._last_frame_index = -1
    
    def __del__(self):
        """Ensure resources are released when the object is destroyed."""
        self.release()
