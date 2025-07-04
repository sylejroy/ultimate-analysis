import cv2
import numpy as np

class VideoPlayer:
    def __init__(self):
        self.cap = None
        self._frame_buffer = None
        self._buffer_size = 0
        self._last_frame_index = -1

    def load_video(self, path):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(path)
        # Clear buffer when loading new video
        self._frame_buffer = None
        self._buffer_size = 0
        self._last_frame_index = -1

    def get_next_frame(self):
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
    
    def get_current_frame(self):
        """Get the current frame without advancing"""
        return self._frame_buffer if self._frame_buffer is not None else None
    
    def skip_frames(self, count):
        """Skip the specified number of frames for faster navigation"""
        if not self.cap:
            return
        current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        new_pos = current_pos + count
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
    
    def get_fps(self):
        """Get video FPS"""
        if not self.cap:
            return 30
        return self.cap.get(cv2.CAP_PROP_FPS)
    
    def get_frame_count(self):
        """Get total frame count"""
        if not self.cap:
            return 0
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))