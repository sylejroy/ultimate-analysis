"""Video utility functions for Ultimate Analysis.

This module contains common video-related functions used across the application.
"""

import cv2
from typing import Optional


def get_video_duration(video_path: str) -> str:
    """Get video duration as formatted string.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Duration string in format "MM:SS" or "Unknown"
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            
            if fps > 0:
                duration_seconds = frame_count / fps
                minutes = int(duration_seconds // 60)
                seconds = int(duration_seconds % 60)
                return f"{minutes:02d}:{seconds:02d}"
        
    except Exception as e:
        print(f"[VIDEO_UTILS] Error getting duration for {video_path}: {e}")
    
    return "Unknown"


def get_video_info(video_path: str) -> Optional[dict]:
    """Get comprehensive video information.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video properties or None if failed
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            duration_seconds = frame_count / fps if fps > 0 else 0
            
            return {
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'duration_seconds': duration_seconds,
                'duration_formatted': get_video_duration(video_path)
            }
        
    except Exception as e:
        print(f"[VIDEO_UTILS] Error getting video info for {video_path}: {e}")
    
    return None
