"""Object tracking module - DeepSORT and histogram-based tracking.

This module handles tracking of detected objects across video frames.
Maintains consistent identities for players and discs throughout the game.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..config.settings import get_setting
from ..constants import MAX_TRACKS_ACTIVE, TRACK_HISTORY_MAX_LENGTH, FALLBACK_DEFAULTS

# Try to import DeepSORT
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    print("[TRACKING] DeepSORT not available, install with: pip install deep-sort-realtime")
    DEEPSORT_AVAILABLE = False
    DeepSort = None


# Global tracking state
_tracker_type = "deepsort"
_deepsort_tracker = None
_histogram_tracker = None
_track_histories = defaultdict(list)
_frame_count = 0

# Histogram tracker state
_active_tracks = {}  # track_id -> track info
_next_track_id = 1


class Track:
    """Represents a tracked object with consistent identity."""
    
    def __init__(self, track_id: int, bbox: List[float], class_id: int, confidence: float, 
                 class_name: str = "unknown"):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.class_id = class_id
        self.confidence = confidence
        self.class_name = class_name
        self.det_class = class_name  # For compatibility with existing code
        
    def to_ltrb(self) -> List[float]:
        """Return bounding box in [x1, y1, x2, y2] format."""
        return self.bbox
        
    def to_tlwh(self) -> List[float]:
        """Return bounding box in [x, y, width, height] format."""
        x1, y1, x2, y2 = self.bbox
        return [x1, y1, x2 - x1, y2 - y1]


def _initialize_deepsort_tracker():
    """Initialize DeepSORT tracker with optimal settings."""
    global _deepsort_tracker
    
    if not DEEPSORT_AVAILABLE:
        print("[TRACKING] Cannot initialize DeepSORT - not available")
        return False
    
    if _deepsort_tracker is not None:
        return True
    
    try:
        # DeepSORT configuration optimized for Ultimate Frisbee
        _deepsort_tracker = DeepSort(
            max_age=get_setting("models.tracking.max_age", 50),           # Frames to keep lost tracks
            n_init=get_setting("models.tracking.n_init", 3),             # Frames needed to confirm track  
            nms_max_overlap=get_setting("models.tracking.nms_overlap", 0.7),  # Non-max suppression
            max_cosine_distance=get_setting("models.tracking.max_cosine_distance", 0.7),  # Feature similarity
            nn_budget=get_setting("models.tracking.nn_budget", 100),     # Feature budget per class
            override_track_class=None,  # Don't override class predictions
            embedder="mobilenet",       # Feature extractor model
            half=True,                  # Use half precision for speed
            bgr=True,                   # Input is BGR format
            embedder_gpu=True,          # Use GPU for feature extraction if available
            embedder_model_name=None,
            embedder_wts=None,
            polygon=False,              # Don't use polygon tracking
            today=None
        )
        
        print("[TRACKING] DeepSORT tracker initialized successfully")
        return True
        
    except Exception as e:
        print(f"[TRACKING] Failed to initialize DeepSORT: {e}")
        _deepsort_tracker = None
        return False


def run_tracking(frame: np.ndarray, detections: List[Dict[str, Any]]) -> List[Track]:
    """Run object tracking on detected objects.
    
    Args:
        frame: Input video frame as numpy array (H, W, C) in BGR format
        detections: List of detection dictionaries from inference
        
    Returns:
        List of Track objects with consistent IDs across frames
        
    Example:
        tracks = run_tracking(frame, detections)
        for track in tracks:
            track_id = track.track_id
            x1, y1, x2, y2 = track.to_ltrb()
    """
    global _frame_count
    _frame_count += 1
    
    print(f"[TRACKING] Processing {len(detections)} detections with {_tracker_type} tracker (frame {_frame_count})")
    
    if not detections:
        return []
    
    if _tracker_type == "deepsort":
        return _run_deepsort_tracking(frame, detections)
    elif _tracker_type == "histogram":
        return _run_histogram_tracking(frame, detections)
    else:
        print(f"[TRACKING] Unknown tracker type: {_tracker_type}")
        return []


def run_batch_tracking(frames: List[np.ndarray], batch_detections: List[List[Dict[str, Any]]], 
                      use_parallel: bool = True) -> List[List[Track]]:
    """Run object tracking on a batch of frames with their detections.
    
    Args:
        frames: List of input video frames
        batch_detections: List of detection lists (one per frame)
        use_parallel: Whether to use parallel processing for track post-processing
        
    Returns:
        List of Track lists (one per frame)
        
    Performance Benefits:
        - Efficient batch processing of tracks
        - Parallel post-processing of track data
        - Maintains temporal consistency across batch
    """
    if not frames or not batch_detections or len(frames) != len(batch_detections):
        return [[] for _ in frames]
    
    print(f"[TRACKING] Processing batch of {len(frames)} frames with tracking")
    
    batch_tracks = []
    
    # For tracking, we need to process frames sequentially to maintain temporal consistency
    # But we can parallelize the post-processing of track results
    for i, (frame, detections) in enumerate(zip(frames, batch_detections)):
        try:
            # Run tracking on this frame (must be sequential for temporal consistency)
            tracks = run_tracking(frame, detections)
            batch_tracks.append(tracks)
        except Exception as e:
            print(f"[TRACKING] Error processing frame {i} in batch: {e}")
            batch_tracks.append([])
    
    # Parallel post-processing of tracks if requested and beneficial
    if use_parallel and len(frames) > 2:
        # Post-process tracks in parallel (coordinate normalization, metadata, etc.)
        def enhance_tracks(frame_data):
            frame_idx, (frame, tracks) = frame_data
            return frame_idx, _enhance_track_metadata(frame, tracks)
        
        # Determine optimal number of workers
        max_workers = min(len(frames), 4)
        enhanced_batch_tracks = [[] for _ in frames]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            print(f"[TRACKING] Using {max_workers} parallel workers for track enhancement")
            
            # Submit all tasks
            indexed_data = [(i, (frame, tracks)) for i, (frame, tracks) in enumerate(zip(frames, batch_tracks))]
            future_to_index = {
                executor.submit(enhance_tracks, frame_data): frame_data[0] 
                for frame_data in indexed_data
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_index):
                try:
                    frame_idx, enhanced_tracks = future.result()
                    enhanced_batch_tracks[frame_idx] = enhanced_tracks
                except Exception as e:
                    frame_idx = future_to_index[future]
                    print(f"[TRACKING] Parallel track enhancement failed for frame {frame_idx}: {e}")
                    enhanced_batch_tracks[frame_idx] = batch_tracks[frame_idx]  # Use original tracks
        
        batch_tracks = enhanced_batch_tracks
    
    print(f"[TRACKING] Batch tracking complete: {[len(tracks) for tracks in batch_tracks]} tracks per frame")
    return batch_tracks


def _enhance_track_metadata(frame: np.ndarray, tracks: List[Track]) -> List[Track]:
    """Enhance track objects with additional metadata for better performance.
    
    Args:
        frame: Video frame for context
        tracks: List of track objects
        
    Returns:
        Enhanced list of track objects
    """
    try:
        enhanced_tracks = []
        frame_height, frame_width = frame.shape[:2]
        
        for track in tracks:
            # Add normalized coordinates for faster processing
            x1, y1, x2, y2 = track.bbox
            track.normalized_bbox = [
                x1 / frame_width,
                y1 / frame_height,
                x2 / frame_width,
                y2 / frame_height
            ]
            
            # Add center point for quick access
            track.center = [(x1 + x2) / 2, (y1 + y2) / 2]
            
            # Add track area for filtering
            track.area = (x2 - x1) * (y2 - y1)
            
            enhanced_tracks.append(track)
        
        return enhanced_tracks
        
    except Exception as e:
        print(f"[TRACKING] Error enhancing track metadata: {e}")
        return tracks  # Return original tracks on error


def _run_deepsort_tracking(frame: np.ndarray, detections: List[Dict[str, Any]]) -> List[Track]:
    """Run DeepSORT tracking on detections.
    
    Args:
        frame: Input video frame
        detections: List of detection dictionaries
        
    Returns:
        List of Track objects with consistent IDs
    """
    if not _initialize_deepsort_tracker():
        print("[TRACKING] DeepSORT not available, falling back to simple tracking")
        return _run_simple_tracking(detections)
    
    try:
        # Convert detections to DeepSORT format: [([x1, y1, x2, y2], confidence, class_id), ...]
        deepsort_detections = []
        
        print(f"[TRACKING] Processing {len(detections)} detections for DeepSORT")
        
        for i, det in enumerate(detections):
            print(f"[TRACKING] Detection {i}: {det}")
            
            bbox = det.get('bbox')
            confidence = det.get('confidence')
            class_id = det.get('class_id')
            
            print(f"[TRACKING] bbox: {bbox} (type: {type(bbox)})")
            print(f"[TRACKING] confidence: {confidence} (type: {type(confidence)})")  
            print(f"[TRACKING] class_id: {class_id} (type: {type(class_id)})")
            
            # Ensure bbox exists and has 4 values
            if bbox is None:
                print(f"[TRACKING] Warning: bbox is None, skipping detection")
                continue
                
            if not hasattr(bbox, '__len__') or len(bbox) != 4:
                print(f"[TRACKING] Warning: Invalid bbox format or length {bbox}, skipping detection")
                continue
                
            try:
                # Convert bbox from [x1, y1, x2, y2] to [x, y, width, height] for DeepSORT
                x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                
                # DeepSORT expects TLWH format: [x, y, width, height]
                x = x1
                y = y1  
                width = x2 - x1
                height = y2 - y1
                
                conf = float(confidence)
                cls = int(class_id)
                
                # DeepSORT expects ([x, y, width, height], confidence, class_id) format
                deepsort_det = ([x, y, width, height], conf, cls)
                deepsort_detections.append(deepsort_det)
                
                print(f"[TRACKING] Formatted detection: LTRB {[x1, y1, x2, y2]} -> TLWH {[x, y, width, height]}")
                
            except (ValueError, TypeError) as e:
                print(f"[TRACKING] Warning: Invalid detection values, skipping: {e}")
                continue
        
        if not deepsort_detections:
            print("[TRACKING] No valid detections for DeepSORT")
            return []
            
        print(f"[TRACKING] Formatted {len(deepsort_detections)} detections for DeepSORT")
        
        # Update tracker with current frame and detections
        tracks_deepsort = _deepsort_tracker.update_tracks(deepsort_detections, frame=frame)
        
        # Convert DeepSORT tracks to our Track format
        tracks = []
        for track in tracks_deepsort:
            if not track.is_confirmed():
                continue  # Skip unconfirmed tracks
                
            # Get track bounding box
            ltrb = track.to_ltrb()
            
            # Get class info (use the most recent detection class)
            class_id = int(track.get_det_class()) if track.get_det_class() is not None else 0
            confidence = float(track.get_det_conf()) if track.get_det_conf() is not None else 0.5
            
            # Map class_id to class_name
            class_name = _get_class_name_from_id(class_id)
            
            # Create our Track object
            our_track = Track(
                track_id=int(track.track_id),
                bbox=[float(ltrb[0]), float(ltrb[1]), float(ltrb[2]), float(ltrb[3])],
                class_id=class_id,
                confidence=confidence,
                class_name=class_name
            )
            
            tracks.append(our_track)
            
            # Update track history for visualization (at player's feet - bottom center)
            foot_x = (ltrb[0] + ltrb[2]) / 2  # Center X
            foot_y = ltrb[3]                  # Bottom Y (feet level)
            _update_track_history(our_track.track_id, (int(foot_x), int(foot_y)))
        
        print(f"[TRACKING] DeepSORT returned {len(tracks)} confirmed tracks")
        return tracks
        
    except Exception as e:
        print(f"[TRACKING] Error in DeepSORT tracking: {e}")
        import traceback
        traceback.print_exc()
        return _run_simple_tracking(detections)


def _run_histogram_tracking(frame: np.ndarray, detections: List[Dict[str, Any]]) -> List[Track]:
    """Run histogram-based tracking on detections using color histograms and IoU matching.
    
    Args:
        frame: Input video frame
        detections: List of detection dictionaries
        
    Returns:
        List of Track objects with consistent IDs
    """
    global _active_tracks, _next_track_id
    
    if not detections:
        # Age all tracks and remove old ones
        _age_and_clean_tracks()
        return []
    
    print(f"[TRACKING] Histogram tracking: processing {len(detections)} detections")
    
    # Extract features for current detections
    current_features = []
    for det in detections:
        try:
            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Clamp coordinates to frame bounds
            x1 = max(0, min(x1, frame.shape[1] - 1))
            y1 = max(0, min(y1, frame.shape[0] - 1))
            x2 = max(x1 + 1, min(x2, frame.shape[1]))
            y2 = max(y1 + 1, min(y2, frame.shape[0]))
            
            # Extract region and compute histogram
            region = frame[y1:y2, x1:x2]
            if region.size > 0:
                hist = _compute_color_histogram(region)
                feature = {
                    'bbox': [x1, y1, x2, y2],
                    'histogram': hist,
                    'center': [(x1 + x2) / 2, (y1 + y2) / 2],
                    'area': (x2 - x1) * (y2 - y1),
                    'class_id': det['class_id'],
                    'confidence': det['confidence'],
                    'class_name': det.get('class_name', 'unknown')
                }
                current_features.append(feature)
        except Exception as e:
            print(f"[TRACKING] Error extracting features for detection: {e}")
            continue
    
    if not current_features:
        _age_and_clean_tracks()
        return []
    
    # Match current detections with existing tracks
    matched_tracks, unmatched_detections = _match_detections_to_tracks(current_features)
    
    # Update matched tracks
    tracks = []
    for track_id, detection_idx in matched_tracks:
        detection_feature = current_features[detection_idx]
        
        # Update track information
        _active_tracks[track_id]['bbox'] = detection_feature['bbox']
        _active_tracks[track_id]['histogram'] = detection_feature['histogram']
        _active_tracks[track_id]['center'] = detection_feature['center']
        _active_tracks[track_id]['area'] = detection_feature['area']
        _active_tracks[track_id]['age'] = 0  # Reset age
        _active_tracks[track_id]['confidence'] = detection_feature['confidence']
        _active_tracks[track_id]['class_id'] = detection_feature['class_id']
        _active_tracks[track_id]['class_name'] = detection_feature['class_name']
        
        # Create Track object
        track = Track(
            track_id=track_id,
            bbox=detection_feature['bbox'],
            class_id=detection_feature['class_id'],
            confidence=detection_feature['confidence'],
            class_name=detection_feature['class_name']
        )
        tracks.append(track)
        
        # Update track history
        _update_track_history(track_id, (int(detection_feature['center'][0]), int(detection_feature['center'][1])))
    
    # Create new tracks for unmatched detections
    for detection_idx in unmatched_detections:
        detection_feature = current_features[detection_idx]
        
        # Create new track
        track_id = _next_track_id
        _next_track_id += 1
        
        _active_tracks[track_id] = {
            'bbox': detection_feature['bbox'],
            'histogram': detection_feature['histogram'],
            'center': detection_feature['center'],
            'area': detection_feature['area'],
            'age': 0,
            'confidence': detection_feature['confidence'],
            'class_id': detection_feature['class_id'],
            'class_name': detection_feature['class_name'],
            'created_frame': _frame_count
        }
        
        # Create Track object
        track = Track(
            track_id=track_id,
            bbox=detection_feature['bbox'],
            class_id=detection_feature['class_id'],
            confidence=detection_feature['confidence'],
            class_name=detection_feature['class_name']
        )
        tracks.append(track)
        
        # Update track history
        _update_track_history(track_id, (int(detection_feature['center'][0]), int(detection_feature['center'][1])))
    
    # Age and clean up tracks
    _age_and_clean_tracks()
    
    print(f"[TRACKING] Histogram tracking: {len(tracks)} tracks (updated: {len(matched_tracks)}, new: {len(unmatched_detections)})")
    return tracks


def _compute_color_histogram(region: np.ndarray, bins: int = None) -> np.ndarray:
    """Compute color histogram for a region.
    
    Args:
        region: Image region as numpy array
        bins: Number of histogram bins per channel (uses config default if None)
        
    Returns:
        Normalized histogram as numpy array
    """
    if bins is None:
        bins = get_setting("models.tracking.histogram_bins", 32)
    
    if region.size == 0:
        return np.zeros(bins * 3)
    
    try:
        # Convert BGR to HSV for better color representation
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Compute histogram for each channel
        hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
        
        # Concatenate and normalize
        hist = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
        hist = hist / (np.sum(hist) + 1e-7)  # Normalize with small epsilon
        
        return hist
        
    except Exception as e:
        print(f"[TRACKING] Error computing histogram: {e}")
        return np.zeros(bins * 3)


def _match_detections_to_tracks(current_features: List[Dict]) -> Tuple[List[Tuple[int, int]], List[int]]:
    """Match current detections to existing tracks using IoU and histogram similarity.
    
    Args:
        current_features: List of detection features
        
    Returns:
        Tuple of (matched_pairs, unmatched_detection_indices)
        matched_pairs: List of (track_id, detection_idx) pairs
    """
    if not _active_tracks:
        return [], list(range(len(current_features)))
    
    # Compute similarity matrix
    track_ids = list(_active_tracks.keys())
    similarity_matrix = np.zeros((len(track_ids), len(current_features)))
    
    for i, track_id in enumerate(track_ids):
        track_info = _active_tracks[track_id]
        
        for j, detection in enumerate(current_features):
            # IoU similarity
            iou = _compute_iou(track_info['bbox'], detection['bbox'])
            
            # Histogram similarity (only if IoU > 0)
            hist_sim = 0.0
            if iou > 0.1:  # Only compute histogram if objects overlap
                hist_sim = _compute_histogram_similarity(track_info['histogram'], detection['histogram'])
            
            # Combined similarity (weighted)
            similarity = 0.7 * iou + 0.3 * hist_sim
            similarity_matrix[i, j] = similarity
    
    # Hungarian algorithm would be ideal, but let's use greedy matching for simplicity
    matched_pairs = []
    unmatched_detections = list(range(len(current_features)))
    matched_tracks = set()
    
    # Greedy matching: find best matches above threshold
    threshold = get_setting("models.tracking.histogram_match_threshold", 0.3)
    
    while True:
        # Find best unmatched pair
        best_similarity = 0
        best_track_idx = -1
        best_det_idx = -1
        
        for i, track_id in enumerate(track_ids):
            if track_id in matched_tracks:
                continue
                
            for j in unmatched_detections:
                if similarity_matrix[i, j] > best_similarity and similarity_matrix[i, j] > threshold:
                    best_similarity = similarity_matrix[i, j]
                    best_track_idx = i
                    best_det_idx = j
        
        if best_track_idx == -1:  # No more matches above threshold
            break
            
        # Match found
        track_id = track_ids[best_track_idx]
        matched_pairs.append((track_id, best_det_idx))
        matched_tracks.add(track_id)
        unmatched_detections.remove(best_det_idx)
    
    return matched_pairs, unmatched_detections


def _compute_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1: First bounding box [x1, y1, x2, y2]
        bbox2: Second bounding box [x1, y1, x2, y2]
        
    Returns:
        IoU value between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Intersection coordinates
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    # Check if there's an intersection
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    # Intersection area
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def _compute_histogram_similarity(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """Compute similarity between two histograms using correlation.
    
    Args:
        hist1: First histogram
        hist2: Second histogram
        
    Returns:
        Similarity value between 0 and 1
    """
    if hist1.size == 0 or hist2.size == 0:
        return 0.0
    
    try:
        # Use correlation coefficient
        correlation = cv2.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_CORREL)
        return max(0.0, correlation)  # Clamp to [0, 1]
    except Exception as e:
        print(f"[TRACKING] Error computing histogram similarity: {e}")
        return 0.0


def _age_and_clean_tracks() -> None:
    """Age all tracks and remove old ones."""
    global _active_tracks
    
    max_age = get_setting("models.tracking.max_age", 10)
    tracks_to_remove = []
    
    for track_id, track_info in _active_tracks.items():
        track_info['age'] += 1
        if track_info['age'] > max_age:
            tracks_to_remove.append(track_id)
    
    # Remove old tracks
    for track_id in tracks_to_remove:
        del _active_tracks[track_id]
        # Remove from track histories as well
        if track_id in _track_histories:
            del _track_histories[track_id]
    
    if tracks_to_remove:
        print(f"[TRACKING] Removed {len(tracks_to_remove)} old tracks")


def _run_simple_tracking(detections: List[Dict[str, Any]]) -> List[Track]:
    """Simple tracking fallback that assigns new IDs to each detection.
    
    Args:
        detections: List of detection dictionaries
        
    Returns:
        List of Track objects with new IDs
    """
    tracks = []
    
    for i, detection in enumerate(detections):
        # Create a simple track with frame-based ID
        track_id = _frame_count * 1000 + i  # Simple ID generation
        
        track = Track(
            track_id=track_id,
            bbox=detection['bbox'],
            class_id=detection['class_id'],
            confidence=detection['confidence'],
            class_name=detection.get('class_name', 'unknown')
        )
        tracks.append(track)
        
        # Update track history (at player's feet - bottom center)
        foot_x = (detection['bbox'][0] + detection['bbox'][2]) / 2  # Center X
        foot_y = detection['bbox'][3]                               # Bottom Y (feet level)
        _update_track_history(track_id, (int(foot_x), int(foot_y)))
    
    return tracks


def _get_class_name_from_id(class_id: int) -> str:
    """Convert class ID to class name.
    
    Args:
        class_id: Numeric class identifier
        
    Returns:
        String class name
    """
    # Map based on our model's class structure
    class_mapping = {
        0: 'disc',
        1: 'player'
    }
    
    return class_mapping.get(class_id, 'unknown')


def set_tracker_type(tracker_type: str) -> bool:
    """Set the type of tracker to use.
    
    Args:
        tracker_type: Type of tracker ("deepsort" or "histogram")
        
    Returns:
        True if tracker type set successfully, False otherwise
        
    Example:
        set_tracker_type("deepsort")
    """
    global _tracker_type, _deepsort_tracker, _histogram_tracker
    
    tracker_type = tracker_type.lower()
    
    if tracker_type not in ["deepsort", "histogram"]:
        print(f"[TRACKING] Unsupported tracker type: {tracker_type}")
        return False
    
    print(f"[TRACKING] Setting tracker type to: {tracker_type}")
    _tracker_type = tracker_type
    
    # Reset tracker instances to force reinitialization
    if tracker_type != "deepsort":
        _deepsort_tracker = None
    if tracker_type != "histogram":
        _histogram_tracker = None
    
    reset_tracker()
    
    return True


def reset_tracker() -> None:
    """Reset the tracker state and clear all tracks.
    
    This should be called when switching videos or when tracking quality degrades.
    """
    global _deepsort_tracker, _histogram_tracker, _track_histories, _frame_count, _active_tracks, _next_track_id
    
    print("[TRACKING] Resetting tracker state")
    
    # Reset DeepSORT tracker
    if _deepsort_tracker is not None:
        _deepsort_tracker = None
    
    # Reset histogram tracker
    if _histogram_tracker is not None:
        _histogram_tracker = None
    
    # Reset histogram tracker state
    _active_tracks.clear()
    _next_track_id = 1
    
    # Clear track histories and reset frame count
    _track_histories.clear()
    _frame_count = 0
    
    # Reset jersey tracking as well
    try:
        from .jersey_tracker import reset_jersey_tracker
        reset_jersey_tracker()
    except ImportError:
        print("[TRACKING] Jersey tracker not available for reset")
    
    print("[TRACKING] Tracker reset complete")


def get_tracker_type() -> str:
    """Get the current tracker type.
    
    Returns:
        Current tracker type string
    """
    return _tracker_type


def get_track_histories() -> Dict[int, List[Tuple[int, int]]]:
    """Get track history for all tracked objects.
    
    Returns:
        Dictionary mapping track_id to list of (center_x, center_y) positions
    """
    return dict(_track_histories)


def _update_track_history(track_id: int, center_point: Tuple[int, int]) -> None:
    """Update the position history for a track.
    
    Args:
        track_id: Unique track identifier
        center_point: Center point (x, y) of the tracked object
    """
    # Add center point to history
    _track_histories[track_id].append(center_point)
    
    # Limit history length
    max_length = get_setting("models.tracking.track_history_length", TRACK_HISTORY_MAX_LENGTH)
    if len(_track_histories[track_id]) > max_length:
        _track_histories[track_id] = _track_histories[track_id][-max_length:]


# Remove the old initialization functions and add proper initialization
def _load_default_tracker():
    """Load the default tracker type."""
    default_tracker = get_setting(
        "models.tracking.default_tracker", 
        "deepsort"
    )
    set_tracker_type(default_tracker)


# Initialize default tracker when module is imported
_load_default_tracker()
