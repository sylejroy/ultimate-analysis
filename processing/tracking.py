from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize DeepSort tracker once (global)
tracker = DeepSort(max_age=10, embedder="mobilenet", n_init=5)

def run_tracking(frame, detections):
    """
    Updates the tracker with the current frame and detections.
    Returns a list of track objects.
    """
    # DeepSort expects: [ [x, y, w, h], confidence, class ]
    tracks = tracker.update_tracks(detections, frame=frame)
    return tracks

def reset_tracker():
    """
    Resets the tracker state.
    """
    global tracker
    tracker = DeepSort(max_age=10, embedder="mobilenet", n_init=5)