def draw_ground_plane_grid(frame, H, grid_size=40, color=(0, 255, 255)):
    """
    Draws a grid on the ground plane, projected onto the current frame using homography H.
    """
    import cv2
    import numpy as np
    h, w = frame.shape[:2]
    # Generate grid points in the previous frame
    pts = []
    for x in range(0, w, grid_size):
        for y in range(0, h, grid_size):
            pts.append([x, y])
    pts = np.float32(pts).reshape(-1, 1, 2)
    # Project points to current frame
    if H is not None:
        pts_proj = cv2.perspectiveTransform(pts, H)
        for (xp, yp) in pts_proj.reshape(-1, 2):
            if 0 <= int(xp) < w and 0 <= int(yp) < h:
                cv2.circle(frame, (int(xp), int(yp)), 2, color, -1)
    return frame
import cv2
import numpy as np

def estimate_ground_homography(prev_frame, curr_frame, detections=None, min_matches=30):
    """
    Estimate the ground plane homography between two frames, filtering out moving objects (players) using detections.
    - prev_frame, curr_frame: BGR or grayscale images
    - detections: list of bounding boxes (x, y, w, h or x1, y1, x2, y2) for moving objects in prev_frame
    Returns: H (3x3 homography matrix), mask (inlier mask), matches (list of cv2.DMatch)
    """
    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if prev_frame.ndim == 3 else prev_frame
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY) if curr_frame.ndim == 3 else curr_frame

    # Detect ORB features
    orb = cv2.ORB_create(1000)
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return None, None, []

    # Filter out keypoints inside any detection bbox (moving objects)
    if detections is not None:
        def is_in_bbox(pt, bbox):
            if len(bbox) == 4:
                x, y, w, h = bbox
                return x <= pt[0] <= x + w and y <= pt[1] <= y + h
            elif len(bbox) == 5:
                x1, y1, x2, y2, _ = bbox
                return x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2
            else:
                x1, y1, x2, y2 = bbox
                return x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2
        filtered_kp1 = []
        filtered_des1 = []
        for k, d in zip(kp1, des1):
            keep = True
            for bbox in detections:
                if is_in_bbox(k.pt, bbox):
                    keep = False
                    break
            if keep:
                filtered_kp1.append(k)
                filtered_des1.append(d)
        if len(filtered_kp1) < 10:
            return None, None, []
        kp1 = filtered_kp1
        des1 = np.array(filtered_des1)

    # Match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) < min_matches:
        return None, None, []
    matches = sorted(matches, key=lambda x: x.distance)
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    # Estimate homography with RANSAC
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    return H, mask, matches
