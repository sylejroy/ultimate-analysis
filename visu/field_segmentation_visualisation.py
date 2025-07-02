import cv2
import numpy as np

def draw_field_segmentation(frame, mask, confs=None):
    """
    Draws segmentation masks for two classes: 0 (Central Field, light blue/teal), 1 (Endzone, pink).
    - mask: (N, H, W) array of binary masks for each class.
    - confs: list of confidences for each class (optional).
    """
    if mask is not None:
        if isinstance(mask, list):
            mask = np.array(mask)
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask)
        overlay = frame.copy()
        color_mask = np.zeros_like(frame)

        # Use the same colors as detection_visualisation
        color_dict = {
            0: (200, 217, 37),   # Central Field: teal (BGR for RGB 37,217,200)
            1: (114, 38, 249)    # Endzone: pink (BGR for RGB 249,38,114)
        }
        name_dict = {0: "Central Field", 1: "Endzone"}

        n_classes = min(mask.shape[0], 2)  # Only process class 0 and 1
        for cls in range(n_classes):
            # Resize each class mask to match frame size
            class_mask = mask[cls]
            class_mask_resized = cv2.resize(class_mask.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask_bool = class_mask_resized > 0.5
            color = color_dict.get(cls, (200, 200, 200))
            color_mask[mask_bool] = color

            # Draw border for the mask
            contours, _ = cv2.findContours(class_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            border_color = tuple(int(c * 0.7) for c in color)
            cv2.drawContours(overlay, contours, -1, border_color, 2)

            # Find center of mask for label
            ys, xs = np.where(mask_bool)
            if len(xs) > 0 and len(ys) > 0:
                cx, cy = int(np.mean(xs)), int(np.mean(ys))
                label = name_dict.get(cls, str(cls))
                if confs and len(confs) > cls:
                    label += f" {confs[cls]:.2f}"
                cv2.putText(overlay, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, border_color, 2, cv2.LINE_AA)

        # Lower alpha for less impact (0.12)
        cv2.addWeighted(color_mask, 0.12, overlay, 0.88, 0, overlay)
        return overlay
    return frame