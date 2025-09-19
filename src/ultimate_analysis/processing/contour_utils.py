"""Contour utility helpers for processing modules.

Split from `field_analysis.py` to keep files within size limits and improve reuse.
"""

from __future__ import annotations

import numpy as np


def normalize_contour_to_points(contour: np.ndarray) -> np.ndarray:
    """Normalize contour input to (N, 2) points format efficiently."""
    if contour.ndim == 3 and contour.shape[1] == 1:
        return contour.reshape(-1, 2)
    elif contour.ndim == 2 and contour.shape[1] == 2:
        return contour
    else:
        return contour.reshape(-1, 2)


def points_to_contour_format(points: np.ndarray) -> np.ndarray:
    """Convert points to standard contour format (N, 1, 2)."""
    return points.reshape(-1, 1, 2) if len(points) > 0 else np.array([]).reshape(0, 1, 2)
