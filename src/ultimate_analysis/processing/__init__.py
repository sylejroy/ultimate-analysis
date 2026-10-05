"""Analysis stages: detection, tracking, jersey numbers, field segmentation, homography."""

import cv2

from . import inference  # noqa: F401  Imports Ultralytics, see below

# Ultralytics switches OpenCV multithreading off when it is imported, to protect its
# training data loaders. Training runs in its own process, so give the app's resizing,
# warping, and mask operations all cores back.
cv2.setNumThreads(-1)
