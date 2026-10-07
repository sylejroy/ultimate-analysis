"""One lock for the models and tracker state that all tabs share.

The processing modules keep their loaded models at module level. The main tab analyses
frames on a worker thread while the other tabs use the same modules from the GUI thread,
so each takes this lock around one unit of work (a frame, a model change).
"""

import threading

MODEL_LOCK = threading.RLock()

# Setting up a TensorRT engine fails if another thread uses the GPU at that moment, and
# the model then runs in PyTorch, several times slower, for the rest of the session. The
# one other thread that uses the GPU is the background jersey reader: it holds this lock
# while it reads, and so does whoever sets up an engine.
GPU_SETUP_LOCK = threading.Lock()
