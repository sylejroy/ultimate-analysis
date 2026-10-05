"""One lock for the models and tracker state that all tabs share.

The processing modules keep their loaded models at module level. The main tab analyses
frames on a worker thread while the other tabs use the same modules from the GUI thread,
so each takes this lock around one unit of work (a frame, a model change).
"""

import threading

MODEL_LOCK = threading.RLock()
