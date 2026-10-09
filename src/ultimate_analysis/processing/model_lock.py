"""One lock for the models and tracker state that all tabs share.

The processing modules keep their loaded models at module level. The main tab analyses
frames on a worker thread while the other tabs use the same modules from the GUI thread,
so each takes this lock around one unit of work (a frame, a model change).
"""

import threading
from contextlib import contextmanager
from typing import Iterator

MODEL_LOCK = threading.RLock()


class _GpuGate:
    """Lets any number of threads run their networks at once, and none while an engine
    is being set up.

    Setting up a TensorRT engine (loading it and running it for the first time) fails if
    another thread uses the GPU at that moment: the model then runs in PyTorch, several
    times slower, or every network fails from then on. Several threads use the GPU: the
    two detectors, the field model, the jersey reader, the network that knows players by
    their looks. Each takes `work()` around a run of its network; whoever sets up an
    engine takes the gate itself (`with GPU_SETUP_LOCK:`), which waits until nobody
    works and lets nobody begin. Never ask for the gate while inside `work()`.
    """

    def __init__(self) -> None:
        self._changed = threading.Condition()
        self._working = 0
        self._setting_up = False
        self._waiting_to_set_up = 0

    @contextmanager
    def work(self) -> Iterator[None]:
        with self._changed:
            # A set-up that is waiting goes first, or it would wait for ever
            while self._setting_up or self._waiting_to_set_up:
                self._changed.wait()
            self._working += 1
        try:
            yield
        finally:
            with self._changed:
                self._working -= 1
                self._changed.notify_all()

    def __enter__(self) -> "_GpuGate":
        with self._changed:
            self._waiting_to_set_up += 1
            while self._setting_up or self._working:
                self._changed.wait()
            self._waiting_to_set_up -= 1
            self._setting_up = True
        return self

    def __exit__(self, *_: object) -> None:
        with self._changed:
            self._setting_up = False
            self._changed.notify_all()


GPU_SETUP_LOCK = _GpuGate()
