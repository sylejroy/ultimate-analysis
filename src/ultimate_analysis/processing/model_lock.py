"""One lock for the models and tracker state that all tabs share.

The processing modules keep their loaded models at module level. The main tab analyses
frames on a worker thread while the other tabs use the same modules from the GUI thread,
so each takes this lock around one unit of work (a frame, a model change).
"""

import threading
from contextlib import contextmanager
from typing import Callable, Iterator

MODEL_LOCK = threading.RLock()

# How long a thread waits for the GPU before it gives up with a TimeoutError: a network
# run or an engine set-up that hangs would otherwise hold up every thread for good. The
# callers report a failure as they report any other (processing/health.py).
WAIT_SECONDS = 60.0


class _GpuGate:
    """Lets any number of threads run their networks at once, and none while an engine
    is being set up.

    Setting up a TensorRT engine (loading it and running it for the first time) fails if
    another thread uses the GPU at that moment: the model then runs in PyTorch, several
    times slower, or every network fails from then on. Several threads use the GPU: the
    two detectors, the field model, the jersey reader, the network that knows players by
    their looks. Each takes `work()` around a run of its network; whoever sets up an
    engine takes the gate itself (`with GPU_SETUP_LOCK:`), which waits until nobody
    works and lets nobody begin. A thread inside `work()` may take `work()` again, but
    not the gate: it would wait for itself.
    """

    def __init__(self, wait_seconds: float = WAIT_SECONDS) -> None:
        self.wait_seconds = wait_seconds
        self._changed = threading.Condition()
        self._working = 0
        self._setting_up = False
        self._waiting_to_set_up = 0
        self._mine = threading.local()  # How deep inside work() this thread is

    def _wait(self, ready: Callable[[], bool], what: str) -> None:
        if not self._changed.wait_for(ready, timeout=self.wait_seconds):
            raise TimeoutError(f"the GPU was not free for {what} in {self.wait_seconds:g} s")

    @contextmanager
    def work(self) -> Iterator[None]:
        depth = getattr(self._mine, "depth", 0)
        if depth == 0:
            with self._changed:
                # A set-up that is waiting goes first, or it would wait for ever
                self._wait(lambda: not (self._setting_up or self._waiting_to_set_up), "a network")
                self._working += 1
        self._mine.depth = depth + 1
        try:
            yield
        finally:
            self._mine.depth = depth
            if depth == 0:
                with self._changed:
                    self._working -= 1
                    self._changed.notify_all()

    def __enter__(self) -> "_GpuGate":
        if getattr(self._mine, "depth", 0):
            raise RuntimeError("an engine set up inside a network run would wait for itself")
        with self._changed:
            self._waiting_to_set_up += 1
            try:
                self._wait(lambda: not (self._setting_up or self._working), "an engine set-up")
            finally:
                self._waiting_to_set_up -= 1
                self._changed.notify_all()
            self._setting_up = True
        return self

    def __exit__(self, *_: object) -> None:
        with self._changed:
            self._setting_up = False
            self._changed.notify_all()


GPU_SETUP_LOCK = _GpuGate()
