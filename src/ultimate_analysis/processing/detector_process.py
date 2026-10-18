"""The player and disc detectors in a process of their own.

The pipeline does much of its work in Python: following the players, possession,
drawing. So do the detectors around their networks, and Python lets one thread at a
time work in a process. With the detectors in another process they no longer take
turns with the rest (`models.inference.own_process`).

The other process loads the models itself and keeps what the detectors remember from
frame to frame (where the disc was). The frame is handed over in memory both processes
see, so it is not copied through a pipe; the detections come back as they are. The
models chosen, the thresholds and a reset are sent along with every frame, so the two
processes cannot come apart.

If the other process cannot be started or dies, that is shown on the frame. The
detectors then run in this process, but only once this process has loaded their
engines when a video was opened: an engine first loaded after frames have been analysed
crashes the app (docs/MEASUREMENTS.md, "Engines are loaded with the video"). Until the
video is opened again no players or discs are found.
"""

import atexit
import importlib
import threading
from multiprocessing import get_context, shared_memory
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config.settings import get_config, get_setting
from ..utils.logger import get_logger
from . import health

logger = get_logger("DETECTOR_PROCESS")

# The settings the detectors read, sent with every frame (parts of `models`)
SENT_SETTINGS = ("inference", "player_detection", "disc_detection", "disc_fusion")
# How long the other process may take to load its models, and to answer for a frame
# (seconds): longer, and it is given up
START_SECONDS = 180.0
ANSWER_SECONDS = 30.0
ALIVE_SECONDS = 0.25  # While waiting, this often it is looked at whether it still runs
# What runs in the other process: module and function, called with a frame and what
# was sent with it
DETECT = "ultimate_analysis.processing.detector_process:_detect"


def is_enabled() -> bool:
    """Whether the detectors are to run in a process of their own."""
    return bool(get_setting("models.inference.own_process", False))


def in_own_process() -> bool:
    """Whether the detectors run in their own process: switched on, and not given up."""
    return is_enabled() and not (_detectors is not None and _detectors.failed)


def _detect(frame: np.ndarray, sent: Dict[str, Any], state: Dict[str, Any]) -> List[dict]:
    """In the other process: bring models, settings and memory in step, then detect."""
    from . import inference

    models = get_config().setdefault("models", {})
    for name, settings in sent["settings"].items():
        models[name] = settings
    player, disc = sent["models"]
    if player and player != inference.current_model_paths()[0]:
        inference.set_player_model(player)
    if disc and disc != inference.current_model_paths()[1]:
        inference.set_disc_model(disc)
    if sent["resets"] != state.get("resets"):
        inference.reset_inference_state()
        state["resets"] = sent["resets"]
    return inference.run_inference(
        frame,
        weak_players_from=sent["weak_players_from"],
        weak_discs_from=sent["weak_discs_from"],
    )


def _echo(frame: np.ndarray, sent: Dict[str, Any], state: Dict[str, Any]) -> List[dict]:
    """For the tests: says what arrived, without any model."""
    state["frames"] = state.get("frames", 0) + 1
    return [
        {
            "shape": tuple(frame.shape),
            "sum": int(frame.sum()),
            "sent": sent["weak_discs_from"],
            "frames": state["frames"],
        }
    ]


def _serve(connection: Any, detect: str) -> None:
    """The other process: answer for frames until told to stop."""
    module, function = detect.split(":")
    work = getattr(importlib.import_module(module), function)
    state: Dict[str, Any] = {}
    memory: Optional[shared_memory.SharedMemory] = None
    connection.send(("ready", None))
    try:
        while True:
            kind, given = connection.recv()
            if kind == "stop":
                break
            try:
                name, shape, sent = given
                if memory is None or memory.name != name:
                    if memory is not None:
                        memory.close()
                    memory = shared_memory.SharedMemory(name=name)
                frame = np.ndarray(shape, dtype=np.uint8, buffer=memory.buf)
                connection.send(("found", work(frame, sent, state)))
            except Exception as error:  # Said to the asking process, which decides
                connection.send(("failed", f"{type(error).__name__}: {error}"))
    except (EOFError, KeyboardInterrupt):
        pass  # The asking process is gone
    finally:
        if memory is not None:
            memory.close()


class DetectorProcess:
    """The other process and the way to it. One frame at a time, whichever thread asks."""

    def __init__(self, detect: str = DETECT, serve: Any = None):
        self._detect = detect
        self._serve = serve or _serve  # What the other process runs
        self._lock = threading.Lock()
        self._process: Any = None
        self._connection: Any = None
        self._memory: Optional[shared_memory.SharedMemory] = None
        self._ready = False
        self.failed = False  # Given up: the detectors run in the asking process

    def start(self) -> None:
        """Begin loading in the other process; does not wait for it."""
        with self._lock:
            self._begin()

    def _begin(self) -> None:
        if self._process is not None or self.failed:
            return
        try:
            # A new interpreter, as on Windows everywhere: nothing of this process's
            # models or windows goes along
            context = get_context("spawn")
            self._connection, theirs = context.Pipe()
            self._process = context.Process(
                target=self._serve, args=(theirs, self._detect), daemon=True, name="detectors"
            )
            self._process.start()
            theirs.close()
            logger.info("The detectors are starting in a process of their own")
        except Exception as error:
            self._give_up(f"could not be started: {error}")

    def _give_up(self, why: str) -> None:
        logger.error(f"The detectors' own process {why}; they run in this one from now on")
        self.failed = True
        self._end()

    def _answer(self, seconds: float) -> Tuple[str, Any]:
        waited = 0.0
        while not self._connection.poll(ALIVE_SECONDS):
            waited += ALIVE_SECONDS
            if not self._process.is_alive():
                raise RuntimeError("the process has ended")
            if waited >= seconds:
                raise TimeoutError(f"no answer in {seconds:.0f} s")
        return self._connection.recv()

    def detect(self, frame: np.ndarray, sent: Dict[str, Any]) -> Optional[List[dict]]:
        """What the other process finds in a frame; None if it cannot be asked (the
        caller then detects itself)."""
        with self._lock:
            self._begin()
            if self.failed:
                return None
            try:
                if not self._ready:
                    self._answer(START_SECONDS)
                    self._ready = True
                frame = np.ascontiguousarray(frame, dtype=np.uint8)
                if self._memory is None or self._memory.size < frame.nbytes:
                    if self._memory is not None:
                        self._memory.close()
                        self._memory.unlink()
                    self._memory = shared_memory.SharedMemory(create=True, size=frame.nbytes)
                np.ndarray(frame.shape, dtype=np.uint8, buffer=self._memory.buf)[:] = frame
                self._connection.send(("detect", (self._memory.name, frame.shape, sent)))
                kind, found = self._answer(ANSWER_SECONDS)
            except Exception as error:
                self._give_up(f"does not answer ({type(error).__name__}: {error})")
                return None
            if kind != "found":
                self._give_up(f"failed ({found})")
                return None
            return found

    def close(self) -> None:
        """End the other process."""
        with self._lock:
            self._end()

    def _end(self) -> None:
        if self._connection is not None:
            try:
                self._connection.send(("stop", None))
            except (OSError, ValueError):
                pass
            self._connection.close()
        if self._process is not None:
            self._process.join(2.0)
            if self._process.is_alive():
                self._process.terminate()
        if self._memory is not None:
            self._memory.close()
            try:
                self._memory.unlink()
            except FileNotFoundError:
                pass
        self._process = self._connection = self._memory = None
        self._ready = False


_detectors: Optional[DetectorProcess] = None
_starting = threading.Lock()
# Whether this process loaded the detectors' engines when the video was opened, so that
# it may detect itself (`ready_here`)
_loaded_here = False


def ready_here() -> None:
    """This process has loaded the detectors and their engines for the video opened."""
    global _loaded_here
    _loaded_here = True


def _the_process() -> DetectorProcess:
    global _detectors
    with _starting:
        if _detectors is None:
            _detectors = DetectorProcess()
            atexit.register(_detectors.close)
        return _detectors


def start() -> None:
    """Begin loading the detectors in their own process, if that is switched on: by
    the time the first frame comes they are ready."""
    if is_enabled():
        _the_process().start()


def detect(
    frame: np.ndarray, weak_players_from: float = 0.0, weak_discs_from: float = 0.0
) -> List[Dict[str, Any]]:
    """The players and discs of a frame, as `inference.run_inference` gives them: from
    the detectors' own process if that is switched on and works, from this one otherwise."""
    from . import inference

    if is_enabled() and not _loaded_here:
        models = get_config().get("models", {})
        found = _the_process().detect(
            frame,
            {
                "settings": {name: models.get(name, {}) for name in SENT_SETTINGS},
                "models": inference.current_model_paths(),
                "resets": inference.reset_count(),
                "weak_players_from": float(weak_players_from),
                "weak_discs_from": float(weak_discs_from),
            },
        )
        if found is not None:
            return found
        health.report(
            "Detection",
            "the detectors' own process stopped; open the video again (see the log)",
        )
        return []
    return inference.run_inference(
        frame, weak_players_from=weak_players_from, weak_discs_from=weak_discs_from
    )
