"""Runs video decoding and the analysis pipeline on a background thread.

The worker owns the video reader and the pipeline; nothing else touches them. The main
tab sends it requests through queued signals, which run one after another in the order
they were sent, and receives the finished frames the same way. The window therefore stays
responsive while frames are analysed and while models load.
"""

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from ...config.settings import get_setting
from ...pipeline import AnalysisPipeline, FrameResult, PipelineOptions
from ...processing.field_segmentation import set_field_model, warmup_field_model
from ...processing.inference import (
    reset_inference_state,
    run_inference,
    set_disc_model,
    set_player_model,
    warmup_models,
)
from ...processing.model_lock import MODEL_LOCK
from ...processing.player_id import initialize_player_id_system, set_player_id_method
from ...utils.logger import get_logger
from ...utils.video import VideoPlayer

logger = get_logger("PIPELINE_WORKER")


@dataclass
class ProcessedFrame:
    """Answer to a frame request."""

    result: Optional[FrameResult]  # None if there was no frame (end of video) or it failed
    mode: str  # "next" or "current", as requested
    generation: int  # As requested; lets the GUI drop frames that became stale
    video_position: int  # Frame index the reader is at after this request


class PipelineWorker(QObject):
    """Video reader and analysis pipeline, living on the worker thread."""

    frame_processed = pyqtSignal(object)  # ProcessedFrame
    video_loaded = pyqtSignal(object)  # Video info dict; info["loaded"] is False on failure

    def __init__(self):
        super().__init__()
        self.pipeline = AnalysisPipeline()
        self.video_player = VideoPlayer()

    # ------------------------------------------------------------------ requests

    @pyqtSlot(str, object, object)
    def load_video(
        self, video_path: str, options: PipelineOptions, homography_matrix: Optional[np.ndarray]
    ) -> None:
        """Open a video, forget the previous one, and prepare the models for it."""
        info: Dict[str, Any] = {"loaded": False, "path": video_path}
        try:
            with MODEL_LOCK:
                self.pipeline.new_video()
                self.pipeline.homography_matrix = homography_matrix

                if self.video_player.load_video(video_path):
                    info = self.video_player.get_video_info()
                    self.pipeline.set_frame_rate(info.get("fps", 0))
                    # Loading models here keeps the first frame from stalling playback
                    if options.player_id:
                        initialize_player_id_system()
                    if options.detection and get_setting("models.inference.warmup_on_load", True):
                        warmup_models()
                    # The engines of all three models are loaded here, one after the
                    # other, whatever is switched on: an engine first loaded after frames
                    # have been analysed crashes the process (an access violation inside
                    # TensorRT, seen only in the app)
                    shape = (info.get("height") or 1080, info.get("width") or 1920, 3)
                    run_inference(np.zeros(shape, dtype=np.uint8))
                    reset_inference_state()
                    warmup_field_model(shape)
        except Exception:
            logger.exception(f"Error loading video {video_path}")
        self.video_loaded.emit(info)

    @pyqtSlot(str, object, int)
    def process_frame(self, mode: str, options: PipelineOptions, generation: int) -> None:
        """Analyse the next frame ("next") or the one at the current position ("current")."""
        result = None
        try:
            start = time.perf_counter()
            frame_index = self.video_player.current_frame_idx
            if mode == "next":
                frame = self.video_player.get_next_frame()
            else:
                frame = self.video_player.get_current_frame()
            io_ms = (time.perf_counter() - start) * 1000

            if frame is not None:
                with MODEL_LOCK:
                    result = self.pipeline.process(frame, frame_index, options)
                total_ms = (time.perf_counter() - start) * 1000
                result.timings = {"Frame I/O": io_ms, **result.timings, "Total Runtime": total_ms}
        except Exception:
            logger.exception("Error processing frame")
            result = None
        self.frame_processed.emit(
            ProcessedFrame(result, mode, generation, self.video_player.current_frame_idx)
        )

    @pyqtSlot(object)
    def execute(self, command: Callable[["PipelineWorker"], None]) -> None:
        """Run a command on the worker thread, in order with the other requests."""
        try:
            with MODEL_LOCK:
                command(self)
        except Exception:
            logger.exception("Error executing worker command")

    # ------------------------------------------------------------------ commands

    def seek(self, frame_index: int) -> None:
        """Move to a frame; tracking starts over from there."""
        if self.video_player.seek_to_frame(frame_index):
            self.pipeline.reset()

    def set_player_model(self, model_path: str) -> None:
        if set_player_model(model_path):
            self.pipeline.reset()

    def set_disc_model(self, model_path: str) -> None:
        if set_disc_model(model_path):
            self.pipeline.reset()

    def set_field_model(self, model_path: str) -> None:
        if set_field_model(model_path):
            self.pipeline.invalidate()

    def set_player_id_method(self, method: str) -> None:
        """Switch the jersey number reader; numbers from the previous reader are dropped."""
        set_player_id_method(method)
        self.pipeline.reset_player_ids()

    def shutdown(self) -> None:
        """Release the video and stop the worker thread's event loop."""
        self.video_player.close_video()
        self.thread().quit()
