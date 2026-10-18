"""The per-frame analysis pipeline: one frame in, results and the drawn views out.

There is no Qt in this package: the GUI owns one AnalysisPipeline and runs it on a
worker thread.

- analysis.py: the stages a frame goes through, and what they keep between frames
- field.py: where the field is, from the field model's masks, beside the frames
- views.py: the camera view and the top-down view drawn from a frame's results
"""

from .analysis import AnalysisPipeline, FrameResult, PipelineOptions

__all__ = ["AnalysisPipeline", "FrameResult", "PipelineOptions"]
