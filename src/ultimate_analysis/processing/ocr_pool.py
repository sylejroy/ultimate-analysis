"""Persistent OCR process pool management for EasyOCR jersey number decoding.

This module centralizes creation and reuse of a ProcessPoolExecutor used for
parallel OCR on player crops. It avoids repeated process + model initialization
per frame.

Design goals:
- Lazy initialization (created on first use)
- Cached per-process EasyOCR reader (created once inside worker process)
- GPU-aware: limit workers to 1 if GPU=True to avoid VRAM duplication
- Safe shutdown via atexit
- Optional warm-up to amortize first-call penalty

Public API:
- submit_ocr_tasks(tasks, ocr_params, readtext_params, timeout) -> generator of results
- get_pool_info() -> diagnostic dict
- shutdown_pool()

Worker contract:
Input: (index, encoded_crop_bytes, shape, dtype, ocr_params, readtext_params)
Output: (index, best_text, best_confidence, raw_results[:detail_limit])

We transfer crops as JPEG-encoded bytes to reduce pickling overhead.
"""

from __future__ import annotations

import atexit
import os
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import easyocr  # type: ignore

    EASYOCR_AVAILABLE = True
except ImportError:  # pragma: no cover
    EASYOCR_AVAILABLE = False

# Global executor reference
_EXECUTOR: Optional[ProcessPoolExecutor] = None
_MAX_WORKERS: int = 0

# Environment variable toggles (optional future extension)
ENV_FORCE_CPU = os.environ.get("UA_OCR_FORCE_CPU", "0") == "1"


def _init_executor(max_workers: int) -> ProcessPoolExecutor:
    global _EXECUTOR, _MAX_WORKERS
    if _EXECUTOR is None:
        _MAX_WORKERS = max_workers
        # 'spawn' is default on Windows; explicitly fine.
        _EXECUTOR = ProcessPoolExecutor(max_workers=max_workers)
    return _EXECUTOR


def get_pool_info() -> Dict[str, int]:
    return {"active": int(_EXECUTOR is not None), "max_workers": _MAX_WORKERS}


def shutdown_pool() -> None:
    global _EXECUTOR
    if _EXECUTOR is not None:
        _EXECUTOR.shutdown(wait=False, cancel_futures=True)
        _EXECUTOR = None


atexit.register(shutdown_pool)


# ---------------- Worker logic ---------------- #

# Per-process cached reader
_WORKER_READER = None  # type: ignore


def _init_worker_reader(ocr_params: Dict) -> None:
    global _WORKER_READER
    if _WORKER_READER is not None:
        return
    if not EASYOCR_AVAILABLE:
        return
    languages = ["en"]
    gpu = bool(ocr_params.get("gpu", True)) and not ENV_FORCE_CPU
    # Limit GPU duplication: if gpu True, we assume single worker externally
    _WORKER_READER = easyocr.Reader(languages, gpu=gpu, verbose=False)


def _decode_crop(encoded_bytes: bytes, shape: Tuple[int, int, int], dtype: str) -> np.ndarray:
    # Decode JPEG bytes back to BGR numpy array
    arr = np.frombuffer(encoded_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def _worker_ocr_task(payload):
    """Worker entry point.

    Payload structure:
      (index, encoded_bytes, shape, dtype, ocr_params, readtext_params, detail_limit)
    Returns:
      (index, best_text|"", best_confidence, raw_results_subset)
    """
    (index, encoded_bytes, shape, dtype, ocr_params, readtext_params, detail_limit) = payload
    try:
        if not EASYOCR_AVAILABLE:
            return index, "", 0.0, []
        _init_worker_reader(ocr_params)
        crop = _decode_crop(encoded_bytes, shape, dtype)
        reader = _WORKER_READER
        if reader is None:
            return index, "", 0.0, []
        raw_results = reader.readtext(crop, **readtext_params)
        # Filter and pick best numeric result here to minimize payload size
        min_conf = 0.5
        filtered = []
        best_text = ""
        best_conf = 0.0
        for bbox, text, conf in raw_results:
            if conf >= min_conf:
                # Extract digits only
                digits = "".join(ch for ch in text if ch.isdigit())
                if digits:
                    filtered.append((bbox, digits, float(conf)))
                    if conf > best_conf:
                        best_conf = float(conf)
                        best_text = digits
        if detail_limit > 0 and len(filtered) > detail_limit:
            filtered = filtered[:detail_limit]
        return index, best_text, best_conf, filtered
    except Exception:  # pragma: no cover
        return index, "", 0.0, []


# ---------------- Submission API ---------------- #


def submit_ocr_tasks(
    crops: List[np.ndarray],
    ocr_params: Dict,
    readtext_params: Dict,
    max_workers: int,
    timeout: float = 3.0,
    detail_limit: int = 5,
) -> List[Tuple[int, str, float, List]]:
    """Submit multiple crops for parallel OCR and gather results.

    Returns ordered list aligned with input crop indices.
    """
    if not crops:
        return []

    if not EASYOCR_AVAILABLE:
        return [(i, "", 0.0, []) for i in range(len(crops))]

    # GPU-aware worker reduction
    gpu = bool(ocr_params.get("gpu", True)) and not ENV_FORCE_CPU
    if gpu:
        effective_workers = 1  # avoid multi GPU reader duplicates
    else:
        effective_workers = max(1, min(max_workers, len(crops)))

    executor = _init_executor(effective_workers)

    # Encode crops to JPEG to shrink transfer size (quality tradeoff acceptable)
    tasks = []
    for idx, crop in enumerate(crops):
        # Ensure color
        if crop is None or crop.size == 0:
            tasks.append(None)
            continue
        success, enc = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not success:
            tasks.append(None)
            continue
        tasks.append(
            (
                idx,
                enc.tobytes(),
                crop.shape,
                str(crop.dtype),
                ocr_params,
                readtext_params,
                detail_limit,
            )
        )

    future_map: Dict[Future, int] = {}
    results_partial: Dict[int, Tuple[int, str, float, List]] = {}

    for payload in tasks:
        if payload is None:
            continue
        fut = executor.submit(_worker_ocr_task, payload)
        future_map[fut] = payload[0]

    for fut in as_completed(future_map, timeout=timeout):
        idx = future_map[fut]
        try:
            res = fut.result(timeout=max(0.1, timeout * 0.9))
            # res: (index, best_text, best_conf, filtered)
            results_partial[idx] = (res[0], res[1], res[2], res[3])
        except Exception:
            results_partial[idx] = (idx, "", 0.0, [])

    # Fill any missing ones (timeout or failure)
    ordered: List[Tuple[int, str, float, List]] = []
    for i in range(len(crops)):
        ordered.append(results_partial.get(i, (i, "", 0.0, [])))
    return ordered


__all__ = [
    "submit_ocr_tasks",
    "shutdown_pool",
    "get_pool_info",
]
