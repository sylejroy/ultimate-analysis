"""Load processing modules without initializing GUI or downloading model weights."""

import importlib
import sys
import types
from pathlib import Path
from unittest.mock import Mock

SOURCE = Path(__file__).resolve().parents[1] / "src" / "ultimate_analysis"
sys.path.insert(0, str(SOURCE.parent))


def load_module(name):
    for suffix in ("", ".processing", ".gui"):
        package_name = "_runtime_checks" + suffix
        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            package.__path__ = [str(SOURCE / suffix.lstrip("."))]
            sys.modules[package_name] = package
    yolo = types.ModuleType("ultralytics")
    yolo.YOLO = Mock()
    deepsort = types.ModuleType("deep_sort_realtime.deepsort_tracker")
    deepsort.DeepSort = Mock()
    # The tracking module subclasses these; a class of its own each, nothing of ByteTrack
    trackers = types.ModuleType("ultralytics.trackers")
    byte_tracker = types.ModuleType("ultralytics.trackers.byte_tracker")
    byte_tracker.BYTETracker = type("BYTETracker", (), {"__init__": lambda self, args: None})
    byte_tracker.STrack = type("STrack", (), {})
    replacements = {
        "ultralytics": yolo,
        "ultralytics.trackers": trackers,
        "ultralytics.trackers.byte_tracker": byte_tracker,
        "deep_sort_realtime.deepsort_tracker": deepsort,
    }
    previous = {key: sys.modules.get(key) for key in replacements}
    sys.modules.update(replacements)
    try:
        return importlib.import_module("_runtime_checks." + name)
    finally:
        # Keep newly imported native modules loaded; OpenCV cannot safely be reimported.
        for key, value in previous.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
