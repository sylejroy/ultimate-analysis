"""Filling the model dropdowns."""

import re
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QComboBox

from ...utils.model_files import (
    find_detection_models,
    find_holder_models,
    find_reid_models,
    find_segmentation_models,
    model_display_name,
    models_root,
)


def populate_detection_model_combo(
    combo: QComboBox, target_class: str, default_model: Optional[str] = None
) -> None:
    """List the detection runs for a class and select the default, without emitting signals.

    The item text is the weights path relative to the models folder.
    """
    combo.blockSignals(True)
    combo.clear()
    combo.addItems(find_detection_models(target_class))
    if default_model:
        try:
            default_text = str(Path(default_model).relative_to(models_root()))
        except ValueError:
            default_text = str(Path(default_model))
        index = combo.findText(default_text)
        if index >= 0:
            combo.setCurrentIndex(index)
    combo.blockSignals(False)


def populate_reid_model_combo(combo: QComboBox, default_model: Optional[str] = None) -> None:
    """List the networks that know players by their looks and select the default, without
    emitting signals.

    The item text is the run's name without what all of them share; the item data is the
    weights path, and the whole name shows when the mouse rests on an entry.
    """
    combo.blockSignals(True)
    combo.clear()
    for weights in find_reid_models():
        run = Path(weights).parent.name
        combo.addItem(reid_display_name(run), weights)
        combo.setItemData(combo.count() - 1, run, Qt.ToolTipRole)
        if default_model and Path(weights) == Path(default_model):
            combo.setCurrentIndex(combo.count() - 1)
    combo.blockSignals(False)


def populate_holder_model_combo(combo: QComboBox, default_model: Optional[str] = None) -> None:
    """List the networks that tell a holder from a picture of a player, the newest
    first, after "Off", and select the default, without emitting signals.

    The item data is the weights path, None for "Off".
    """
    combo.blockSignals(True)
    combo.clear()
    combo.addItem("Off", None)
    for weights in find_holder_models():
        run = Path(weights).parent.name
        combo.addItem(holder_display_name(run), weights)
        combo.setItemData(combo.count() - 1, run, Qt.ToolTipRole)
        if default_model and Path(weights) == Path(default_model):
            combo.setCurrentIndex(combo.count() - 1)
    combo.blockSignals(False)


def holder_display_name(run: str) -> str:
    """A run of scripts/train_holder.py by its day and what sets it apart: the runs are
    called <date>_holder_<network>_<dataset>_<name>."""
    found = re.fullmatch(r"\d{4}(\d{2})(\d{2})_holder_(.+)", run)
    if not found:
        return run
    month, day, name = found.groups()
    name = re.sub(r"_holder_[a-z]+_v\d+", "", name)
    return f"{int(day)}.{int(month)}. {name}"


def reid_display_name(run: str) -> str:
    """A run of scripts/train_reid.py by its day and what sets it apart: the runs are
    called <date>_reid_<network and options>_<dataset>."""
    found = re.fullmatch(r"\d{4}(\d{2})(\d{2})_reid_(.+?)(_reid_players_v\d+)?", run)
    if not found:
        return run
    month, day, name, _ = found.groups()
    return f"{int(day)}.{int(month)}. {name}"


def populate_segmentation_model_combo(
    combo: QComboBox, default_model: Optional[str] = None
) -> None:
    """List the field segmentation runs and select the default, without emitting signals.

    The item text is the run name; the item data is the weights path.
    """
    combo.blockSignals(True)
    combo.clear()
    for weights in find_segmentation_models():
        combo.addItem(model_display_name(weights), weights)
        # Compared as paths: the setting uses forward slashes, Windows paths do not
        if default_model and Path(weights) == Path(default_model):
            combo.setCurrentIndex(combo.count() - 1)
    combo.blockSignals(False)
