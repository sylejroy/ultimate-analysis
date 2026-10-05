"""Filling the model dropdowns."""

from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import QComboBox

from ...utils.model_files import (
    find_detection_models,
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
