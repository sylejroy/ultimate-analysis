"""The Labelling tab: boxes of players and discs, or the lines and marks of the field."""

from typing import Optional

from PyQt5.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from .labelling_tab import LabellingTab


class LabellingModes(QWidget):
    """Both kinds of labelling, each under its own heading.

    The field labelling is built when it is first opened: it is not needed to label boxes.
    """

    def __init__(self):
        super().__init__()
        self.boxes = LabellingTab()
        self.field: Optional[QWidget] = None

        self.modes = QTabWidget()
        self.modes.setDocumentMode(True)
        self.modes.addTab(self.boxes, "Players and discs")
        self._field_holder = QWidget()
        QVBoxLayout(self._field_holder).setContentsMargins(0, 0, 0, 0)
        self.modes.addTab(self._field_holder, "Field lines")
        self.modes.currentChanged.connect(self._on_mode_changed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.modes)

    def _on_mode_changed(self, index: int) -> None:
        if index == 1 and self.field is None:
            from .field_labelling import FieldLabellingWidget

            self.field = FieldLabellingWidget()
            self._field_holder.layout().addWidget(self.field)
