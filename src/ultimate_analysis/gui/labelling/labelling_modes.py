"""The Labelling tab: boxes of players and discs, the lines and marks of the field, and
whether two tracks are one player, and who holds the disc: by a yes or no for one
player, or by a click on the holder after the second before the frame was played."""

from typing import Callable, Dict, Optional

from PyQt5.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from .labelling_tab import LabellingTab


def _field() -> QWidget:
    from .field_labelling import FieldLabellingWidget

    return FieldLabellingWidget()


def _same_player() -> QWidget:
    from .judgement_tabs import SamePlayerWidget

    return SamePlayerWidget()


def _holder() -> QWidget:
    from .judgement_tabs import HolderWidget

    return HolderWidget()


def _holder_click() -> QWidget:
    from .holder_click_tab import HolderClickWidget

    return HolderClickWidget()


class LabellingModes(QWidget):
    """The kinds of labelling, each under its own heading.

    All but the boxes are built when they are first opened: they are not needed to label
    boxes.
    """

    def __init__(self):
        super().__init__()
        self.boxes = LabellingTab()
        self.field: Optional[QWidget] = None

        self.modes = QTabWidget()
        self.modes.setDocumentMode(True)
        self.modes.addTab(self.boxes, "Players and discs")
        # Page -> what builds it, until it is opened
        self._to_build: Dict[int, Callable[[], QWidget]] = {}
        for title, build in (
            ("Field lines", _field),
            ("Same player?", _same_player),
            ("Holding the disc?", _holder),
            ("Click the holder", _holder_click),
        ):
            holder = QWidget()
            QVBoxLayout(holder).setContentsMargins(0, 0, 0, 0)
            self._to_build[self.modes.addTab(holder, title)] = build
        self.modes.currentChanged.connect(self._on_mode_changed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.modes)

    def _on_mode_changed(self, index: int) -> None:
        build = self._to_build.pop(index, None)
        if build is None:
            return
        page = build()
        if build is _field:
            self.field = page
        self.modes.widget(index).layout().addWidget(page)
