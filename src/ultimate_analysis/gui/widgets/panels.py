"""Building blocks all tabs lay their controls out with."""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QComboBox, QFrame, QGroupBox, QScrollArea, QWidget

# Width of the column of controls at the left of a tab
PANEL_WIDTH = 360


def side_panel(content: QWidget, width: int = PANEL_WIDTH) -> QScrollArea:
    """A column of controls of fixed width that scrolls when the window is too low for it.

    Without the scrolling, the controls are squeezed until their text no longer fits.
    """
    area = QScrollArea()
    area.setWidget(content)
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    area.setFixedWidth(width)
    # A scroll area paints its own light background otherwise
    area.viewport().setAutoFillBackground(False)
    content.setAutoFillBackground(False)
    return area


def collapsible(group: QGroupBox, expanded: bool) -> QGroupBox:
    """Give a group a tick box in its title that shows and hides its contents."""

    def show_contents(visible: bool) -> None:
        for child in group.findChildren(QWidget, options=Qt.FindDirectChildrenOnly):
            child.setVisible(visible)

    group.setCheckable(True)
    group.setChecked(expanded)
    group.toggled.connect(show_contents)
    show_contents(expanded)
    return group


def compact_combo(combo: QComboBox) -> QComboBox:
    """Keep a dropdown as narrow as its column, whatever the length of its entries."""
    combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(12)
    return combo
