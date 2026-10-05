"""Building a form of parameter controls from a description.

A form is a list of Parameter, Section, and Note items. build_form creates one control per
Parameter, stores it on the owning widget under Parameter.attr, and adds it to the layout.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

from PyQt5.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QWidget,
)


@dataclass(frozen=True)
class Parameter:
    """One editable parameter."""

    key: str  # Name in the parameter dictionary
    attr: str  # Attribute of the owner the control is stored under
    label: str
    kind: str  # "float", "int", "bool", or "text"
    minimum: float = 0
    maximum: float = 0
    step: Optional[float] = None
    decimals: Optional[int] = None
    tooltip: str = ""
    placeholder: str = ""


@dataclass(frozen=True)
class Section:
    """A heading between groups of parameters."""

    title: str


@dataclass(frozen=True)
class Note:
    """A line of small explanatory text."""

    text: str


FormItem = Union[Parameter, Section, Note]


def control_names(items: List[FormItem]) -> Dict[str, str]:
    """Parameter key -> attribute of its control, for the parameters of a form."""
    return {item.key: item.attr for item in items if isinstance(item, Parameter)}


def build_form(
    owner: QWidget,
    layout: QFormLayout,
    items: List[FormItem],
    values: Dict[str, Any],
    on_change: Callable[[], None],
    make_header: Callable[[str], QWidget],
) -> None:
    """Create the controls of a form and add them to a layout.

    Args:
        owner: Widget that gets each control as an attribute
        layout: Form layout to fill
        items: The form description
        values: Initial value per parameter key
        on_change: Called whenever a control changes
        make_header: Creates the widget for a Section title
    """
    for item in items:
        if isinstance(item, Section):
            layout.addRow(make_header(item.title))
        elif isinstance(item, Note):
            note = QLabel(item.text)
            note.setStyleSheet("color: #cccccc; font-size: 10px;")
            layout.addRow("", note)
        else:
            control = _make_control(item, values[item.key], on_change)
            setattr(owner, item.attr, control)
            layout.addRow(item.label, control)


def _make_control(parameter: Parameter, value: Any, on_change: Callable[[], None]) -> QWidget:
    if parameter.kind == "bool":
        control = QCheckBox()
        control.setChecked(value)
        control.stateChanged.connect(on_change)
    elif parameter.kind == "text":
        control = QLineEdit()
        if value:
            control.setText(value)
        control.textChanged.connect(on_change)
        control.setPlaceholderText(parameter.placeholder)
    else:
        control = QDoubleSpinBox() if parameter.kind == "float" else QSpinBox()
        control.setRange(parameter.minimum, parameter.maximum)
        if parameter.step is not None:
            control.setSingleStep(parameter.step)
        if parameter.decimals is not None:
            control.setDecimals(parameter.decimals)
        control.setValue(value)
        control.valueChanged.connect(on_change)
    if parameter.tooltip:
        control.setToolTip(parameter.tooltip)
    return control


def read_control(control: QWidget) -> Any:
    """Value of a parameter control."""
    if isinstance(control, QCheckBox):
        return control.isChecked()
    if isinstance(control, QLineEdit):
        return control.text()
    return control.value()


def write_control(control: QWidget, value: Any) -> None:
    """Show a value in a parameter control."""
    if isinstance(control, QCheckBox):
        control.setChecked(value)
    elif isinstance(control, QLineEdit):
        control.setText(str(value) if value else "")
    else:
        control.setValue(value)
