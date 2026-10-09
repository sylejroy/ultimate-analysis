"""The players of the video and what each of them did, as a table."""

from typing import List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor, QFont
from PyQt5.QtWidgets import QAbstractItemView, QHeaderView, QTableWidget, QTableWidgetItem

from ..theme import FAINT_TEXT

# A player seen for less than this many seconds is not listed: a track that was never
# matched to anybody for long
MIN_SECONDS_SEEN = 3.0


def clock(seconds: float) -> str:
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


class PlayerStatsTable(QTableWidget):
    """A row per player: number, time on the field, distance run, thrown and received.

    Players are listed by team, those whose number is known first. A player whose number
    was never read has a dot for a number: they are known by their looks alone.
    """

    # Heading, what it says in full, and how its cell is made from a row of the roster
    COLUMNS = (
        ("#", "Jersey number, in the colour of the team", None),
        ("Field", "Time seen on the field (minutes:seconds)", None),
        ("Run", "Distance run, in {unit}", "distance_run"),
        (
            "Throw",
            "Length of the passes thrown that a teammate caught, in {unit}",
            "distance_thrown",
        ),
        ("Catch", "Length of the passes caught, in {unit}", "distance_received"),
        ("Disc", "Times the player had the disc", "times_with_disc"),
    )

    def __init__(self, unit: str = "yd", parent=None):
        super().__init__(0, len(self.COLUMNS), parent)
        self.setHorizontalHeaderLabels([heading for heading, _, _ in self.COLUMNS])
        for column, (_, tip, _) in enumerate(self.COLUMNS):
            self.horizontalHeaderItem(column).setToolTip(tip.format(unit=unit))
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(22)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.NoSelection)
        self.setFocusPolicy(Qt.NoFocus)
        self.setShowGrid(False)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setMinimumHeight(260)
        self.setToolTip(
            f"What each player did so far, distances in {unit}. A dot for a number: it was "
            "never read, and the player is known by their looks. Kept per video."
        )

    @staticmethod
    def listed(rows: List[dict]) -> List[dict]:
        """The rows worth showing, in the order they are shown in."""
        seen = [row for row in rows if row.get("seconds_seen", 0.0) >= MIN_SECONDS_SEEN]

        def order(row: dict):
            number = str(row.get("number") or "")
            return (
                row.get("team", 0),
                not number,
                int(number) if number.isdigit() else 0,
                -row.get("seconds_seen", 0.0),
            )

        return sorted(seen, key=order)

    def show_players(self, rows: Optional[List[dict]]) -> None:
        """Fill the table from the roster's rows (AnalysisPipeline.player_stats)."""
        rows = self.listed(rows or [])
        self.setRowCount(len(rows))
        bold = QFont(self.font())
        bold.setBold(True)
        for index, row in enumerate(rows):
            number = str(row.get("number") or "")
            cells = [number or "·", clock(row.get("seconds_seen", 0.0))]
            for _, _, key in self.COLUMNS[2:]:
                value = row.get(key, 0.0)
                cells.append(f"{value:.0f}" if value else "")
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(
                    Qt.AlignCenter if column == 0 else Qt.AlignRight | Qt.AlignVCenter
                )
                if column == 0:
                    item.setFont(bold)
                    colour = row.get("colour")
                    if colour is not None:
                        blue, green, red = (int(part) for part in colour)
                        # A dark team is drawn dark on the field; here it has to be read
                        if max(blue, green, red) < 90:
                            blue, green, red = 150, 150, 150
                        item.setForeground(QBrush(QColor(red, green, blue)))
                elif not number:
                    item.setForeground(QBrush(QColor(FAINT_TEXT)))
                self.setItem(index, column, item)
