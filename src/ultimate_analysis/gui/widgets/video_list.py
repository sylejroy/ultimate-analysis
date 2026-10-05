"""List of the available videos."""

from pathlib import Path
from typing import List

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QListWidget, QListWidgetItem

from ...utils.video import find_video_files, get_video_duration


class VideoListWidget(QListWidget):
    """List of the videos found in the data folders."""

    def __init__(self, show_duration: bool = True):
        super().__init__()
        self._show_duration = show_duration
        self.video_files: List[str] = []

        # Names differ at their end (the clip number): shorten them in the middle
        self.setTextElideMode(Qt.ElideMiddle)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.setStyleSheet(
            """
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #444;
            }
            QListWidget::item:selected {
                background-color: #2a2a2a;
            }
        """
        )

    def reload(self) -> List[str]:
        """Search the data folders again and list what is found; returns the paths."""
        self.video_files = find_video_files()

        # Repopulating must not look like the user picking a video
        self.blockSignals(True)
        self.clear()
        for video_path in self.video_files:
            text = Path(video_path).name
            if self._show_duration:
                text = f"{text} ({get_video_duration(video_path)})"
            item = QListWidgetItem(text)
            item.setToolTip(video_path)
            self.addItem(item)
        self.setCurrentRow(-1)
        self.blockSignals(False)
        return self.video_files
