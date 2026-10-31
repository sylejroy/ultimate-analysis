"""List of the available videos."""

from pathlib import Path
from typing import Dict, List

from PyQt5.QtCore import QCoreApplication, Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import QListWidget, QListWidgetItem

from ...utils.video import find_video_files, get_video_duration

# The length of each video as text, by path. Reading it means opening the video, which
# takes a moment each; the list is filled again at every refresh, so it is read once
_durations: Dict[str, str] = {}


class _DurationReader(QThread):
    """Reads the lengths of videos one after the other, off the GUI thread."""

    read = pyqtSignal(str, str)  # A video's path and its length as text

    def __init__(self, paths: List[str]):
        super().__init__()
        self._paths = paths

    def run(self) -> None:
        for path in self._paths:
            if self.isInterruptionRequested():
                return
            self.read.emit(path, get_video_duration(path))


class VideoListWidget(QListWidget):
    """List of the videos found in the data folders."""

    def __init__(self, show_duration: bool = True):
        super().__init__()
        self._show_duration = show_duration
        self.video_files: List[str] = []
        self._readers: List[_DurationReader] = []

        # Names differ at their end (the clip number): shorten them in the middle
        self.setTextElideMode(Qt.ElideMiddle)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # A thread that is still running when the program ends would crash it
        QCoreApplication.instance().aboutToQuit.connect(self._stop_readers)

    def reload(self) -> List[str]:
        """Search the data folders again and list what is found; returns the paths.

        The names are shown at once; the lengths not yet known arrive one by one.
        """
        self.video_files = find_video_files()

        # Repopulating must not look like the user picking a video
        self.blockSignals(True)
        self.clear()
        for video_path in self.video_files:
            item = QListWidgetItem(self._text(video_path))
            item.setToolTip(video_path)
            item.setData(Qt.UserRole, video_path)
            self.addItem(item)
        self.setCurrentRow(-1)
        self.blockSignals(False)

        if self._show_duration:
            self._read_durations([path for path in self.video_files if path not in _durations])
        return self.video_files

    def _text(self, video_path: str) -> str:
        text = Path(video_path).name
        if self._show_duration and video_path in _durations:
            text = f"{text} ({_durations[video_path]})"
        return text

    def _read_durations(self, paths: List[str]) -> None:
        """Read these videos' lengths on a thread of their own; each row is filled in as
        its length arrives."""
        if not paths:
            return
        reader = _DurationReader(paths)
        reader.read.connect(self._on_duration_read)
        # Kept until it is done: a thread object freed while it runs crashes the program
        reader.finished.connect(lambda: self._readers.remove(reader))
        self._readers.append(reader)
        reader.start()

    def _on_duration_read(self, video_path: str, duration: str) -> None:
        _durations[video_path] = duration
        for row in range(self.count()):
            item = self.item(row)
            if item.data(Qt.UserRole) == video_path:
                item.setText(self._text(video_path))

    def _stop_readers(self) -> None:
        for reader in list(self._readers):
            reader.requestInterruption()
            reader.wait()
