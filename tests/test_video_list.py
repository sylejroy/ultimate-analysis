"""The video list shows the names at once and reads the lengths off the GUI thread,
once per video."""

import threading
import unittest
from unittest.mock import patch

from PyQt5.QtWidgets import QApplication
from support import load_module, qt_app

VIDEOS = ["C:/videos/first.mp4", "C:/videos/second.mp4"]


class VideoListTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = qt_app()
        cls.module = load_module("gui.widgets.video_list")

    def setUp(self):
        self.module._durations.clear()
        self.read_on = []  # The thread each length was read on

        def duration(path):
            self.read_on.append(threading.current_thread())
            return "01:00" if path.endswith("first.mp4") else "02:30"

        patches = [
            patch.object(self.module, "find_video_files", return_value=list(VIDEOS)),
            patch.object(self.module, "get_video_duration", side_effect=duration),
        ]
        for started in patches:
            started.start()
            self.addCleanup(started.stop)

    def _wait_for_readers(self, widget):
        for reader in list(widget._readers):
            reader.wait()
        QApplication.processEvents()

    def test_names_come_at_once_and_lengths_from_a_thread_of_their_own(self):
        widget = self.module.VideoListWidget()
        self.assertEqual(widget.reload(), VIDEOS)
        self.assertEqual([widget.item(i).text() for i in range(2)], ["first.mp4", "second.mp4"])
        self.assertEqual(widget.currentRow(), -1)

        self._wait_for_readers(widget)
        self.assertEqual(
            [widget.item(i).text() for i in range(2)],
            ["first.mp4 (01:00)", "second.mp4 (02:30)"],
        )
        self.assertEqual(len(self.read_on), 2)
        self.assertNotIn(threading.main_thread(), self.read_on)

        # A refresh knows the lengths already and opens no video again
        widget.reload()
        self.assertEqual(widget.item(0).text(), "first.mp4 (01:00)")
        self.assertEqual(widget._readers, [])
        self.assertEqual(len(self.read_on), 2)

    def test_without_lengths_no_video_is_opened(self):
        widget = self.module.VideoListWidget(show_duration=False)
        widget.reload()
        self.assertEqual(widget._readers, [])
        self.assertEqual(widget.item(1).text(), "second.mp4")
        self.assertEqual(self.read_on, [])


if __name__ == "__main__":
    unittest.main()
