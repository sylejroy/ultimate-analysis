"""The picture display: the user's zoom stays from frame to frame, and the picture
follows the view's size until the user zooms."""

import unittest

from PyQt5.QtCore import QPoint, QPointF, QSize, Qt
from PyQt5.QtGui import QPixmap, QWheelEvent
from PyQt5.QtWidgets import QApplication, QScrollArea
from support import load_module, qt_app


class ZoomableImageLabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = qt_app()
        cls.module = load_module("gui.widgets.zoomable_image_label")

    def _label_in_view(self, width: int = 400, height: int = 300):
        area = QScrollArea()
        area.setWidgetResizable(True)
        label = self.module.ZoomableImageLabel()
        area.setWidget(label)
        area.resize(width, height)
        area.show()
        self.addCleanup(area.close)
        QApplication.processEvents()
        return area, label

    @staticmethod
    def _wheel(label, up: bool):
        event = QWheelEvent(
            QPointF(10, 10),
            QPointF(10, 10),
            QPoint(0, 0),
            QPoint(0, 120 if up else -120),
            Qt.NoButton,
            Qt.NoModifier,
            Qt.NoScrollPhase,
            False,
        )
        label.wheelEvent(event)

    def test_the_users_zoom_stays_from_frame_to_frame(self):
        area, label = self._label_in_view()
        label.set_image(QPixmap(1920, 1080))
        fit = label.zoom_factor
        self.assertLess(fit, 1.0)
        self.assertLessEqual(label.pixmap().width(), area.viewport().width())
        self.assertFalse(label.user_zoomed)

        self._wheel(label, up=True)
        zoomed = label.zoom_factor
        self.assertAlmostEqual(zoomed, fit + 0.15)
        self.assertTrue(label.user_zoomed)

        # The next frame of the same size keeps the zoom; one of another size fits again
        label.set_image(QPixmap(1920, 1080))
        self.assertEqual(label.zoom_factor, zoomed)
        self.assertEqual(label.pixmap().width(), round(1920 * zoomed))
        label.set_image(QPixmap(640, 480))
        self.assertFalse(label.user_zoomed)
        self.assertLessEqual(label.pixmap().width(), area.viewport().width())

    def test_the_picture_follows_the_view_until_the_user_zooms(self):
        area, label = self._label_in_view()
        label.set_image(QPixmap(1920, 1080))
        fit = label.zoom_factor
        # Nothing holds the label at the picture's size, so the view may shrink
        self.assertEqual(label.minimumSizeHint(), QSize(0, 0))

        area.resize(200, 150)
        QApplication.processEvents()
        self.assertLess(label.zoom_factor, fit)
        self.assertLessEqual(label.pixmap().width(), area.viewport().width())

        # Zoomed in, the picture is larger than the view and the bars reach all of it
        self._wheel(label, up=True)
        self._wheel(label, up=True)
        self.assertEqual(label.minimumSizeHint(), label.pixmap().size())
        self.assertGreater(label.width(), area.viewport().width())
        zoomed = label.zoom_factor
        area.resize(400, 300)
        QApplication.processEvents()
        self.assertEqual(label.zoom_factor, zoomed)

        # A double click fits the picture again
        label.fit_image()
        self.assertFalse(label.user_zoomed)
        self.assertLessEqual(label.pixmap().width(), area.viewport().width())

    def test_a_picture_at_its_own_size_is_not_scaled(self):
        area, label = self._label_in_view(800, 600)
        picture = QPixmap(100, 50)
        label.set_image(picture)
        self.assertEqual(label.zoom_factor, 1.0)
        self.assertEqual(label.pixmap().cacheKey(), picture.cacheKey())


if __name__ == "__main__":
    unittest.main()
