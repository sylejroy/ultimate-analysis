"""Image display that zooms with the mouse wheel and can overlay a grid."""

from typing import Optional

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QColor, QPainter, QPen, QPixmap, QWheelEvent
from PyQt5.QtWidgets import (
    QLabel,
    QScrollArea,
    QSizePolicy,
)

MIN_ZOOM = 0.1
MAX_ZOOM = 10.0
WHEEL_STEP = 0.15


class ZoomableImageLabel(QLabel):
    """A picture that fits its view until the user zooms with the wheel; the zoom then
    stays from picture to picture and through resizes. A double click fits it again."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.zoom_factor = 1.0
        self.original_pixmap: Optional[QPixmap] = None
        self.user_zoomed = False  # Whether the zoom is the user's choice, not the fit
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setScaledContents(False)

        # Grid overlay properties
        self.show_grid = True
        self.grid_spacing = 50  # pixels at 1.0 zoom
        self.grid_color = QColor(255, 255, 255, 80)  # Semi-transparent white
        self.grid_line_width = 1

    def _scroll_area(self) -> Optional[QScrollArea]:
        parent = self.parent()
        while parent is not None:
            if isinstance(parent, QScrollArea):
                return parent
            parent = parent.parent()
        return None

    def _view_size(self) -> QSize:
        """The size the picture has to fit in: the scroll area's view, or this widget's
        own size when there is none."""
        area = self._scroll_area()
        return area.viewport().size() if area is not None else self.size()

    def _fit_zoom(self) -> float:
        """The zoom at which the whole picture is seen, not enlarged beyond its own size."""
        if self.original_pixmap is None or self.original_pixmap.isNull():
            return 1.0
        view, size = self._view_size(), self.original_pixmap.size()
        scale_w = view.width() / size.width() if size.width() > 0 else 1.0
        scale_h = view.height() / size.height() if size.height() > 0 else 1.0
        return max(MIN_ZOOM, min(scale_w, scale_h, 1.0))

    def minimumSizeHint(self) -> QSize:  # noqa: N802  (name given by the base class)
        """The zoomed picture's size when the user zoomed, so that the scroll area scrolls
        it; otherwise nothing, so that the picture may shrink with the view."""
        if self.original_pixmap is None:
            return super().minimumSizeHint()
        return self.pixmap().size() if self.user_zoomed and self.pixmap() else QSize(0, 0)

    def wheelEvent(self, event: QWheelEvent):
        """Zoom in or out around the mouse position."""
        if self.original_pixmap is None:
            return

        zoom_in = event.angleDelta().y() > 0
        old_zoom = self.zoom_factor
        step = WHEEL_STEP if zoom_in else -WHEEL_STEP
        new_zoom = max(MIN_ZOOM, min(MAX_ZOOM, old_zoom + step))
        if new_zoom == old_zoom:
            return

        scroll_area = self._scroll_area()
        if scroll_area is None or not self.pixmap():
            self.user_zoomed = True
            self.zoom_factor = new_zoom
            self._update_display()
            return

        mouse_pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
        h_scroll = scroll_area.horizontalScrollBar()
        v_scroll = scroll_area.verticalScrollBar()
        old_h = h_scroll.value()
        old_v = v_scroll.value()

        # The point of the picture under the mouse, in the picture's own pixels: the
        # label centres a picture smaller than itself, and is scrolled when larger
        widget_rect = self.rect()
        pixmap_size = self.pixmap().size()
        x_offset = max(0, (widget_rect.width() - pixmap_size.width()) // 2)
        y_offset = max(0, (widget_rect.height() - pixmap_size.height()) // 2)
        orig_img_x = (mouse_pos.x() - x_offset + old_h) / old_zoom
        orig_img_y = (mouse_pos.y() - y_offset + old_v) / old_zoom

        self.user_zoomed = True
        self.zoom_factor = new_zoom
        self._update_display()

        # Scroll so that the same point of the picture stays under the mouse
        new_pixmap_size = self.pixmap().size()
        new_x_offset = max(0, (widget_rect.width() - new_pixmap_size.width()) // 2)
        new_y_offset = max(0, (widget_rect.height() - new_pixmap_size.height()) // 2)
        new_h = orig_img_x * new_zoom - (mouse_pos.x() - new_x_offset)
        new_v = orig_img_y * new_zoom - (mouse_pos.y() - new_y_offset)
        h_scroll.setValue(max(0, min(h_scroll.maximum(), int(new_h))))
        v_scroll.setValue(max(0, min(v_scroll.maximum(), int(new_v))))

    def mouseDoubleClickEvent(self, event):
        self.fit_image()
        super().mouseDoubleClickEvent(event)

    def fit_image(self) -> None:
        """Show the whole picture again, at the zoom that fits the view."""
        self.user_zoomed = False
        self.zoom_factor = self._fit_zoom()
        self._update_display()

    def set_image(self, pixmap: QPixmap):
        """Show a picture. It fits the view unless the user chose a zoom; a picture of
        another size than the last one fits the view again."""
        if self.original_pixmap is None or pixmap.size() != self.original_pixmap.size():
            self.user_zoomed = False
        self.original_pixmap = pixmap
        if not self.user_zoomed:
            self.zoom_factor = self._fit_zoom()
        self._update_display()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # The picture follows the view's size until the user chooses a zoom
        if self.original_pixmap is not None and not self.user_zoomed:
            fit = self._fit_zoom()
            if fit != self.zoom_factor:
                self.zoom_factor = fit
                self._update_display()

    def _update_display(self):
        """Show the picture at the current zoom."""
        if self.original_pixmap is None:
            return

        # Scaling a 1080p frame to the panel's size takes about a millisecond, smooth or
        # not; smooth keeps the thin lines of the field steady while the video plays
        if self.zoom_factor == 1.0:
            scaled_pixmap = self.original_pixmap
        else:
            scaled_pixmap = self.original_pixmap.scaled(
                self.original_pixmap.size() * self.zoom_factor,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        self.setPixmap(scaled_pixmap)

        # What the scroll area does when it is resized: the label takes the view's size,
        # or the picture's when that is larger, so that the bars reach all of it
        area = self._scroll_area()
        if area is not None:
            self.resize(area.viewport().size().expandedTo(self.minimumSizeHint()))

    def paintEvent(self, event):
        """Override paint event to draw grid overlay on top of the image."""
        # First, let the parent QLabel draw the image
        super().paintEvent(event)

        # Draw grid overlay if enabled and we have an image
        if self.show_grid and self.pixmap() and not self.pixmap().isNull():
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing, True)

            # Set up the grid pen
            pen = QPen(self.grid_color)
            pen.setWidth(self.grid_line_width)
            painter.setPen(pen)

            # Get the image area within the widget
            pixmap_rect = self.pixmap().rect()
            widget_rect = self.rect()

            # Calculate the image position (centered in widget)
            x_offset = max(0, (widget_rect.width() - pixmap_rect.width()) // 2)
            y_offset = max(0, (widget_rect.height() - pixmap_rect.height()) // 2)

            # Calculate actual grid spacing based on current zoom
            actual_grid_spacing = self.grid_spacing * self.zoom_factor

            # Draw vertical lines
            image_left = x_offset
            image_right = x_offset + pixmap_rect.width()
            image_top = y_offset
            image_bottom = y_offset + pixmap_rect.height()

            # Start from the first grid line within the image
            start_x = (
                image_left
                + (actual_grid_spacing - (image_left % actual_grid_spacing)) % actual_grid_spacing
            )
            x = start_x
            while x < image_right:
                painter.drawLine(int(x), image_top, int(x), image_bottom)
                x += actual_grid_spacing

            # Draw horizontal lines
            start_y = (
                image_top
                + (actual_grid_spacing - (image_top % actual_grid_spacing)) % actual_grid_spacing
            )
            y = start_y
            while y < image_bottom:
                painter.drawLine(image_left, int(y), image_right, int(y))
                y += actual_grid_spacing

            painter.end()

    def set_grid_visible(self, visible: bool):
        """Toggle grid visibility."""
        self.show_grid = visible
        self.update()  # Trigger a repaint

    def set_grid_spacing(self, spacing: int):
        """Set grid spacing in pixels at 1.0 zoom."""
        self.grid_spacing = spacing
        self.update()  # Trigger a repaint
