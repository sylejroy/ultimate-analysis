"""Zoomable image display widget with optional grid overlay.

Extracted from homography_tab to keep files small and focused. This QLabel subclass
supports scroll-aware mouse wheel zooming and a configurable grid overlay to help
judge perspective alignment while tuning homography.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPen, QPixmap, QWheelEvent
from PyQt5.QtWidgets import QLabel, QScrollArea, QSizePolicy


class ZoomableImageLabel(QLabel):
    """Custom QLabel that supports mouse wheel zooming and grid overlay."""

    zoom_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.zoom_factor = 1.0
        self.original_pixmap: Optional[QPixmap] = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setScaledContents(False)

        # Grid overlay properties
        self.show_grid = True
        self.grid_spacing = 50  # pixels at 1.0 zoom
        self.grid_color = QColor(255, 255, 255, 80)  # Semi-transparent white
        self.grid_line_width = 1

    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zooming to mouse position (if in a QScrollArea)."""
        if self.original_pixmap is None:
            return

        # Find ancestor scroll area to support anchor-point zooming
        scroll_area = None
        parent = self.parent()
        while parent:
            if isinstance(parent, QScrollArea):
                scroll_area = parent
                break
            parent = parent.parent()

        if scroll_area is None:
            # Fallback: center zoom if no scroll area found
            zoom_in = event.angleDelta().y() > 0
            zoom_delta = 0.15 if zoom_in else -0.15
            new_zoom = max(0.1, min(10.0, self.zoom_factor + zoom_delta))

            if new_zoom != self.zoom_factor:
                self.zoom_factor = new_zoom
                self._update_display()
                self.zoom_changed.emit(self.zoom_factor)
            return

        # Mouse position
        mouse_pos = event.position().toPoint() if hasattr(event, "position") else event.pos()

        # Calculate zoom change
        zoom_in = event.angleDelta().y() > 0
        zoom_delta = 0.15 if zoom_in else -0.15
        old_zoom = self.zoom_factor
        new_zoom = max(0.1, min(10.0, old_zoom + zoom_delta))

        if new_zoom == old_zoom:
            return

        # Get scrollbars
        h_scroll = scroll_area.horizontalScrollBar()
        v_scroll = scroll_area.verticalScrollBar()

        # Store old scroll positions
        old_h = h_scroll.value()
        old_v = v_scroll.value()

        # Image coordinate under mouse before zoom
        widget_rect = self.rect()
        if self.pixmap():
            pixmap_size = self.pixmap().size()

            # QLabel centers the pixmap when smaller than the widget
            x_offset = max(0, (widget_rect.width() - pixmap_size.width()) // 2)
            y_offset = max(0, (widget_rect.height() - pixmap_size.height()) // 2)

            # Mouse position relative to the actual image (accounting for centering)
            img_mouse_x = mouse_pos.x() - x_offset
            img_mouse_y = mouse_pos.y() - y_offset

            # Account for scroll position and current zoom to get original image coordinates
            orig_img_x = (img_mouse_x + old_h) / old_zoom
            orig_img_y = (img_mouse_y + old_v) / old_zoom

            # Apply new zoom
            self.zoom_factor = new_zoom
            self._update_display()

            # Calculate new offsets after zoom
            if self.pixmap():
                new_pixmap_size = self.pixmap().size()
                new_x_offset = max(0, (widget_rect.width() - new_pixmap_size.width()) // 2)
                new_y_offset = max(0, (widget_rect.height() - new_pixmap_size.height()) // 2)

                # Keep the same original image point under the mouse
                target_img_x = orig_img_x * new_zoom
                target_img_y = orig_img_y * new_zoom

                new_h = target_img_x - (mouse_pos.x() - new_x_offset)
                new_v = target_img_y - (mouse_pos.y() - new_y_offset)

                # Apply new scroll positions with bounds checking
                h_scroll.setValue(max(0, min(h_scroll.maximum(), int(new_h))))
                v_scroll.setValue(max(0, min(v_scroll.maximum(), int(new_v))))
        else:
            # If no pixmap, just update zoom
            self.zoom_factor = new_zoom
            self._update_display()

        self.zoom_changed.emit(self.zoom_factor)

    def set_image(self, pixmap: QPixmap):
        """Set the image and reset zoom to fit the container."""
        self.original_pixmap = pixmap
        # Calculate initial zoom to fit the container while maintaining aspect ratio
        if pixmap and not pixmap.isNull():
            container_size = self.size()
            pixmap_size = pixmap.size()

            # Calculate scale factors for width and height
            scale_w = (
                container_size.width() / pixmap_size.width() if pixmap_size.width() > 0 else 1.0
            )
            scale_h = (
                container_size.height() / pixmap_size.height() if pixmap_size.height() > 0 else 1.0
            )

            # Use the smaller scale factor to ensure the image fits completely
            initial_zoom = min(
                scale_w, scale_h, 1.0
            )  # Don't scale up beyond original size initially
            self.zoom_factor = max(0.1, initial_zoom)
        else:
            self.zoom_factor = 1.0

        self._update_display()
        self.zoom_changed.emit(self.zoom_factor)

    def set_zoom(self, zoom_factor: float):
        """Set zoom factor programmatically."""
        self.zoom_factor = max(0.1, min(10.0, zoom_factor))
        self._update_display()

    def _update_display(self):
        """Update the displayed image with current zoom."""
        if self.original_pixmap is None:
            return

        # Scale the pixmap
        scaled_size = self.original_pixmap.size() * self.zoom_factor
        scaled_pixmap = self.original_pixmap.scaled(
            scaled_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        self.setPixmap(scaled_pixmap)

        # Set the minimum size so the scroll area recognizes the content size
        self.setMinimumSize(scaled_pixmap.size())

        # Also set the size hint for proper scroll area calculation
        self.resize(scaled_pixmap.size())

        # Update the scroll area geometry
        parent = self.parent()
        if isinstance(parent, QScrollArea):
            parent.updateGeometry()

    def paintEvent(self, event):  # noqa: N802 - Qt override
        """Override paint event to draw a grid overlay on top of the image."""
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

    def set_grid_color(self, color: QColor):
        """Set grid color."""
        self.grid_color = color
        self.update()  # Trigger a repaint
