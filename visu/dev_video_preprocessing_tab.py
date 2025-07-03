from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QLabel, QPushButton, QSlider, QSpinBox, QFileDialog
from PyQt5.QtCore import Qt
import os
import cv2

class DevVideoPreprocessingTab(QWidget):
    def __init__(self, input_folder="input"):
        super().__init__()
        self.input_folder = input_folder
        self.cap = None
        self.current_video = None
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30

        layout = QHBoxLayout()
        self.video_list = QListWidget()
        self.video_list.itemSelectionChanged.connect(self.load_selected_video)
        layout.addWidget(self.video_list, 1)

        right = QVBoxLayout()
        self.video_label = QLabel("No video loaded")
        self.video_label.setAlignment(Qt.AlignCenter)
        right.addWidget(self.video_label, 8)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.slider_moved)
        right.addWidget(self.slider)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Snippet length (s):"))
        self.snippet_length = QSpinBox()
        self.snippet_length.setRange(1, 600)
        self.snippet_length.setValue(10)
        controls.addWidget(self.snippet_length)
        self.snip_button = QPushButton("Snip")
        self.snip_button.clicked.connect(self.snip)
        controls.addWidget(self.snip_button)
        right.addLayout(controls)

        layout.addLayout(right, 4)
        self.setLayout(layout)
        self.populate_video_list()

    def populate_video_list(self):
        self.video_list.clear()
        if not os.path.exists(self.input_folder):
            return
        for f in os.listdir(self.input_folder):
            path = os.path.join(self.input_folder, f)
            if os.path.isfile(path) and os.path.getsize(path) > 100*1024*1024 and f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                self.video_list.addItem(f)

    def load_selected_video(self):
        items = self.video_list.selectedItems()
        if not items:
            return
        filename = items[0].text()
        path = os.path.join(self.input_folder, filename)
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(path)
        self.current_video = path
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.slider.setMaximum(self.total_frames-1)
        self.slider.setValue(0)
        self.show_frame(0)

    def slider_moved(self, value):
        if not self.cap:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, value)
        self.show_frame(value)

    def show_frame(self, frame_idx):
        if not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        import numpy as np
        from PyQt5.QtGui import QImage, QPixmap
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def snip(self):
        if not self.cap or not self.current_video:
            return
        start_frame = self.slider.value()
        length_sec = self.snippet_length.value()
        fps = self.fps
        end_frame = min(self.total_frames, int(start_frame + length_sec * fps))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Save to input\dev_data with similar filename
        base_name = os.path.splitext(os.path.basename(self.current_video))[0]
        ext = os.path.splitext(self.current_video)[1]
        out_dir = os.path.join("input", "dev_data")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{base_name}_snippet_{start_frame}_{end_frame}{ext}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v') if ext.lower() == ".mp4" else cv2.VideoWriter_fourcc(*'XVID')
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        for i in range(start_frame, end_frame):
            ret, frame = self.cap.read()
            if not ret:
                break
            out.write(frame)
        out.release()