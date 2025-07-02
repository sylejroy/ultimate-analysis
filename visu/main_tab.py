import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QLabel, QCheckBox, QPushButton
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage, QKeySequence
from PyQt5.QtWidgets import QShortcut

from video_player import VideoPlayer
from processing.inference import run_inference
from processing.tracking import run_tracking, reset_tracker

class MainTab(QWidget):
    def __init__(self):
        super().__init__()
        self.video_folder = "input/dev_data"
        self.player = VideoPlayer()
        self.video_files = []
        self.current_video_index = 0
        self.is_paused = False
        self.init_ui()
        self.init_shortcuts()  # <-- Add this line

    def init_ui(self):
        layout = QHBoxLayout()
        # Video list
        self.video_list = QListWidget()

        # Video display and controls
        right_layout = QVBoxLayout()
        self.video_label = QLabel("Select a video to play")
        self.video_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.video_label, 8)

        # Checkboxes with keybinds in labels
        self.inference_checkbox = QCheckBox("Show Inference Results [I]")
        self.tracking_checkbox = QCheckBox("Show Inference Tracking [T]")
        self.jersey_checkbox = QCheckBox("Show Jersey Number Tracking [J]")
        right_layout.addWidget(self.inference_checkbox)
        right_layout.addWidget(self.tracking_checkbox)
        right_layout.addWidget(self.jersey_checkbox)

        # Play/Pause button with keybind
        self.play_pause_button = QPushButton("Play [Space]")
        self.play_pause_button.clicked.connect(self.toggle_play_pause)
        right_layout.addWidget(self.play_pause_button)

        # Navigation buttons with keybinds
        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous Video [←]")
        self.prev_button.clicked.connect(self.prev_video)
        nav_layout.addWidget(self.prev_button)

        self.next_button = QPushButton("Next Video [→]")
        self.next_button.clicked.connect(self.next_video)
        nav_layout.addWidget(self.next_button)
        right_layout.addLayout(nav_layout)

        # Reset Tracker button with keybind
        self.reset_tracker_button = QPushButton("Reset Tracker [R]")
        self.reset_tracker_button.clicked.connect(self.reset_tracker)
        right_layout.addWidget(self.reset_tracker_button)

        layout.addWidget(self.video_list, 1)
        layout.addLayout(right_layout, 4)
        self.setLayout(layout)

        # Timer for video playback
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)

        # Now connect signals and load videos
        self.inference_checkbox.stateChanged.connect(self.handle_inference_checkbox)
        self.video_list.currentRowChanged.connect(self.load_selected_video)
        self.load_videos()

    def init_shortcuts(self):
        # Space: Play/Pause
        QShortcut(QKeySequence(Qt.Key_Space), self, self.toggle_play_pause)
        # Left Arrow: Previous Video
        QShortcut(QKeySequence(Qt.Key_Left), self, self.prev_video)
        # Right Arrow: Next Video
        QShortcut(QKeySequence(Qt.Key_Right), self, self.next_video)
        # R: Reset Tracker
        QShortcut(QKeySequence(Qt.Key_R), self, self.reset_tracker)
        # I: Toggle Inference Checkbox
        QShortcut(QKeySequence(Qt.Key_I), self, lambda: self.inference_checkbox.toggle())
        # T: Toggle Tracking Checkbox
        QShortcut(QKeySequence(Qt.Key_T), self, lambda: self.tracking_checkbox.toggle())
        # J: Toggle Jersey Checkbox
        QShortcut(QKeySequence(Qt.Key_J), self, lambda: self.jersey_checkbox.toggle())

    def handle_inference_checkbox(self, state):
        if state == Qt.Checked:
            self.tracking_checkbox.setEnabled(True)
        else:
            self.tracking_checkbox.setChecked(False)
            self.tracking_checkbox.setEnabled(False)

    def load_videos(self):
        self.video_list.clear()
        self.video_files = []
        if not os.path.exists(self.video_folder):
            os.makedirs(self.video_folder)
        for f in os.listdir(self.video_folder):
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                self.video_files.append(f)
                self.video_list.addItem(f)
        if self.video_files:
            self.video_list.setCurrentRow(0)
            self.load_selected_video(0)

    def load_selected_video(self, index):
        if not self.video_files or index < 0 or index >= len(self.video_files):
            return
        filename = self.video_files[index]
        path = os.path.join(self.video_folder, filename)
        self.player.load_video(path)
        self.video_label.setText("Ready to play: " + filename)
        self.current_video_index = index
        self.is_paused = False
        self.play_pause_button.setText("Play")

    def toggle_play_pause(self):
        if not self.player.cap:
            return
        if self.timer.isActive():
            self.timer.stop()
            self.is_paused = True
            self.play_pause_button.setText("Play")
        else:
            self.timer.start(30)  # ~30 FPS
            self.is_paused = False
            self.play_pause_button.setText("Pause")

    def next_frame(self):
        frame = self.player.get_next_frame()
        if frame is None:
            self.timer.stop()
            self.play_pause_button.setText("Play")
            return

        # --- Inference step ---
        if self.inference_checkbox.isChecked() or self.tracking_checkbox.isChecked():
            # Always run inference if either visualisation is needed
            self.detections = self.run_inference(frame)
        else:
            self.detections = []

        # --- Tracking step ---
        if self.tracking_checkbox.isChecked():
            self.tracks = self.run_tracking(frame, self.detections)
        else:
            self.tracks = []

        # --- Visualization step ---
        vis_frame = frame.copy()
        if self.tracking_checkbox.isChecked():
            from tracking_visualisation import draw_track_history
            vis_frame = draw_track_history(vis_frame, self.tracks, self.detections)
        elif self.inference_checkbox.isChecked():
            from detection_visualisation import draw_yolo_detections
            vis_frame = draw_yolo_detections(vis_frame, self.detections)

        # Display
        h, w, ch = vis_frame.shape
        bytes_per_line = ch * w
        qimg = QImage(vis_frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def next_video(self):
        if not self.video_files:
            return
        next_index = (self.current_video_index + 1) % len(self.video_files)
        self.video_list.setCurrentRow(next_index)
        self.load_selected_video(next_index)
        self.reset_tracker()  # Reset tracker when switching videos

    def prev_video(self):
        if not self.video_files:
            return
        prev_index = (self.current_video_index - 1) % len(self.video_files)
        self.video_list.setCurrentRow(prev_index)
        self.load_selected_video(prev_index)
        self.reset_tracker()  # Reset tracker when switching videos

    def reset_tracker(self):
        reset_tracker()  # resets DeepSort tracker
        # Also reset the track histories used for visualization
        from tracking_visualisation import draw_track_history
        if hasattr(draw_track_history, "track_histories"):
            draw_track_history.track_histories = {}
        print("Tracker and track histories reset.")

    def run_inference(self, frame):
        return run_inference(frame)

    def run_tracking(self, frame, detections):
        return run_tracking(frame, detections)