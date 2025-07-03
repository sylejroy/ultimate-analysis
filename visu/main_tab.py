import os
import time
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QLabel, QCheckBox, QPushButton
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage, QKeySequence
from PyQt5.QtWidgets import QShortcut

from video_player import VideoPlayer
from processing.inference import run_inference, set_detection_model
from processing.tracking import run_tracking, reset_tracker
from processing.field_segmentation import set_field_model  # <-- Add this import
from processing.player_id import run_player_id

class MainTab(QWidget):
    def __init__(self, dev_runtimes_tab=None):
        super().__init__()
        self.dev_runtimes_tab = dev_runtimes_tab
        self.video_folder = "input/dev_data"
        self.player = VideoPlayer()
        self.video_files = []
        self.current_video_index = 0
        self.is_paused = False
        self.tracks = []
        self.track_histories = {}  # or whatever structure you use
        self.init_ui()
        self.init_shortcuts()

    def init_ui(self):
        layout = QHBoxLayout()
        # Video list
        self.video_list = QListWidget()

        # Video display and controls
        right_layout = QVBoxLayout()
        self.video_label = QLabel("Select a video to play")
        self.video_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.video_label, 8)

        # Progress bar for video position
        from PyQt5.QtWidgets import QSlider
        self.progress_bar = QSlider(Qt.Horizontal)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setSingleStep(1)
        self.progress_bar.sliderMoved.connect(self.seek_video)
        right_layout.addWidget(self.progress_bar)

        # Checkboxes with keybinds in labels
        self.inference_checkbox = QCheckBox("Show Inference Results [I]")
        self.tracking_checkbox = QCheckBox("Show Inference Tracking [T]")
        self.player_id_checkbox = QCheckBox("Show Player Identification [J]")
        self.field_checkbox = QCheckBox("Show Field Segmentation [F]")
        right_layout.addWidget(self.inference_checkbox)
        right_layout.addWidget(self.tracking_checkbox)
        right_layout.addWidget(self.player_id_checkbox)
        right_layout.addWidget(self.field_checkbox)

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
        self.tracking_checkbox.stateChanged.connect(self.handle_tracking_checkbox)
        self.player_id_checkbox.stateChanged.connect(self.handle_player_id_checkbox)
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
        QShortcut(QKeySequence(Qt.Key_J), self, lambda: self.player_id_checkbox.toggle())
        # F: Toggle Field Segmentation Checkbox
        QShortcut(QKeySequence(Qt.Key_F), self, lambda: self.field_checkbox.toggle())

    def handle_inference_checkbox(self, state):
        if state == Qt.Checked:
            self.tracking_checkbox.setEnabled(True)
        else:
            self.tracking_checkbox.setChecked(False)
            self.tracking_checkbox.setEnabled(False)
            self.player_id_checkbox.setChecked(False)
            self.player_id_checkbox.setEnabled(False)

    def handle_tracking_checkbox(self, state):
        if state == Qt.Checked:
            # Automatically enable inference if tracking is enabled
            if not self.inference_checkbox.isChecked():
                self.inference_checkbox.setChecked(True)
            self.player_id_checkbox.setEnabled(True)
        else:
            self.player_id_checkbox.setChecked(False)
            self.player_id_checkbox.setEnabled(False)

    def handle_player_id_checkbox(self, state):
        if state == Qt.Checked:
            # Automatically enable tracking and inference if player ID is enabled
            if not self.tracking_checkbox.isChecked():
                self.tracking_checkbox.setChecked(True)
            if not self.inference_checkbox.isChecked():
                self.inference_checkbox.setChecked(True)
        # No further dependencies to set here, but could add logic if needed

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
        # Set progress bar range
        if self.player.cap:
            total_frames = int(self.player.cap.get(7))  # cv2.CAP_PROP_FRAME_COUNT == 7
            self.progress_bar.setMaximum(max(0, total_frames - 1))
            self.progress_bar.setValue(0)

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

        # Update progress bar
        if self.player.cap:
            current_frame = int(self.player.cap.get(1))  # cv2.CAP_PROP_POS_FRAMES == 1
            self.progress_bar.setValue(current_frame)

        # --- Inference step ---
        t0 = time.time()
        if self.inference_checkbox.isChecked() or self.tracking_checkbox.isChecked():
            self.detections = self.run_inference(frame)
        else:
            self.detections = []
        t1 = time.time()
        if self.dev_runtimes_tab:
            self.dev_runtimes_tab.log_runtime("Inference", (t1 - t0) * 1000)

        # --- Tracking step ---
        t2 = time.time()
        if self.tracking_checkbox.isChecked():
            self.tracks = self.run_tracking(frame, self.detections)
        else:
            self.tracks = []
        t3 = time.time()
        if self.dev_runtimes_tab:
            self.dev_runtimes_tab.log_runtime("Tracking", (t3 - t2) * 1000)

        # --- Field segmentation step ---
        t4 = time.time()
        vis_frame = frame.copy()
        if self.field_checkbox.isChecked():
            from processing.field_segmentation import run_field_segmentation
            results = run_field_segmentation(frame)
            mask = results[0].masks.data.cpu().numpy()
            from field_segmentation_visualisation import draw_field_segmentation
            vis_frame = draw_field_segmentation(vis_frame, mask)
        t5 = time.time()
        if self.dev_runtimes_tab:
            self.dev_runtimes_tab.log_runtime("Field Segmentation", (t5 - t4) * 1000)

        # --- Detection/Tracking overlays ---
        t6 = time.time()
        if self.tracking_checkbox.isChecked():
            from tracking_visualisation import draw_track_history
            vis_frame = draw_track_history(vis_frame, self.tracks, self.detections)
        elif self.inference_checkbox.isChecked():
            from detection_visualisation import draw_yolo_detections
            vis_frame = draw_yolo_detections(vis_frame, self.detections)
        t7 = time.time()
        if self.dev_runtimes_tab:
            self.dev_runtimes_tab.log_runtime("Overlay", (t7 - t6) * 1000)

        # --- Player ID step ---
        t8 = time.time()
        if self.player_id_checkbox.isChecked() and self.tracking_checkbox.isChecked() and self.inference_checkbox.isChecked():
            from player_id_visualisation import draw_player_id
            import numpy as np
            for track in self.tracks:
                bbox = None
                if hasattr(track, "to_tlwh"):
                    bbox = track.to_tlwh()
                elif hasattr(track, "to_ltrb"):
                    bbox = track.to_ltrb()
                elif hasattr(track, "bbox"):
                    bbox = track.bbox
                elif isinstance(track, dict):
                    bbox = track.get('bbox', None)
                if bbox is not None:
                    bbox = np.round(np.array(bbox)).astype(int)
                    if len(bbox) == 4:
                        x, y, w, h = bbox
                        x, y = max(0, x), max(0, y)
                        w, h = max(1, w), max(1, h)
                        if y + h <= frame.shape[0] and x + w <= frame.shape[1]:
                            obj_crop = frame[y:y+h, x:x+w]
                            if obj_crop.size > 0:
                                digit_str, digits = run_player_id(obj_crop)
                                vis_frame = draw_player_id(vis_frame, (x, y, w, h), digit_str, digits)
        t9 = time.time()
        if self.dev_runtimes_tab:
            self.dev_runtimes_tab.log_runtime("Player ID", (t9 - t8) * 1000)

        # --- Display step ---
        t10 = time.time()
        h, w, ch = vis_frame.shape
        bytes_per_line = ch * w
        qimg = QImage(vis_frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        t11 = time.time()
        if self.dev_runtimes_tab:
            self.dev_runtimes_tab.log_runtime("Display", (t11 - t10) * 1000)

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
        self.reset_visualisation()
        reset_tracker()  # resets tracker
        # Also reset the track histories used for visualization
        print("Tracker and track histories reset.")

    def reset_visualisation(self):
        self.tracks = []
        self.track_histories = {}
        try:
            from tracking_visualisation import reset_track_histories
            reset_track_histories()
        except ImportError:
            pass

    def run_inference(self, frame):
        return run_inference(frame)

    def run_tracking(self, frame, detections):
        return run_tracking(frame, detections)

    def set_detection_model(self, model_path):
        print(f"[DEBUG] MainTab.set_detection_model called with: {model_path}")
        set_detection_model(model_path)

    def set_field_model(self, model_path):
        print(f"[DEBUG] MainTab.set_field_model called with: {model_path}")
        set_field_model(model_path)  # <-- Add this line to actually update the field segmentation model

    def seek_video(self, frame_idx):
        if self.player.cap:
            self.player.cap.set(1, frame_idx)  # cv2.CAP_PROP_POS_FRAMES == 1
            frame = self.player.get_next_frame()
            if frame is not None:
                vis_frame = frame.copy()
                h, w, ch = vis_frame.shape
                bytes_per_line = ch * w
                qimg = QImage(vis_frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
                pixmap = QPixmap.fromImage(qimg)
                self.video_label.setPixmap(pixmap.scaled(
                    self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                ))