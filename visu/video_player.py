import cv2

class VideoPlayer:
    def __init__(self):
        self.cap = None

    def load_video(self, path):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(path)

    def get_next_frame(self):
        if not self.cap:
            return None
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            self.cap = None
            return None
        return frame