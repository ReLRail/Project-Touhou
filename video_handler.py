import cv2
from PIL import Image


class VideoHandler:

    def __init__(self, width, height, channels):
        self.size = (width, height)
        self.channels = channels
        self.out = cv2.VideoWriter('video_out.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, self.size)
    def write_frame(self,frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        self.out.write(frame)

    def close(self):
        self.out.release()