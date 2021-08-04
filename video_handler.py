import cv2
from PIL import Image


class VideoHandler:

    def __init__(self, width, height, channels):
        self.size = (width, height)
        self.channels = channels
        self.out = cv2.VideoWriter(filename='video_out.avi',
                         fourcc=cv2.VideoWriter_fourcc(*'MJPG'),
                         fps=10, frameSize=self.size)
    def write_frame(self,frame):
        self.out.write(frame)

    def close(self):
        self.out.release()