import time
import random

import cv2
import keyboard
from keyboard import press, release

from video_handler import VideoHandler
from window_loader import WindowLoader


class ProjectTouhou:

    def __init__(self):
        self.TouHou = WindowLoader()
        self.data = self.TouHou.get_window()
        self.video_handler = VideoHandler(640,480,4)

    def get_frame(self):
        return next(self.data)

    def set_input(self, inputs):
        if inputs:
            for i in inputs:
                press(i)

    def release_input(self, inputs):
        if inputs:
            for i in inputs:
                release(i)

    def GameOn(self):
        self.start = time.time()
        input = None
        while(True):
            time.sleep(max(1 / 10 - (time.time() - self.start), 0))
            print(1 / (time.time() - self.start))
            self.release_input(input)
            self.video_handler.write_frame(self.get_frame())
            #self.set_input([random.choice(['left', 'right', 'up', 'down'])])
            if keyboard.is_pressed('/'):  # if key 'q' is pressed
                print('You Pressed A Key!')
                self.release_input(input)
                self.video_handler.close()
                break  # finishing the loop
            self.start = time.time()


if __name__ == "__main__":
    tmp = ProjectTouhou()
    tmp.GameOn()
