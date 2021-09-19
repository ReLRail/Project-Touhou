import datetime
import glob
import os
import time
import random

import torchvision.models as models
import matplotlib.pyplot as plt
import cv2
import keyboard

from input_handler import InputHandler
from men_handler import MemHandler
from video_handler import VideoHandler
from window_loader import WindowLoader
import numpy as np
import skimage.color
import skimage.io
import fpstimer
from torch import nn, optim
import torch
from PIL import Image
from numpy import asarray


def rgb2gray(rgb):
    return skimage.color.rgb2gray(rgb)


def save_frame(frame):
    plt.imsave('test.png', frame, cmap=plt.cm.gray)


class touhou_handler:

    def __init__(self, load=False):
        self.death_reset = 0
        self.timer = fpstimer.FPSTimer(5)
        self.window_handler = WindowLoader()
        self.frame_handler = self.window_handler.get_window()
        self.video_handler = VideoHandler(640, 480, 4)
        self.input_handler = InputHandler()

        self.frame_heap = []
        self.frame_count = 0

        self.available_moves = (('z'), ('shift', 'up', 'z'), ('shift', 'left', 'z'), ('shift', 'down', 'z'), ('shift', 'right', 'z'))

        if 'data' not in os.listdir():
            os.mkdir('data')

        self.data_path = 'data/exp' + str(len(glob.glob('data/exp*'))) + '/'
        os.mkdir('data/exp' + str(len(glob.glob('data/exp*'))))
        self.mem_handler = MemHandler()

        self.hp = 255

        self.net = models.resnet18(pretrained=True)
        self.net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.net.load_state_dict(torch.load('Touhou.pth'))
        self.net = self.net.cuda()
        self.net.eval()

    def get_frame(self):
        return cv2.cvtColor(next(self.frame_handler).reshape(480, 640, 4), cv2.COLOR_BGRA2RGB)

    def save_frame(self, frame, input):
        self.frame_heap.append((frame,input))
        if len(self.frame_heap) > 5:
            tmp = self.frame_heap.pop(0)
            plt.imsave(self.data_path + str(self.frame_count) + '_' + '_'.join(tmp[1])+ '.png', tmp[0])
            self.frame_count += 1

    def select(self, frame):
        frame = asarray(Image.fromarray(np.uint8(frame)).convert('L'))
        print(frame)
        print(frame.shape)
        conf, move = torch.max(self.net(torch.tensor([[frame]]).cuda() / 255), 1)
        return self.available_moves[move[0]]

    def GameOn(self):
        self.frame = self.get_frame()
        self.start = time.time()
        inputs = []
        while (True):
            if keyboard.is_pressed('/'):  # if key 'q' is pressed
                print('You Pressed the Key! :3')
                self.input_handler.clear()
                break

            self.frame = self.get_frame()

            res = self.get_status()
            self.input_handler.release_input(inputs)
            if self.death_reset > 0:
                self.death_reset -= 1
                self.frame_heap = []
                inputs = ['z']
            elif res == -10:
                inputs = ['z']
                self.death_reset = 10
                self.frame_heap = []
            elif res == -20:
                inputs = ['z']
                self.death_reset = 10
                self.frame_heap = []
                time.sleep(0.5)
            else:
                inputs = self.select(self.frame)
                print(inputs)
                self.save_frame(self.frame, inputs)
            #time.sleep(0.05)
            self.input_handler.set_input(inputs)
            self.timer.sleep()
            print(1 / (time.time() - self.start), inputs,self.death_reset)
            self.start = time.time()

        self.video_handler.close()
        self.input_handler.close()

    def get_status(self):
        hp, bm, dg, score, power = self.mem_handler.get_score()

        if hp < self.hp:
            self.hp = hp
            return -10

        if hp > 255:
            self.hp = 255
            return -20
        return 10


if __name__ == "__main__":
    tmp = touhou_handler()
    tmp.GameOn()
