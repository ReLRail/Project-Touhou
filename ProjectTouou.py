import datetime
import glob
import os
import sys
import time
import random
import traceback
import torchvision.models as models
from PIL import Image
from torchsummary import summary
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import keyboard
import torch
from torch import nn, optim
from keyboard import press, release
from DeadExp import DeadExp as DE, DeadExp
from input_handler import InputHandler
from men_handler import MemHandler
from models.cnn import alexnet
from video_handler import VideoHandler
from window_loader import WindowLoader
import numpy as np
from numpy import log as ln
import skimage.color
import skimage.io


def rgb2gray(rgb):
    return skimage.color.rgb2gray(rgb)


def save_frame(frame):
    plt.imsave('test.png', frame, cmap=plt.cm.gray)


class ProjectTouhou:

    def __init__(self, load=False):
        self.window_handler = WindowLoader()
        self.frame_handler = self.window_handler.get_window()
        self.video_handler = VideoHandler(640, 480, 4)
        self.input_handler = InputHandler()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 6)
        if load:
            self.model.load_state_dict(torch.load('model'))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

        #summary(self.model, (3, 460, 640))
        self.mem_handler = MemHandler()
        self.frame = self.get_frame()
        self.frames = []
        self.frames.append(rgb2gray(self.frame))
        self.frames.append(rgb2gray(self.frame))
        self.frames.append(rgb2gray(self.frame))
        #print(self.frames)
        #print(len(self.frames))
        #print(len(self.frames[0]))
        #print(len(self.frames[0][0]))

        if 'run' not in os.listdir():
            os.mkdir('run')

        self.path = 'run/exp' + str(len(glob.glob('run/exp*'))) + '/'
        self.death_path = 'run/exp' + str(len(glob.glob('run/exp*'))) + '/death/'
        os.mkdir('run/exp' + str(len(glob.glob('run/exp*'))))
        os.mkdir(self.path + 'death')
        self.save_model()

        self.actions = []
        self.hp = 10
        self.mem_handler = MemHandler()
        self.success = 0
        self.power = 0
        self.dg = 0
        self.available_moves = (['shift', 'up', 'z'], ['shift', 'down', 'z'], ['shift', 'left', 'z'], ['shift', 'right', 'z'], ['z'], ['x'])
        self.move_sequels = []


    def get_frame(self):
        return cv2.cvtColor(next(self.frame_handler).reshape(480, 640, 4), cv2.COLOR_BGRA2BGR)

    def save_death_frame(self):
        plt.imsave(
            self.death_path + 'death@' + str(datetime.datetime.now()).replace(' ', '-').replace('-', '_').replace(':','_') + '.png',
            self.frame, cmap=plt.cm.gray)

    def loss(self, output, target):
        return torch.mean(torch.abs(output - target))

    def carrot(self, action):
        _, tmp = torch.max(action, 0)
        ret = [0] * len(action)
        ret[tmp] = 1
        print('🥕', action)
        return torch.Tensor(ret)

    def stick(self, action):
        _, tmp = torch.max(action, 0)
        ret = [float(x) + (float(action[tmp] / (len(action) - 1))) for x in action]
        ret[tmp] = 0
        print('🏒', action)
        return torch.Tensor(ret)

    def soso(self, action):
        print('😶', action)
        return None

    def get_incentive(self):
        _, tmp = torch.max(self.actions[-1], 0)
        hp, bm, dg, score, power = self.mem_handler.get_score()

        if hp < self.hp:
            self.hp = hp
            self.save_death_frame()
            return -10

        if bm == 0 and tmp == 5:
            return -10

        if hp > 255:
            if self.hp != 255:
                self.save_death_frame()
            self.hp = 255
            self.save_model()
            return -20
        self.hp = hp
        if dg > self.dg:
            self.dg = dg
            return 10
        self.dg = dg

    def backward(self, reward):
        #print(reward)
        discount = 1
        if reward < 0:
            print(self.actions)
        else:
            return
        for i in reversed(self.actions):
            if reward < 0:
                if reward == -20:
                    self.input_handler.clear()
                    self.start = time.time()
                    time.sleep(max(1 / 3 - (time.time() - self.start), 0))
                    self.input_handler.set(['z'])
                    self.start = time.time()
                    time.sleep(max(1 / 3 - (time.time() - self.start), 0))
                loss = self.loss(i, self.stick(i))
                loss.backward()
            #elif reward > 0:
                #pass
                #loss = self.loss(i, self.carrot(i))
            #else:
                #loss = self.loss(i, self.soso(i))
            #loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.actions = []

    def save_model(self):
        torch.save(self.model.state_dict(),
                   self.path + 'model@' + str(datetime.datetime.now()).replace(' ', '-').replace('-', '_').replace(':',
                                                                                                                   '_'))

    def GameOn(self):
        self.frame = self.get_frame()
        self.start = time.time()
        while (True):
            if keyboard.is_pressed('/'):  # if key 'q' is pressed
                print('You Pressed the Key! :3')
                break  # finishing the loop\
            self.frame = self.get_frame()
            if len(self.actions) != 0:
                inc = self.get_incentive()
                #print(inc)
                if inc is not None:
                    self.backward(inc)
                    #self.actions = []
                if inc == -20:
                    continue

            self.frames.pop(0)
            self.frames.append(rgb2gray(self.frame) / 255)
            action = self.model(torch.Tensor([self.frames]) / 255)
            self.actions.append(action[0])
            if(len(self.actions)>3):
                self.actions.pop(0)
            #print(action)
            #print(action[0])
            conf, move = torch.max(action, 1)
            self.input_handler.set(self.available_moves[move[0]])
            if move[0] == 5:
                time.sleep(1 / 5)
                self.input_handler.clear()
                time.sleep(1 / 5)
            print(self.available_moves[move[0]],conf)
            # self.save_death_frame()
            print(1 / (time.time() - self.start))
            self.start = time.time()


        # self.save_frame(self.rgb2gray(frame))///
        torch.save(self.model.state_dict(), self.path + 'last')
        self.video_handler.close()
        self.input_handler.close()
        # self.window_handler.close()


if __name__ == "__main__":
    tmp = ProjectTouhou(load=True)
    #tmp = ProjectTouhou()
    tmp.GameOn()
