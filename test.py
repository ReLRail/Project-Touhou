import datetime
import glob
import os
import sys
import time
import random
import traceback
import torchvision.models as models
import cv2
import keyboard
import skimage.color
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from keyboard import press, release

from DeadExp import DeadExp as DE, DeadExp
from men_handler import MemHandler
from models.cnn import alexnet
from video_handler import VideoHandler
from window_loader import WindowLoader
import numpy as np
from numpy import log as ln

def rgb2gray(rgb):
    return skimage.color.rgb2gray(rgb)

class ProjectTouhou:

    def __init__(self,load = False):
        self.TouHou = WindowLoader()
        self.data = self.TouHou.get_window()
        self.video_handler = VideoHandler(640,480,4)
        self.net = models.resnet18(pretrained=True)
        self.net.fc = nn.Linear(self.net.fc.in_features, 5)
        self.moves = (['shift', 'up', 'z'], ['shift', 'down', 'z'], ['shift', 'left', 'z'], ['shift', 'right', 'z'], ['z'])
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001)
        self.frame = []
        self.frame_inv = []
        self.move_sequels = []
        self.death_count = 0

        self.last_action = []
        self.hp = 10
        self.mem_handler = MemHandler()
        self.success = 0
        self.power = 0
        self.dg = 0

        if 'run' not in os.listdir():
            os.mkdir('run')
        self.path = 'run/exp'+str(len(glob.glob('run/exp*'))) + '/'
        os.mkdir('run/exp'+str(len(glob.glob('run/exp*'))))
        os.mkdir(self.path+'death')
        torch.save(self.net.state_dict(), self.path + 'model@' + str(datetime.datetime.now()).replace(' ','-').replace('-','_').replace(':','_'))

        if load:
            self.net.load_state_dict(torch.load('model'))

    def get_frame(self):
        return cv2.cvtColor(next(self.data).reshape(480, 640, 4), cv2.COLOR_BGRA2BGR)

    def set_input(self, inputs):
        if inputs is not None:
            for i in inputs:
                if i is not None:
                    press(i)

    def release_input(self, inputs):
        if inputs is not None:
            for i in inputs:
                if i is not None:
                    release(i)
    def loss(self,output, target):
        return torch.mean(torch.abs(output - target))

    def carrot(self, action):
        ret = [None]*len(action)
        _, tmp = torch.max(action, 1)
        for i in range(len(action)):
            tmper = [0] * len(action[0])
            tmper[tmp[i]] = 1
            ret[i] = tmper
        print('🥕', action)
        #return None
        return torch.Tensor(ret)

    def stick(self, action):
        ret = [None]*len(action)
        conf, tmp = torch.max(action, 1)
        #print(conf, tmp,action)
        for i in range(len(action)):
            mov = [float(x) for x in action[i]]
            tmper = [float(conf[i]) / (len(action[i]) - 1)] * len(action[0])
            tmper[tmp[i]] = 0
            #print(mov,tmper)

            ret[i] = np.array(mov) + np.array(tmper)
            ret[i][tmp[i]]=0
            #print(tmp[i],ret[i])
        print('🏒', action)
        return torch.Tensor(ret)

    def soso(self, action):
        print('😶', action)
        return None

    def get_incentive(self, action):

        hp = self.mem_handler.get_mem(offsets=[0x4A57f4])
        bm = self.mem_handler.get_mem(offsets=[0x4A5800])
        dg = self.mem_handler.get_mem(offsets=[0x4A57C0])
        score = self.mem_handler.get_mem(offsets=[0x4A57B0])
        power = self.mem_handler.get_mem(offsets=[0x4A57E4])

        if hp < self.hp:
            self.hp = hp
            return self.stick(action)
        self.hp = hp
        success = ln(score)

        _, tmper = torch.max(action, 1)
        tmper = tmper[0]
        if not self.last_action == []:
            if self.last_action[-1] == tmper:
                if len(self.last_action) > 3:
                    success = self.success - 1
            else:
                self.last_action = []
        self.last_action.append(tmper)

        print(self.moves[tmper], hp, bm, dg, score, ln(score), power)

        if hp > 12:
            raise DeadExp
        if tmper == 5 and bm == 0:
            return self.stick(action)
        if power > self.power:
            self.power = power
            #return selfZ.carrot(action)
        if dg > self.dg:
            self.dg = dg
            #return self.carrot(action)
        self.dg = dg
        if success > self.success:
            self.success = success
            #return self.carrot(action)
        elif success == self.success:
            self.success = success
            #return self.soso(action)
        else:
            self.success = success
            return self.stick(action)

    def backward(self, reward):
        for i in reversed(self.move_sequels):
            if reward:
                loss = self.loss(i, self.stick(i))
            else:
                loss = self.loss(i, self.carrot(i))
            loss.backward()
            self.optimizer.step()

    def save_model(self):
        torch.save(self.net.state_dict(), self.path + 'model@' + str(datetime.datetime.now()).replace(' ','-').replace('-','_').replace(':','_'))

    def save_death_frame(self):
        plt.imsave(self.death_path + 'death@' + str(datetime.datetime.now()).replace(' ', '-').replace('-', '_').replace(':', '_') + '.png', self.frame, cmap=plt.cm.gray)

    def GameOn(self):
        self.start = time.time()
        move = None
        self.optimizer.zero_grad()
        frame = self.get_frame()

        tmp_frame = np.array(rgb2gray(frame))
        print(tmp_frame.shape)
        for x in range(3):
            self.frame.append(tmp_frame)

        print(np.array(self.frame).shape)

        action = self.net(torch.Tensor([self.frame]) / 255)

        while(True):
            if keyboard.is_pressed('/'):  # if key 'q' is pressed
                print('You Pressed A Key!')
                self.release_input(move)
                self.video_handler.close()
                break  # finishing the loop\
            #self.video_handler.write_frame(frame)
            #print('outputs -> ',outputs)

            try:
                frame = self.get_frame()
                target = self.get_incentive(action)
            except DeadExp:
                if self.death_count == 0:
                    self.save_death_frame()
                if self.death_count == 10:
                    break
                print('Dead :3')
                #e = sys.exc_info()[0]
                #print(e)
                #traceback.print_exc()

                self.release_input(move)
                self.start = time.time()
                self.death_count += 1
                try:
                    pass
                    #loss = self.loss(action, self.stick(action))
                    #loss.backward()
                    #self.optimizer.step()
                except:
                    pass
                #self.mem_handler = MemHandler()
                time.sleep(max(1 / 4 - (time.time() - self.start), 0))
                move = self.moves[4]
                self.set_input(move)
                print('=>', move)
                self.start = time.time()
                time.sleep(max(1 / 4 - (time.time() - self.start), 0))
                continue
            except:
                e = sys.exc_info()[0]
                print(e)
                traceback.print_exc()
                target = self.stick(action)
                flag = False
                print('exp and saved')
                self.release_input(move)
                break
            self.death_count = 0

            if target is not None:
                self.optimizer.zero_grad()
                loss = self.loss(action, target)
                loss.backward()
                self.optimizer.step()


            self.frame.pop(0)
            self.frame.append(rgb2gray(frame))

            action = self.net(torch.Tensor([self.frame]) / 255)
            #print(action)
            self.release_input(move)
            conf, move = torch.max(action, 1)
            move = self.moves[move[0]]
            #print(round(float(conf[0]),2), move, action)
            self.set_input(move)

            time.sleep(max(1 / 10 - (time.time() - self.start), 0))
            print(1 / (time.time() - self.start))
            self.start = time.time()
        self.save_model()
        torch.save(self.net.state_dict(), self.path + 'model')


if __name__ == "__main__":
    tmp = ProjectTouhou(load=False)
    #tmp = ProjectTouhou()
    tmp.GameOn()