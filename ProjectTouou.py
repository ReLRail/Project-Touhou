import sys
import time
import random
import traceback

import cv2
import keyboard
import torch
from torch import nn, optim
from keyboard import press, release

from DeadExp import DeadExp as DE, DeadExp
from men_handler import MemHandler
from models.cnn import alexnet
from models.dopamine_handler import DopamineHandler
from video_handler import VideoHandler
from window_loader import WindowLoader
import numpy as np


class ProjectTouhou:

    def __init__(self,load = False):
        self.dopamine_handler = DopamineHandler()
        self.TouHou = WindowLoader()
        self.data = self.TouHou.get_window()
        self.video_handler = VideoHandler(640,480,4)
        self.net = alexnet()
        self.moves = (['shift', 'up', 'z'], ['shift', 'down', 'z'], ['shift', 'left', 'z'], ['shift', 'right', 'z'], ['z'], ['x'])
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001)
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
        #print('Delt -> ',output - torch.Tensor([[0,0,0,0,1]]))
        #print('Delt -> ',torch.abs(output - torch.Tensor([[0,0,0,0,1]])))
        #print('Delt -> ',torch.mean(torch.abs(output - torch.Tensor([[0,0,0,0,1]]))))
        #return torch.mean(torch.abs(output - torch.Tensor([[0,0,0,0.5,0.5]])))
        return torch.mean(torch.abs(output - target))

    def GameOn(self):
        self.start = time.time()
        move = None
        self.optimizer.zero_grad()
        frame = self.get_frame()
        action = self.net(torch.Tensor([np.einsum('ijk->kji', frame)]) / 255)
        flag = True
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
                target = self.dopamine_handler.get_incentive(frame, action)
            except DeadExp:

                e = sys.exc_info()[0]
                #print(e)
                traceback.print_exc()

                self.release_input(move)
                self.start = time.time()
                try:
                    loss = self.loss(action, self.dopamine_handler.stick(action))
                    loss.backward()
                    self.optimizer.step()
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
                target = self.dopamine_handler.stick(action)
                flag = False
                print('exp and saved')
                self.release_input(move)
                break

            if target is not None:
                try:
                    loss = self.loss(action, target)
                    loss.backward()
                    self.optimizer.step()
                except:
                    e = sys.exc_info()[0]
                    print(e)
                    traceback.print_exc()




            self.optimizer.zero_grad()
            action = self.net(torch.Tensor([np.einsum('ijk->kji',frame)])/255)
            self.release_input(move)
            conf, move = torch.max(action, 1)
            move = self.moves[move[0]]
            #print(round(float(conf[0]),2), move, action)
            self.set_input(move)

            #print(1 / (time.time() - self.start))
            time.sleep(max(1 / 10 - (time.time() - self.start), 0))
            self.start = time.time()
        torch.save(self.net.state_dict(), 'model')


if __name__ == "__main__":
    tmp = ProjectTouhou(load=True)
    #tmp = ProjectTouhou()
    tmp.GameOn()
