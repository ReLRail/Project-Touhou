import numpy as np
import torch
import plotly.graph_objects as go
from numpy import log as ln
from DeadExp import DeadExp
from men_handler import MemHandler


class DopamineHandler:

    def __init__(self):
        # self.dopamine_level = [0]
        # self.dopamine_norm = 0
        # self.memory = []
        self.last_action = []
        self.last_hp = 10
        self.mem_handler = MemHandler()
        self.success = 0
        # self.fig = go.FigureWidget()/////
        # self.fig.add_scatter()
        # self.fig.show()

    def carrot(self, action):
        _, tmp = torch.max(action, 1)
        ret = [0] * len(action[0])
        ret[tmp[0]] = 1
        print('🥕', self.success)
        return None
        #return torch.Tensor(ret)

    def stick(self, action):
        _, tmp = torch.max(action, 1)
        ret = [1 / (len(action[0]) - 1)] * len(action[0])
        ret[tmp[0]] = 0
        print('🏒', self.success)
        return torch.Tensor(ret)

    def soso(self, action):
        print('😶', self.success)
        return None

    def get_incentive(self, frame, action):
        '''if self.memory == []:
            self.memory.append(frame)
            return self.carrot(action)
        delt = 100000
        for m in self.memory:
            tmp = np.mean(np.absolute(frame - m))
            delt = min(tmp,delt)
        self.dopamine_level.append(delt)
        _, tmper = torch.max(action, 1)
        if not self.last_action == []:aaaaaa
            if self.last_action[-1] == tmper:
                delt = delt - (delt/10)*len(self.last_action)
            else:
                self.last_action = []
        self.last_action.append(tmper)
        #with self.fig.batch_update():
            #self.fig.data[0].y = self.dopamine_level'''

        hp = self.mem_handler.get_mem(offsets=[0x4A57f4])
        bm = self.mem_handler.get_mem(offsets=[0x4A5800])
        dg = self.mem_handler.get_mem(offsets=[0x4A57C0])
        score = self.mem_handler.get_mem(offsets=[0x4A57B0])
        power = self.mem_handler.get_mem(offsets=[0x4A57E4])

        if hp < self.last_hp:
            self.last_hp = hp
            return self.stick(action)

        success = ln(score)

        _, tmper = torch.max(action, 1)
        if not self.last_action == []:
            if self.last_action[-1] == tmper:
                if len(self.last_action) > 3:
                    success = self.success - 1
            else:
                self.last_action = []
        self.last_action.append(tmper)

        print(hp, bm, dg, score, ln(score), power)

        if hp > 10:
            raise DeadExp
        if success > self.success:
            self.success = success
            return self.carrot(action)
        elif success == self.success:
            self.success = success
            return self.soso(action)
        else:
            self.success = success
            return self.stick(action)
