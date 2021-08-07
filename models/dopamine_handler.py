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
        self.hp = 10
        self.mem_handler = MemHandler()
        self.success = 0
        self.power = 0
        self.dg = 0
        self.moves = (['shift', 'up', 'z'], ['shift', 'down', 'z'], ['shift', 'left', 'z'], ['shift', 'right', 'z'], ['z'], ['x'])
        # self.fig = go.FigureWidget()/////
        # self.fig.add_scatter()
        # self.fig.show()

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
