#!/usr/bin/env python
#coding:utf-8
"""
  Author:  doublestar_l --<Dr.>
  Purpose: 以微操的形式，对飞机进行机动控制
  Created: 2020/11/19
"""

import unittest
import torch
import random
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
from torch.distributions import Categorical
from .my_utils.prioritized_memory import Memory
from .common.utils import *

########################################################################
class RL_Agent():
    """"""

    #----------------------------------------------------------------------
    def __init__(self,args,share_queue=None,share_model=None,lock=True):
        """Constructor"""
        self.obs_n_space=args.obs_shape
        self.state_n_space=args.state_shape
        self.action_n_space=args.n_actions
        self.agents_n_size=args.n_agents
        
        if args.per:
            self.replay_buffer=Memory(args.buffer_size) 
            self.per_init()
        if args.mdn_rnn_gen:
            self.init_state_memory=deque(maxlen=10000)
            self.replay_buffer_mdn=Memory(args.buffer_size) 
        
        if args.alg == 'coma':
            from .policy.coma import COMA
            self.policy = COMA(args)
            

        self.args=args
    
    def per_init(self):
        '''
        对PER下的内容进行计算
        '''
        self.per_mean=0
        self.per_std=0
        self.per_step=0            
    def select_action(self,obs, last_action, avail_actions, evaluate=False,maven_z=None):
        '''
        决策神经网络，选择动作
        '''
        inputs = obs.copy()
        #avail_actions_ind = np.nonzero(avail_actions)[0]  # index of actions which can be choose
    
        # transform agent_num to onehot vector
   
        hidden_state = self.policy.eval_hidden
        '''
        3.25 在state中已经包含了，故而不再需要记录last_action
        '''
        #if self.args.last_action:
            #inputs = np.hstack((inputs, np.hstack(last_action)))
    
        # transform the shape of inputs from (42,) to (1,42)
        inputs = torch.tensor(inputs, dtype=torch.float32).view(1,1,-1)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
            hidden_state = hidden_state.cuda()
        
        if self.args.vae:
            inputs=self.policy.vae_model(inputs)[1]
            
        if self.args.mdn_rnn:
            action_padding=torch.zeros((1,1,self.args.n_agents*self.args.n_actions))
            if self.args.cuda:
                action_padding=action_padding.cuda()
            _,_,_,_,self.policy.eval_hidden=self.policy.mdn_model(inputs,action_padding,hidden_state)
            q_value = self.policy.eval_rnn(self.policy.eval_hidden)
        else:
            
            # get q value
            q_value, self.policy.eval_hidden, _ = self.policy.eval_rnn(inputs, hidden_state)
    
        # choose action from q value
        actions,prob_n = self._choose_action_from_softmax(q_value.cpu(), avail_actions, evaluate)
        action=actions.squeeze(0).detach().cpu().numpy()

        return action,prob_n      
        

    def _choose_action_from_softmax(self, inputs, avail_actions, epsilon=0, evaluate=False):
        """
        :param inputs: # q_value of all actions
        """
        ##action_num = avail_actions.sum(dim=1, keepdim=True).float().repeat(1, avail_actions.shape[-1])  # num of avail_actions
        action_num=avail_actions.sum(dim=2,keepdim=True)
        #限定inputs的大小，[-20,20]
        inputs=torch.clamp(inputs,min=-self.args.clamp_q,max=self.args.clamp_q)
        inputs[avail_actions.unsqueeze(0) == 0]=-self.args.clamp_q
        # 先将Actor网络的输出通过softmax转换成概率分布,取消第一维的batch
        prob = torch.nn.functional.softmax(inputs, dim=-1).squeeze(0)
        # add noise of epsilon
        prob = ((1 - epsilon) * prob + torch.ones_like(prob) * epsilon / action_num)
        prob[avail_actions == 0] = 0.0  # 不能执行的动作概率为0
        
        '''
        3.25获得归一化动作概率
        '''
        prob_n=prob/prob.sum(-1,keepdim=True)
        
        """
        不能执行的动作概率为0之后，prob中的概率和不为1，这里不需要进行正则化，因为torch.distributions.Categorical
        会将其进行正则化。要注意在训练的过程中没有用到Categorical，所以训练时取执行的动作对应的概率需要再正则化。
        """

        if epsilon == 0 and evaluate:
            action = torch.argmax(prob)
        else:
            action = Categorical(prob).sample().long()
        return action,torch.gather(prob_n, dim=2, index=action.unsqueeze(2)).squeeze(2).detach().numpy()

    def _get_max_episode_len(self, batch):
        if True:
            shape=batch['terminated'].shape
            return shape[1]
        
        terminated = batch['terminated']
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                ####
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        return max_episode_len

    def train(self, batch, train_step=0, epsilon=None,vir=False):  # coma needs epsilon for training

        # different episode has different length, so we need to get max length of the batch
        max_episode_len = self._get_max_episode_len(batch)#原代码设定了episode的最大长度，是一个episode进行整体训练，我们这里切成小块训练
        #max_episode_len=self.args.episode_limit
        for key in batch.keys():
            if key != 'z':
                batch[key] = batch[key][:, :max_episode_len]
        self.policy.learn(batch, max_episode_len, train_step, epsilon,vir)
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            self.policy.save_model(train_step)

class DQN(nn.Module):
    """
    此处使用dqn实现对单元的控制
    """

    #----------------------------------------------------------------------
    def __init__(self, entity_size=[(6,13),(52,12),(6,6),(52,10),(14)],action_size=27):
        super(DQN, self).__init__()
        self.cnn1=nn.Sequential(nn.BatchNorm2d(num_features=1),
                                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1,entity_size[0][1])),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(entity_size[0][0],1))
                                )
        
        self.cnn2=nn.Sequential(nn.BatchNorm2d(num_features=1),
                                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1,entity_size[1][1])),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(entity_size[1][0],1))
                                )   
               
        self.cnn3=nn.Sequential(nn.BatchNorm2d(num_features=1),
                                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1,entity_size[2][1])),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(entity_size[2][0],1))
                                )  
                
        self.cnn4=nn.Sequential(nn.BatchNorm2d(num_features=1),
                                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1,entity_size[3][1])),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(entity_size[3][0],1))
                                )  
               
        
        self.state_fc=nn.Sequential(nn.LayerNorm(entity_size[4]),
                                    nn.Linear(in_features=entity_size[4], out_features=256),
                                     nn.ReLU(),
                                     nn.LayerNorm(256),
                                     nn.Linear(in_features=256, out_features=512),
                                     nn.ReLU(),
                                     nn.LayerNorm(512),
                                     nn.Linear(in_features=512, out_features=256) ,
                                     nn.ReLU(),
                                     nn.LayerNorm(256),
                                     nn.Linear(in_features=256, out_features=128) 
                                     )
        self.out_fc=nn.Sequential(nn.LayerNorm(256),
                                  nn.Linear(in_features=256, out_features=256) ,
                                     nn.ReLU(),
                                     nn.LayerNorm(256),
                                     nn.Linear(in_features=256, out_features=128),
                                     nn.ReLU(),  
                                     nn.LayerNorm(128),
                                     nn.Linear(in_features=128, out_features=27) 
                                    )        
        ##############
        
        self.fc1=nn.Sequential(nn.BatchNorm1d(entity_size[0][1]*entity_size[0][0]),
                               nn.Linear(entity_size[0][1]*entity_size[0][0], 128),nn.ReLU(),
                               nn.LayerNorm(128),
                               nn.Linear(128,64),nn.ReLU(),
                               nn.LayerNorm(64),
                               nn.Linear(64,16)
                               )        
        
        self.fc2=nn.Sequential(nn.BatchNorm1d(entity_size[1][1]*entity_size[1][0]),
                               nn.Linear(entity_size[1][1]*entity_size[1][0], 1024),nn.ReLU(),
                               nn.LayerNorm(1024),
                               nn.Linear(1024,256),nn.ReLU(),
                               nn.LayerNorm(256),
                               nn.Linear(256,64),nn.ReLU(),
                               nn.LayerNorm(64),
                               nn.Linear(64,16)
                               )         
        
        self.fc3=nn.Sequential(nn.BatchNorm1d(entity_size[2][1]*entity_size[2][0]),
                               nn.Linear(entity_size[2][1]*entity_size[2][0], 128),nn.ReLU(),
                               nn.LayerNorm(128),
                               nn.Linear(128,64),nn.ReLU(),
                               nn.LayerNorm(64),
                               nn.Linear(64,16)
                               )        
        
        self.fc4=nn.Sequential(nn.BatchNorm1d(entity_size[3][1]*entity_size[3][0]),
                               nn.Linear(entity_size[3][1]*entity_size[3][0], 1024),nn.ReLU(),
                               nn.LayerNorm(1024),
                               nn.Linear(1024,256),nn.ReLU(),
                               nn.LayerNorm(256),
                               nn.Linear(256,64),nn.ReLU(),
                               nn.LayerNorm(64),
                               nn.Linear(64,16)
                               )          
        
        self.state_fc5=nn.Sequential(nn.LayerNorm(entity_size[4]),
                                    nn.Linear(in_features=entity_size[4], out_features=256),
                                     nn.ReLU(),
                                     nn.LayerNorm(256),
                                     nn.Linear(in_features=256, out_features=128),
                                     nn.ReLU(),
                                     nn.LayerNorm(128),
                                     nn.Linear(in_features=128, out_features=64)
                                     )
        self.out_fc2=nn.Sequential(nn.LayerNorm(128),
                                  nn.Linear(in_features=128, out_features=256) ,
                                     nn.ReLU(),
                                     nn.LayerNorm(256),
                                     nn.Linear(in_features=256, out_features=128),
                                     nn.ReLU(),  
                                     nn.LayerNorm(128),
                                     nn.Linear(in_features=128, out_features=27) 
                                    )        
        
    def forward2(self,NN_input_my_airs,NN_input_my_weapons,NN_input_en_airs,NN_input_en_weapons,NN_input_state):
        '''
        
        '''
        out_cnn1=self.cnn1(NN_input_my_airs).view(-1,32)
        out_cnn2=self.cnn2(NN_input_my_weapons).view(-1,32)
        out_cnn3=self.cnn3(NN_input_en_airs).view(-1,32)
        out_cnn4=self.cnn4(NN_input_en_weapons).view(-1,32)
        out_state=self.state_fc(NN_input_state)
        
        state=torch.cat((out_cnn1,out_cnn2,out_cnn3,out_cnn4),1).view(-1,1,128)
        size=state.size()
        state=state.expand((size[0],6,128))
        state=torch.cat((state,out_state),2)
        
        out=self.out_fc(state)
        return out
    def forward(self,NN_input_my_airs,NN_input_my_weapons,NN_input_en_airs,NN_input_en_weapons,NN_input_state):
        out_cnn1=self.fc1(NN_input_my_airs.view(-1,13*6)).view(-1,16)
        out_cnn2=self.fc2(NN_input_my_weapons.view(-1,52*12)).view(-1,16)
        out_cnn3=self.fc3(NN_input_en_airs.view(-1,6*6)).view(-1,16)
        out_cnn4=self.fc4(NN_input_en_weapons.view(-1,52*10)).view(-1,16)
        out_state=self.state_fc5(NN_input_state)
        
        state=torch.cat((out_cnn1,out_cnn2,out_cnn3,out_cnn4),1).view(-1,1,64)
        size=state.size()
        state=state.expand((size[0],6,64))
        state=torch.cat((state,out_state),2)
        
        out=self.out_fc2(state)
        return out  

class QmixNet(nn.Module):
    def __init__(self,entity_size=[(6,13),(52,12),(6,6),(52,10)],action_size=27, agent_size=6):
        super(QmixNet, self).__init__()
        
        self.fc1=nn.Sequential(nn.BatchNorm1d(entity_size[0][1]*entity_size[0][0]),
                                   nn.Linear(entity_size[0][1]*entity_size[0][0], 128),nn.ReLU(),
                                   nn.LayerNorm(128),
                                   nn.Linear(128,64),nn.ReLU(),
                                   nn.LayerNorm(64),
                                   nn.Linear(64,32)
                                   )        
    
        self.fc2=nn.Sequential(nn.BatchNorm1d(entity_size[1][1]*entity_size[1][0]),
                                   nn.Linear(entity_size[1][1]*entity_size[1][0], 1024),nn.ReLU(),
                                   nn.LayerNorm(1024),
                                   nn.Linear(1024,256),nn.ReLU(),
                                   nn.LayerNorm(256),
                                   nn.Linear(256,64),nn.ReLU(),
                                   nn.LayerNorm(64),
                                   nn.Linear(64,32)
                                   )         
    
        self.fc3=nn.Sequential(nn.BatchNorm1d(entity_size[2][1]*entity_size[2][0]),
                                   nn.Linear(entity_size[2][1]*entity_size[2][0], 128),nn.ReLU(),
                                   nn.LayerNorm(128),
                                   nn.Linear(128,64),nn.ReLU(),
                                   nn.LayerNorm(64),
                                   nn.Linear(64,32)
                                   )        
    
        self.fc4=nn.Sequential(nn.BatchNorm1d(entity_size[3][1]*entity_size[3][0]),
                                   nn.Linear(entity_size[3][1]*entity_size[3][0], 1024),nn.ReLU(),
                                   nn.LayerNorm(1024),
                                   nn.Linear(1024,256),nn.ReLU(),
                                   nn.LayerNorm(256),
                                   nn.Linear(256,64),nn.ReLU(),
                                   nn.LayerNorm(64),
                                   nn.Linear(64,32)
                                   )          
    
        self.state_fc5=nn.Sequential(nn.LayerNorm(128),
                                         nn.Linear(in_features=128, out_features=256),
                                        nn.ReLU(),
                                         nn.LayerNorm(256),
                                         nn.Linear(in_features=256, out_features=128),
                                         nn.ReLU(),
                                         nn.LayerNorm(128),
                                         nn.Linear(in_features=128, out_features=64)
                                         ) 
        #生成的向量为状态
        self.hyper_w1 = nn.Sequential(nn.LayerNorm(64),
                                      nn.Linear(in_features=64, out_features=256),
                                      nn.LayerNorm(256),
                                      nn.Linear(in_features=256, out_features=agent_size*16)
                                      )
        self.hyper_b1 = nn.Sequential(nn.LayerNorm(64),
                                      nn.Linear(in_features=64, out_features=128),
                                      nn.LayerNorm(128),
                                      nn.Linear(in_features=128, out_features=16)
                                      )
        self.hyper_w2 = nn.Sequential(nn.LayerNorm(64),
                                      nn.Linear(in_features=64, out_features=32),
                                      nn.LayerNorm(32),
                                      nn.Linear(in_features=32, out_features=16)
                                      )  
        self.hyper_b2 = nn.Sequential(nn.LayerNorm(64),
                                      nn.Linear(in_features=64, out_features=32),
                                      nn.LayerNorm(32),
                                      nn.Linear(in_features=32, out_features=1)
                                      )
        
    def forward(self,NN_input_my_airs,NN_input_my_weapons,NN_input_en_airs,NN_input_en_weapons,q_values):
        out_cnn1=self.fc1(NN_input_my_airs.view(-1,13*6)).view(-1,16)
        out_cnn2=self.fc2(NN_input_my_weapons.view(-1,52*12)).view(-1,16)
        out_cnn3=self.fc3(NN_input_en_airs.view(-1,6*6)).view(-1,16)
        out_cnn4=self.fc4(NN_input_en_weapons.view(-1,52*10)).view(-1,16)
        
        
        state=torch.cat((out_cnn1,out_cnn2,out_cnn3,out_cnn4),1).view(-1,1,128)
        
        f_state=self.state_fc5(state)
        q_values=q_values.unsqueeze(1)
        w1=torch.abs(self.hyper_w1(f_state)).view(-1,6,16)#(batch_size,6,16)
        b1=self.hyper_b1(f_state)#(batch_size,6,16)
        hidden = F.elu(torch.bmm(q_values, w1) + b1)
        
        w2=torch.abs(self.hyper_w2(f_state)).view(-1,16,1)
        b2=self.hyper_b2(f_state)
        q_total = torch.bmm(hidden, w2) + b2
        
        return q_total.view(-1)
        

def get_latitude(latitude):
    '''
    将纬度缩放到0-1之间，定义边界北纬22-29
    '''
    lat=(latitude-22)/7.0
    order=sorted([0,lat,1])
    return order[1]
    
def get_longitude(longitude):
    '''
    将经度缩放到0-1之间，定义边界东经148.5-163.5
    '''
    lon=(longitude-148.5)/7.0
    order=sorted([0,lon,1])
    return order[1]

def get_speed(speed):
    '''
    对速度进行缩放，648.2-1703.84
    '''
    speed=(speed*1.852)/(4300.0)
    order=sorted([0,speed,1])
    return order[1] 
def get_weapon(weaponsValid):
    '''
    对武器进行统计，统计51,945,826
    '''
    weapon_list=[]
    for weapon_id,weapon_maxNum in zip([51,945,826],[6.0,2.0,6.0]):
        if weapon_id in weaponsValid:
            weapon_list.append(weaponsValid[weapon_id]/weapon_maxNum)
        else:
            weapon_list.append(0)
    return weapon_list

if __name__ == '__main__':
    unittest.main()