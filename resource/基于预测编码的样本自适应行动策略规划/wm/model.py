#coding:utf-8

import torch
import random
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

PI = torch.FloatTensor([3.1415926])
       

class VAE(nn.Module):
    def __init__(self,args):
        super(VAE,self).__init__()
        self.args=args
        self.encoder = nn.Sequential(nn.Linear(args.state_shape, 512),
                                     nn.ReLU(),
                                     nn.Linear(512, 256) ,
                                     nn.ReLU(),
                                     nn.Linear(256, 128) ,
                                     nn.ReLU(),
                                     nn.Linear(128, args.vae_hidden_num*2) 
                                     )
        self.decoder = nn.Sequential(nn.Linear(args.vae_hidden_num, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 256) ,
                                     nn.ReLU(),
                                     nn.Linear(256, 512) ,
                                     nn.ReLU(),
                                     nn.Linear(512, args.state_shape) 
                                     )
        self._enc_mu = torch.nn.Linear(args.vae_hidden_num*2, args.vae_hidden_num)
        self._enc_log_sigma = torch.nn.Linear(args.vae_hidden_num*2, args.vae_hidden_num)


    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()

        self.z_mean = mu
        self.z_sigma = sigma
        if self.args.cuda:
            std_z=std_z.to('cuda:{}'.format(self.args.vae_device))
        
        return mu + sigma * Variable(std_z, requires_grad=False)  # Reparameterization trick    

    def forward(self, state):
        h_enc = self.encoder(state)
        z = self._sample_latent(h_enc)

        return self.decoder(z),z
 
class RNN(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, args):
        super(RNN, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions*args.n_agents)

    def forward(self, obs, hidden_state):
        x = F.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q.view(-1,self.args.n_agents,self.args.n_actions), h
    
class MDN_RNN(nn.Module):
    def __init__(self,args):
        super(MDN_RNN,self).__init__()
        self.args=args
        
        
        #组合信息经过两层，得到隐层，作为预测的输入
        self.fc=nn.Sequential(nn.Linear(in_features=32+self.args.rnn_hidden_dim, out_features=256),
                                nn.ReLU(),
                                nn.Linear(in_features=256, out_features=128),
                                nn.ReLU(),
                                )
        
        ##预测done标签
        self.done=nn.Linear(in_features=128, out_features=2)
                                
        #对onehot的动作进行编码
        self.acLiner=nn.Sequential(nn.Linear(in_features=self.args.n_agents*self.args.n_actions, out_features=256),
                                   nn.ReLU(),
                                   nn.Linear(in_features=256, out_features=128),
                                   nn.ReLU(),
                                   nn.Linear(in_features=128, out_features=32)
                                   )
        ##预测reward
        self.reward_pre=nn.Linear(in_features=128, out_features=1)
                                    
        
        ##预测可用动作
        self.action_avail=nn.Linear(in_features=128, out_features=self.args.n_agents*self.args.n_actions)
                        

        self._KI=nn.Linear(in_features=128, out_features=128)
        self._KI2=nn.Linear(in_features=128, out_features=self.args.vae_hidden_num*self.args.mdn_rnn_headings_num)

        self._enc_mu=nn.Linear(in_features=128,out_features=128)
        self._enc_mu2=nn.Linear(in_features=128,out_features=self.args.vae_hidden_num*self.args.mdn_rnn_headings_num)

        self._enc_log_sigma=nn.Linear(in_features=128,out_features=128)
        self._enc_log_sigma2=nn.Linear(in_features=128,out_features=self.args.vae_hidden_num*self.args.mdn_rnn_headings_num)

    def _sample_latent(self, h_enc,tao=1):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu2(F.relu(self._enc_mu(h_enc))).view(h_enc.size(0),h_enc.size(1),self.args.mdn_rnn_headings_num,self.args.vae_hidden_num)
        log_sigma = self._enc_log_sigma2(F.relu(self._enc_log_sigma(h_enc))).view(h_enc.size(0),h_enc.size(1),self.args.mdn_rnn_headings_num,self.args.vae_hidden_num)
        KI=  self._KI2(F.relu(self._KI(h_enc))).view(h_enc.size(0),h_enc.size(1),self.args.mdn_rnn_headings_num,self.args.vae_hidden_num)

        self.K=F.softmax(KI,dim=2)
        sigma =torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()

        self.z_mean = mu
        self.z_sigma = sigma

        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()

        z_n=mu + tao*sigma * Variable(std_z.cuda(), requires_grad=False)
        z_n=z_n*self.K

        return  z_n.sum(2) # Reparameterization trick  

    def forward(self,state,action,tao=1):
        a_em=self.acLiner(action) 
        st=torch.cat((state,a_em),dim=2)
        st=self.fc(st)
        
        #预测的是差值
        z_n=self._sample_latent(st,tao)
        done=F.softmax(self.done(st),dim=2)
        reward=self.reward_pre(st)
        action_avail=self.action_avail(st)
        
        return z_n,done,reward,action_avail
    
    def log_prob(self,delta):
        log_prob_cur= (torch.log((torch.exp((delta.unsqueeze(2).expand_as(self.z_mean) - self.z_mean).pow(2) / (-2 * self.z_sigma.pow(2) )) / (2*Variable(PI.cuda()).expand_as(
            self.z_sigma)).pow(0.5)/self.z_sigma*self.K).sum(2)+1e-8)).mean()
        
        return -log_prob_cur
    def loss_KL(self):
        mean_sq = (self.K*self.z_mean).mean(dim=2)
        mean_sq=mean_sq*mean_sq
        stddev_sq = (self.K*self.z_sigma).mean(dim=2)
        stddev_sq=stddev_sq*stddev_sq
        ###########
        return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)      