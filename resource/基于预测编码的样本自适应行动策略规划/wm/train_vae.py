#coding:utf-8

'''
利用vae网络对态势编码，展开训练与测试
'''

import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
#import torchvision
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from ..network.base_net import VAE
#from model import * 

from tensorboardX import SummaryWriter
from sklearn.model_selection import train_test_split


def train_vae(data,model_state_dict,args,train_step=0,pipe=None):
    '''
    首先对获取数据进行处理，之后展开训练
    返回model_state_dict
    '''
    
    path_w=args.tensorboard_path+'/vae_{}/ts_{}'.format(args.alg_name,train_step)
    if os.path.exists(path_w):
        import shutil
        shutil.rmtree(path_w)    
    writer=SummaryWriter(log_dir=path_w) 
    
    if pipe!=None:
        model_state_dict=pipe.recv()
    data_npy=process_data(data)
    train_set,test_set,_,_=train_test_split(data_npy,data_npy,test_size=0.1)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=32,shuffle=True)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.vae_batch,shuffle=True)
    
    vae=VAE(args)
    if args.vae_soft_update:
        vae_target=VAE(args)    
    
    if args.cuda:
        vae=vae.to('cuda:{}'.format(args.vae_device))
        if args.vae_soft_update:
            vae_target=vae_target.to('cuda:{}'.format(args.vae_device))   
    
        
    vae.load_state_dict(model_state_dict)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(vae.parameters(), lr=args.vae_lr)    
    for epoch in range(args.vae_maxepoch):
        for i, data in enumerate(trainloader, 0):
            inputs= data.float()
            if args.cuda:
                inputs=inputs.to('cuda:{}'.format(args.vae_device))
            
            dec,_ = vae(inputs)
            ll = latent_loss(vae.z_mean, vae.z_sigma)
            loss = criterion(dec, inputs)*1000 + ll
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            l = loss.item()
            #writer.add_graph(vae)
        if args.vae_soft_update:
            for param, target_param in zip(vae.parameters(), vae_target.parameters()):
                target_param.data.copy_(args.vae_tau * param.data + (1 - args.vae_tau) * target_param.data) 
            vae.load_state_dict(vae_target.state_dict())
        print(train_step,epoch)
        writer.add_scalar(tag='train_loss', scalar_value=l,global_step=epoch)
        test_total=test_vae(vae, testloader,args)
        writer.add_scalar(tag='test_loss', scalar_value=test_total,global_step=epoch)  
    if pipe!=None:
        #pipe.send([vae.cpu().state_dict()])
        torch.save(obj=vae.state_dict(), f=args.vae_load_model_path+'vae.pkl')
        pipe.send(0)
        
        
    writer.close()
    return vae.cpu().state_dict()

def process_data(data):
    '''
    将原始数据处理成为npy
    '''
    npy_data=[]
    for episode in data.tree.data:
        if episode==0:
            continue
        else:
            episode=episode['data']
            npy_data +=episode['s'][0].tolist()
            npy_data +=[episode['s_next'][0][-1].tolist()]
    
    return np.array(npy_data)
            

def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

def test_vae(vae_model,test_data,args):
    '''
    对测试集数据进行验证
    '''
    total=0
    criterion = nn.MSELoss()
    for i, data in enumerate(test_data, 0):
        inputs= data.float()
        if args.cuda:
            inputs= inputs.to('cuda:{}'.format(args.vae_device))
        
        dec,_ = vae_model(inputs)
        ll = latent_loss(vae_model.z_mean, vae_model.z_sigma)
        loss = criterion(dec, inputs)*1000 + ll

        total += loss.item()
    
    return total/(i+1)

if __name__=='__main__':
    from get_arg import *
    args=get_common_args()
    model_save_path='./model/vae/'
    if os.path.exists(args.tensorboard_path+'/vae'):
        import shutil
        shutil.rmtree(args.tensorboard_path+'/vae')    
    writer=SummaryWriter(log_dir=args.tensorboard_path+'/vae')
    
    data=np.load('./data/vae_data.npy')
    train_set,test_set,_,_=train_test_split(data,data,test_size=0.2)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=32,shuffle=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=32,shuffle=True)
    vae=VAE(args).to(args.device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(vae.parameters(), lr=args.vae_lr)
    
    for epoch in range(4000):
        for i, data in enumerate(trainloader, 0):
            inputs= data
            inputs= inputs.to(args.device).float()
            
            dec,_ = vae(inputs)
            ll = latent_loss(vae.z_mean, vae.z_sigma)
            loss = criterion(dec, inputs)*1000 + ll
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            l = loss.item()
            #writer.add_graph(vae)
        print(epoch,l)
        writer.add_scalar(tag='train_loss', scalar_value=l,global_step=epoch)
        if epoch%10==0 :
            torch.save(vae.state_dict(), model_save_path+str(epoch)+'.pkl') 
        test_total=test(vae, testloader)
        writer.add_scalar(tag='test_loss', scalar_value=test_total,global_step=epoch)
    writer.close()
            