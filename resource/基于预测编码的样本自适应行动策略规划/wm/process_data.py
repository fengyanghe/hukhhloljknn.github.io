# -*- coding:utf-8 -*-

import numpy as np
import os
from get_arg import *
'''
根据收集到的数据，整理成为vae 和mdn_rnn网络训练所需要的数据
存储数据为 episode, step, tuple
vae网络：（state）
mdn_rnn网络：（state, action, reward, done, next_state, avail_action）
'''

def get_data(file_path='../replay_data'):
    '''
    遍历保存数据目录下的所有文件，从保存数据的npy中取出所有数据
    return：list
    '''
    data=[]
    for root,dirs,files in os.walk(file_path):
        print(root,dirs,files)
        for file in files:
            temp_data=np.load(file=root+'/'+file, allow_pickle=True)
            data.append(temp_data)
    return data

def get_vae_data(data):
    '''
    根据获得的原始data，获得vae的网络学习数据
    '''
    vae_data=[]
    for episode in data:
        episode=episode.item()
        for state in episode['s']:
            vae_data.append(state)
        vae_data.append(episode['s_next'][-1])
    
    np.save('./data/vae_data.npy',np.array(vae_data))

def get_mdn_rnn(data,args):
    '''
    根据数据，获得mdn_rnn网络的学习数据
    首先将数据首位相连，之后再进行切割，按照指定长度
    return: list
    '''
    mdn_length=args.mdn_rnn_length
    episode_batch=data[0].item()
    data.pop(0)
    for episode in data:
        episode=episode.item()
        for key in episode_batch.keys():
            episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)    
    
    import math        
    length= math.ceil(len(episode_batch['s'])/mdn_length) 
    mdn_rnn_date=[]
    for i in range(1,length-1):
        start_ind=(i-1)*mdn_length
        end_ind=i*mdn_length
        temp_dict={}
        for k,v in episode_batch.items():
            temp_dict[k]=v[start_ind:end_ind]
        
        mdn_rnn_date.append(temp_dict)
    
    temp_dict={}
    for k,v in episode_batch.items():
        temp_dict[k]=v[len(episode_batch['s'])-mdn_length:]
        
    mdn_rnn_date.append(temp_dict)
    
    np.save('./data/mdn_rnn_data.npy',np.array(mdn_rnn_date))    

if __name__=='__main__':
    data=get_data()
    #get_vae_data(data)
    args=get_common_args()
    get_mdn_rnn(data,args)
    