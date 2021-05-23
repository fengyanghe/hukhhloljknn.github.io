#coding:utf-8

'''
利用mdn_rnn网络对环境的转移，展开训练与测试
'''
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import torch.multiprocessing as mp
import pylab
import copy
#from model import *
from ..network.base_net import MDN_RNN,VAE,Critic, Actor
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
#from get_arg import *
from sklearn.model_selection import train_test_split
from torch.distributions import Categorical
import datetime
import traceback

class TData(Dataset):
    def __init__(self,data):
        self.dataset=data
        
    def __len__(self):
        return len(self.dataset)
   
    def __getitem__(self,idx): 

        states=torch.from_numpy(self.dataset[idx]['s']).float()
        actions=torch.from_numpy(self.dataset[idx]['u'])
        dones=torch.from_numpy(self.dataset[idx]['terminated']).long()
        rewards=torch.from_numpy(self.dataset[idx]['r']).float()
        n_states=torch.from_numpy(self.dataset[idx]['s_next']).float()
        avail_u_next=torch.from_numpy(self.dataset[idx]['avail_u_next']).long()
        
        return states,actions,rewards,dones,n_states,avail_u_next

def get_data(data,args):
    mdn_length=args.mdn_rnn_length
    episode_batch=[]
    for episode in data.tree.data:
        if episode==0:
            continue
        else: 
            episode=episode['data']
            if episode_batch==[]:
                episode_batch=episode.copy()
                continue
            for key in episode_batch.keys():
                episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=1)    
    for k,v in episode_batch.items():
        episode_batch[k]=v.squeeze(0)
    
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
    return np.array(mdn_rnn_date)

def train_mdn_rnn(data, mdn_rnn_model, vae_model,cirtic_model, args, pipe=None):
    '''
    根据传来的data 做处理
    '''
    
    ##模型加注
    vae_net=VAE(args)
    critic_net=Critic(args.mdn_rnn_hidden_num, args)
    mdn_rnn_net=MDN_RNN(args) 
    if args.mdn_rnn_soft_update:
        mdn_rnn_net_target=MDN_RNN(args) 
        mdn_rnn_net_target.load_state_dict(mdn_rnn_model)
    
    vae_net.load_state_dict(vae_model)
    mdn_rnn_net.load_state_dict(mdn_rnn_model)
    critic_net.load_state_dict(cirtic_model)
    
    if args.cuda:
        vae_net = vae_net.cuda()
        mdn_rnn_net = mdn_rnn_net.cuda()
        critic_net = critic_net.cuda() 
        if args.mdn_rnn_soft_update:
            mdn_rnn_net_target=mdn_rnn_net_target.cuda()        
        
    ##数据加注
    p_data=get_data(data, args)
    train_set,test_set,_,_=train_test_split(p_data,p_data,test_size=0.1)
    train_set = TData(train_set)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.mdn_rnn_batch,shuffle=True)
    #testloader = torch.utils.data.DataLoader(test_set, batch_size=args.mdn_rnn_batch,shuffle=True)    
    
    ##参数加注
    eval_parameters=list(critic_net.parameters()) + list(mdn_rnn_net.parameters())
    opt = torch.optim.Adam(eval_parameters, lr=args.mdn_rnn_lr)    
    loss_function_avail_action=nn.BCEWithLogitsLoss()
    loss_function_reward=nn.MSELoss()
    loss_function_done=nn.CrossEntropyLoss()    
    
      
    
    for epoch in range(args.mdn_rnn_maxepoch):
        
        for i, data in enumerate(trainloader):
            states,actions,rewards,dones,n_states,avail_u_next=data
            states,actions,rewards,dones,n_states,avail_u_next=states,actions,rewards,dones,n_states,avail_u_next
            if args.cuda:
                states,actions,rewards,dones,n_states,avail_u_next=\
                states.cuda(),actions.cuda(),rewards.cuda(),dones.cuda(),\
                n_states.cuda(),avail_u_next.cuda()            
            #获得状态的编码
            _, e_z = vae_net(states)
            _, e_n_z = vae_net(n_states)
            e_z, e_n_z=e_z.detach(), e_n_z.detach()
            
            #此处通过循环获得隐层信息
            args.mdn_rnn_batch=states.size()[0]
            h_n=torch.ones((1,args.mdn_rnn_batch,args.mdn_rnn_hidden_num))
            if args.cuda:
                h_n=h_n.cuda()
            rnn_out=[]
            done_pre=[]
            reward_pre=[]
            action_avail_pre=[]
            h_n_pre=[]
            
            shape=states.shape
            one_hot_action = torch.FloatTensor(shape[0],args.mdn_rnn_length,args.n_agents,args.n_actions).zero_()
            
            
            if args.cuda:
                one_hot_action = one_hot_action.cuda()
            one_hot_action.scatter_(3, actions, 1) 
            one_hot_action = one_hot_action.view(shape[0],shape[1],-1)
            
            delta_z = (e_n_z-e_z).detach()
            loss_delta_z = 0            
            
            for i in range(args.mdn_rnn_length):
                _,done,reward,action_avail,h_n=mdn_rnn_net(e_z[:,i,:].unsqueeze(1),one_hot_action[:,i].unsqueeze(1),h_n)
                h_n=h_n*(1-dones[:,i,:].unsqueeze(1))
                rnn_out.append(h_n)
                done_pre.append(done)
                reward_pre.append(reward)
                action_avail_pre.append(action_avail) 
                h_n_pre.append(h_n)
                loss_delta_z += mdn_rnn_net.log_prob(delta_z[:,i,:].unsqueeze(1))/ shape[0] 
                h_n=h_n.view(1,-1,args.mdn_rnn_hidden_num)
                
                
            rnn_outs=torch.stack(rnn_out,dim=1).float()
            done_pre=torch.stack(done_pre,dim=1).float()
            reward_pre=torch.stack(reward_pre,dim=1).float()
            action_avail_pre=torch.stack(action_avail_pre,dim=1).float()
            #h_n_pre=torch.stack(h_n_pre,dim=1) 
            #此处，对预测信息进行输出
            
                      
            #_,done_pre,reward_pre,action_avail_pre=mdn_rnn_net(rnn_outs,one_hot_action.view(args.mdn_rnn_batch,args.mdn_rnn_length,-1))
            #print('a')
            
            loss_action_pre=loss_function_avail_action(action_avail_pre.view(shape[0],args.mdn_rnn_length,args.n_agents,args.n_actions),
                                                       avail_u_next.float())
            loss_reward=loss_function_reward(reward_pre.squeeze(-1),rewards)
            loss_done=loss_function_done(done_pre.view(-1,2),dones.view(-1).long())
            
            ##计算critic
            pred_v=critic_net(rnn_outs.float())
            h_n_t=torch.ones((1,args.mdn_rnn_batch,args.mdn_rnn_hidden_num))
            if args.cuda:
                h_n_t = h_n_t.cuda()
            _,_,_,_,h_n_t=mdn_rnn_net(e_n_z,one_hot_action,h_n_t)
            target_v=critic_net(h_n_t).detach()
            target_v=rewards+0.99*target_v*(1-dones)
            
            loss_value=F.mse_loss(pred_v.view(-1,1), target_v.detach().view(-1,1))

            
            loss_total=loss_action_pre+loss_reward+loss_done+loss_delta_z+0.5*loss_value
            
            opt.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(eval_parameters, args.grad_norm_clip)
            opt.step()
            
        
        if args.mdn_rnn_soft_update:
            for param, target_param in zip(mdn_rnn_net.parameters(), mdn_rnn_net_target.parameters()):
                target_param.data.copy_(args.mdn_rnn_tau * param.data + (1 - args.mdn_rnn_tau) * target_param.data) 
            mdn_rnn_net.load_state_dict(mdn_rnn_net_target.state_dict())            
        print('{} loss：{}'.format(epoch,loss_total.item()))
    if True:
        if args.mdn_rnn_soft_update:
            torch.save(obj=mdn_rnn_net_target.state_dict(), f=args.vae_load_model_path+'mdn_rnn_net.pkl')
        else:
            torch.save(obj=mdn_rnn_net.state_dict(), f=args.vae_load_model_path+'mdn_rnn_net.pkl')
        torch.save(obj=critic_net.state_dict(), f=args.vae_load_model_path+'critic_net.pkl')
        #pipe.send(0)
    #return mdn_rnn_net.state_dict()
        
                
def mdn_rnn_gen_data(init_data,vae_state_dict,mdn_state_dict,actor_state_dict,args,pipe=None):
    '''
    根据初始状态，获得虚假数据
    '''
    try:
        vae_net=VAE(args).to(args.mdn_rnn_device)
        mdn_net=MDN_RNN(args).to(args.mdn_rnn_device)
        mdn_net_pre=MDN_RNN(args).to(args.mdn_rnn_device)
        actor_net=Actor(args.mdn_rnn_hidden_num, args).to(args.mdn_rnn_device)
        vae_net.load_state_dict(vae_state_dict)
        mdn_net.load_state_dict(mdn_state_dict)
        mdn_net_pre.load_state_dict(mdn_state_dict)
        actor_net.load_state_dict(actor_state_dict)
        
        hidden=torch.zeros((1,1,args.mdn_rnn_hidden_num)).to(args.mdn_rnn_device)
        hidden_pre=torch.zeros((1,1,args.mdn_rnn_hidden_num)).to(args.mdn_rnn_device)
        episode_sample={
            's':[],#整体状态，1 x state_n_space
            'u':[],#智能体的动作，数字 agent_size x 1
            'u_onehot':[],#智能体的动作onehot编码，agent_size x 32
            'avail_u':[],#智能体的可用动作，agent_size x 32
            'r':[],#智能体的奖赏，agent_size x 1
            'terminate':[],#环境是否结束 1
            's_next':[], #下一时刻的s
            'next_avail_u':[],#下时刻的可用动作
            'padded':[],#智能体可能会消亡，此处为标示 agent_size x 1
            'prob_n':[],#记录智能体当时选择动作的概率
                                 }
        
        
        vae_init_state=np.reshape(init_data[0], (1,1,-1))
        avail_actions = init_data[1]
        
        vae_init_state=torch.from_numpy(vae_init_state).float().to(args.mdn_rnn_device)
        _, vae_state=vae_net(vae_init_state)
        action_tensor=torch.zeros((1,1,args.n_agents*args.n_actions)).to(args.mdn_rnn_device)
        terminated=False
        index=0
        while not terminated and index<args.episode_limit:
            index +=1
            with torch.no_grad():
                ##根据hidden，生成动作概率
                z_n,done,reward,action_avail,hidden=mdn_net(vae_state,action_tensor,hidden)
                action_prob=actor_net(hidden)
                
                ##根据可选动作的概率，选择动作
                prob = torch.nn.functional.softmax(action_prob, dim=-1).squeeze(0)
                avail_actions_tensor=torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
                prob=prob.cpu()
                prob[avail_actions_tensor == 0] = 0.0
                prob_value,_=prob.max(-1)
                for ind, v in enumerate(prob_value[0]):
                    if not v:
                        prob[0,ind,0]=1.0
                
                with open('tr.txt',mode='a') as fo:
                    fo.write('action_prob:{} \n prob:{} \n'.format(action_prob,prob))
                prob=prob.to(args.mdn_rnn_device)
                actions = Categorical(prob).sample().long()
                with open('tr.txt',mode='a') as fo:
                    fo.write('action:{}\n '.format(actions))                
                #获得归一化动作概率
                #prob_n=prob/(prob.sum(-1,keepdim=True)+1e-10)
                #prob_select=prob_n*
                #torch.gather(prob_n, dim=2, index=actions.unsqueeze(2)).squeeze(2).detach().cpu().numpy()
                
                    
                actions=actions.squeeze(0).cpu().numpy()
                
                
                ##获得动作的one-hot编码
                actions_onehot=[]
                for ind in range(args.n_agents):
                    action=actions[ind]
                    action_onehot = [0 for _ in range(args.n_actions)]
                    action_onehot[action] = 1
                    actions_onehot.append(action_onehot)
                
                #prob_select=(prob_n*torch.Tensor(actions_onehot).to(args.mdn_rnn_device)).sum(-1)    
                ##根据动作的编码，以及当前vae信息，以及hidden信息，推导下时刻的预测信息
                action_tensor=torch.Tensor(actions_onehot).view(1,1,args.n_agents*args.n_actions).to(args.mdn_rnn_device)
                delta_z,done,reward,action_avail,hidden_pre=mdn_net_pre(vae_state,action_tensor,hidden_pre)
                
                ##根据推导信息，整理形成下一时刻的输入信息
                next_vae_state=vae_state+delta_z
                terminated=False if done[0,0,1]<0.8 else True
                action_avail=action_avail.view(1,1,args.n_agents,args.n_actions)
                next_action_avail=(action_avail>0.5).long().cpu().squeeze(0).squeeze(0).numpy().tolist()
                reward=reward[0,0,0].item()
                
                episode_sample['s'].append(vae_state.cpu().numpy().squeeze(0).squeeze(0))
                episode_sample['u'].append(np.reshape(actions, [args.n_agents, 1]))
                episode_sample['u_onehot'].append(np.array(actions_onehot))
                episode_sample['r'].append([reward])
                episode_sample['terminate'].append([terminated])
                episode_sample['padded'].append([0])
                episode_sample['avail_u'].append(np.array(avail_actions))
                episode_sample['s_next'].append(next_vae_state.cpu().view(-1).numpy())
                episode_sample['next_avail_u'].append(np.array(next_action_avail))
                '''
                获得原始概率值
                '''
                #episode_sample['prob_n'].append(prob_select[0])
                
                vae_state=next_vae_state
                avail_actions=next_action_avail
        
        
    
        s=episode_sample['s']
        u=episode_sample['u']
        #prob=episode_sample['prob_n']
        u_onehot=episode_sample['u_onehot']
        avail_u=episode_sample['avail_u']
        
        r=episode_sample['r']
        episode_sample['terminate'][-1]=[True]
        terminate=episode_sample['terminate']
        padded=episode_sample['padded']
        s_next=episode_sample['s']
        avail_u_next=episode_sample['avail_u'] 
        
        
        episode = dict(
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy(),
                       #prob_n=prob.copy()
                       )
        for key in episode.keys():  
            episode[key] = np.array([episode[key]])  
        
        if pipe!=None:
            pipe.send(episode)
        
        return  episode  
    except:
        with open('error.txt',mode='a') as f:
            f.write('env_id:{}\t time:{}\n\t error_info:{}\n\n'.format(mp.current_process().name,datetime.datetime.now().strftime('%Y/%m/%d  %H:%M:%S'),traceback.format_exc()))        
        
    
if __name__=='__main__':
    
    args=get_common_args()
    vae=VAE(args).to(args.device)
    
    actor_net=RNN(args.vae_hidden_num, args).to(args.device)
    mdn_rnn_net=MDN_RNN(args).to(args.device)
    
    ##记录
    if os.path.exists(args.tensorboard_path+'/mdn_rnn'):
        import shutil
        shutil.rmtree(args.tensorboard_path+'/mdn_rnn')    
    writer=SummaryWriter(log_dir=args.tensorboard_path+'/mdn_rnn')    
    
    
    eval_parameters=list(actor_net.parameters()) + list(mdn_rnn_net.parameters())
    opt = torch.optim.Adam(eval_parameters, lr=args.mdn_rnn_lr)
    
    data=np.load('./data/mdn_rnn_data.npy',allow_pickle=True)
    train_set,test_set,_,_=train_test_split(data,data,test_size=0.2)
    train_set = TData(train_set)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.mdn_rnn_batch,shuffle=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=args.mdn_rnn_batch,shuffle=True)
    
    #定义损失函数
    loss_function_avail_action=nn.BCEWithLogitsLoss()
    loss_function_reward=nn.MSELoss()
    loss_function_done=nn.CrossEntropyLoss()
    
    
    for epoch in range(3000):
        
        for i, data in enumerate(trainloader, 0):
            states,actions,rewards,dones,n_states,avail_u_next=data
            states,actions,rewards,dones,n_states,avail_u_next=\
                states.to(args.device),actions.to(args.device),rewards.to(args.device),dones.to(args.device),\
                n_states.to(args.device),avail_u_next.to(args.device)
            #获得状态的编码
            _, e_z = vae(states)
            _, e_n_z = vae(n_states)
            e_z, e_n_z=e_z.detach(), e_n_z.detach()
            
            #此处通过循环获得隐层信息
            args.mdn_rnn_batch=states.size()[0]
            h_n=torch.ones((args.mdn_rnn_batch,args.rnn_hidden_dim)).to(args.device)
            rnn_out=[]
            
            for i in range(args.mdn_rnn_length):
                _, h_n=actor_net(e_z[:,i,:],h_n)
                h_n=h_n*(1-dones[:,i,:])
                rnn_out.append(h_n)
            rnn_outs=torch.stack(rnn_out,dim=1)
            
            #此处，对预测信息进行输出
            
            one_hot_action = torch.FloatTensor(args.mdn_rnn_batch,args.mdn_rnn_length,args.n_agents,args.n_actions).zero_().to(args.device)
            one_hot_action.scatter_(3, actions, 1)             
            _,done_pre,reward_pre,action_avail_pre=mdn_rnn_net(rnn_outs,one_hot_action.view(args.mdn_rnn_batch,args.mdn_rnn_length,-1))
            print('a')
            
            loss_action_pre=loss_function_avail_action(action_avail_pre.view(args.mdn_rnn_batch,args.mdn_rnn_length,args.n_agents,args.n_actions),
                                                       avail_u_next.float())
            loss_reward=loss_function_reward(reward_pre,rewards)
            loss_done=loss_function_done(done_pre.view(-1,2),dones.view(-1).long())
            
            delta_z=(e_n_z-e_z).detach()
            loss_delta_z=mdn_rnn_net.log_prob(delta_z)
            
            loss_total=loss_action_pre+loss_reward+loss_done+loss_delta_z
            
            opt.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(eval_parameters, args.grad_norm_clip)
            opt.step()
        writer.add_scalars('Train_Loss', 
                           {
                               'loss_action_pre':loss_action_pre.item(),
                                'loss_reward':loss_reward.item(),
                                'loss_done':loss_done.item(),
                                'loss_delta_z':loss_delta_z.item(),
                                },
                           epoch)
        writer.add_scalar('Train_Loss/'+'loss_action_pre', 
                           loss_action_pre.item(),
                           epoch) 
        writer.add_scalar('Train_Loss/'+'loss_reward', 
                           loss_reward.item(),
                           epoch)
        writer.add_scalar('Train_Loss/'+'loss_done', 
                           loss_done.item(),
                           epoch)  
        writer.add_scalar('Train_Loss/'+'loss_delta_z', 
                           loss_delta_z.item(),
                           epoch)         
    writer.close()
            
            
            
            
            