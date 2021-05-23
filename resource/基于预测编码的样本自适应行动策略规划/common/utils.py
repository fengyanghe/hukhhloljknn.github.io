# -*- coding:utf-8 -*-
import inspect
import functools
import torch
import os
import numpy as np

#存储data数据
def store_per_data(args,agent,train_step,updata_sample_idx):
    '''
    存储数据，以字典的形势存储树结构
    '''
   
    if not os.path.exists('{}/{}/data'.format(args.model_dir,args.alg_name)):
        os.mkdir('{}/{}/data'.format(args.model_dir,args.alg_name))
    #else:
        #file_name='{}/{}/data'.format(args.model_dir,args.alg_name)
        #import shutil
        #shutil.rmtree(file_name) 
        #os.mkdir(file_name)
    if not os.path.exists('{}/{}/actor'.format(args.model_dir,args.alg_name)):
        os.mkdir('{}/{}/actor'.format(args.model_dir,args.alg_name))
    if not os.path.exists('{}/{}/critic'.format(args.model_dir,args.alg_name)):
        os.mkdir('{}/{}/critic'.format(args.model_dir,args.alg_name))        
    
    #new_data={'train_idx':train_step,'write_num':}
    #np.save(file='{}/{}/data/data_{}.npy'.format(args.model_dir,args.alg_name,-1), arr=np.array(new_data))
    
    data=agent.replay_buffer
    
    new_data={'train_idx':train_step+1,'write_num':data.tree.write}
    np.save(file='{}/{}/data/data_{}.npy'.format(args.model_dir,args.alg_name,-1), arr=np.array(new_data))    
    for idx in updata_sample_idx:
        p_idx=idx + data.capacity - 1
        new_data={'idx':idx,'p':data.tree.tree[p_idx],'data':data.tree.data[idx]}
        np.save(file='{}/{}/data/data_{}.npy'.format(args.model_dir,args.alg_name,idx), arr=np.array(new_data))        
    
    #for idx,episode in enumerate(data.tree.data):
        #if episode!=0:
            #new_data={'idx':idx,'p':data.tree.tree[idx],'data':episode}
            #np.save(file='{}/{}/data/data_{}.npy'.format(args.model_dir,args.alg_name,idx), arr=np.array(new_data))
    #np.save(file='{}/{}/data/data.npy'.format(args.model_dir,args.alg_name), arr=np.array(new_data))
    torch.save(obj=agent.policy.eval_rnn.state_dict(), f='{}/{}/actor/actor.pkl'.format(args.model_dir,args.alg_name))
    torch.save(obj=agent.policy.eval_critic.state_dict(), f='{}/{}/critic/critic.pkl'.format(args.model_dir,args.alg_name))
    if args.vae:
        if not os.path.exists('{}/{}/vae'.format(args.model_dir,args.alg_name)):
            os.mkdir('{}/{}/vae'.format(args.model_dir,args.alg_name))         
        torch.save(obj=agent.policy.vae_model.state_dict(), f='{}/{}/vae/vae.pkl'.format(args.model_dir,args.alg_name))
    if args.mdn_rnn:
        if not os.path.exists('{}/{}/vae'.format(args.model_dir,args.alg_name)):
            os.mkdir('{}/{}/vae'.format(args.model_dir,args.alg_name)) 
        if not os.path.exists('{}/{}/mdn'.format(args.model_dir,args.alg_name)):
            os.mkdir('{}/{}/mdn'.format(args.model_dir,args.alg_name))         
        torch.save(obj=agent.policy.vae_model.state_dict(), f='{}/{}/vae/vae.pkl'.format(args.model_dir,args.alg_name))    
        torch.save(obj=agent.policy.mdn_model.state_dict(), f='{}/{}/mdn/mdn.pkl'.format(args.model_dir,args.alg_name))
    #清理缓存
    sudopw='011216'
    cmd_line=' sh -c \'echo 3 > /proc/sys/vm/drop_caches\''
    os.system('echo {}|sudo -S {}'.format(sudopw,cmd_line))    
    
#加载过往数据
def load_old_data(args,agent):
    '''
    加载过往数据
    '''
    main_path='./{}/{}'.format(args.model_dir,args.alg_name)
        
    try:
        old_data={}
        file_path='{}/data'.format(main_path)
        #file_path='./model/fc_naf/data'
        for root, dirs, files in os.walk(file_path):
            for file in files:
                file_idx=file.split('_')[1].split('.')[0]
                temp_data=np.load('{}/{}'.format(file_path,file)).item()
                old_data[file_idx]=temp_data
        
        for k,v in old_data.items():
            if k=='-1':
                args.start_episodes=v['train_idx']
                #agent.replay_buffer.tree.write=v['write_num']
                continue
            #agent.replay_buffer.tree.tree[k]=v['p']
            #agent.replay_buffer.tree.data[k]=v['data']
            #agent.replay_buffer.tree.update(k, v['p'])
            agent.replay_buffer.tree.add(v['p'],v['data'])
    except:
        print('No Old Data!')
    
    try:
        agent.policy.eval_rnn.load_state_dict(torch.load('{}/{}/actor/actor.pkl'.format(args.model_dir,args.alg_name)))
        agent.policy.eval_rnn_old.load_state_dict(agent.policy.eval_rnn.state_dict())
    except:
        print('No Actor Model!')
        
    try:
        agent.policy.eval_critic.load_state_dict(torch.load('{}/{}/critic/critic.pkl'.format(args.model_dir,args.alg_name)))
        agent.policy.target_critic.load_state_dict(torch.load('{}/{}/critic/critic.pkl'.format(args.model_dir,args.alg_name)))
    except:
        print('No Critic Model!')
    
    if args.vae:
        try:
            agent.policy.vae_model.load_state_dict(torch.load('{}/{}/vae/vae.pkl'.format(args.model_dir,args.alg_name)))
        except:
            print('No VAE Model!')
    if args.mdn_rnn:
        try:
            agent.policy.vae_model.load_state_dict(torch.load('{}/{}/vae/vae.pkl'.format(args.model_dir,args.alg_name)))
        except:
            print('No VAE Model!')
        
        try:
            agent.policy.mdn_model.load_state_dict(torch.load('{}/{}/mdn/mdn.pkl'.format(args.model_dir,args.alg_name)))
        except:
            print('No MDN Model!')         
    
    
    

def store_args(method):
    """Stores provided method args as instance attributes.
    """
    argspec = inspect.getfullargspec(method)
    defaults = {}
    if argspec.defaults is not None:
        defaults = dict(
            zip(argspec.args[-len(argspec.defaults):], argspec.defaults))
    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)
    arg_names = argspec.args[1:]

    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        # Get default arg values
        args = defaults.copy()
        # Add provided arg values
        for name, value in zip(arg_names, positional_args[1:]):
            args[name] = value
        args.update(keyword_args)
        self.__dict__.update(args)
        return method(*positional_args, **keyword_args)

    return wrapper


def gae_return(batch,values,args,gamma=0.99,tau=0.95):
    pre_value=values[:,-1]
    gae=0
    masks = (1 - batch["padded"].float())##.repeat(1, 1, args.n_agents)
    terminated = (1 - batch["terminated"].float())##.repeat(1, 1, args.n_agents)
    rewards = batch['r']
    if args.cuda:
        rewards = rewards.cuda()
        masks = masks.cuda()
        terminated = terminated.cuda()
    returns=[]
    for step in reversed(range(rewards.size()[-2])):
        delta=rewards[:,step]+gamma*pre_value*masks[:,step]*terminated[:,step]-values[:,step]
        gae=delta+gamma*tau*masks[:,step]*gae
        pre_value=values[:,step]
        returns.insert(0,gae+values[:,step])
    return returns    


def td_lambda_target(batch, max_episode_len, q_targets, args):
    # batch.shep = (episode_num, max_episode_len， n_agents，n_actions)
    # q_targets.shape = (episode_num, max_episode_len， n_agents)
    episode_num = batch['o'].shape[0]
    mask = (1 - batch["padded"].float())##.repeat(1, 1, args.n_agents)
    terminated = (1 - batch["terminated"].float())##.repeat(1, 1, args.n_agents)
    r = batch['r']##.repeat((1, 1, args.n_agents))
    # --------------------------------------------------n_step_return---------------------------------------------------
    '''
    1. 每条经验都有若干个n_step_return，所以给一个最大的max_episode_len维度用来装n_step_return
    最后一维,第n个数代表 n+1 step。
    2. 因为batch中各个episode的长度不一样，所以需要用mask将多出的n-step return置为0，
    否则的话会影响后面的lambda return。第t条经验的lambda return是和它后面的所有n-step return有关的，
    如果没有置0，在计算td-error后再置0是来不及的
    3. terminated用来将超出当前episode长度的q_targets和r置为0
    '''
    n_step_return = torch.zeros((episode_num, max_episode_len, max_episode_len))
    for transition_idx in range(max_episode_len - 1, -1, -1):
        # 最后计算1 step return
        n_step_return[:, transition_idx, 0] = ((r[:, transition_idx] + args.gamma * q_targets[:, transition_idx] * terminated[:, transition_idx]) * mask[:, transition_idx]).squeeze(1)        
        # 经验transition_idx上的obs有max_episode_len - transition_idx个return, 分别计算每种step return
        # 同时要注意n step return对应的index为n-1
        for n in range(1, max_episode_len - transition_idx):
            # t时刻的n step return =r + gamma * (t + 1 时刻的 n-1 step return)
            # n=1除外, 1 step return =r + gamma * (t + 1 时刻的 Q)
            n_step_return[:, transition_idx,  n] = ((r[:, transition_idx] + args.gamma * n_step_return[:, transition_idx + 1, n - 1].unsqueeze(1)) * mask[:, transition_idx]).squeeze(1) 
    # --------------------------------------------------n_step_return---------------------------------------------------

    # --------------------------------------------------lambda return---------------------------------------------------
    '''
    lambda_return.shape = (episode_num, max_episode_len，n_agents)
    '''
    lambda_return = torch.zeros((episode_num, max_episode_len))
    for transition_idx in range(max_episode_len):
        returns = torch.zeros((episode_num))
        for n in range(1, max_episode_len - transition_idx):
            returns += pow(args.td_lambda, n - 1) * n_step_return[:, transition_idx,  n - 1]
        lambda_return[:, transition_idx] = (1 - args.td_lambda) * returns + \
                                           pow(args.td_lambda, max_episode_len - transition_idx - 1) * \
                                           n_step_return[:, transition_idx, max_episode_len - transition_idx - 1]
    # --------------------------------------------------lambda return---------------------------------------------------
    return lambda_return
