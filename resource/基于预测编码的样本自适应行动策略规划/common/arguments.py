# -*- coding:utf-8 -*-
import argparse
import torch
"""
Here are the param for the training

"""


def get_common_args():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--n_actions', type=int, default=33)
    parser.add_argument('--n_agents', type=int, default=6)
    parser.add_argument('--state_shape', type=int, default=678)
    parser.add_argument('--obs_shape', type=int, default=312)
    parser.add_argument('--last_action', type=bool, default=False)
    parser.add_argument('--cuda', type=bool, default=True if torch.cuda.is_available() else False)
    parser.add_argument('--model_dir', type=str, default='model')
    parser.add_argument('--model_name', type=str, default='6000.pkl')
    parser.add_argument('--alg', type=str, default='coma')
    parser.add_argument('--load_model', type=bool, default=False)###########
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=0.001) 
    parser.add_argument('--reuse_network', type=bool, default=False, help='whether to use one network for all agents')
    parser.add_argument('--n_episodes', type=int, default=1, help='the number of episodes before once training')
    parser.add_argument('--episode_limit', type=int, default=400,help='6000/decision_interval')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--params_clip', type=float, default=0.2, help='params clip in ppo')
    parser.add_argument('--evaluate', type=bool, default=False, help='whether to evaluate the model')
    #parser.add_argument('--map', type=str, default='3m', help='the map of the game')
    
    parser.add_argument('--replay_data', type=bool, default=False, help='store the replay data')
    parser.add_argument('--replay_data_file', type=str, default='./agent/pymt_c/replay_data/')
    parser.add_argument('--decision_interval', type=int, default=15, help='决策时间间隔（单位：秒）')
    parser.add_argument('--continue_train', type=bool, default=True, help='是否接续训练')#########
    
    
    ####多进程配置
    parser.add_argument('--alg_name', type=str, default='mdn_af') ###################
    parser.add_argument('--mp', type=bool, default=True, help='是否多进程')#############
    parser.add_argument('--Max_episodes', type=int, default=5000, help='每个进行最大推演局数')
    parser.add_argument('--start_episodes', type=int, default=0, help='中断后继续训练')
    parser.add_argument('--tensorboard_path', type=str, default='./tensorboard_output/', help='数据记录位置')
    parser.add_argument('--per', type=bool, default=True, help='是否使用PER')
    parser.add_argument('--buffer_size', type=int, default=2**10, help='缓存的大小')
    parser.add_argument('--clamp_q', type=float, default=10, help='裁剪rnn输出的Q，约束不可用动作概率')
    
    
    parser.add_argument('--platform', type=int, default=1, help='0:windows, 1:linux 服务器的操作系统配置')###########
    parser.add_argument('--scenario', type=str, default='NUDT_mission' if parser.parse_args().platform==1 else '科大教学版.scen'  ,
                        help='NUDT_mission, 科大教学版-limit.scen,科大教学版-1.scen,科大教学版.scen')###########
    parser.add_argument('--test_num', type=int, default=5, help='每次测试选择几个环境')
    parser.add_argument('--process_num', type=int, default=5, help='同时开启几个通道')
    parser.add_argument('--train_steps', type=int, default=5)
    parser.add_argument('--test_steps', type=int, default=500/5, help='100次训练进行测试')
    
    parser.add_argument('--vae', type=bool, default=False, help='是否使用VAE')
    parser.add_argument('--vae_hidden_num', type=int, default=32, help='VAE的隐层大小')
    parser.add_argument('--vae_batch', type=int, default=32, help='VAE的batch大小')
    parser.add_argument('--vae_device', type=int, default=0, help='VAE的训练设备')
    parser.add_argument('--vae_lr', type=float, default=1e-3, help='VAE的学习率')
    parser.add_argument('--vae_maxepoch', type=int, default=100, help='VAE的最大训练次数')
    parser.add_argument('--vae_load_model_path', type=str, default='./model/', help='vae模型在多进程中无法直接传递')
    parser.add_argument('--vae_soft_update', type=bool, default=True, help='软更新')
    parser.add_argument('--vae_tau', type=float, default=0.01, help='软更新下的参数值')
    
    
    parser.add_argument('--mdn_rnn', type=bool, default=False, help='是否使用MDN_RNN')
    parser.add_argument('--mdn_rnn_hidden_num', type=int, default=128, help='mdn-rnn中的隐层信息')
    parser.add_argument('--mdn_rnn_headings_num', type=int, default=5)
    parser.add_argument('--mdn_rnn_maxepoch', type=int, default=60, help='MDN-RNN的最大训练次数')
    parser.add_argument('--mdn_rnn_length', type=int, default=32, help='MDN-RNN的step长度')
    parser.add_argument('--mdn_rnn_batch', type=int, default=32, help='MDN-RNN的batch大小')
    parser.add_argument('--mdn_rnn_lr', type=float, default=1e-4, help='MDN-RNN的学习率')
    parser.add_argument('--mdn_rnn_gen', type=int, default=0, help='开启几个虚拟环境生成')
    parser.add_argument('--mdn_rnn_device', type=str, default='cpu' if not torch.cuda.is_available() else 'cuda:{}'.format(0), help='开启几个虚拟环境生成')
    parser.add_argument('--mdn_rnn_soft_update', type=bool, default=True, help='软更新')
    parser.add_argument('--mdn_rnn_tau', type=float, default=0.01, help='软更新下的参数值')
    
    
    parser.add_argument('--af', type=bool, default=False, help='是否使用样本自适应')
    
    

    args=parser.parse_args()
    return args

def get_FC_NAF(args):
    args.vae=False
    args.af=False
    return args

def get_FC_AF(args):
    args.vae=False
    args.af=True
    return args

def get_VAE_NAF(args):
    args.vae=True
    args.af=False
    return args

def get_VAE_AF(args):
    args.vae=True
    args.af=True
    return args
def get_MDN_AF(args):
    args.vae=True
    args.mdn_rnn=True
    args.af=True
    return args

def get_MDN_NAF(args):
    args.vae=True
    args.mdn_rnn=True
    args.af=False
    return args

def get_alg_name_args(args):
    
    if args.alg_name=='fc_naf':
        args=get_FC_NAF(args)
    elif args.alg_name=='fc_af':
        args=get_FC_AF(args)
    elif args.alg_name=='vae_af':
        args=get_VAE_AF(args) 
    elif args.alg_name=='vae_naf':
        args=get_VAE_NAF(args) 
    elif args.alg_name=='mdn_naf':
        args=get_MDN_NAF(args) 
    elif args.alg_name=='mdn_af':
        args=get_MDN_AF(args)         
        
    return args


# arguments of coma
def get_coma_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'episode'

    # lambda of td-lambda return
    args.td_lambda = 0.8

    # how often to save the model
    args.save_cycle = 10#5000

    # how often to update the target_net
    args.target_update_cycle = 5

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


# arguments of vnd、 qmix、 qtran
def get_mixer_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.qmix_hidden_dim = 32
    args.two_hyper_layers = False
    args.hyper_hidden_dim = 64
    args.qtran_hidden_dim = 64
    args.lr = 5e-4

    # epsilon greedy
    args.epsilon = 1
    args.min_epsilon = 0.05
    anneal_steps = 100
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
    args.epsilon_anneal_scale = 'episode'

    # the number of the train steps in one epoch
    args.train_steps = 10

    # experience replay
    args.batch_size = 16
    args.buffer_size = int(2e3)

    # how often to save the model
    args.save_cycle = 5000

    # how often to update the target_net
    args.target_update_cycle = 10

    # QTRAN lambda
    args.lambda_opt = 1
    args.lambda_nopt = 1

    # prevent gradient explosion
    args.grad_norm_clip = 10

    # MAVEN
    args.noise_dim = 16
    args.lambda_mi = 0.001
    args.lambda_ql = 1
    args.entropy_coefficient = 0.001
    return args


# arguments of central_v
def get_centralv_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'episode'

    # lambda of td-lambda return
    args.td_lambda = 0.8

    # how often to save the model
    args.save_cycle = 20

    # how often to update the target_net
    args.target_update_cycle = 5

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


# arguments of central_v
def get_reinforce_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'episode'

    # how often to save the model
    args.save_cycle = 5000

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


# arguments of coma+commnet
def get_commnet_args(args):
    if args.map == '3m':
        args.k = 2
    else:
        args.k = 3
    return args


def get_g2anet_args(args):
    args.attention_dim = 32
    args.hard = True
    return args

