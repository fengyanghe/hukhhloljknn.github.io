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
    parser.add_argument('--obs_shape', type=int, default=0)
    parser.add_argument('--last_action', type=bool, default=False)
    parser.add_argument('--cuda', type=bool, default=True if torch.cuda.is_available() else False)
    parser.add_argument('--model_dir', type=str, default='model')
    parser.add_argument('--alg', type=str, default='coma')
    parser.add_argument('--load_model', type=bool, default=False)
    parser.add_argument('--optimizer', type=str, default='RMS')
    parser.add_argument('--vae_lr', type=float, default=1e-6) 
    parser.add_argument('--reuse_network', type=bool, default=False, help='whether to use one network for all agents')
    parser.add_argument('--n_episodes', type=int, default=4, help='the number of episodes before once training')
    parser.add_argument('--episode_limit', type=int, default=400,help='6000/decision_interval')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--params_clip', type=float, default=0.2, help='params clip in ppo')
    parser.add_argument('--evaluate', type=bool, default=False, help='whether to evaluate the model')
    parser.add_argument('--map', type=str, default='3m', help='the map of the game')
    parser.add_argument('--train_steps', type=int, default=10)
    parser.add_argument('--mdn_rnn_length', type=int, default=64)
    parser.add_argument('--rnn_hidden_dim', type=int, default=256, help='同决策网络的隐层中间数目一致')
    parser.add_argument('--mdn_rnn_headings_num', type=int, default=5)
    parser.add_argument('--mdn_rnn_lr', type=float, default=1e-4)
    parser.add_argument('--grad_norm_clip', type=float, default=10)
    parser.add_argument('--mdn_rnn_batch', type=int, default=32)
    parser.add_argument('--vae_hidden_num', type=int, default=32)
    parser.add_argument('--tensorboard_path', type=str, default='./tensorboard_record')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args=parser.parse_args()
    return args



