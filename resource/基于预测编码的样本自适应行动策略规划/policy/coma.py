# -*- coding:utf-8 -*-
import torch
import os
from ..network.base_net import RNN, VAE,Actor,Critic,MDN_RNN

from ..network.coma_critic import ComaCritic
from ..common.utils import * #td_lambda_target


class COMA:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        ##self.obs_shape = args.obs_shape
        self.obs_shape = args.state_shape
        actor_input_shape = self.obs_shape  # actor网络输入的维度，和vdn、qmix的rnn输入维度一样，使用同一个网络结构
        critic_input_shape = self._get_critic_input_shape()  # critic网络输入的维度
        # 根据参数决定RNN的输入维度
        if args.last_action:
            actor_input_shape += self.n_actions
        if args.reuse_network:
            actor_input_shape += self.n_agents
        self.args = args
        
        if self.args.af:
            self.af_tau=0.5
            self.af_beta=0.999
            self.af_step=100
            self.af_train_step=0

        if self.args.vae:
            self.vae_model=VAE(self.args)
            actor_input_shape=self.args.vae_hidden_num
            if self.args.cuda:
                self.vae_model.cuda()
            
            if self.args.mdn_rnn:
                self.mdn_model=MDN_RNN(self.args)
                #self.mdn_model.load_state_dict(torch.load('mdn_20.pkl'))
                if self.args.cuda:
                    self.mdn_model.cuda()                
            
        # 神经网络
        # 每个agent选动作的网络,输出当前agent所有动作对应的概率，用该概率选动作的时候还需要用softmax再运算一次。
        if self.args.alg == 'coma':
            print('Init alg coma')
            if args.mdn_rnn:
                actor_input_shape=args.mdn_rnn_hidden_num
                self.eval_rnn=Actor(actor_input_shape, args)
                self.eval_rnn_old=Actor(actor_input_shape, args)
            else:
                self.eval_rnn = RNN(actor_input_shape, args)
                self.eval_rnn_old=RNN(actor_input_shape, args)

        else:
            raise Exception("No such algorithm")

        # 得到当前agent的所有可执行动作对应的联合Q值，得到之后需要用该Q值和actor网络输出的概率计算advantage
        if args.mdn_rnn:
            self.eval_critic = Critic(actor_input_shape, self.args)
            self.target_critic = Critic(actor_input_shape, self.args)
            
        else:
            self.eval_critic = ComaCritic(critic_input_shape, self.args)
            self.target_critic = ComaCritic(critic_input_shape, self.args)

        if self.args.cuda:
            self.eval_rnn.cuda()
            self.eval_critic.cuda()
            self.target_critic.cuda()
            self.eval_rnn_old.cuda()
        self.model_dir = args.model_dir + '/' + args.alg_name
        # 如果存在模型则加载模型
        if self.args.load_model:
            try:
                
                path_rnn = self.model_dir + '/actor/actor_'+args.model_name
                path_coma = self.model_dir + '/critic/critic_'+args.model_name
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
                self.eval_critic.load_state_dict(torch.load(path_coma, map_location=map_location))
                print('Successfully load the model: {} and {}'.format(path_rnn, path_coma))
            except:
                print("No model!")

        # 让target_net和eval_net的网络参数相同
        self.target_critic.load_state_dict(self.eval_critic.state_dict())
        self.eval_rnn_old.load_state_dict(self.eval_rnn.state_dict())
        self.rnn_parameters = list(self.eval_rnn.parameters())
        self.critic_parameters = list(self.eval_critic.parameters())

        if args.optimizer == "RMS":
            self.critic_optimizer = torch.optim.RMSprop(self.critic_parameters, lr=args.lr_critic)
            self.rnn_optimizer = torch.optim.RMSprop(self.rnn_parameters, lr=args.lr_actor)
        elif args.optimizer == "Adam":
            self.critic_optimizer = torch.optim.Adam(self.critic_parameters, lr=args.lr_critic)
            self.rnn_optimizer = torch.optim.Adam(self.rnn_parameters, lr=args.lr_actor)
        self.args = args

        # 执行过程中，要为每个agent都维护一个eval_hidden
        # 学习过程中，要为每个episode的每个agent都维护一个eval_hidden
        self.eval_hidden = None
        
    def load_model_from_train(self,model):
        '''
        不同使用，不同的model
        [rnn,vae]
        '''
        if self.args.mdn_rnn:
            self.eval_rnn.load_state_dict(model[0])
            self.vae_model.load_state_dict(model[1])
            self.mdn_model.load_state_dict(model[2])
        elif self.args.vae:
            self.eval_rnn.load_state_dict(model[0])
            self.vae_model.load_state_dict(model[1])
        else:
            self.eval_rnn.load_state_dict(model)
            
            
    
    def _get_critic_input_shape(self):
        # state
        input_shape = self.state_shape  # 48
        # obs
        #input_shape += self.obs_shape  # 30
        # agent_id
        #input_shape += self.n_agents  # 3
        # 所有agent的当前动作和上一个动作
        #input_shape += self.n_actions * self.n_agents * 2  # 54

        return input_shape

    def learn(self, batch, max_episode_len, train_step, epsilon,vir=False):  # train_step表示是第几次学习，用来控制更新target_net网络的参数
        episode_num = batch['s'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():  # 把batch里的数据转化成tensor
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long) if type(batch[key]) is not torch.Tensor else batch[key]
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32) if type(batch[key]) is not torch.Tensor else batch[key]
        u, r, avail_u, terminated = batch['u'], batch['r'],  batch['avail_u'], batch['terminated']
        ##mask = (1 - batch["padded"].float()).repeat(1, 1, self.n_agents)  # 用来把那些填充的经验的TD-error置0，从而不让它们影响到学习
        mask = (1 - batch["padded"].float()).repeat(1,1,self.args.n_agents)
        shape=u.shape
        final_v=torch.zeros((shape[0],1,1))
        if self.args.cuda:
            u = u.cuda()
            mask = mask.cuda()
            final_v = final_v.cuda()
        # 根据经验计算每个agent的Ｑ值,从而跟新Critic网络。然后计算各个动作执行的概率，从而计算advantage去更新Actor。
        #q_values = self._train_critic(batch, max_episode_len, train_step)  # 训练critic网络，并且得到每个agent的所有动作的Ｑ值
        
        action_prob, pred_v = self._get_action_prob(batch, max_episode_len, epsilon,vir=vir)  # 每个agent的所有动作的概率

        #q_taken = q_values.repeat(1,1,self.args.n_agents)#torch.gather(q_values, dim=3, index=u).squeeze(3)  # 每个agent的选择的动作对应的Ｑ值
        pi_taken = torch.gather(action_prob, dim=3, index=u).squeeze(3)  # 每个agent的选择的动作对应的概率
        pi_taken[mask == 0] = 1.0  # 因为要取对数，对于那些填充的经验，所有概率都为0，取了log就是负无穷了，所以让它们变成1
        #log_pi_taken = torch.log(pi_taken)

        ##取旧的
        action_prob_old, _ = self._get_action_prob(batch, max_episode_len, epsilon,old=True,vir=vir)  # 每个agent的所有动作的概率
    
        #q_taken = q_values.repeat(1,1,self.args.n_agents)#torch.gather(q_values, dim=3, index=u).squeeze(3)  # 每个agent的选择的动作对应的Ｑ值
        pi_taken_old = torch.gather(action_prob_old, dim=3, index=u).squeeze(3)  # 每个agent的选择的动作对应的概率
        pi_taken_old[mask == 0] = 1.0  # 因为要取对数，对于那些填充的经验，所有概率都为0，取了log就是负无穷了，所以让它们变成1
        #log_pi_taken_old = torch.log(pi_taken_old)        

        # 计算advantage
        #baseline = (q_values * action_prob).sum(dim=3, keepdim=True).squeeze(3).detach()
        #advantage = (q_taken - baseline).detach()
        #q_evals, q_next_target = self._get_q_values(batch, max_episode_len)
        #q_values = q_evals.clone()  # 在函数的最后返回，用来计算advantage从而更新actor
        # 取每个agent动作对应的Q值，并且把最后不需要的一维去掉，因为最后一维只有一个值了
        
        q_next_target = torch.cat((pred_v[:,:],final_v), dim=1)
        targets =gae_return(batch,q_next_target.detach(),self.args) #td_lambda_target(batch, max_episode_len, q_next_target.cpu(), self.args).unsqueeze(2)
        targets = torch.stack(targets, dim=1)
        
        if self.args.af:
            self.af_train_step+=1
            loss_c=(0.5*(pred_v-targets)**2*(torch.softmax(abs(pred_v-targets)**self.af_tau,1))).sum()
            if self.af_train_step%(self.af_step*self.args.process_num*self.args.train_steps)==0:
                self.af_tau *= self.af_beta
        else:
            loss_c=torch.nn.functional.mse_loss(pred_v, targets)
        
        advantage =( targets - pred_v).detach()
        
        ##
        ratio=pi_taken/(1e-10+pi_taken_old.detach())
        surr1=ratio*advantage
        surr2=ratio.clamp(1-self.args.params_clip, 1+self.args.params_clip)*advantage
        ##loss = - ((advantage * log_pi_taken) * mask).sum() / mask.sum()
        
        loss_a = -torch.mean(torch.min(surr1, surr2))
        ##
        loss=loss_a+0.5*loss_c
        self.eval_rnn_old.load_state_dict(self.eval_rnn.state_dict())
        
        self.rnn_optimizer.zero_grad()
        loss.backward()
        #for k in self.rnn_parameters:
            #if k.grad.isnan().max():
                #self._get_action_prob(batch, max_episode_len, epsilon)
        torch.nn.utils.clip_grad_norm_(self.rnn_parameters, self.args.grad_norm_clip)
        self.rnn_optimizer.step()
        

    def _get_critic_inputs(self, batch, transition_idx, max_episode_len):
        # 取出所有episode上该transition_idx的经验
        s, s_next = batch['s'][:, transition_idx], batch['s_next'][:, transition_idx]

        return s, s_next

    def _get_q_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        q_evals, q_targets = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_critic_inputs(batch, transition_idx, max_episode_len)
            if self.args.cuda:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
            # 神经网络输入的是(episode_num * n_agents, inputs)二维数据，得到的是(episode_num * n_agents， n_actions)二维数据
            q_eval = self.eval_critic(inputs)
            q_target = self.target_critic(inputs_next)

            # 把q值的维度重新变回(episode_num, n_agents, n_actions)
            q_eval = q_eval.view(episode_num, -1)
            q_target = q_target.view(episode_num, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)
        # 得的q_evals和q_targets是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
        # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        return q_evals, q_targets

    def _get_actor_inputs(self, batch, transition_idx):
        # 取出所有episode上该transition_idx的经验，u_onehot要取出所有，因为要用到上一条
        obs = batch['s'][:, transition_idx]

        return obs

    def _get_action_prob(self, batch, max_episode_len, epsilon=0,old=False,vir=False):
        #episode_num = batch['s'].shape[0]
        avail_actions = batch['avail_u']
        if self.args.cuda:
            avail_actions = avail_actions.cuda()        
            # coma不用target_actor，所以不需要最后一个obs的下一个可执行动作
       
        inputs=batch['s']
        actions_one_hot=batch['u_onehot']
        if self.args.cuda:
            inputs = inputs.cuda()
            shape=inputs.shape
            actions_one_hot=actions_one_hot.cuda().view(shape[0],shape[1],-1)
            self.eval_hidden = self.eval_hidden.cuda() 
            self.eval_hidden_old=self.eval_hidden_old.cuda()
            if self.args.vae:
                self.vae_model = self.vae_model.cuda()
        if self.args.vae:
            if not vir:
                inputs=self.vae_model(inputs)[1].detach()
        
        if old:
            if self.args.mdn_rnn:
                _,_,_,_,h=self.mdn_model(inputs, actions_one_hot,self.eval_hidden_old)
                inputs = h.detach()                
                q_value = self.eval_rnn_old(inputs)
                v = self.eval_critic(inputs)
                
            else:
                q_value,_,v=self.eval_rnn_old(inputs, self.eval_hidden_old)
        else:
            if self.args.mdn_rnn:
                _,_,_,_,h=self.mdn_model(inputs, actions_one_hot,self.eval_hidden)
                inputs = h.detach()
                q_value = self.eval_rnn(inputs)
                v = self.eval_critic(inputs)
            else:
                q_value,_,v=self.eval_rnn(inputs, self.eval_hidden)            
        
        q_value=torch.clamp(q_value,min=-self.args.clamp_q,max=self.args.clamp_q)
        q_value[avail_actions==0]=-self.args.clamp_q
        action_prob=torch.nn.functional.softmax(q_value, dim=-1)
        
        #action_num = avail_actions.sum(dim=-1, keepdim=True).float().repeat(1, 1, 1, avail_actions.shape[-1])   # 可以选择的动作的个数
        #action_prob = ((1 - epsilon) * action_prob + torch.ones_like(action_prob) * epsilon / action_num)
        #action_prob[avail_actions == 0] = 0.0  # 不能执行的动作概率为0
        action_prob=action_prob*avail_actions
        # 因为上面把不能执行的动作概率置为0，所以概率和不为1了，这里要重新正则化一下。执行过程中Categorical会自己正则化。
        action_prob = action_prob / (action_prob.sum(dim=-1, keepdim=True))
        # 因为有许多经验是填充的，它们的avail_actions都填充的是0，所以该经验上所有动作的概率都为0，在正则化的时候会得到nan。
        # 因此需要再一次将该经验对应的概率置为0
        action_prob=action_prob*avail_actions
        #action_prob[avail_actions == 0] = 0.0
        
        return action_prob,v

    def init_hidden(self, episode_num):
        # 为每个episode中初始化一个eval_hidden
        ##self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        if self.args.mdn_rnn:
            self.eval_hidden = torch.zeros((1, episode_num, self.args.mdn_rnn_hidden_num))
            self.eval_hidden_old = torch.zeros((1,episode_num,  self.args.mdn_rnn_hidden_num))
        else:
            self.eval_hidden = torch.zeros((1, episode_num, self.args.rnn_hidden_dim))
            self.eval_hidden_old = torch.zeros((1,episode_num,  self.args.rnn_hidden_dim))
    def _train_critic(self, batch, max_episode_len, train_step):
        u, r, avail_u, terminated = batch['u'], batch['r'], batch['avail_u'], batch['terminated']
        u_next = u[:, 1:]
        padded_u_next = torch.zeros(*u[:, -1].shape, dtype=torch.long).unsqueeze(1)
        u_next = torch.cat((u_next, padded_u_next), dim=1)
        ##mask = (1 - batch["padded"].float()).repeat(1, 1, self.n_agents)  # 用来把那些填充的经验的TD-error置0，从而不让它们影响到学习
        mask = (1 - batch["padded"].float())
        if self.args.cuda:
            u = u.cuda()
            u_next = u_next.cuda()
            mask = mask.cuda()
        # 得到每个agent对应的Q值，维度为(episode个数, max_episode_len， n_agents，n_actions)
        # q_next_target为下一个状态-动作对应的target网络输出的Q值，没有包括reward
        q_evals, q_next_target = self._get_q_values(batch, max_episode_len)
        q_values = q_evals.clone()  # 在函数的最后返回，用来计算advantage从而更新actor
        # 取每个agent动作对应的Q值，并且把最后不需要的一维去掉，因为最后一维只有一个值了

        #q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)
        #q_next_target = torch.gather(q_next_target, dim=3, index=u_next).squeeze(3)
        targets = td_lambda_target(batch, max_episode_len, q_next_target.cpu(), self.args).unsqueeze(2)
        if self.args.cuda:
            targets = targets.cuda()
        td_error = targets.detach() - q_evals
        masked_td_error = mask * td_error  # 抹掉填充的经验的td_error

        # 不能直接用mean，因为还有许多经验是没用的，所以要求和再比真实的经验数，才是真正的均值
        loss = (masked_td_error ** 2).sum() / mask.sum()
        # print('Loss is ', loss)
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_parameters, self.args.grad_norm_clip)
        self.critic_optimizer.step()
        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_critic.load_state_dict(self.eval_critic.state_dict())
        return q_values

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_critic.state_dict(), self.model_dir + '/' + num + '_critic_params.pkl')
        torch.save(self.eval_rnn.state_dict(),  self.model_dir + '/' + num + '_rnn_params.pkl')