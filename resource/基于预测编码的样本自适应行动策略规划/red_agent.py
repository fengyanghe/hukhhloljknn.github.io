# -*- coding:utf-8 -*-

import logging
logging.basicConfig(format="%(asctime)s %(name)s:%(levelname)s:    %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

from entitys.global_util import *
from entitys.geo import *
from entitys.player import Player

from .RL_Agent import *
import numpy as np
from .common.arguments import *
from tensorboardX import SummaryWriter 
import os
import multiprocessing as mp
"""
红方bot
"""

Pos_my_ship={'latitude': 25.798576, 'longitude': 155.266136}
class RedAgent(Player):
    def __init__(self, side_name,kwarg):
        Player.__init__(self, side_name)   
        if kwarg!={}:
            args=kwarg['args']
            self.Pipe=kwarg['Pipe']
        else:    
            args=get_common_args()
            if args.alg.find('coma') > -1:
                args = get_coma_args(args)        
        #设置对应算法的参数
        
        
        self.args=args
        self.agent=RL_Agent(args)
        
        file_name=self.args.tensorboard_path
        if self.args.mp:
            model_state_dict=self.Pipe.recv()
            self.agent.policy.load_model_from_train(model_state_dict)
            file_name += '{}/{}'.format(args.alg_name,mp.current_process().name) 
        else:
            file_name += '{}/{}'.format(args.alg_name,mp.current_process().name)
        ##数据记录
        
        
        if os.path.exists(file_name):
            import shutil
            shutil.rmtree(file_name)
        self.tensorboard_writer=SummaryWriter(file_name)        
        self.episode_num=0
        
        self.train_steps=0
        self.episodes=[]
        
        logging.info('%s:%s play game' % (side_name, self.__class__.__name__))

    def initial(self, situation):
        # 初始化函数，每局推演开始前被调用
        logging.info("%s initial, units count:%d" % (self.__class__.__name__, len(situation[0])))
        self.my_init(situation)
        self.agent.policy.init_hidden(1)

    def step(self, time_elapse, situation):
        # 每步决策函数
        # logging.info(
        #    "%s step, units count:%d, contact:%d" % (self.__class__.__name__, len(situation[0]), len(situation[1])))
        self.current_time=time_elapse
        #决策开始判断，只要当有飞机在空才开始决策，所有飞机失去规划能力后，仿真进入自由仿真阶段
        if not self.ok_plan(situation):
            return

        #获得局部观测
        obs=self.get_obs(situation,time_elapse)
        #获得全局观测
        state=self.get_state(situation,time_elapse)
        actions, avail_actions, actions_onehot,last_actions = [], [], [], []
        ind=0
        
        #利用state获得六个agent的动作
        for k,v in self.mine_air.items():
            avai_action=self.get_avai_action(k)
            last_action=self._get_last_action(k)
            
            avail_actions.append(avai_action)
            last_actions.append(last_action)
        
        actions,prob_n=self.agent.select_action(state, last_actions, avail_actions)
        ind=0
        for k,v in self.mine_air.items():
            action =actions[ind]
            if action!=0:
                self._exect_action(k, action)
                action_onehot = [0 for _ in range(self.args.n_actions)]
                action_onehot[action] = 1                
            else:
                action_onehot =[0 for _ in range(self.args.n_actions)]
                action_onehot[action] = 1 
            
            actions_onehot.append(action_onehot)
            
            ind+=1
        
        terminated=1 if self.record_info['terminate'] else 0
        reward=self.record_info['delta_score']/1843+self.get_units_reward()
        
        self.episode_sample['o'].append(obs)
        self.episode_sample['s'].append(state)
        self.episode_sample['u'].append(np.reshape(actions, [self.args.n_agents, 1]))
        self.episode_sample['u_onehot'].append(np.array(actions_onehot))
        self.episode_sample['r'].append([reward])
        self.episode_sample['terminate'].append([terminated])
        self.episode_sample['padded'].append([0])
        self.episode_sample['avail_u'].append(np.array(avail_actions))
        '''
        获得原始概率值
        '''
        self.episode_sample['prob_n'].append(prob_n[0])
        
        #存储本步信息，将其作为上一步信息（动作存储）
        ind=0
        for k,v in self.mine_air.items():
            self.air_last_action[k]=actions_onehot[ind]
            ind+=1

    def deduction_end(self):
        # agent结束后自动被调用函数， 可保存模型，比如记录本局策略后的得分等
        logging.info("%s deduction_end" % self.__class__.__name__)
        #print('----------------red_score:{}----------------'.format(self.iTotalScore))
        self.tensorboard_writer.add_scalar('score', self.iTotalScore, global_step=self.episode_num)
        self.episode_num+=1
        
        if self.args.epsilon_anneal_scale == 'episode':
            self.args.epsilon = self.args.epsilon - self.args.anneal_epsilon if self.args.epsilon > self.args.min_epsilon else self.args.epsilon           
        #此处对最后一个样本状态修改
        if self.record_info['score']!=self.current_time:
            duration_score=self.iTotalScore-self.record_info['score']
            final_r=self.episode_sample['r'][-1][0]
            self.episode_sample['r'][-1]=[final_r+duration_score/1843]
        
        #整理数据，按照指定length
        o=self.episode_sample['o'][:-1]
        s=self.episode_sample['s'][:-1]
        u=self.episode_sample['u'][:-1]
        prob=self.episode_sample['prob_n'][:-1]
        u_onehot=self.episode_sample['u_onehot'][:-1]
        avail_u=self.episode_sample['avail_u'][:-1]
        padded=self.episode_sample['padded'][:-1]
        
        r=self.episode_sample['r'][1:]
        terminate=self.episode_sample['terminate'][1:]
        o_next=self.episode_sample['o'][1:]
        s_next=self.episode_sample['s'][1:]
        avail_u_next=self.episode_sample['avail_u'][1:]
        
        if self.args.replay_data:
            store_episode=dict(
                               s=s.copy(),
                               u=u.copy(),
                               r=r.copy(),
                               s_next=s_next.copy(),
                               avail_u_next=avail_u_next.copy(),
                               terminated=terminate.copy()
                       ) 
            np.save(file=self.args.replay_data_file+"replay_dat_{}.npy".format(self.episode_num), arr=np.array(store_episode))
        
        start_index=len(o)
        for i in range(start_index,self.args.episode_limit):
            o.append(np.zeros((self.args.n_agents, self.args.obs_shape)))
            u.append(np.zeros([self.args.n_agents, 1]))
            s.append(np.zeros(self.args.state_shape))
            r.append([0.])
            o_next.append(np.zeros((self.args.n_agents, self.args.obs_shape)))
            s_next.append(np.zeros(self.args.state_shape))
            u_onehot.append(np.zeros((self.args.n_agents, self.args.n_actions)))
            avail_u.append(np.zeros((self.args.n_agents, self.args.n_actions)))
            avail_u_next.append(np.zeros((self.args.n_agents, self.args.n_actions)))
            padded.append([1.])
            terminate.append([1.])
            prob.append(np.array([1.]*self.args.n_agents))
        
       
        #通过队列将数据发送至训练进程
        
        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy(),
                       prob_n=prob.copy()
                       ) 
            
        for key in episode.keys():
            #if key in ['r','padded','terminated']:
                #episode[key]=np.expand_dims(np.array([episode[key]]),2)
            #else:   
            episode[key] = np.array([episode[key]])        
 
        self.episodes.append(episode)
        #self.episodes.append(episode)
        if len(self.episodes)==self.args.n_episodes:
            
            
            episode_batch = self.episodes[0]
            self.episodes.pop(0)

            for episode in self.episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
            if self.args.mp:
                self.Pipe.send({'data':episode_batch,'score':self.iTotalScore})
                
                model_state_dict=self.Pipe.recv()
                
                self.agent.policy.load_model_from_train(model_state_dict)           
            else:   
                #样本存储
                #train_steps=0
                if self.args.alg.find('coma') > -1:
                    #用通道进行多进程沟通
                    for train_step in range(self.args.train_steps):
                        self.agent.train(episode_batch, self.train_steps, self.args.epsilon)
                        self.train_steps += 1
    
                
                self.episodes=[]

    def is_done(self):
        # agent主动结束本局推演，比如一定胜利或失败后, 注意：
        # 只能训练时调用此函数，比赛对战时此函数不可用
        if self.iTotalScore >= 1000 or self.iTotalScore <= -1000:
            return True
        return False
    
    def my_init(self,situation):
        '''
        初始化设置，飞机开启自动开火（初始开火距离为最小那一档）、（燃油状态）自动返航，驱逐舰开火设置
        '''
        #######1.对平台的设置
        #燃油预先规划与返航
        self.fuel_rtb=False
        self.doctrine_fuel_state_planned(fuel_state_plannedEnum=FuelState.Bingo)
        self.doctrine_fuel_state_rtb(fuel_state_rtbEnum=FuelStateRTB.YesLeaveGroup)
        #对歼击机，一直支持燃油返航
        #for unit in situation[0]:
            #if unit['DBID']==753:
                #air=self.get_unit(unit['guid'])
                #air.doctrine_fuel_state_rtb(fuel_state_rtbEnum=FuelStateRTB.YesLeaveGroup)
        
        #武器预先规划与返航
        self.doctrine_weapon_state_planned(weapon_state_plannedEnum=WeaponStatePlanned.WinchesterDisengage)
        self.doctrine_weapon_state_rtb(weapon_state_rtbEnum=WeaponStateRTB.No)
        
        #设置对空，对地，对海的打击规则
        self.doctrine_weapon_control_status_air(weapon_control_status_airEnum=WeaponControlStatus.Free)
        self.doctrine_weapon_control_status_land(weapon_control_status_landEnum=WeaponControlStatus.Free)
        self.doctrine_weapon_control_status_surface(control_status=WeaponControlStatus.Free)
        ####################缺少对海自动打击的指令
        
        #进攻时，忽略计划航线
        self.doctrine_ignore_plotted_course(ignore_plotted_courseEnum=IgnorePlottedCourseWhenAttacking.Yes)
        #自动接战临机目标
        self.doctrine_engage_opportunity_targets(engage_opportunity_targetsEnum=EngageWithContactTarget.Yes_AnyTarget)
        #自动规避
        self.doctrine_automatic_evasion(AutomaticEvasion.Yes)
        
        #雷达全开
        self.doctrine_switch_radar(switch_on=True)
        
        #设置武器,945-响尾蛇（默认）；51-先进中程空空（禁止自动开火）；826-防区外(默认)；15-海麻雀
        self.doctrine_WRA(weapon_id=51, target_type=WRA_WeaponTargetType.Air_Contact_Unknown_Type, quantity_salvo='1', firing_range='none')
        self.doctrine_WRA(weapon_id=51, target_type=WRA_WeaponTargetType.Aircraft_Unspecified,  quantity_salvo='1',firing_range='none')
        self.doctrine_WRA(weapon_id=51, target_type=WRA_WeaponTargetType.Aircraft_5th_Generation, quantity_salvo='1',firing_range='none')
        self.doctrine_WRA(weapon_id=51, target_type=WRA_WeaponTargetType.Aircraft_4th_Generation, quantity_salvo='1',firing_range='none')
        self.doctrine_WRA(weapon_id=51, target_type=WRA_WeaponTargetType.Guided_Weapon_Unspecified,  quantity_salvo='1',firing_range='none')
        
        self.doctrine_WRA(weapon_id=15, target_type=WRA_WeaponTargetType.Aircraft_Unspecified,  quantity_salvo='1',firing_range='none')
        self.doctrine_WRA(weapon_id=15, target_type=WRA_WeaponTargetType.Air_Contact_Unknown_Type, quantity_salvo='1', firing_range='none')
        self.doctrine_WRA(weapon_id=15, target_type=WRA_WeaponTargetType.Aircraft_5th_Generation, quantity_salvo='1', firing_range='none')
        self.doctrine_WRA(weapon_id=15, target_type=WRA_WeaponTargetType.Aircraft_4th_Generation, quantity_salvo='1', firing_range='none')
        self.doctrine_WRA(weapon_id=15, target_type=WRA_WeaponTargetType.Aircraft_4th_Generation, quantity_salvo='1', firing_range='none')
        
        self.doctrine_WRA(weapon_id=15, target_type=WRA_WeaponTargetType.Guided_Weapon_Unspecified,  quantity_salvo='1',firing_range='25')
        self.doctrine_WRA(weapon_id=15, target_type=WRA_WeaponTargetType.Guided_Weapon_Supersonic_Sea_Skimming, quantity_salvo='1', firing_range='25')
        self.doctrine_WRA(weapon_id=15, target_type=WRA_WeaponTargetType.Guided_Weapon_Subsonic_Sea_Skimming, quantity_salvo='1', firing_range='25')
        self.doctrine_WRA(weapon_id=15, target_type=WRA_WeaponTargetType.Guided_Weapon_Supersonic,  quantity_salvo='1',firing_range='25')
        
        
        #飞机单机起飞出动
        #self.air_single_out(list(self.aircrafts.keys()))
        
        '''
        3.25 设置巡逻任务区域，首先将飞机运达己方驱逐舰上侧，再开展决策
        '''
        ################
        self.mission_name='巡逻任务_0'
        self.set_my_mission()
        for air_key in list(self.aircrafts.keys()):
            unit=self.get_unit(air_key)
            unit.assign_to_mission(self.mission_name)
            
        ###############
        
        #####2.智能体进行分析的统计量设计
        self._Statistic_init()
        
        ##3.设置禁飞区，保证飞机不出边界
        
        zone_guid=self.zone_add_no_navigate([(self.max_lat,self.min_lon-0.5),(self.max_lat,self.max_lon+0.5),
                                             (self.max_lat+0.5,self.max_lon+0.5),(self.max_lat+0.5,self.min_lon-0.5)])
        self.zone_set(zone_guid,Isactive=1,Affects=0,RPVISIBLE=0)
        zone_guid=self.zone_add_no_navigate([(self.max_lat+0.5,self.max_lon),(self.min_lat-0.5,self.max_lon),
                                             (self.min_lat-0.5,self.max_lon+0.5),(self.max_lat+0.5,self.max_lon+0.5)])
        self.zone_set(zone_guid,Isactive=1,Affects=0,RPVISIBLE=0)  
        zone_guid=self.zone_add_no_navigate([(self.min_lat,self.max_lon+0.5),(self.min_lat,self.min_lon-0.5),
                                             (self.min_lat-0.5,self.min_lon-0.5),(self.min_lat-0.5,self.max_lon+0.5)])
        self.zone_set(zone_guid,Isactive=1,Affects=0,RPVISIBLE=0) 
        zone_guid=self.zone_add_no_navigate([(self.min_lat-0.5,self.min_lon),(self.max_lat+0.5,self.min_lon),
                                             (self.max_lat+0.5,self.min_lon-0.5),(self.min_lat-0.5,self.min_lon-0.5)])
        self.zone_set(zone_guid,Isactive=1,Affects=0,RPVISIBLE=0)          
    
    def _Statistic_init(self):
        '''
        智能体进行分析的统计量设计
        '''
        
        self.mine_air_reward={}
        self.mine_air={
            'c8dca793-a20f-4056-af88-e354289fcbcd':None,
            'bfd543f7-ed60-4940-9c9e-06cb4ca22c7d':None,
            '6d87e0ba-8105-4769-a19f-bc1c9efe8632':None,
            '114f1923-942e-4d3b-a8fb-87ed9a231e50':None,
            '4da452b4-e136-4660-b270-90eafcd73bfd':None,
            'fedd3f61-d2eb-40ff-a5dd-45cc29bcf6d8':None,
        }
        self.mine_weapon={}
        self.mine_ship={}
        
        self.enemy_air={}
        self.enemy_weapon={}
        self.enemy_ship={}
        self.enemy_Tguid_Fguid={} #记录映射，目标真实guid和虚假guid的映射
        
        self.current_situation=None
        self.last_situation=None
        
        
        self.air_att_max={3836:{'max_fuel':8900.0,'max_speed':1734.0,'spy_distance':185.2},
                          753:{'max_fuel':10910.5,'max_speed':1734.0,'spy_distance':222.24}
                          }
        
        #记录维护己方武器的信息
        self.mine_weapon_type={51:{'name':'AIM-120D型先进中程空空导弹','max_distance':138.9,'max_num':2},
                               945:{'name':'AIM-9X型“响尾蛇”空空导弹','max_distance':18.52,'max_num':6},
                               826:{'name':'AGM-154C联合防区外武器','max_distance':83.34,'max_num':6}}
        self.history_situation={} #对situation信息进行存储，一步只会计算一次
        self.simulation_decision_interval=self.args.decision_interval #仿真决策时间间隔
        self.attack_distance=self.mine_weapon_type[51]['max_distance']*0.75 #'AIM-120D型先进中程空空导弹'的有效射程
        self.current_time=None
        self.air_last_action={} #记录智能体的上一步动作
        
        #对作战地图的限定
        self.min_lon=151.5
        self.max_lon=161.0
        self.min_lat=23.5
        self.max_lat=28.0
        self.map_central=((self.min_lat+self.max_lat)/2.0,(self.max_lon+self.min_lon)/2.0)
        
        #记录当前推演仿真的信息转移
        self.episode_sample={'o':[],#智能体的单个观测，agent_size x obs_n_space
                             's':[],#整体状态，1 x state_n_space
                             'u':[],#智能体的动作，数字 agent_size x 1
                             'u_onehot':[],#智能体的动作onehot编码，agent_size x 32
                             'avail_u':[],#智能体的可用动作，agent_size x 32
                             'r':[],#智能体的奖赏，agent_size x 1
                             'terminate':[],#环境是否结束 1
                             'padded':[],#智能体可能会消亡，此处为标示 agent_size x 1
                             'prob_n':[],#记录智能体当时选择动作的概率
                             }
        
        self.record_info={'airs_state':[0 for i in range(self.args.n_agents)],#用于记录当前所有智能体的在空状态，从全0到出现1，标示推演开始，从包含1到全0，智能体决策全部结束
                          'terminate':False,#智能体决策是否结束
                          'score':0,#智能体推演结束的得分，用于统计智能体推演结束到仿真推演结束的得分差，作为最后的reward
                          'delta_score':0,#同上一决策步的差值
                          'time':self.current_time,#记录当前时刻
                          }
    def ok_plan(self,situation):
        '''
        根据飞机的状态信息，判断本步是否是规划步
        '''
        #全局状态信息记录，如果terminate规划飞机全部失去规划能力
        
        '''
        3.25增加了飞机到达巡逻区域才开始规划的选项，以时间判断，改动最小
        '''
        if self.current_time<800:
            return False
        elif self.mission_name:
            self.delete_mission(self.mission_name)
            self.mission_name=False
        
        if self.record_info['terminate']:
            return False
        #假定所有飞机状态为0
        air_state=[0 for i in range(self.args.n_agents)]
        
        #检测situation中的存活飞机，是否有人为1
        ind=0
        for unit in situation[0]:
            if unit['type']=='Aircraft':
                if unit['airStatus'].value ==1:
                    air_state[ind]=1
                ind+=1
        
        #如果无飞机可规划，且上一时刻有飞机可规划，则当前时刻为最终时刻，修改terminate，记录当前的得分
        if  (max(air_state)!=1 and max(self.record_info['airs_state'])==1) or abs(self.iTotalScore-self.record_info['score'])>1000:
            self.record_info['airs_state']=air_state
            self.record_info['terminate']=True
            self.record_info['delta_score']=self.iTotalScore-self.record_info['score']
            self.record_info['score']=self.iTotalScore
            return True
        #如果有飞机处于在空可规划，则记录在空状态，记录在当前步的差值分数，记录当前步下的总得分
        elif max(air_state)==1:
            self.record_info['airs_state']=air_state
            self.record_info['delta_score']=self.iTotalScore-self.record_info['score']#初始化为0
            self.record_info['score']=self.iTotalScore
            return True
        return False
    
        
    def _get_situation(self,situation,time):
        '''
        从原始的situation信息中获得规范化的单元信息,更新了飞机单元的私有奖励，靠近和返航。
        '''
        mine_air={}
        mine_weapon={}
        mine_ship={}
        
        enemy_air={}
        enemy_weapon={}
        enemy_ship={}        
        
        #额外统计需要统计的我方空空弹、空地弹、地空单，敌方空空弹、空地弹、地空弹剩余数量
        for unit in situation[0]:
            #飞机类记录：经度、纬度、朝向、速度、燃油、可用武器、飞机状态、单元DBID、单元名字
            if unit['type']=='Aircraft':
                guid=unit['guid']
                temp={'latitude':unit['latitude'],
                      'longitude':unit['longitude'],
                      'heading':unit['heading'],
                      'speed':unit['speed'],
                      'fuel':unit['fuel'],
                      'weapons':unit['weaponsValid'],
                      'status':unit['airStatus'].value,
                      'DBID':unit['DBID'],
                      'name':unit['name']
                      }
                
                mine_air[guid]=temp

            #武器类记录（空空）：经度、纬度、朝向、速度、目标ID（目标无则为None）、武器类型、武器名字
            elif unit['DBID'] in self.mine_weapon_type:
                guid=unit['guid']
                temp={'latitude':unit['latitude'],
                      'longitude':unit['longitude'],
                      'heading':unit['heading'],
                      'speed':unit['speed'],
                      'target':unit['target'] if unit['target'] in self.enemy_Tguid_Fguid.values() else None,
                      'type': list(self.mine_weapon_type.keys()).index(unit['DBID']),
                      'name':unit['name']
                      }
                mine_weapon[guid]=temp
            
            #记录己方舰船信息：经度、纬度、朝向、速度、可用武器、损管值、单元DBID、单元名字
            elif unit['type']=='Ship':
                if '驱逐舰' not in unit['name']:
                    continue
                guid=unit['guid']
                temp={'latitude':unit['latitude'],
                      'longitude':unit['longitude'],
                      'heading':unit['heading'],
                      'speed':unit['speed'],
                      'weapons':unit['weaponsValid'],
                      'damage':math.floor(float(unit['damage']))/100.0,
                      'DBID':unit['DBID'],
                      'name':unit['name']
                      }
                mine_ship[guid]=temp

        
        for unit in situation[1]:
            
            #飞机类记录：经度、纬度、朝向、速度、位置精准、单元名字
            if unit['type']== 'Air':
                guid=unit['guid']
                temp={'latitude':unit['latitude'],
                      'longitude':unit['longitude'],
                      'heading':unit['heading'],
                      'speed':unit['speed'],
                      'visiable':1 if unit['area']==[] else 0,
                      'name':unit['name']
                      }
                
                #获得单元的真实guid，用于记录单元信息，确保在一局对抗中，目标位置不串
                t_guid=self.get_contact(guid).m_ActualUnit
                enemy_air[t_guid]=temp
                self.enemy_Tguid_Fguid[t_guid]=guid
                
                
            #导弹类记录（只记录空空弹）：经度、纬度、朝向、速度、单元名字
            elif unit['type']== 'Missile':
                guid=unit['guid']
                if 'VAMPIRE' in unit['name'] or 'GuidedWeapon' in unit['name']:
                    continue
                temp={'latitude':unit['latitude'],
                      'longitude':unit['longitude'],
                      'heading':unit['heading'],
                      'speed':unit['speed'],
                      'name':unit['name']
                      }                 
                enemy_weapon[guid]=temp
                
            #舰船类：经度、纬度、朝向、速度、单元名字
            elif unit['type']=='Surface':
                if '驱逐舰' not in unit['name']:
                    continue
                guid=unit['guid']
                temp={'latitude':unit['latitude'],
                      'longitude':unit['longitude'],
                      'heading':unit['heading'],
                      'speed':unit['speed'],
                      'name':unit['name']
                      }
                
                enemy_ship[guid]=temp            
                
        #将信息填充至信息，没有的补None
        
        ##己方飞机
        #首先更新单元
        for k,v in mine_air.items():
            #对reward进行记录与更新
            if self.mine_air[k]!=None:#表明上一步是有值的，记录的为上一时刻的信息
                unit=self.mine_air[k]
                if k in ['bfd543f7-ed60-4940-9c9e-06cb4ca22c7d','c8dca793-a20f-4056-af88-e354289fcbcd']:
                    
                    last_distance=get_horizontal_distance(geopoint1=(unit['latitude'],unit['longitude']), geopoint2=(25.80,157.5))
                    current_dianstance=get_horizontal_distance(geopoint1=(v['latitude'],v['longitude']), geopoint2=(25.80,157.5))
                else:
                    last_distance=get_horizontal_distance(geopoint1=(unit['latitude'],unit['longitude']), geopoint2=(25.80,155.26))
                    current_dianstance=get_horizontal_distance(geopoint1=(v['latitude'],v['longitude']), geopoint2=(25.80,155.26))                    
                
                #如果k不在self.mine_air_reward，则飞机为首次规划，记录都为0
                if k not in self.mine_air_reward:#初始化为空
                    self.mine_air_reward[k]={'approch_reward':0, 'return_reward':0}
                    continue
                
                #返航的飞机，不再有approch_reward
                if v['status']!=1:
                    self.mine_air_reward[k]['approch_reward']=0
                else:
                    if k in ['bfd543f7-ed60-4940-9c9e-06cb4ca22c7d','c8dca793-a20f-4056-af88-e354289fcbcd']:
                        if last_distance<60 and current_dianstance<60:
                            self.mine_air_reward[k]['approch_reward']=0
                        else:
                            self.mine_air_reward[k]['approch_reward']=(last_distance-current_dianstance)/400#400/sum(self.record_info['airs_state'])#初一当前在规划飞机的总数
                    else:
                        if last_distance<160 and current_dianstance<160:
                            self.mine_air_reward[k]['approch_reward']=0
                        else:
                            self.mine_air_reward[k]['approch_reward']=(last_distance-current_dianstance)/400#400/sum(self.record_info['airs_state'])
                #上一步飞机信息！=None,上一步为在空1，本步非1，则有返航惩罚
                time_step=time-self.simulation_decision_interval
                if time_step in self.history_situation and self.history_situation[time_step]['mine']['airs'][k]!=None \
                   and self.history_situation[time_step]['mine']['airs'][k]['status']==1 and v['status']!=1:
                    self.mine_air_reward[k]['return_reward']=-1
                else:
                    self.mine_air_reward[k]['return_reward']=0
                
            #更新单元信息至全局变量
            self.mine_air[k]=v
        #其次，删除消失的单元  
        for k,v in self.mine_air.items():
            #飞机不在本步的统计信息，则为死亡状态。奖赏为0
            if k not in mine_air.keys():
                self.mine_air[k]=None
                self.mine_air_reward[k]['approch_reward']=0
                self.mine_air_reward[k]['return_reward']=0
        
        ##己方武器
        #首先更新武器
        for k,v in mine_weapon.items():
            self.mine_weapon[k]=v
        #其次，删除消失的武器   
        for k,v in self.mine_weapon.items():
            if k not in mine_weapon.keys():
                self.mine_weapon[k]=None
                
        ##己方舰船
        #首先更新舰船
        for k,v in mine_ship.items():
            self.mine_ship[k]=v
        #其次删除消失的舰船    
        for k,v in self.mine_ship.items():
            if k not in mine_ship.keys():
                self.mine_ship[k]=None
        ##敌方飞机
        for k,v in enemy_air.items():
            self.enemy_air[k]=v
            
        for k,v in self.enemy_air.items():
            if k not in enemy_air.keys():
                self.enemy_air[k]=None
        ##敌方舰船       
        for k,v in enemy_ship.items():
            self.enemy_ship[k]=v
            
        for k,v in self.enemy_ship.items():
            if k not in enemy_ship.keys():
                self.enemy_ship[k]=None
        ##敌方导弹       
        for k,v in enemy_weapon.items():
            self.enemy_weapon[k]=v
            
        for k,v in self.enemy_weapon.items():
            if k not in enemy_weapon.keys():
                self.enemy_weapon[k]=None
        
        
        return {'mine':{'airs':self.mine_air,
                        'weapon':self.mine_weapon,
                        'ship':self.mine_ship},
                
                'enemy':{'airs':self.enemy_air,
                        'weapon':self.enemy_weapon,
                        'ship':self.enemy_ship}
                }
    
    def get_current_situation(self,situation,time):
        '''
        将解析数据返回
        '''
        import copy
        if time not in self.history_situation.keys():
            self.history_situation[time]=copy.deepcopy(self._get_situation(situation, time))
        
        return self.history_situation[time]

    def get_state(self, situation, time):
        '''
        获得全局输入
        '''
        self.get_current_situation(situation, time)
        mine_air=np.zeros((self.args.n_agents,77))
        ind=0
        for k,v in self.mine_air.items():
            if v==None:
                ind+=1
                continue
            lat_relative=(v['latitude']-self.map_central[0])*2/(self.max_lat-self.min_lat)
            lon_relative=(v['longitude']-self.map_central[1])*2/(self.max_lon-self.min_lon)
            fuel=v['fuel']/self.air_att_max[v['DBID']]['max_fuel'] #if v['DBID']==3538 else v['fuel']/self.air_att_max[753]['max_fuel']
            weapon=[0 for _ in self.mine_weapon_type]
            status=[int(i==v['status']) for i in range(3)]
            ind=0
            for k_w,v_w in self.mine_weapon_type.items():
                if k_w in v['weapons']:
                    weapon[ind]=v['weapons'][k_w]/v_w['max_num']
                ind+=1            
            type_i=[int(v['DBID']==k) for k,_ in self.air_att_max.items()]   #[1,0] if v['DBID']==3836 else [0,1]
            last_action=self._get_last_action(k)
            temp=[lat_relative,lon_relative,fuel]+weapon+status+type_i+last_action+self.get_avai_action(k)
            mine_air[ind]=np.array(temp)
            ind+=1
        
        mine_ship=np.zeros((1,4))
        for k,v in self.mine_ship.items():
            unit=self.get_unit(k)
            #unit.attack_weapon_allocate_to_target(target=(self.max_lat,self.max_lon), weaponDBID='hsfw-dataweapon-00000000000586', weapon_count=2)
            if v!=None:        
                lat_relative=(v['latitude']-self.map_central[0])*2/(self.max_lat-self.min_lat)
                lon_relative=(v['longitude']-self.map_central[1])*2/(self.max_lon-self.min_lon)
                weapon=0 if 15 not in v['weapons'] else v['weapons'][15]/8.0##
                damage=v['damage']
                mine_ship[0]=np.array([lat_relative,lon_relative,weapon,damage])
                
        enemy_air=np.zeros((6,3))
        ind=0
        for k,v in self.enemy_air.items():
            if v==None:
                ind+=1
                continue 
            lat_relative=(v['latitude']-self.map_central[0])*2/(self.max_lat-self.min_lat)
            lon_relative=(v['longitude']-self.map_central[1])*2/(self.max_lon-self.min_lon) 
            attacked=1 if self._get_target_attacked(k) else 0
            temp=[lat_relative,lon_relative,attacked]
            enemy_air[ind]=np.array(temp)
            ind+=1
        
        enemy_ship=np.zeros((1,2))
        for k,v in self.enemy_ship.items():
            if v!=None:        
                lat_relative=(v['latitude']-self.map_central[0])*2/(self.max_lat-self.min_lat)
                lon_relative=(v['longitude']-self.map_central[1])*2/(self.max_lon-self.min_lon)
                enemy_ship[0]=np.array([lat_relative,lon_relative])
            
        enemy_weapon=np.zeros((48,4))
        ind=0
        for k,v in self.enemy_weapon.items():
            if v==None:
                ind+=1
                continue 
            lat_relative=(v['latitude']-self.map_central[0])*2/(self.max_lat-self.min_lat)
            lon_relative=(v['longitude']-self.map_central[1])*2/(self.max_lon-self.min_lon) 
            type_i=0 if self.enemy_weapon[k]['name'].find('防空')==-1 else 1
            temp=[lat_relative,lon_relative]+[int(i==type_i) for i in range(2)] 
            enemy_weapon[ind]=np.array(temp)
            ind+=1        
        
        state=np.concatenate(
            (
                np.array(mine_air).flatten(),
                np.array(mine_ship).flatten(),
                np.array(enemy_air).flatten(),
                np.array(enemy_ship).flatten(),
                np.array(enemy_weapon).flatten(),
            )
        )  
        return state
    
    def get_single_obs(self,guid):
        '''
        根据传来的guid，获得指定的观测
        
        '''
        if self.mine_air[guid]==None:
            return [0 for i in range(0)]
        else:
            unit=self.mine_air[guid]
            #可移动方向与速度
            avai_move_action=self._get_avai_move_action(guid)
            
            #敌方飞机信息
            enemy_air=[[0 for _ in range(5)] for _ in range(self.args.n_agents)]
            ind=0
            for k,v in self.enemy_air.items():
                if v!=None:
                    dis=get_horizontal_distance(geopoint1=(unit['latitude'],unit['longitude']), geopoint2=(v['latitude'],v['longitude']))
                    if dis<self.air_att_max[unit['DBID']]['spy_distance']:
                        attackable=1 if dis<self.mine_weapon_type[51]['max_distance'] and 51 in unit['weapons'] else 0
                        distance_relative=dis/self.air_att_max[unit['DBID']]['spy_distance']
                        lat_relative=get_horizontal_distance(geopoint1=(unit['latitude'],unit['longitude']), geopoint2=(v['latitude'],unit['longitude']))/self.air_att_max[unit['DBID']]['spy_distance']
                        lon_distance=get_horizontal_distance(geopoint1=(unit['latitude'],unit['longitude']), geopoint2=(unit['latitude'],v['longitude']))/self.air_att_max[unit['DBID']]['spy_distance']
                        attacked=1 if self._get_target_attacked(k) else 0
                        enemy_air[ind]=[attackable,distance_relative,lat_relative,lon_distance,attacked]
                ind+=1   
            
            #敌方舰船信息
            enemy_ship=[0 for _ in range(4)]
            for k,v in self.enemy_ship.items():
                if v!=None:
                    dis=get_horizontal_distance(geopoint1=(unit['latitude'],unit['longitude']), geopoint2=(v['latitude'],v['longitude']))
                    if dis<self.air_att_max[unit['DBID']]['spy_distance']:
                        attackable=1 if dis<self.mine_weapon_type[826]['max_distance'] and 826 in unit['weapons'] else 0
                        distance_relative=dis/self.air_att_max[unit['DBID']]['spy_distance']
                        lat_relative=get_horizontal_distance(geopoint1=(unit['latitude'],unit['longitude']), geopoint2=(v['latitude'],unit['longitude']))/self.air_att_max[unit['DBID']]['spy_distance']
                        lon_distance=get_horizontal_distance(geopoint1=(unit['latitude'],unit['longitude']), geopoint2=(unit['latitude'],v['longitude']))/self.air_att_max[unit['DBID']]['spy_distance']
                        enemy_ship=[attackable,min(distance_relative,1),min(lat_relative,1),min(lon_distance,1)]
            
            #敌方导弹信息
            enemy_weapon=[[0 for _ in range(5)] for _ in range(3)]
            ind=0
            weapon_dis={}
            for  k,v in self.enemy_weapon.items():
                if v==None:
                    continue
                dis=get_horizontal_distance(geopoint1=(unit['latitude'],unit['longitude']), geopoint2=(v['latitude'],v['longitude']))
                if self._judge_weapon_attack(unit, v,dis):
                    weapon_dis[k]=dis
            if len(weapon_dis)>0:
                weapon_sorted=sorted(weapon_dis.items(),key=lambda d:d[1])
                for item in weapon_sorted:
                    if ind==3:
                        break
                    v=self.enemy_weapon[item[0]]
                    distance_relative=item[1]/self.air_att_max[unit['DBID']]['spy_distance']
                    lat_relative=get_horizontal_distance(geopoint1=(unit['latitude'],unit['longitude']), geopoint2=(v['latitude'],unit['longitude']))/self.air_att_max[unit['DBID']]['spy_distance']
                    lon_distance=get_horizontal_distance(geopoint1=(unit['latitude'],unit['longitude']), geopoint2=(unit['latitude'],v['longitude']))/self.air_att_max[unit['DBID']]['spy_distance']
                    type_i=0 if self.enemy_weapon[item[0]]['name'].find('防空')==-1 else 1
                    
                    enemy_weapon[ind]=[min(distance_relative,1),min(lat_relative,1),min(lon_distance,1)]+[int(i==type_i) for i in range(2)]  
                    ind+=1
                
            #友方飞机信息
            al_air=[[0 for _ in range(45)] for _ in range(5)]
            ind=0
            for k,v in self.mine_air.items():
                if v!=None and k!=guid:
                    dis=get_horizontal_distance(geopoint1=(unit['latitude'],unit['longitude']), geopoint2=(v['latitude'],v['longitude']))
                    if dis<self.air_att_max[unit['DBID']]['spy_distance']:
                        distance_relative=dis/self.air_att_max[unit['DBID']]['spy_distance']
                        lat_relative=get_horizontal_distance(geopoint1=(unit['latitude'],unit['longitude']), geopoint2=(v['latitude'],unit['longitude']))/self.air_att_max[unit['DBID']]['spy_distance']
                        lon_distance=get_horizontal_distance(geopoint1=(unit['latitude'],unit['longitude']), geopoint2=(unit['latitude'],v['longitude']))/self.air_att_max[unit['DBID']]['spy_distance']
                        fuel=v['fuel']/self.air_att_max[3538]['max_fuel'] if v['DBID']==3538 else v['fuel']/self.air_att_max[753]['max_fuel']
                        status=[int(i==v['status']) for i in range(3)]
                        weapon=[0,0,0]
                        ind=0
                        for k_w,v_w in self.mine_weapon_type.items():
                            if k_w in v['weapons']:
                                weapon[ind]=v['weapons'][k_w]/v_w['max_num']
                            ind+=1
                        type_i=[1,0] if v['DBID']==3836 else [0,1]
                        al_air[ind]=[distance_relative,lat_relative,lon_distance,fuel]+status+weapon+self._get_last_action(k)+type_i
                ind+=1
                
            
            #友方舰船信息
            al_ship=[0 for _ in range(5)]
            for k,v in self.mine_ship.items():
                if v!=None:
                    dis=get_horizontal_distance(geopoint1=(unit['latitude'],unit['longitude']), geopoint2=(v['latitude'],v['longitude']))
                    if dis<self.air_att_max[unit['DBID']]['spy_distance']:
                        distance_relative=dis/self.air_att_max[unit['DBID']]['spy_distance']
                        lat_relative=get_horizontal_distance(geopoint1=(unit['latitude'],unit['longitude']), geopoint2=(v['latitude'],unit['longitude']))/self.air_att_max[unit['DBID']]['spy_distance']
                        lon_relative=get_horizontal_distance(geopoint1=(unit['latitude'],unit['longitude']), geopoint2=(unit['latitude'],v['longitude']))/self.air_att_max[unit['DBID']]['spy_distance']
                        weapon=0 if 15 not in v['weapons'] else v['weapons'][15]/8.0
                        damage=v['damage']
                        al_ship=[damage,min(distance_relative,1),min(lat_relative,1),min(lon_relative,1),weapon]                
            
            
            #自身消息
            lat_relative=(unit['latitude']-self.map_central[0])*2/(self.max_lat-self.min_lat)
            lon_relative=(unit['longitude']-self.map_central[1])*2/(self.max_lon-self.min_lon)
            fuel=unit['fuel']/self.air_att_max[3538]['max_fuel'] if unit['DBID']==3538 else unit['fuel']/self.air_att_max[753]['max_fuel']
            weapon=[0,0,0]
            ind=0
            for k_w,v_w in self.mine_weapon_type.items():
                if k_w in v['weapons']:
                    weapon[ind]=v['weapons'][k_w]/v_w['max_num']
                ind+=1            
            type_i=[1,0] if v['DBID']==3836 else [0,1]
            mine_info=[lat_relative,lon_relative,fuel]+weapon+type_i
            
            obs=np.concatenate(
                (
                    np.array(avai_move_action).flatten(),
                    np.array(enemy_ship).flatten(),
                    np.array(enemy_air).flatten(),
                    np.array(enemy_weapon).flatten(),
                    np.array(al_air).flatten(),
                    np.array(al_ship).flatten(),
                    np.array(mine_info).flatten(),
                )
            )
            
            return obs
    def _judge_weapon_attack(self,unit,weapon,dis=None):
        '''
        根据导弹朝向以及距离
        '''
        if dis==None:
            dis=get_horizontal_distance(geopoint1=(unit['latitude'],unit['longitude']), geopoint2=(weapon['latitude'],weapon['longitude']))
        if dis>=self.mine_weapon_type[51]['max_distance']*0.75:
            return False
        #导弹朝向的一定范围内，包含该目标
        w_u_heading=get_azimuth((weapon['latitude'],weapon['longitude']), (unit['latitude'],unit['longitude']))
        min_heading,max_heading=weapon['heading']-50,weapon['heading']+50
        range_heading=[]
        if min_heading<0:
            range_heading=[[min_heading+360,0],[0,max_heading]]
        elif max_heading>360:
            range_heading=[[min_heading,360],[0,max_heading-360]]
        else:
            range_heading=[[min_heading,max_heading]]
        
        u_in_range=False
        for interval in range_heading:
            if interval[0]<=w_u_heading<=interval[1]:
                u_in_range=True
                break
        
        if u_in_range and 20<dis<self.mine_weapon_type[51]['max_distance']*0.75:
            return True
        if dis<=20:
            return True
        
        return False
            
    def _get_last_action(self,guid):
        '''
        返回上一步的实体动作信息，
        '''
        if guid in self.air_last_action:
            return self.air_last_action[guid]
        else:
            return [0 for _ in range(self.args.n_actions)]
        
            
    def _get_target_attacked(self,guid):
        '''
        获取敌方单元是否被打击
        '''
        f_guid=self.enemy_Tguid_Fguid[guid]
        for k,v in self.mine_weapon.items():
            if v!=None and v['target']==f_guid:
                return True
        return False
        
        
    def get_obs(self,situation,time):
        '''
        获得所有agent的观测
        '''
        self.get_current_situation(situation, time)
        obs=np.zeros((self.args.n_agents,self.args.obs_shape))
        ind=0
        for k,v in self.mine_air.items():
            if v!=None and v['status']!=0:
                obs[ind]=self.get_single_obs(k)
            ind+=1
            
        return obs
    
    def _get_avai_move_action(self,guid):
        '''
        根据当前的飞机位置，判断之后的动作飞行会不会越界，进而检测当前机动动作是否可用
        '''    

        unit=self.mine_air[guid]
         
        speeds=[648.2,888.96,963.06,1703.84] if unit['DBID']!=3836 else [648.2,888.96,1703.84,10**10]
        headings=[i*60 for i in range(6)]
        
        temp=[0 for i in range(len(speeds)*len(headings))]
        
        
        ind=0
        
        for speed in speeds:
            for heading in headings:
                new_point=get_geopoint_from_distance(geo_point=(unit['latitude'],unit['longitude']), 
                                                     azimuth=heading, 
                                                     distance_m=3*speed/3.6*(self.simulation_decision_interval))
                
                if self.min_lat<new_point[0]<self.max_lat and self.min_lon<new_point[1]<self.max_lon:
                    temp[ind]=1
                
                ind+=1
                
        #什么也不做
        temp.append(1)
        
        return temp
    
    def _get_avai_attack_action(self,guid):
        '''
        获得可打击目标，目标如果在打击范围中，则可打击（船对于轰炸机始终可以打击）
        '''
        unit=self.mine_air[guid]
        
        temp=[0 for i in range(7)]
        if unit==None:
            return temp
        ind=0
        if 51 in unit['weapons']:    
            for k,v in self.enemy_air.items():
                if v!=None:
                    dis=get_horizontal_distance(geopoint1=(unit['latitude'],unit['longitude']), geopoint2=(v['latitude'],v['longitude']))
                    if dis<self.attack_distance:
                        temp[ind]=1
                ind+=1

        ind=self.args.n_agents   
        
        '''
        3.25 修改了轰炸机的可打击目标，
        如果敌方舰船存在，且轰炸机携带有对地导弹，则可打击
        '''
        for k,v in self.enemy_ship.items():
            if v!=None and 826 in unit['weapons']:
                #根据射程设置的可打击
                #dis=get_horizontal_distance(geopoint1=(unit['latitude'],unit['longitude']), geopoint2=(v['latitude'],v['longitude']))
                #if dis<self.mine_weapon_type[826]['max_distance']:  
                    #temp[ind]=1  
                #根据有无舰船设计的可打击（3.25修改）
                temp[ind]=1 
        return temp
    
    def get_avai_action(self,guid):
        '''
        根据单元的guid，获得其可选动作
        '''
        #可决策的智能体不能选择 no_op
        unit=self.get_unit(guid)
        
        #如果飞机没了或者返航，则no_op为1，反之为0
        if self.mine_air[guid]==None or unit.get_status_type().value!=1:
            temp=[0]*self.args.n_actions
            temp[0]=1
            return temp
        else:
            no_op=[0]
            
        avai_move=self._get_avai_move_action(guid)   
        avai_attack=self._get_avai_attack_action(guid)
        
        return no_op+avai_move+avai_attack
    
    def _exect_action(self,guid,action):
        '''
        首先清除上一步的所有指令，即打击和机动动作
        '''
        unit=self.get_unit(guid)
        unit_info=self.mine_air[guid]
        
        #清除上一步指令，包含打击和机动。
        unit.delete_coursed_point(clear=True)
        unit.attack_drop_target_all()
        
        #进入本步决策的单元都是可以执行的，因而删除第一位的no_op动作
        action=action-1
        
        if action<24:
            speed=int(action/6)
            heading=action%6
            
            new_point=get_geopoint_from_distance(geo_point=(unit_info['latitude'],unit_info['longitude']), azimuth=heading*60, distance_m=15000)
            unit.plotted_course([new_point])
            if speed==0:
                unit.set_throttle(Throttle.Loiter)
            elif speed==1:
                unit.set_throttle(Throttle.Cruise)
            elif speed==2:
                unit.set_throttle(Throttle.Full)
            elif speed==3:
                unit.set_throttle(Throttle.Flank)           
            
        elif 24<action<=30:
            target_guids=list(self.enemy_Tguid_Fguid.values())
            #loadout_DBID=int(unit.loadout.strDBGUID.split('-')[-1])
            #unit.attack_manual(contact_guid=target_guids[action-25],mount_num=loadout_DBID,weapon_num='hsfw-dataweapon-00000000000051',qty_num=1)
            unit.attack_weapon_allocate_to_target(target=target_guids[action-25], weaponDBID='hsfw-dataweapon-00000000000051', weapon_count=1)
        elif action==31:
            target_guid=list(self.enemy_ship.keys())[0]
            #loadout_DBID=int(unit.loadout.strDBGUID.split('-')[-1])
            #unit.attack_manual(contact_guid=target_guid,mount_num=loadout_DBID,weapon_num=826,qty_num=2)    
            unit.attack_weapon_allocate_to_target(target=target_guid, weaponDBID='hsfw-dataweapon-00000000000826', weapon_count=2)
            
    def get_units_reward(self):
        '''
        加入智能体的靠近奖励
        '''
        reward=0

        for k,v in self.mine_air_reward.items():
            #if k in ['bfd543f7-ed60-4940-9c9e-06cb4ca22c7d','c8dca793-a20f-4056-af88-e354289fcbcd']:
                
            reward+=(v['approch_reward']+v['return_reward'])
            #else:
                #reward+=(v['return_reward'])
            
            '''
            3.25不再考虑靠近奖励
            '''
            #reward+=v['return_reward']
            #print(k,v)
  
        return reward
        
        
    def set_my_point(self):
        '''
        根据点位计算
        '''
        point=(Pos_my_ship['latitude']-0.2,Pos_my_ship['longitude']-0.2,
                                         'RP_{}'.format(0))
        self.reference_point_add(point)
        
        point=(Pos_my_ship['latitude']+0.2,Pos_my_ship['longitude']-0.2,
                                         'RP_{}'.format(1))
        self.reference_point_add(point) 
        
        point=(Pos_my_ship['latitude']+0.2,Pos_my_ship['longitude']+0.2,
                                         'RP_{}'.format(2))
        self.reference_point_add(point)  
        
        point=(Pos_my_ship['latitude']-0.2,Pos_my_ship['longitude']+0.2,
                                         'RP_{}'.format(3))
        self.reference_point_add(point)        
        
        
    def set_my_mission(self):
        '''
        设置任务
        '''
        self.set_my_point()
        self. create_patrol_mission(
            name= self.mission_name,
            mission_type= MissionPatrolType.ANTI_SHIP,
            point_list= ['RP_{}'.format(0), 'RP_{}'.format(1), 
                         'RP_{}'.format(2), 'RP_{}'.format(3)]
        )
        
        mission=self.get_mission_by_name(self.mission_name)
        mission.set_one_third_rule(False)
        mission.patrol_checkOPA(False)
        mission.doctrine_fuel_state_rtb(fuel_state_rtbEnum=FuelStateRTB.YesLeaveGroup)
        
        return self.mission_name