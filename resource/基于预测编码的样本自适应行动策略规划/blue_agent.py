# -*- coding:utf-8 -*-

import logging
logging.basicConfig(format="%(asctime)s %(name)s:%(levelname)s:    %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

from entitys.player import Player


"""
蓝方bot
"""


class BlueAgent(Player):
    def __init__(self, side_name,kwarg):
        Player.__init__(self, side_name)
        logging.info('%s:%s play game' % (side_name, self.__class__.__name__))

    def initial(self, situation):
        # 初始化函数，每局推演开始前被调用
        logging.info("%s initial, units count:%d" % (self.__class__.__name__, len(situation[0])))

    def step(self, time_elapse, situation):
        # 每步决策函数
        # logging.info(
        #    "%s step, units count:%d, contact:%d" % (self.__class__.__name__, len(situation[0]), len(situation[1])))
        
        
        pass

    def deduction_end(self):
        # agent结束后自动被调用函数， 可保存模型，比如记录本局策略后的得分等
        logging.info("%s deduction_end" % self.__class__.__name__)

    def is_done(self):
        # agent主动结束本局推演，比如一定胜利或失败后, 注意：
        # 只能训练时调用此函数，比赛对战时此函数不可用
        if self.iTotalScore >= 1000 or self.iTotalScore <= -1000:
            return True
        return False

