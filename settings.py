import os
import numpy
import xml.dom.minidom

class Settings():
    def __init__(self):
        self.targ_mat_1 = numpy.loadtxt(os.path.join(os.path.dirname(__file__), 'input_data', 'targ_1.txt'))
        self.obs_mat = numpy.loadtxt(os.path.join(os.path.dirname(__file__), 'input_data', 'obs.txt'))

    #打开xml文档
        self.dom = xml.dom.minidom.parse(os.path.join(os.getcwd(), 'config_swarm.xml'))
    #得到文档元素对象
        self.root = self.dom.documentElement
        '''
        swarm:
        componemts comp
        scene
        problem_setting
        algorithm_setting
        '''

        #.nodeValue    ->    .toxml()  也可以

        #窄道宽度
        self.width=int(self.root.getElementsByTagName('scene')[0].getElementsByTagName('scene_width')[0].childNodes[0].nodeValue)

    # 初始化
        self.INIT_DEPTH_MIN = int(self.root.getElementsByTagName('algorithm_setting')[0].getElementsByTagName('min_depth')[0].childNodes[0].nodeValue)
        self.INIT_DEPTH_MAX = int(self.root.getElementsByTagName('algorithm_setting')[0].getElementsByTagName('max_depth')[0].childNodes[0].nodeValue)

    # 优化的结构
        self.CX_PB = float(self.root.getElementsByTagName('algorithm_setting')[0].getElementsByTagName('struct_cross_rate')[0].childNodes[0].nodeValue)    # 交叉概率
        self.MUT_PB = float(self.root.getElementsByTagName('algorithm_setting')[0].getElementsByTagName('struct_mutation_rate')[0].childNodes[0].nodeValue)   # 变异概率
        self.NUM_GEN = int(self.root.getElementsByTagName('algorithm_setting')[0].getElementsByTagName('max_iter')[0].childNodes[0].nodeValue)   # number of generations
        self.POP_SIZE = int(self.root.getElementsByTagName('algorithm_setting')[0].getElementsByTagName('pop_struct_size')[0].childNodes[0].nodeValue)# population size

    # DE优化的结构参数（内部定义）
        self.THETA_CX_RATE = 0.9   # theta交叉概率
        self.THETA_TOTAL_GEN = 1  # 优化theta的代数     #因为在DE优化中更新individual.theta，所以一定不能为0
        self.THETA_RANGE = [0, 2]  # theta取值范围 (开区间)
        self.THETA_F = 0.5         # DE的变异率
        self.hof_size = 1          # theta的名人堂size
        self.pop_size = 3         # 种群数量 (theta组数量)

    '''
    print(width)
    print(INIT_DEPTH_MIN)
    print(INIT_DEPTH_MAX)
    print(CX_PB)
    print(MUT_PB)
    print(NUM_GEN)
    print(POP_SIZE)
    '''




