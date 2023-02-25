import numpy
import math
import matplotlib.pyplot as plt

import numpy as np

from settings import Settings

#obs和tar由width决定

class Setting_Scene():
    def __init__(self, settings):

        self.width = settings.width
        self.midline=40
        #中轴：40

        # target列表， obstacles列表，机器人与目标最大dmax、最小距离din、机器人与窄道最小距离domin
        # 都搞成全局变量
        self.obstacles = []
        self.obstacles.append(   [(self.midline-self.width/2, 88), (self.midline-self.width/2, 68), (self.midline+self.width/2, 88), (self.midline+self.width/2, 68)]   )
        #先只考虑条形窄道            每个元素是个列表，[左上, 左下, 右上, 右下]
        #                                          [(30, 88), (30, 68), (50, 88), (50, 68)]
        self.targets = []
        self.targets.append(   ((self.obstacles[0][1][0]+self.obstacles[0][3][0])/2, (self.obstacles[0][1][1]+self.obstacles[0][0][1])/2)   )


        #targets_mat和obstacles_mat
        self.targets_mat = []  #按targets顺序排列  targ_mat_0 = settings.targ_mat_0等
        self.targets_mat.append(settings.targ_mat_1)
        self.obstacles_mat = [] #obs_mat = settings.obs_mat
        self.obstacles_mat.append(settings.obs_mat)

        self.dmax = 6
        self.dmin = 3
        self.domin = 6
        self.k=(1, 2, 2)

def sigmoid(x, a, k):
    return 1 / (1 + numpy.exp(-k * (x - a)))

#Contours类和Contour类，Contour类基本不用看
class Contour(list):#继承list
    def __init__(self, pos):
        #generate_contours中contour = Contour(contour_pos)
        #contour是contours中的元素
        #contour_pos是等高线上取的点，类型list，[(x=, y=),    ...   ,(x=, y=)]
        list.__init__(self, pos)
        self._fitness = None

    @property
    def contour_fitness(self):
        return self._fitness

    @contour_fitness.setter
    def contour_fitness(self, v):
        self._fitness = v

#主要是generate_contours和find_best_contour函数
class Contours(object):
    def __init__(self, individual, i, settings_scene):
        self.settings_scene = settings_scene
        self.pattern = individual.compile()
        self.contours = self.generate_contours(self.settings_scene.targets_mat[i], self.settings_scene.obstacles_mat[i])

        # 计算每一条等高线的contour_fitness，并取fitness最大的一条即最差的一条等高线
        self.minfit_contour = self.find_best_contour(i)  #
        self.minfit_contour.cv = self.get_contour_cv(i)  # minfit_contour的cv值（最佳等高线的cv值）

    # 由self.pattern、self.target_mat、self.obstacle_mat产生等高线图
    def generate_contours(self, target_mat, obstacles_mat):
        #numpy.array生成 numpy.ndarray类型,
        #第i个元素是最外层括号下的第i个元素，len是最外层括号下的元素个数，.shape是最外层至最里层括号下的元素个数
        #两层    .shape即矩阵形状
        #print(type(pattern))
        Z = np.zeros([100,100])

        for i in range(len(target_mat)):
            #pattern:eval(code, components_dict, {})
            Z[i]=numpy.array(list(map(lambda x: self.pattern(x[0], x[1]), zip(target_mat[i], obstacles_mat[i]))))

        CS = plt.contour(Z,30)  #类型matplotlib.contour.QuadContourSet
        plt.close()
        contours = []     #contours是列表

        for collec in CS.collections:#CS.collections   <class 'matplotlib.cbook.silent_list'>
            paths = collec.get_paths()

            if len(paths) != 0:
                path = paths[0]

                contour_pos = list(zip(path.vertices[::, 0], path.vertices[::, 1]))
                contour = Contour(contour_pos)   #带_fitness属性的contour_pos list
                contours.append(contour)

        return contours

    #由self.contours和self.f计算所有等高线的适应度函数，找到适应度最小的等高线
    def find_best_contour(self, i):
        # 确保至少有一条等高线
        assert len(self.contours) != 0, "no contour has been selected."

        for vertices in self.contours:  # vertices是带_fitness属性的contour_pos list
            dpt, dpo= self.generate_dpt_dpo(vertices, i)  ###############robot类的作用
            vertices.contour_fitness = self.f(dpt, dpo, [], in_chennel = i)

        minfit_contour = min(self.contours, key=lambda x: x.contour_fitness)  #找到fitness最小的contour

        return minfit_contour

    #每个等高线的适应度计算
    def f(self, dpt, dpo, A, in_chennel, Nt=1, No=2):  # k:sig的参数

        Np = dpt.shape[1]  # 等高线上采样点的个数
        bottom1 = Np * Nt  # 分母

        dmax = self.settings_scene.dmax
        dmin = self.settings_scene.dmin
        domin = self.settings_scene.domin
        k = self.settings_scene.k

        t1 = 0
        for i in range(Nt):  # 目标个数
            for j in range(Np):  # 采样点个数
                # t1 = t1 + (sigmoid(dpt[i][j], dmax, k[0]) + sigmoid(dmin, dpt[i][j], k[1]) - A[i][j]) / bottom1
                # 更改: fitness 第一部分的惩罚项A去掉了
                t1 = t1 + (sigmoid(dpt[i][j], dmax, k[0]) + sigmoid(dmin, dpt[i][j], k[1])) / bottom1

        t2 = 0
        if in_chennel != 0:
            bottom2 = Np * No
            for i in range(No):
                for j in range(Np):
                    t2 = t2 + sigmoid(domin, dpo[i][j], k[2]) / bottom2
        f = t1 + t2
        return f

    #每个等高线的cv计算
    def get_contour_cv(self, i):

        if  i == 0:
            cv = 0

        if i != 0:
            ###左上 左下 右上 右下
            #obstacles[i] = [(30, 88), (30, 68), (50, 88), (50, 68)]
            left = self.settings_scene.obstacles[i][0][0]
            right = self.settings_scene.obstacles[i][2][0]

            in_num = 0
            for ii in self.minfit_contour:
                if left < ii[0] < right: #在窄道内的点
                    in_num = in_num + 1
            total_num = len(self.minfit_contour)
            cv = 1 - in_num / total_num

        return cv

    def generate_dpt_dpo(self, vertices, i):

        def generate_dpt(pos):
            dpt = numpy.array([[math.sqrt((tx - px) ** 2 + (ty - py) ** 2) for px, py in pos] for tx, ty in self.settings_scene.targets])
            return dpt

        def generate_dpo(pos):#考虑不同窄道的不同计算方式
            #[左上, 左下, 右上, 右下]
            least = self.settings_scene.obstacles[i][0][0]
            most = self.settings_scene.obstacles[i][2][0]

            def condition(p):  # pos中的每个元素
                return least < p[0] < most

            #print(obstacles[i])#[(35.0, 88), (35.0, 68), (45.0, 88), (45.0, 68)]

            dpo = numpy.array([[abs(p[0] - o[0])
                                if condition(p) else -abs(p[0] - o[0]) for p in pos] for o in self.settings_scene.obstacles[i]])
            return dpo

        dpt = generate_dpt(vertices)  # 机器人到目标tar
        dpo = generate_dpo(vertices)  # 机器人到障碍物obs     用等高线来算dpo，dpt

        return dpt, dpo