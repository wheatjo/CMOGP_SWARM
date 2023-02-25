import random
import sys
import copy
from collections import OrderedDict, namedtuple
from deap import gp, base, tools, creator
import time
import re

from csvpictxt import create_csv, create_pic, create_txt, create_graph, rm_file
from cdp import cdp_selNSGA2

from settings import Settings
from scene import Setting_Scene,Contours

#from function_set import get_primitive
import function_set
from xml_component_operate import read_components, count_fs_file_num, reinit_fs_file
from importlib import reload

class Individual(gp.PrimitiveTree):     #gp.PrimitiveTree继承list
    #首先自己是个list，list由genHalfAndHalf来填充
    #然后还有components、de等属性
    def __init__(self, content):
        super().__init__(content)  # 继承父类的__init__方法
        self.theta = None

    def load_settings(self, settings, settings_scene):
        self.settings = settings
        self.settings_scene = settings_scene

    def cal_fitness_cv(self):

        num = len(self.settings_scene.targets)  # 要计算适应度的位置的个数
        sum_fitness = 0
        sum_cv = 0  #是否要cv
        for i in range(num):
            contours = Contours(self, i, self.settings_scene)
            sum_fitness = sum_fitness + contours.minfit_contour.contour_fitness
            sum_cv = sum_cv + contours.minfit_contour.cv

        nodes, edges, lables = gp.graph(self)
        nodes_len = len(nodes)

        self.cv = sum_cv

        return (sum_fitness, nodes_len)

    # 得到self.components
    def read_components(self):
        self.components = OrderedDict()
        for i, prim in enumerate(self):#individual     individual里装的是genHalfAndHalf生成的树
            if isinstance(prim, Component):
                self.components[i] = copy.deepcopy(prim)

    def de(self):
        DE(len(self.components), self.settings).de(self)

    def compile(self): # expr, components, arguments
        expr = self    #expr 即 individual
        arguments = ['x1', 'x2']

        #values都是compo.primitive
        components_list = self.components.values()   #关键词是原始的元件
        components_dict = {}     #关键词是改名后的元件

        #num_name记录结构中每个元件的次数，开始都是0    （为了生成components_dict）
        num_name = {}
        for compo in components_list:
            name = compo.name
            num_name[name] = 0

        #生成components_dict并生成num_name
        for compo in components_list:
            name = compo.name

            # 生成num_name
            num_name[name] = num_name[name] + 1

            # 生成components_dict
            if name in components_dict:
                name = name + (num_name[name]-1) * 'a'
            components_dict[name] = compo.primitive

        self.components_dict_keys = list(components_dict.keys())
        self.components_list_len = len(components_list)
        self.components_dict_len = len(components_dict)

        code = str(expr)   #ANDN(XOR(x2, NOR(NEGATIVE(x1), NEGATIVE(x2))), x2)
        code = self.change_code(code, num_name)

        if len(arguments) > 0:
            args = ",".join(arg for arg in arguments)
            code = "lambda {args}: {code}".format(args=args, code=code)
            print('code:',code)
            # lambda x1,x2: ANDN(XOR(x2, NOR(NEGATIVE(x1), NEGATIVEa(x2))), x2)
            print('code:', components_dict)
        try:
            return eval(code, components_dict, {})
            #self.pattern = individual.compile()
            #self.pattern(x[0], x[1])

        except MemoryError:
            _, _, traceback = sys.exc_info()
            raise MemoryError("DEAP : Error in tree evaluation :"
                              " Python cannot evaluate a tree higher than 90. "
                              "To avoid this problem, you should use bloat control on your "
                              "operators. See the DEAP documentation for more information. "
                              "DEAP will now abort.").with_traceback(traceback)

    # 将theta应用于个体中的元件
    def apply_thetas(self, thetas):   #thetas是theta_population中的个体

        for c, t in zip(self.components.values(), thetas): #想使用其中的对应关系所以用zip

            c.apply_theta(t)   #c.primitive.theta = t  ，function_set那里的theta

        return self

    @staticmethod
    def change_code(code, num_name):
        for name in num_name.keys():  #结构中出现的所有元件
            if num_name[name] != 1 :
                new_code = ''
                code_seg = code.split(name)
                #print(code_seg)
                for i in range(len(code_seg)-1):
                    new_code = new_code + code_seg[i] + name + i * 'a'
                    #print(new_code)
                new_code = new_code + code_seg[len(code_seg)-1]
                #print(new_code) #ANDN(XOR(x2, NOR(NEGATIVE(x1), NEGATIVEa(x2))), x2)
                code = new_code
        return code

class Component(gp.Primitive):

    # class Terminal是gp.py中的Terminal   ##这啥意思？？

    def __init__(self, in_types, ret_type, name):
        self.primitive = function_set.get_primitive(name)
        self.apply_theta(0)  #Component.primitive.theta = 0
        self.name = name

        self.in_types = in_types
        self.ret_type = ret_type
        super().__init__(self.name, self.in_types, self.ret_type)
        #               (self, name, args, ret)
        #self.name = name
        #self.args = self.in_types
        #self.ret = self.ret_type
        #self.arity = len(self.in_types)

    def apply_theta(self, theta):
        self.primitive.theta = theta



#对某一个参数，其fitness是 把参数放到结构上后计算得到的fitness ，然后通过hof选出该结构当前最佳的参数。
#怎么选：通过fitness权重的(-1.0,-1.0)，知道要选的是最小值
#然后再把选出的参数装到结构上（apply_theta），得到结构当前的fitness

#装的过程：apply thetas函数，即设置个体结构对应的多个函数的.theta值

#所以这里的fitness是用在hof处的，并不是没用。hof源码有说明对fitness的使用
creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0))
creator.create("ThetaIndividual", list, fitness=creator.FitnessMin)

#############参数theta的类：参数初始化和参数de进化，被individual类的self.de = DE(len(self.components))调用
class DE:

    def __init__( self, NDIM, settings):
        self.NDIM = NDIM
        self.CR = settings.THETA_CX_RATE  # DE交叉概率
        self.F = settings.THETA_F  # DE的变异率
        self.TGEN = settings.THETA_TOTAL_GEN
        self.settings = settings
        pass


    def init_population(self, individual):#初始化theta种群
        self.toolbox = base.Toolbox()
        self.toolbox.register("random_theta", random.uniform, self.settings.THETA_RANGE[0], self.settings.THETA_RANGE[1])
        self.toolbox.register("theta_individual", tools.initRepeat, creator.ThetaIndividual, self.toolbox.random_theta,
                         self.NDIM)  # 元件的个数即为theta的个数
        self.toolbox.register("theta_population", tools.initRepeat, list, self.toolbox.theta_individual)  # 生成种群

        self.toolbox.register("theta_select", tools.selRandom, k=3)  # 随机选择3个theta
        self.toolbox.register("theta_evaluate", individual.cal_fitness_cv)  #

        print(3*'==','theta 随机生成',3*'==')
        self.population = self.toolbox.theta_population(self.settings.pop_size)
        self.hof = tools.HallOfFame(self.settings.hof_size)  #1

        for i, theta_ind in enumerate(self.population):
            #self.population是一个list，theta_ind是里面的一个个体，也是list

            individual.apply_thetas(theta_ind)
            theta_ind.fitness.values = self.toolbox.theta_evaluate()
            print('第', i+1, '个参数：', theta_ind)
            print('fitness: ', theta_ind.fitness.values[0])


        self.hof.update(self.population)
        self.hof.update(self.population)
        individual.apply_thetas(self.hof[0])
        individual.fitness.values = self.hof[0].fitness.values
        individual.theta = self.hof[0]


    def de(self, individual):#这个individual是外层的individual
        #参数随机生成
        self.init_population(individual)

        #参数差分进化
        print(3*'==','theta DE优化',3*'==')
        for g in range(1, self.TGEN+1):
            print("DE优化次数:", g)

            for k, theta_ind in enumerate(self.population):#population是参数种群，一个元素是一套参数
                y = self.toolbox.clone(theta_ind)   #按顺序克隆的个体
                a, b, c = self.toolbox.theta_select(self.population)  # 在population中随机选择3个参数

                for i, value in enumerate(theta_ind):
                    if random.random() < self.CR:
                        y[i] = a[i] + self.F * (b[i] - c[i])
                        if y[i] < 0:
                            y[i] = 0
                        if y[i] > 2:
                            y[i] = 2

                individual.apply_thetas(y)
                print('individual type:',type(individual))
                y.fitness.values = self.toolbox.theta_evaluate()

                if y.fitness.values[0] < theta_ind.fitness.values[0]:
                    self.population[k] = y

                    #这一句是否多余？
                    self.population[k].fitness.values = y.fitness.values

                print('第',k+1,'个参数个体', self.population[k])
                print('fitness:', self.population[k].fitness.values[0])

        self.hof.update(self.population)
        print(2*'==', '结构最优theta', 2*'==')
        print(self.hof[0])
        print('fitness:', self.hof[0].fitness.values[0])

        #将最优theta应用于个体中的元件
        individual.apply_thetas(self.hof[0])
        individual.fitness.values = self.hof[0].fitness.values
        individual.theta = self.hof[0]


#########重写generate
def copy_generate(min_, max_, condition, type_, component_name_list, one_variable_list):

    expr = []
    height = random.randint(min_, max_)
    stack = [(0, type_)]
    while len(stack) != 0:
        depth, type_ = stack.pop()
        if condition(height, depth):
            term = random.choice([gp.Terminal('x1', True, float), gp.Terminal('x2', True, float)])

            expr.append(term)
        else:

            name = random.choice(component_name_list)

            if name in one_variable_list:
                prim = Component([float, ], float, name)
            else:
                prim = Component([float, float], float, name)

            expr.append(prim)
            for arg in reversed(prim.args):
                stack.append((depth + 1, arg))

    return expr


def genFull(min_, max_, type_, component_name_list, one_variable_list):

    def condition(height, depth):
        """当深度等于高度时，表达式生成将停止."""
        return depth == height

    return copy_generate(min_, max_, condition, type_, component_name_list, one_variable_list)

def genGrow(min_, max_, type_, component_name_list, one_variable_list):

    def condition(height, depth):
        terminalRatio = 2 / ( 2  + 10 )
        return depth == height or \
               (depth >= min_ and random.random() < terminalRatio)

    return copy_generate(min_, max_, condition, type_, component_name_list, one_variable_list)

#czhczhczhczhczhczhczhczhczhczhczhczhczhczhczhczhczhczhczhczhczhczhczh这些也应该放在元件连接部分吧
def genHalfAndHalf(min_, max_, type_=float, component_name_list=None, one_variable_list=None):
    '''
    :param type_: pset.ret
    :returns: Either, a full or a grown tree.
    '''

    method = random.choice((genGrow, genFull))
    return method(min_, max_, type_, component_name_list, one_variable_list)

def mutUniform(individual, expr):  #expr = genHalfAndHalf

    index = random.randint(0, len(individual)-1)
    slice_ = individual.searchSubtree(index)

    type_ = individual[index].ret     #Component的return：float

    #print(sys.getrefcount(individual[slice_][0]))
    #print(123)
    #for i in range(len(individual[slice_])):
    #    print(sys.getrefcount(individual[slice_][i]))

    individual[slice_] = expr(type_=type_)  #生成新子树, 其theta都为0
    #在计算invalids适应值时individual._initIndividual()

def cxOnePoint(ind1, ind2):

    index1 = random.randint(0, len(ind1)-1)
    index2 = random.randint(0, len(ind2)-1)

    slice1 = ind1.searchSubtree(index1)
    slice2 = ind2.searchSubtree(index2)

    ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]
    #在计算invalids适应值时ind1._initIndividual()和ind2._initIndividual()


# 检测个体的树是否取到了x1和x2
def check_get_all_inputs(ind_str):#传入str
    x1_mat = re.search(r"x1", ind_str)#如果不出现则返回一个NoneType的对象
    x2_mat = re.search(r"x2", ind_str)#
    return x1_mat and x2_mat  #True：两个同时为True才是True


# 删除掉pop中没有同时取到x1和x2的树
def remove_unsatisfied_tree(population):
    population_copy = copy.deepcopy(population)
    for i, ind in enumerate(population_copy):
        if not check_get_all_inputs(str(ind)):  # 如果没有同时取到x1 和 x2
            population.remove(ind)

def evolve_main():
    lines_num = count_fs_file_num()  #function_set.py一开始的行数
    component_name_list, one_variable_list = read_components()
    print(component_name_list)
    print(one_variable_list)
    reload(function_set)
    reinit_fs_file(lines_num)    #恢复function_set.py一开始的行数

    #这两个类都可以改成类似   类的静态函数   那样
    settings = Settings()
    settings_scene = Setting_Scene(settings)

    # fitness的权重装到个体ind上，在场景中更新ind.fitness.values，然后用ind.fitness.values作为后续的进化的依据
    creator.create("Fitness", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", Individual, fitness=creator.Fitness)

    toolbox = base.Toolbox()
    toolbox.register('main_expr', genHalfAndHalf, min_=settings.INIT_DEPTH_MIN,
                     max_=settings.INIT_DEPTH_MAX, component_name_list=component_name_list, one_variable_list = one_variable_list)  # 采用自定义genHalfAndHalf

    toolbox.register('generateIndividual', tools.initIterate,
                     creator.Individual,
                     toolbox.main_expr)  # 后一个的内容 迭代填充（fill）前一个

    toolbox.register('population', tools.initRepeat, list, toolbox.generateIndividual)  # 后一个的内容被调用n次来 重复填充（fill）前一个
    toolbox.register('select', tools.selTournament, tournsize=3)
    toolbox.register('mate', cxOnePoint)  # 单点交叉
    toolbox.register('mutate', mutUniform, expr=toolbox.main_expr)  # uniform变异

    start = time.time()
    print("started.")
    random.seed(1024)

    print(20*'==','iteration0',20*'==')
    print('随机产生树结构，删除、维持数量......')
    pop = toolbox.population(n=settings.POP_SIZE)


    # 删除掉pop中没有同时取到x1和x2的树
    remove_unsatisfied_tree(pop)
    # 维持种群中个体数量POP_SIZE恒定
    while len(pop) < settings.POP_SIZE:
        pop_extended = toolbox.population(n=1)
        if check_get_all_inputs(str(pop_extended[0])):  # 如果同时取到了x1 和 x2
            pop.extend(pop_extended)

    print(8 * '==','产生的所有结构是：',8 * '==')
    for ind in pop:
        print(ind)

    print(5 * '==','每个结构的theta随机生成和DE优化：',5 * '==')
    for i,ind in enumerate(pop):
        print(15*'~','结构'+str(i+1),15*'~')
        ind.load_settings(settings, settings_scene)
        ind.read_components() #生成个体（GRN结构）的de属性和components属性
        ind.de() #更新ind的value并得到cv

    rm_file()
    create_csv(pop, 0)  # 保存每代所有的结构及结构的最优的参数
    create_pic(pop, 0)
    create_txt(pop, 0)
    create_graph(pop, 0)
    print("create csv, pic, txt, gragh")

    CXPB = settings.CX_PB
    MUTPB = settings.MUT_PB
    NGEN = settings.NUM_GEN
    for g in range(1, NGEN + 1):
        print(20 * '==', 'iteration' +str(g), 20 * '==')
        print('选择上一代树结构，交叉、变异、删除、维持数量......')

        #offspring在前一代pop的基础上生成，再进行交叉变异删除维持等操作
        offspring = toolbox.select(pop, len(pop)) #在pop里选全部即把全部拿出来,作为offspring
        offspring = [toolbox.clone(ind) for ind in offspring]

        #交叉
        for ind1, ind2 in zip(offspring[0::2], offspring[1::2]):#从0开始间隔，从1开始间隔
            if random.random() < CXPB:
                toolbox.mate(ind1, ind2)  #在offspring中更新ind1和ind2
                del ind1.fitness.values
                del ind2.fitness.values

        #变异
        for ind in offspring:
            # for tree, pset in zip(ind, pset):
            if random.random() < MUTPB:
                toolbox.mutate(individual=ind)  #在offspring中更新ind1和ind2
                del ind.fitness.values

        #删除掉offspring中没有同时取到x1和x2的树
        remove_unsatisfied_tree(offspring)

        #维持种群中个体数量POP_SIZE恒定
        while len(offspring) < settings.POP_SIZE:
            offspring_extended = toolbox.population(n=1)
            if check_get_all_inputs(str(offspring_extended[0])):  # 如果同时取到了x1 和 x2
                offspring.extend(offspring_extended)
            print('offspring_extended:',offspring_extended)

        # Evaluate the individuals with an invalid fitness
        invalids = [ind for ind in offspring if not ind.fitness.valid]
        print(7 * '==', '需要计算fitness的结构', 7 * '==')
        for ind in invalids:
            print(ind)
        print(5 * '==', '每个结构的theta随机生成和DE优化', 5 * '==')
        for i, ind in enumerate(invalids):#offspring中发生了变化的个体
            print(15*'~','结构'+str(i+1),15*'~')
            ind.load_settings(settings, settings_scene)
            ind.read_components()
            ind.de() #更新ind的value

        # pop是上一代的population，offspring是从上一代的population交叉变异去掉补充得到的
        pop.extend(invalids)
        pop = cdp_selNSGA2(pop, settings.POP_SIZE)
        # pop = tools.selNSGA2(pop, settings.POP_SIZE)
        print(8*'==','iteration' +str(g)+'中最好的个体',8*'==')
        for ind in pop:
            print(ind)

        create_csv(pop, g)  #保存每代所有的结构及结构的最优的参数
        create_pic(pop, g)
        create_txt(pop, g)   #保存每代的最佳fitness和平均fitness
        create_graph(pop,g)
        print("create csv, pic, txt, gragh")

    elapsed = (time.time() - start)
    print("Time used:", elapsed)

    return pop  # stats,