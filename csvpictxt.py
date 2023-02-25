import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import graphviz
from deap import gp
import shutil

OUTPUT_FILE_DIR=os.path.join(os.path.dirname(__file__), 'output')

#个体信息 -> 'output/individual_info.csv'
def create_csv(population, NGEN):
    FILE_NAME = os.path.join(OUTPUT_FILE_DIR, 'individual_info.csv')

    individual_info_dict = {'NGEN': [],  'NO': [], 'ind': [], 'fitness': [], 'theta': []}

    for i,ind in enumerate(population):
        individual_info_dict['NGEN'].append(NGEN)
        individual_info_dict['NO'].append(i)
        individual_info_dict['ind'].append(str(ind))

        fitness = (ind.fitness.values[0], int(ind.fitness.values[1]))
        individual_info_dict['fitness'].append(fitness)

        theta = ind.theta
        individual_info_dict['theta'].append(str(theta))

    data_fame = pd.DataFrame(individual_info_dict)
    if NGEN == 0:
        data_fame.to_csv(FILE_NAME, index=False)
    else:
        data_fame.to_csv(FILE_NAME, mode='a', header=False, index=False)

#Pareto前沿 -> 'output/fitnessnode_pic'
def create_pic(pop,g):

    f = [ind.fitness.values[0] for ind in pop]
    n = [ind.fitness.values[1] for ind in pop]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(n, f)  # y轴fitness
    plt.xlabel('nodes')
    plt.ylabel('fitness')
    plt.title('iteration:' + str(g))
    plt.savefig(os.path.join(OUTPUT_FILE_DIR, 'fitnessnode_pic', str(g) + '.jpg'))

    plt.close(fig)


# 计算当代的平均fitness, 添加到'output/log_generations.txt'
def create_txt(pop, NGEN):
    FILE_NAME = os.path.join(OUTPUT_FILE_DIR, 'generation_average_fitness.txt')
    avg_fitness = tuple(sum([np.array(ind.fitness.values) for ind in pop])/len(pop))
                 #tuple(sum([np.array((1, 2)), np.array((1, 2)), np.array((1, 2))]) / 3)

    # 每代总览信息, 添加到log文件
    with open(FILE_NAME, 'a') as f:

        generation_info = str(NGEN) + " "  + str(avg_fitness) + "\n"
        f.write(generation_info)

def create_graph(pop, g):

    dir = os.path.join(OUTPUT_FILE_DIR, 'best ind', "GEN" + str(g))
    os.makedirs(dir)
    for i in range(len(pop)):
        expr = pop[i]   #个体

        #个体的信息
        nodes, edges, labels = gp.graph(expr)
        s_nodes = list(map(str, nodes))
        s_edges = [tuple(map(str, t)) for t in edges]
        s_labels = list(labels.values())
        dot = graphviz.Digraph(comment='NGEN')
        dot.format = 'png'
        for n, l in zip(s_nodes, s_labels):
            dot.node(n, l)
        dot.edges(s_edges)

        #画树状图
        dot.render(dir + "/" + "ind" + str(i))   #这个是画图？？


def delete_file(file_name):
    if os.path.exists(file_name):
        os.remove(file_name)

#删掉文件夹
def rm_file():
    #删除已有的文件
    delete_file(os.path.join(OUTPUT_FILE_DIR, 'generation_average_fitness.txt'))
    delete_file(os.path.join(OUTPUT_FILE_DIR, 'individual_info.csv'))

    #删除已有文件夹并创建新文件夹
    shutil.rmtree(os.path.join(OUTPUT_FILE_DIR, 'best ind'))
    shutil.rmtree(os.path.join(OUTPUT_FILE_DIR, 'fitnessnode_pic'))
    os.mkdir(os.path.join(OUTPUT_FILE_DIR, 'best ind'))
    os.mkdir(os.path.join(OUTPUT_FILE_DIR, 'fitnessnode_pic'))
