import matplotlib.pyplot as plt
import numpy as np
import time

"""
绘制fitness收敛趋势图
所用到的数据: 代数, 每一代的当代最低(最好的)fitness(注意) 和每一代的平均fitness
数据存于log_generations.txt当中

e.g.
1 3.7323432 4.212132
2 3.7132322 4.123433
3 3.6745345 4.032313
"""


def plot_convergence():
    gen_li = list()
    best_fitness_li = list()
    avg_fitness_li = list()

    try:
        with open("log_generations.txt", "r") as f:
            for line in f.readlines():
                col1, col2, col3 = line.split(" ")
                col3 = col3.strip()  # 删掉换行符
                gen_li.append(float(col1))
                best_fitness_li.append(float(col2))
                avg_fitness_li.append(float(col3))
    except:
        print("ERROR: Make sure the file exist, and the ")


    # 开始画图
    plt.title('Fitness Convergence Graph')
    plt.ylabel("Fitness")
    plt.xlabel("Number of generation")
    plt.plot(gen_li, avg_fitness_li, 'g-s', label='Average Fitness')
    plt.plot(gen_li, best_fitness_li, 'b-o', label='Best Fitness')
    plt.legend(loc='best')
    plt.show()

    time.sleep(2)

    plt.close()


if __name__ == '__main__':
    plot_convergence()





