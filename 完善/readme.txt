2019.08.24
add the statement “plt.close”after plt，Solve the time accumulation problem

2019.08.25
增加my_graph.py，提供创建gif的接口：gif_creat(hof_set)函数


input_data 文件夹放目标浓度 障碍物浓度
settings 调出参数供调节

### 此处省略一堆更新 ###

requirements中的PIL改为python 3.X 版本通用的Pillow


2019年9月5日
point_to_line:点到线段的版本。更正了dpo的计算方法。
nsga-2

2019-09-05 Integrating version
初始化种群后, 检查并移除只用到x1, x2之一的个体
填充同时有x1,x2个体至指定种群数量
交叉变异后同样检查是否x1, x2都取到;

- save_fitness_log将每代当代最好fitness和平均fitness存入txt文件

- output_data2CVS.py新增个体theta存储, 格式为 元件名+theta值; 元件名+theta值; ...;

- 运行plot_fitnessConvergenceGraph.py画出每代当代最好fitness和当代fitness均值的折线图, 数据来源为自动生成的log_generations.txt

- 运行Refreshing.py删除程序每次运行所生成的相关png, gif图像, 及每代average fitness, best fitness