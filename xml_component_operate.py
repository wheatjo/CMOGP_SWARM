#操作function_set.py文件

import xml.dom.minidom # 加载模块
import os
def read_components():

    dom = xml.dom.minidom.parse(os.path.join(os.getcwd(), 'config_swarm.xml')) # 加载和读取xml文件
    root = dom.documentElement # 创建根节点。每次都要用DOM对象来创建任何节点
    component = root.getElementsByTagName('components')[0].getElementsByTagName('comp') #The function getElementsByTagName returns NodeList.
    component_num = len(component) # 获得基本函数个数

    component_name_list = []
    one_variable_list = []

    for i in range(component_num):
        name = component[i].getElementsByTagName('name')[0].childNodes[0].nodeValue
        function_name = 'c' + name.lower()

        component_name_list.append(name)

        variable = component[i].getElementsByTagName('input')[0].childNodes[0].nodeValue

        content = component[i].getElementsByTagName('function')[0].childNodes[0].nodeValue
        content = content.replace('theta', function_name+'.theta')

        #写入
        with open('function_set.py', 'a') as f:
            f.write('\n')
            if i==0:
                f.write(4*' ' + 'if name == ' + '\''+name + '\'' + ':' + '\n')
            else:
                f.write(4 * ' ' + 'elif name == ' + '\''+name + '\'' + ':' + '\n')

            #这里要判断x1和x2是否都有
            if 'x1' in variable and 'x2' in variable:
                f.write(2*4*' '+'def ' + function_name + '(x1, x2):' + '\n')
            else:
                f.write(2 * 4 * ' ' + 'def ' + function_name + '(x):' + '\n')
                one_variable_list.append(name)

            f.write(3*4*' '+'return '+ content + '\n')
            f.write(2*4*' ' + 'return ' + function_name + '\n')

    return component_name_list, one_variable_list

def count_fs_file_num():
    # 用with open(....) as ... 来读取一个文件，这里“r”是指open(...)括号里的内容
    with open('function_set.py', 'r') as r:
        lines = r.readlines() # readlines()方法是指一次性读取整个文件；自动将文件分析成一个行的列表
    return len(lines) # 返回该文件一共多少行

def reinit_fs_file(lines_num):
    with open('function_set.py', 'r') as r:
        lines = r.readlines()
    with open('function_set.py','w') as w:
        for l in lines[:lines_num]:
            w.write(l)

if __name__ == '__main__':
    #a=haha()
    reinit_fs_file()

