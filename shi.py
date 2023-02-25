# import time
# import threading
# import random
# from queue import Queue
#
# class Producer(threading.Thread):
#
#     def __init__(self, queue):
#         threading.Thread.__init__(self)
#         self.queue = queue
#
#     def run(self):
#         while True:
#             random_integer = random.randint(0, 100)
#             self.queue.put(random_integer)
#             print('add {}'.format(random_integer))
#             time.sleep(random.random())
#
#
# class Consumer(threading.Thread):
#
#     def __init__(self, queue):
#         threading.Thread.__init__(self)
#         self.queue = queue
#
#     def run(self):
#         while True:
#             get_integer = self.queue.get()
#             print('lose {}'.format(get_integer))
#             time.sleep(random.random())
#
#
# def main():
#     queue = Queue()
#     th1 = Producer(queue)
#     th2 = Consumer(queue)
#     th1.start()
#     th2.start()
#
# if __name__ == '__main__':
#     main()

#practice 1
# def myfunc(l=[]):
#     print(id(l))
#     l.append('add')
#     print(l)
#
# myfunc()
# myfunc()
# myfunc()

#practice 2
# def myfunc(l):
#     l.append(1)
#
# def myfunc1(l):
#     m=l[:]
#     m.append(1)
#
# a = [1, 2, 3]
# myfunc(a)
# print(a)
#
# b = [1, 2, 3]
# myfunc1(b)
# print(b)

#practice 3
# import numpy as np
#
# def myfunc(a):
#     a = a*1
#     a[0,0]=100
#
# a = np.array([[1,1],[1,1]])
# myfunc(a)
# print(a)

#practice 4
####################3####145
# m = [1, 2, [3]]
# n = m[:]#是浅复制？
# n[1] = 4
# n[2][0] = 5  #前者
# #区分：n[2] = [5]  #后者
# #其实根本在于是改变内部的元素（前者，地址不变，所以有关联）还是赋值（后者，新的地址，所以无关联）
# print(m)
#
#对比
# m = [1, 2, [3]]
# n = m#对比
# n[1] = 4
# n[2][0] = 5  #前者
# print(m)

# 对比
# import copy
# m = [1, 2, [3]]
# n = copy.copy(m)#对比
# n[1] = 4
# n[2][0] = 5  #前者
# print(m)

#practice 5
################my_tuple
# my_tuple = (1, 2, [3, 4])
# my_tuple[2] += [5, 6]

#practice 6
# import copy
# a=[['123',[0]],[1,1]]
# b= copy.copy(a)
# c= copy.deepcopy(a)
# print(id(a[0]))
# print(id(b[0]))
# print(id(c[0])) #不一样
#
# #浅复制、深复制不可变对象，地址不变
# print(id(a[0][0]))
# print(id(b[0][0])) #一样
# print(id(c[0][0])) #一样

#practice 7
##########will wilber
# will = ["Will", 28, ["Python", "C#", "JavaScript"]]
# wilber = will
#
# will[0] = "Wilber"
# print(wilber[0])
# #
# a=1
# b=a
# a=2
# print(b)

# print([id(ele) for ele in wilber])

# import copy
# will = ["Will", 28, ["Python", "C#", "JavaScript"]]
# wilber = copy.copy(will)
#
# will[0] = "Wilber"
# print(wilber[0])

#practice 8
##########person team
# person = {'name': '', 'id': 0}
# team = []
#
# for i in range(3):
#     x = person
#     x['id'] = i
#     team.append(x)
#
# # [{'id':2},{'id':2},{'id':2}]
#
# team[0]['name'] = 'Jack'
# team[1]['name'] = 'Pony'
# team[2]['name'] = 'Crossin'
#
# print(team[1])
# print(person)

#practice 9
# a='123'
# b=a
# a+='1'   #a是不可变对象，地址发生变化，b还是指向原来的地址
# print(b)
#
# a=['123']
# b=a    #a和b是同一块内存
# a[0]+='1'
# print(b)
#
# import copy
# a=['123']
# b=copy.copy(a)   #a和b已经不是同一块内存
# a[0]+='1'
# print(b)

#practice 10
# def fun1(m):
#     # m=1
#     return m
#
# print(fun1(3))

# def func(m):
#     m[0] = 20
#     m = [4, 5, 6]
#     return m
#
# l = [1, 2, 3]
# func(l)
# print('l =', l)

# def fun1(m):
#     m[0]=20
#     m=[1,2,3]
#     return m
#
# l=[10]
# print(fun1(l))

#practice 11
# a = [1,2,3]
# def foo(b):
#     b.append(4)
# foo(a)
# print(a)  #  [1,2,3,4]
#
# def bar(c):
#     c = [0,0,0]
# bar(a)
# print(a)  # [1,2,3,4]

# a = [1,2,3]
# def foo(b):
#     print(b is a)
#     b.append(4)
#     print(b is a)
# foo(a)    # 会打印出 True  True
# print(a)  #  [1,2,3,4]
#
# def bar(c):
#     print(c is a)
#     c = [0,0,0]
#     print(c is a)
# bar(a)    # 会打印出 True  False
# print(a)  # [1,2,3,4]

#practice 12
# def func1(b):
#     print(b is a)
#
# a=1
# func1(a)

# m=1
# n=1
# def func2():
#    return m+n
#
# print(func2())

#practice 13
#函数的参数              #类中
#ab实例里面函数的地址相同，函数.theta的地址也相同
# class A:
#     def __init__(self):
#         pass
#
#     def fun(self):
#         pass
#
# A.fun.theta=2   #只可以这样定义，不可以在实例中定义
# print(A.fun.theta)
#
# a=A()
# print(a.fun.theta)
# print(id(a.fun.theta))
#
# b=A()
# print(b.fun.theta)
# print(id(b.fun.theta))
#
# print(id(a.fun))
# print(id(b.fun))

#函数的参数              #一般的函数
# def fun():
#     fun.theta_1=1
#     pass

# fun.theta =2
# print(fun.theta)
# print(id(fun.theta))
# fun()
# print(fun.theta_1)

#practice 14
# set
# mmm=set({1,12,123})
# print(mmm)
# print(mmm[0])


#practice 15
# class A:
#     def __init__(self):
#         self.p1='[1,2,3]'
#
# a=A()
# b=A()
#
# print(id(a.p1))
# print(id(b.p1))
# b.p1='123'
# print(id(b.p1))

#practice 16
#函数是可变对象还是不可变对象
# def fun():
#      pass
#
# print(id(fun))
# fun.theta =2
# print(fun.theta)
# print(id(fun))    #fun.theta =2不会改变fun的地址



'''
1.变量作用域
2.函数也是对象？可变对象还是不可变对象？
3.类与对象
'''

# x=input()
# print(eval(x))


# import numpy
#
# def sigmoid(x, a, k):
#     return 1 / (1 + numpy.exp(-k * (x - a)))
#
# def get_primitive(name):
#     k = 1
#     if name == 'OR':
#         def cor(x1, x2):
#             return sigmoid(x1 + x2, 3/2, k)
#         return cor
#
#     elif name == 'ORN':
#         def corn(x1, x2):
#             return sigmoid(x1 + (1 - x2), 1, k)
#         return corn
#
#     elif name == 'XOR':
#         def cxor(x1, x2):
#             return sigmoid(x1 * (1 - x2), 0, k) + sigmoid((1 - x1) * x2, 0, k)
#         return cxor
#
# def compile():
#     code = 'lambda x1,x2: OR(ORN(x1,x2),XOR(x1,x2))'
#     components_dict = {'OR':get_primitive('OR'), "ORN":get_primitive('ORN'), "XOR":get_primitive('XOR')}
#     return eval(code, components_dict, {})
#
# target_mat = numpy.array([[1,1],[1,1]])
# obstacles_mat = numpy.array([[1,1],[1,1]])
# pattern = compile()
# Z = numpy.zeros([2,2])
# for i in range(len(target_mat)):
#     Z[i] = numpy.array(list(map(lambda x: pattern(x[0], x[1]), zip(target_mat[i], obstacles_mat[i]))))
#
# print(Z)

a=1
import numpy as np
b=np.array([1,1])
print(b)