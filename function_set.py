import numpy

def sigmoid(x, a, k):
    return 1 / (1 + numpy.exp(-k * (x - a)))

def get_primitive(name):
    k = 1
