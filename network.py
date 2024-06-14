import random
import numpy as np



class Network:

    def __init__(self, sizes):
        self.__num_layers = len(sizes)
        self.__sizes = sizes
        self.__biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.__weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]