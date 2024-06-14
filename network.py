import random
import numpy as np



class Network:

    def __init__(self, sizes):
        #number of layers in the network
        self.__num_layers = len(sizes)
        #number of neurons in each layer
        self.__sizes = sizes
        #initialize biases for each layer
        self.__biases = [np.random.randn(y, 1) for y in sizes[1:]]
        #initialize weights for each layer
        self.__weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

        #activation function and its derivative for all layers exept output layer (for now)
        self.__activation_func = self.sigmoid
        self.__activation_prime = self.sigmoid_prime



    # activation functions and their derivatives

    #--relu--
    def relu(self, z):
        return np.maximum(z, 0)
    
    def relu_prime(self, z):
        return np.where(z > 0, 1, 0)
    
    #--sigmoid--
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))
    
    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    #--tanh--
    def tanh(self, z):
        return np.tanh(z)
    
    def tanh_prime(self, z):
        return 1 - np.tanh(z)**2
    
    #--softmax--
    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=0)
    
    def softmax_prime(self, z):
        return self.softmax(z) * (1 - self.softmax(z))
    
    #--linear--
    def linear(self, z):
        return z
    
    def linear_prime(self, z):
        return 1