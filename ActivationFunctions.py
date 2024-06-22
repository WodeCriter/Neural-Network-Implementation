import numpy as np


class ActivationFunctions:
    def __init__(self):
        # dictionary (of tuples) with all activation functions and their derivatives
        self.__activation_functions = {
            'relu': (ActivationFunctions.relu, ActivationFunctions.relu_prime),
            'sigmoid': (ActivationFunctions.sigmoid, ActivationFunctions.sigmoid_prime),
            'tanh': (ActivationFunctions.tanh, ActivationFunctions.tanh_prime),
            'softmax': (ActivationFunctions.softmax, ActivationFunctions.softmax_prime),
            'linear': (ActivationFunctions.linear, ActivationFunctions.linear_prime),
            'argmax': (np.argmax, 1)
        }

    @property
    def Activation_functions(self):
        return self.__activation_functions

    @staticmethod
    def get_activation_function_by_name(func_name: str):
        activation_function_generator = ActivationFunctions()
        # if the function is in the dictionary, set the activation function and its derivative
        if func_name in activation_function_generator.Activation_functions:
            return activation_function_generator.Activation_functions[func_name]
        else:
            raise ValueError(f'Activation function {func_name} not recognized')

    @staticmethod
    def relu(z):
        return np.maximum(z, 0)
    @staticmethod
    def relu_prime(z):
        return np.where(z > 0, 1, 0)

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))
    @staticmethod
    def sigmoid_prime(z):
        return ActivationFunctions.sigmoid(z) * (1 - ActivationFunctions.sigmoid(z))

    @staticmethod
    def tanh(z):
        return np.tanh(z)
    @staticmethod
    def tanh_prime(z):
        return 1 - np.tanh(z) ** 2

    @staticmethod
    def softmax(z):
        return np.exp(z) / np.sum(np.exp(z), axis=0)
    @staticmethod
    def softmax_prime(z):
        return ActivationFunctions.softmax(z) * (1 - ActivationFunctions.softmax(z))

    @staticmethod
    def linear(z):
        return z
    @staticmethod
    def linear_prime(z):
        return 1