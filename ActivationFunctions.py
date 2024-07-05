import numpy as np


class ActivationFunctions:
    def __init__(self):
        # dictionary (of tuples) with all activation functions and their derivatives
        self.__activation_functions = {
            'relu': (ActivationFunctions.relu, ActivationFunctions.relu_prime),
            'leaky_relu': (ActivationFunctions.leaky_relu, ActivationFunctions.leaky_relu_prime),
            'sigmoid': (ActivationFunctions.sigmoid, ActivationFunctions.sigmoid_prime),
            'tanh': (ActivationFunctions.tanh, ActivationFunctions.tanh_prime),
            'softmax': (ActivationFunctions.softmax, ActivationFunctions.softmax_prime),
            'linear': (ActivationFunctions.linear, ActivationFunctions.linear_prime),
            'argmax': (np.argmax, 1)
        }

    @property
    #activation functions property
    def Activation_functions(self):
        return self.__activation_functions

    #Note: A more descriptive name might be get_cost_derivative_by_name
    @staticmethod
    def get_activation_function_by_name(func_name: str):
        activation_function_generator = ActivationFunctions()
        # if the function is in the dictionary, set the activation function and its derivative
        if func_name in activation_function_generator.Activation_functions:
            return activation_function_generator.Activation_functions[func_name]
        else:
            raise ValueError(f'Activation function {func_name} not recognized')

    @staticmethod
    # Rectified Linear Unit (ReLU) activation function
    def relu(z):
        return np.maximum(z, 0)
    
    @staticmethod
    # Derivative of ReLU activation function
    def relu_prime(z):
        return np.where(z > 0, 1, 0)
    
    @staticmethod
    # Leaky ReLU activation function (small slope for z < 0)
    def leaky_relu(z):
        return np.where(z > 0, z, 0.01 * z)
    
    @staticmethod
    # Derivative of Leaky ReLU activation function
    def leaky_relu_prime(z):
        return np.where(z > 0, 1, 0.01)

    @staticmethod
    # Sigmoid activation function
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))
    
    @staticmethod
    # Derivative of Sigmoid activation function
    def sigmoid_prime(z):
        return ActivationFunctions.sigmoid(z) * (1 - ActivationFunctions.sigmoid(z))

    @staticmethod
    # Hyperbolic Tangent (tanh) activation function
    def tanh(z):
        return np.tanh(z)
    
    @staticmethod
    # Derivative of Hyperbolic Tangent (tanh) activation function
    def tanh_prime(z):
        return 1 - np.tanh(z) ** 2

    @staticmethod
    # Softmax activation function for multi-class classification output layer (stabliized version)
    def softmax(z):
        # Subtracting the maximum value from z to avoid numerical instability
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)
   
    @staticmethod
    # Derivative of Softmax activation function
    def softmax_prime(z):
        softmax_z = ActivationFunctions.softmax(z)
        return softmax_z * (1 - softmax_z)

    @staticmethod
    # Linear activation function
    def linear(z):
        return z
    
    @staticmethod
    # Derivative of Linear activation function
    def linear_prime(z):
        return 1