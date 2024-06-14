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
        #TODO(1) level 1 : instead of a single function, we can use a list of function pointers (one function for each layer)
        self.__activation_func = self.sigmoid
        self.__activation_prime = self.sigmoid_prime
        
        self.__cost_derivative = self.quadratic_cost_derivative

    #given the input a, return the output of the network (for now, using sigmoid only)
    def feedforward(self, a):
        #feedforward the input a through the network
        for b, w in zip(self.__biases, self.__weights):
            #TODO(1)
            a = self.__activation_func(np.dot(w, a) + b)
        return a


    #TODO(1)
    def backpropagation(self, x, y):
        #initialize the lists for the gradients of the cost function
        #with respect to the biases and weights
        nabla_b = [np.zeros(b.shape) for b in self.__biases]
        nabla_w = [np.zeros(w.shape) for w in self.__weights]

        #feedforward
        #TODO(1)
        activation = x
        #list to store all the activations, layer by layer
        activations = [x]
        #list to store all the z vectors, layer by layer
        zs = []
        for b, w in zip(self.__biases, self.__weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            #TODO(1)
            activation = self.__activation_func(z)
            activations.append(activation)

        #backward pass
        #calculate the error in the output layer
        delta = self.__cost_derivative(activations[-1], y) * self.__activation_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        #propagate the error to the previous layers
        for l in range(2, self.__num_layers):
            z = zs[-l]
            #TODO(1)
            ap = self.__activation_prime(z)
            delta = np.dot(self.__weights[-l + 1].transpose(), delta) * ap
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return nabla_b, nabla_w



    #looks scary, but the only thing it does is setting the activation function
    #input: string with the name of the activation function
    def __set_activation_function(self, func):

        #dictionary (of tuples) with all activation functions and their derivatives
        activation_functions = {
            'relu': (self.relu, self.relu_prime),
            'sigmoid': (self.sigmoid, self.sigmoid_prime),
            'tanh': (self.tanh, self.tanh_prime),
            'softmax': (self.softmax, self.softmax_prime),
            'linear': (self.linear, self.linear_prime)
        }
        
        #if the function is in the dictionary, set the activation function and its derivative
        if func in activation_functions:
            self.__activation_func, self.__activation_prime = activation_functions[func]
        else:
            raise ValueError('Activation function not recognized')


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
    


    #loss function derivatives

    #--quadratic cost--
    def quadratic_cost_derivative(self, output_activations, y):
        return (output_activations - y)
    
    #--huber loss--
    def huber_loss_derivative(self, output_activations, y):
        return np.where(np.abs(output_activations - y) <= 1, output_activations - y, np.sign(output_activations - y))
