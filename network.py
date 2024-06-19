import random
import numpy as np

#TODO(1) level 1 : instead of a single function, we can use a list of function pointers (one function for each layer)

class Network:

    def __init__(self, sizes, activation_func_name = "sigmoid"):
        #number of layers in the network
        self.__num_layers = len(sizes)
        #number of neurons in each layer
        self.__sizes = sizes
        self.__biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.__weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

        #activation function and its derivative for all layers exept output layer (for now)
        #TODO(1)
        self.__init_activation_func_for_all_layers(activation_func_name)
        self.__cost_derivative = self.quadratic_cost_derivative

    def __init_activation_func_for_all_layers(self, activation_func_name):
        match activation_func_name:
            case "relu":
                self.__activation_func = Activation_Functions.relu
                self.__activation_prime = Activation_Functions.relu_derivative
            case "sigmoid":
                self.__activation_func = Activation_Functions.sigmoid
                self.__activation_prime = Activation_Functions.sigmoid_derivative
            case "tanh":
                self.__activation_func = Activation_Functions.tanh
                self.__activation_prime = Activation_Functions.tanh_derivative
            case "softmax":
                self.__activation_func = Activation_Functions.softmax
                self.__activation_prime = Activation_Functions.softmax_derivative
            case "linear":
                self.__activation_func = Activation_Functions.linear
                self.__activation_prime = Activation_Functions.linear_derivative
            case other:
                raise ValueError(f"{self.__kernel} does not exist")

    #given the input a, return the output of the network (for now, using sigmoid only)
    def __feedforward(self, a):
        #feedforward the input a through the network
        for b, w in zip(self.__biases, self.__weights):
            #TODO(1)
            a = self.__activation_func(np.dot(w, a) + b)
        return a

    def train(self, training_data,  mini_batch_size, learningRate = 0.1, epochs = 1000):

        #for each epoch
        for j in range(epochs):
            #shuffle the training data
            random.shuffle(training_data)
            #divide the training data into mini batches
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]
            #for each mini batch
            for mini_batch in mini_batches:
                #update the weights and biases using the gradients of the cost function
                self.update_mini_batch(mini_batch, learningRate)

    #input: eta - learning rate
    def update_mini_batch(self, mini_batch, eta):
        #initialize the lists for the gradients of the cost function
        #with respect to the biases and weights
        nabla_b = [np.zeros(b.shape) for b in self.__biases]
        nabla_w = [np.zeros(w.shape) for w in self.__weights]

        #for each training example x, y in the mini_batch, accumulate the gradients
        for x, y in mini_batch:
            #compute the gradients of the cost function using backpropagation
            delta_nabla_b, delta_nabla_w = self.__backpropagation(x, y)
            #accumulate the gradients of the batch
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        #update the weights and biases using the gradients and the learning rate eta
        self.__weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.__weights, nabla_w)]
        self.__biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.__biases, nabla_b)]

    #TODO(1)
    def __backpropagation(self, x, y):
        #initialize the lists for the gradients of the cost function
        #with respect to the biases and weights
        nabla_b = [np.zeros(b.shape) for b in self.__biases]
        nabla_w = [np.zeros(w.shape) for w in self.__weights]

        #feedforward
        #TODO(1)
        activation = x
        #list to store all the activations, layer by layer
        #TODO(1)
        activations = [x]
        #list to store all the z vectors, layer by layer
        zs = []
        #feedforward the input x through the network and store the activations and z vectors
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

    #TODO (1)
    def predict(self, x):
        return np.argmax(self.__feedforward(x))


    #input: test_data - list of tuples (x, y) where x is the input and y is the expected output
    #output: list of tuples (prediction, expected) for each input in the test_data
    #TODO (1)
    def __predict_batch(self, test_data):
        #get the number of correct predictions
        test_results = [(np.argmax(self.__feedforward(x)), np.argmax(y)) for (x, y) in test_data]
        return test_results


    #input: test_data - list of tuples (x, y) where x is the input and y is the expected output
    #output: the accuracy of the network on the test_data
    def score(self, test_data):
        return sum(int(x == y) for (x, y) in self.__predict_batch(test_data)) / len(test_data)
    #setters

    #input: string with the name of the activation function
    def __set_cost_function(self, cost_function):

        cost_functions = {
            'quadratic': self.quadratic_cost_derivative,
            'huber': self.huber_loss_derivative
        }

        if cost_function in cost_functions:
            self.__cost_derivative = cost_functions[cost_function]
        else:
            raise ValueError('Cost function not recognized')


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


    #loss function derivatives

    #--quadratic cost--
    def quadratic_cost_derivative(self, output_activations, y):
        return (output_activations - y)
    
    #--huber loss--
    def huber_loss_derivative(self, output_activations, y):
        return np.where(np.abs(output_activations - y) <= 1, output_activations - y, np.sign(output_activations - y))
