import random
import numpy as np

#TODO(1) level 1 : instead of a single function, we can use a list of function pointers (one function for each layer)
#TODO 1 change the feedforward function to use the list of activation functions and so on
class Network:

    def __init__(self, sizes , activations = None , cost_function = 'quadratic' , output_activation = 'softmax'):
        #number of layers in the network
        self.__num_layers = len(sizes)
        #number of neurons in each layer
        self.__sizes = sizes
        #initialize biases for each layer
        self.__biases = [np.random.randn(y, 1) for y in sizes[1:]]
        #initialize weights for each layer
        self.__weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        #stores a activation function name for each layer
        self.activation_function_list = None

        #activation function and its derivative for all layers exept output layer (for now)
        #TODO(1)
        self.__activation_func , self.__activation_prime  = self.___create_activation_function_lists(activations , self.__num_layers , output_activation)
        
        
        self.__cost_derivative = self.quadratic_cost_derivative

    #given the input a, return the output of the network (for now, using sigmoid only)
    def __feedforward(self, a):
        i = 0
        #feedforward the input a through the network
        for b, w in zip(self.__biases, self.__weights):
            #TODO(1)
            a = self.__activation_func[i](np.dot(w, a) + b)
            i += 1
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
            #DEBUG
            epoch_completion = 100 * (j + 1) / epochs
            print(f"Epoch {j+1}/{epochs} complete: {epoch_completion:.2f}% of total training complete")

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
        i = 0

        #feedforward
        activation = x
        #list to store all the activations, layer by layer
        activations = [x]
        #list to store all the z vectors, layer by layer
        zs = []
        #feedforward the input x through the network and store the activations and z vectors
        for b, w in zip(self.__biases, self.__weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            #TODO(1)
            activation = self.__activation_func[i](z)
            activations.append(activation)
            i += 1

        #backward pass
        #calculate the error in the output layer
        delta = self.__cost_derivative(activations[-1], y) * self.__activation_prime[-1](zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        i -= 1
        #propagate the error to the previous layers
        for l in range(2, self.__num_layers):
            z = zs[-l]
            #TODO(1)
            ap = self.__activation_prime[i](z)
            delta = np.dot(self.__weights[-l + 1].transpose(), delta) * ap
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
            i -= 1

        return nabla_b, nabla_w

    #TODO (1)
    def predict(self, x):
        return self.__activation_prime[self.__num_layers -2 ](self.__feedforward(x))


    #input: test_data - list of tuples (x, y) where x is the input and y is the expected output
    #output: list of tuples (prediction, expected) for each input in the test_data
    #TODO (1)
    def __predict_batch(self, test_data):
        #get the number of correct predictions
        test_results = [(self.__activation_prime[self.__num_layers -2](self.__feedforward(x)), self.__activation_prime[self.__num_layers-2](y)) for (x, y) in test_data]
        return test_results


    #input: test_data - list of tuples (x, y) where x is the input and y is the expected output
    #output: the accuracy of the network on the test_data
    def score(self, test_data):
        predictions = [np.argmax(self.__feedforward(x)) for x, _ in test_data]  # Assuming __feedforward returns a probability distribution
        correct_count = sum(1 for pred, (_, y) in zip(predictions, test_data) if pred == np.argmax(y))
        return correct_count / len(test_data)
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
    def ___create_activation_function_lists(self, activations , size_list_len , output_activation):
        activation_func_list = []
        activation_prime_list = []
        if activations is None:
            activations = ['sigmoid'] * (size_list_len - 2) + [output_activation]

        if len(activations) != size_list_len - 1:
            raise ValueError('The number of activation functions must be equal to : (the number of layers) - 1 ')

        self.activation_function_list = activations

        for name in activations:
            func , prime = self.__get_activation_function(name)
            activation_func_list.append(func)
            activation_prime_list.append(prime)
        
        return activation_func_list , activation_prime_list

    def __get_activation_function(self, func_name):

        #dictionary (of tuples) with all activation functions and their derivatives
        activation_functions = {
            'relu': (self.relu, self.relu_prime),
            'sigmoid': (self.sigmoid, self.sigmoid_prime),
            'tanh': (self.tanh, self.tanh_prime),
            'softmax': (self.softmax, self.softmax_prime),
            'linear': (self.linear, self.linear_prime),
            'argmax': (np.argmax, 1)
        }
        
        #if the function is in the dictionary, set the activation function and its derivative
        if func_name in activation_functions:
            return activation_functions[func_name]
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
        y = y.reshape(-1, 1)
        return (output_activations - y)
    
    #--huber loss--
    def huber_loss_derivative(self, output_activations, y):
        y = y.reshape(-1, 1)
        return np.where(np.abs(output_activations - y) <= 1, output_activations - y, np.sign(output_activations - y))
