import random
import numpy as np
from ActivationFunctions import ActivationFunctions
from CostFunctions import CostFunctions

#TODO(1) level 1 : instead of a single function, we can use a list of function pointers (one function for each layer)
#TODO 1 change the feedforward function to use the list of activation functions and so on
class Network:
    def __init__(self, sizes , activations_functions_names=None, cost_function_name ='quadratic', output_activation_name='softmax', train_learning_rate=0.1):
        self.__num_layers = len(sizes)
        #number of neurons in each layer
        self.__sizes = sizes
        #initialize weights and biases for each layer
        self.__biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.__weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        #stores a activation function name for each layer
        #activation function and its derivative for all layers exept output layer (for now)
        #TODO(1)
        self.__activation_func_list, self.__activation_derivatives_list = self.__init_activation_function_lists(
            activations_functions_names, output_activation_name)
        self.__cost_derivative = None
        self.__init_cost_function_derivative(cost_function_name)
        self.__learning_rate = train_learning_rate

    def __init_activation_function_lists(self, activations_funcs_names, output_activation_name):
        activation_func_list = []
        activation_prime_list = []
        if activations_funcs_names is None:
            activations_funcs_names = ['sigmoid'] * (self.__num_layers - 2) + [output_activation_name]

        if len(activations_funcs_names) != self.__num_layers - 1:
            raise ValueError('The number of activation functions must be equal to : (the number of layers) - 1 ')

        for name in activations_funcs_names:
            func, prime = ActivationFunctions.get_activation_function_by_name(name)
            activation_func_list.append(func)
            activation_prime_list.append(prime)

        return activation_func_list, activation_prime_list

    # input: string with the name of the cost function
    def __init_cost_function_derivative(self, cost_function_name):
        self.__cost_derivative = CostFunctions.get_cost_function_by_name(cost_function_name)

    def __feedforward(self, input_vector):
        i = 0
        current_layer_vector = input_vector
        #feedforward the input a through the network
        for bias, weight in zip(self.__biases, self.__weights):
            #TODO(1)
            z = np.dot(weight, current_layer_vector) + bias
            current_layer_vector = self.__activation_func_list[i](z)
            i += 1
        return current_layer_vector

    def train(self, training_data, mini_batch_size, epochs=1000):
        for j in range(epochs):
            random.shuffle(training_data)
            #divide the training data into mini batches
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]
            for mini_batch in mini_batches:
                #update the weights and biases using the gradients of the cost function
                self.__update_mini_batch(mini_batch)
            #DEBUG
            epoch_completion = 100 * (j + 1) / epochs
            print(f"Epoch {j+1}/{epochs} complete: {epoch_completion:.2f}% of total training complete")

    def __update_mini_batch(self, mini_batch):
        # Initialize the lists for the gradients of the cost function
        # with respect to the biases and weights
        bias_gradients = [np.zeros(bias.shape) for bias in self.__biases]
        weight_gradients = [np.zeros(weight.shape) for weight in self.__weights]

        for feature_vector, true_label in mini_batch:
            # Compute the gradients of the cost function using backpropagation
            delta_bias_gradients, delta_weight_gradients = self.__backpropagation(feature_vector, true_label)
            # Accumulate the gradients of the batch
            bias_gradients = [bg + dbg for bg, dbg in zip(bias_gradients, delta_bias_gradients)]
            weight_gradients = [wg + dwg for wg, dwg in zip(weight_gradients, delta_weight_gradients)]

        # Update the weights and biases using the gradients and the learning rate
        self.__weights = [weight - (self.__learning_rate / len(mini_batch)) * wg for weight, wg in
                          zip(self.__weights, weight_gradients)]
        self.__biases = [bias - (self.__learning_rate / len(mini_batch)) * bg for bias, bg in
                         zip(self.__biases, bias_gradients)]

    def __backpropagation(self, feature_vector, true_label):
        # Perform feedforward pass to compute activations and weighted inputs
        all_vectors_after_activations, weighted_inputs = self.__compute_activation_vectors(feature_vector)
        # Perform backward pass to compute gradients
        bias_gradients, weight_gradients = self.__compute_gradients(all_vectors_after_activations, weighted_inputs,
                                                                    true_label)

        return bias_gradients, weight_gradients

    def __compute_activation_vectors(self, feature_vector):
        # Initialize lists to store activations and weighted inputs
        all_vectors_after_activations = [feature_vector]
        weighted_inputs = []

        # Perform the feedforward pass through the network
        current_activation = feature_vector

        for layer_index, (bias, weight) in enumerate(zip(self.__biases, self.__weights)):
            weighted_input = np.dot(weight, current_activation) + bias
            weighted_inputs.append(weighted_input)
            current_activation = self.__activation_func_list[layer_index](weighted_input)
            all_vectors_after_activations.append(current_activation)

        return all_vectors_after_activations, weighted_inputs

    def __compute_gradients(self, all_vectors_after_activations, weighted_inputs, true_label):
        # Initialize gradients
        bias_gradients = [np.zeros(bias.shape) for bias in self.__biases]
        weight_gradients = [np.zeros(weight.shape) for weight in self.__weights]

        # Compute initial delta
        final_output = all_vectors_after_activations[-1]
        delta = self.__cost_derivative(final_output, true_label) * self.__activation_derivatives_list[-1](
            weighted_inputs[-1])
        bias_gradients[-1] = delta
        weight_gradients[-1] = np.dot(delta, all_vectors_after_activations[-2].transpose())

        # propagate the error to the previous layers
        layer_index = len(self.__weights) - 1
        for l in range(2, self.__num_layers):
            weighted_input = weighted_inputs[-l]
            activation_prime = self.__activation_derivatives_list[layer_index](weighted_input)
            delta = np.dot(self.__weights[-l + 1].transpose(), delta) * activation_prime
            bias_gradients[-l] = delta
            weight_gradients[-l] = np.dot(delta, all_vectors_after_activations[-l - 1].transpose())
            layer_index -= 1

        return bias_gradients, weight_gradients

    #TODO (1)
    def predict(self, x):
        return self.__activation_derivatives_list[self.__num_layers - 2](self.__feedforward(x))

    #input: test_data - list of tuples (x, y) where x is the input and y is the expected output
    #output: the accuracy of the network on the test_data
    def score(self, test_data):
        predictions = [np.argmax(self.__feedforward(x)) for x, _ in test_data]  # Assuming __feedforward returns a probability distribution
        correct_count = sum(1 for pred, (_, y) in zip(predictions, test_data) if pred == np.argmax(y))
        return correct_count / len(test_data)
