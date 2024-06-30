import random
import numpy as np
from ActivationFunctions import ActivationFunctions
from CostFunctions import CostFunctions
import logging


class Network:
    def __init__(self, sizes , activations_functions_names=None, cost_function_name ='quadratic', output_activation_name='softmax'
                 , train_learning_rate=0.1, train_mini_batch_size=10, train_epochs=100):
        self.__num_layers = len(sizes)
        #number of neurons in each layer
        self.__sizes = sizes
        #initialize weights and biases for each layer
        self.__biases = [np.random.randn(y, 1)  for y in sizes[1:]]
        self.__weights = [np.random.randn(y, x) * np.sqrt(1./x) for x, y in zip(sizes[:-1], sizes[1:])]
        #stores a activation function name for each layer
        #activation function and its derivative for all layers exept output layer (for now)
        #TODO(1)
        self.__activation_func_list, self.__activation_derivatives_list = self.__init_activation_function_lists(
            activations_functions_names, output_activation_name)
        self.__cost_derivative = None
        self.__init_cost_function_derivative(cost_function_name)
        self.__learning_rate = train_learning_rate
        self.__mini_batch_size = train_mini_batch_size
        self.__epochs = train_epochs
        self.__original_labels = None
        self.__fit_completed = False
        self.__logger = self.__setup_logger()


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

    def __setup_logger(self):
        logger = logging.getLogger('hyperparams-logger')
        logger.propagate = False
        logger.setLevel(logging.DEBUG)
        if not logger.hasHandlers():
            # Create formatter and add it to the handlers
            formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(levelname)s: %(message)s',
                                          datefmt='%d-%m-%Y %H:%M:%S')
            # Create file handler for logging to a file
            file_handler = logging.FileHandler('hyperparams.log')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            # Create console handler for logging to a file
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(formatter)
            # Add handlers to the logger
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        return logger

    def __feedforward(self, input_vector):
        i = 0
        current_layer_vector = input_vector.reshape(-1, 1)
        #feedforward the input a through the network
        for bias, weight in zip(self.__biases, self.__weights):
            z = np.dot(weight, current_layer_vector) + bias
            current_layer_vector = self.__activation_func_list[i](z)
            i += 1
        return current_layer_vector

    def fit(self, X, y, validation_X=None, validation_y=None):
        self.__fit_completed = True
        # TODO: add try and catch
        training_data = self.__init_fit_params(X, y)
        for j in range(self.__epochs):
            random.shuffle(training_data)
            #divide the training data into mini batches
            mini_batches = [training_data[k:k + self.__mini_batch_size]
                            for k in range(0, len(training_data), self.__mini_batch_size)]
            for mini_batch in mini_batches:
                #update the weights and biases using the gradients of the cost function
                self.__update_mini_batch(mini_batch)
            #DEBUG
            self.__log(j, validation_X, validation_y)

    def __log(self, curr_epoch_num, validation_X, validation_y):
        if validation_X is not None and validation_y is not None:
            epoch_completion = 100 * (curr_epoch_num + 1) / self.__epochs
            self.__logger.debug(
                f"Epoch {curr_epoch_num + 1}/{self.__epochs} complete: {epoch_completion:.2f}% of total training complete")
            accuracy = self.score(validation_X, validation_y)
            self.__logger.info(f"Test Accuracy in the {curr_epoch_num + 1} iteration: {accuracy * 100:.2f}%")

    def __init_fit_params(self, X, y):
        self.__original_labels = np.unique(y)
        # Reshape data to fit the network input (each input is a vector)
        feature_vectors = X.reshape(X.shape[0], -1, 1)
        # Convert labels to one-hot encoding
        true_labels = np.eye(len(self.__original_labels))[y]
        # Reshape each one-hot encoded label to be (num_classes, 1)
        true_labels = true_labels.reshape(true_labels.shape[0], -1, 1)
        # Combine data and targets into a list of tuples
        training_data = list(zip(feature_vectors, true_labels))

        return training_data


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
        for l in range(2, self.__num_layers):
            weighted_input = weighted_inputs[-l]
            activation_prime = self.__activation_derivatives_list[-l](weighted_input)
            delta = np.dot(self.__weights[-l + 1].transpose(), delta) * activation_prime
            bias_gradients[-l] = delta
            weight_gradients[-l] = np.dot(delta, all_vectors_after_activations[-l - 1].transpose())

        return bias_gradients, weight_gradients

    def predict(self, X):
        if self.__fit_completed:
            X = np.array(X)
            ROWS = 1
            predictions = np.apply_along_axis(self.__predict_single_feature_vector, arr=X, axis=ROWS)
            return predictions
        else:
            raise RuntimeError("Model has not been fitted yet. Please call fit() first.")

    def __predict_single_feature_vector(self, feature_vector):
        index_of_predicted_value = np.argmax(self.__feedforward(feature_vector))
        return self.__original_labels[index_of_predicted_value]

    def score(self, X, y):
        if self.__fit_completed:
            predictions = np.array(self.predict(X))
            true_labels = np.array(y)
            num_of_correct_classifications = np.sum(predictions == true_labels)
            num_of_samples = len(X)
            return num_of_correct_classifications / num_of_samples
        else:
            raise RuntimeError("Model has not been fitted yet. Please call fit() first.")

