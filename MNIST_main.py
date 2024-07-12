import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from network import Network

def MNIST_main():
    train_X_y, test_X_y, validation_X_y = prepare_data('MNIST-train.csv', 'MNIST-test.csv')
    network = create_network_for_MNIST(train_X_y[0], train_X_y[1])
    accuracy = train_and_test_network_for_MNIST(network, train_X_y, test_X_y, validation_X_y)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

def load_and_scale_data(data_csv_path):
    # Load the data from the CSV file
    df = pd.read_csv(data_csv_path)
    true_labels = df["y"]
    data = df.drop(columns=["y"])
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    data = np.array(data)
    true_labels = np.array(true_labels)

    return data, true_labels

def prepare_data(train_data_csv_path, test_data_csv_path=None, load_and_scale_data_function=load_and_scale_data, validation_size=0.2):
    train_data, true_labels_train = load_and_scale_data_function(train_data_csv_path)

    # Split data into training and validation sets
    X_train, X_validation, y_train, y_validation = train_test_split(train_data, true_labels_train, test_size=validation_size, shuffle=True)

    # Combine data and targets into a list of tuples
    train_X_y = (X_train, y_train)
    validation_X_y = (X_validation, y_validation)

    if test_data_csv_path != None:
        test_data, true_labels_test = load_and_scale_data_function(test_data_csv_path)
        test_X_y = (test_data, true_labels_test)
        return train_X_y, test_X_y, validation_X_y
    else:
        return train_X_y, validation_X_y

def create_network_for_MNIST(X_train, y_train):
    num_of_input_features = X_train.shape[1]
    num_of_possible_outputs = len(np.unique(y_train))

    network_sizes = [num_of_input_features, 256, 128, 64, num_of_possible_outputs]  # configuration
    activations = ['leaky_relu', 'leaky_relu', 'leaky_relu', 'softmax']

    network = Network(sizes=network_sizes, activations_functions_names=activations, output_activation_name='softmax'
                      , train_learning_rate=0.01, train_mini_batch_size=10, train_epochs=100,
                      cost_function_name='cross_entropy', regularization_lambda=0.00001)

    return network

def train_and_test_network_for_MNIST(network, train_X_y, test_X_y, validation_X_y):
    X_train, y_train = train_X_y
    X_test, y_test = test_X_y
    X_validation, y_validation = validation_X_y

    # Training the network
    network.fit(X_train, y_train, X_validation, y_validation)
    # Evaluate the network on test data
    accuracy = network.score(X_test, y_test)

    return accuracy