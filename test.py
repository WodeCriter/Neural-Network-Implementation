import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from network import Network

def load_and_prepare_data():
    # Load the data from the CSV file
    df = pd.read_csv('MNIST-train.csv')
    targets = df["y"]
    data = df.drop(columns=["y"])

    # Normalize the data
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # Reshape data to fit the network input (each input is a vector)
    data = data.reshape(data.shape[0], -1, 1)

    # Convert labels to one-hot encoding
    targets = np.eye(10)[targets]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.2, random_state=42)

    # Combine data and targets into a list of tuples
    train_data = list(zip(X_train, y_train))
    test_data = list(zip(X_test, y_test))

    return train_data, test_data

def create_and_train_network(train_data, test_data):
    network_sizes_list = [[784, 128, 64, 10], [784, 256, 128, 64, 10]]
    learning_rates = [0.25, 0.1, 0.01, 0.001]
    mini_batch_sizes = [10, 30, 50]
    epochs_list = [20]

    # Iterate over each set of hyperparameters
    for network_sizes in network_sizes_list:
        for learning_rate in learning_rates:
            for mini_batch_size in mini_batch_sizes:
                for epochs in epochs_list:
                    activations = ['sigmoid'] * (len(network_sizes) - 2) + ['softmax']

                    # Create a new instance of Network with current hyperparameters
                    network = Network(sizes=network_sizes, activations=activations, output_activation='softmax')

                    # Training the network
                    network.train(train_data, mini_batch_size=mini_batch_size, learningRate=learning_rate,
                                  epochs=epochs, validation_data=test_data)

                    del network



def main():
    train_data, test_data = load_and_prepare_data()
    create_and_train_network(train_data, test_data)

if __name__ == '__main__':
    main()