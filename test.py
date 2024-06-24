import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from network import Network
from ActivationFunctions import ActivationFunctions

def load_and_prepare_data():
    # Load the data from the CSV file
    df = pd.read_csv('MNIST-train.csv')
    targets = df["y"]
    data = df.drop(columns=["y"])
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    data = np.array(data)
    targets = np.array(targets)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.2, shuffle=True)

    # Combine data and targets into a list of tuples
    train_X_y = (X_train, y_train)
    test_X_y = (X_test, y_test)

    return train_X_y, test_X_y

def create_and_train_network(train_X_y, test_X_y):
    X_train, y_train = train_X_y
    X_test, y_test = test_X_y

    network_sizes = [784, 128, 64, 10]  # configuration
    activations = ['relu', 'relu', 'softmax']

    network = Network(sizes=network_sizes, activations_functions_names=activations, output_activation_name='softmax'
                      , train_learning_rate=0.01, train_mini_batch_size=10, train_epochs=10)

    # Training the network
    network.fit(X_train, y_train, X_test, y_test)

    # Evaluate the network on test data
    accuracy = network.score(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

def main():
    train_X_y, test_X_y = load_and_prepare_data()
    create_and_train_network(train_X_y, test_X_y)

if __name__ == '__main__':
    main()