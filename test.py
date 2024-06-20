import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from network import Network

def load_and_prepare_data():
    # Load the data from the CSV file
    df = pd.read_csv('MNIST-train.csv')
    data = df.iloc[:, 1:].values  
    targets = df.iloc[:, 0].values  

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
    network_sizes = [784, 128, 64, 10]  # configuration
    activations = ['relu', 'relu', 'softmax']  # Using ReLU for hidden layers and softmax for output

    network = Network(sizes=network_sizes, activations=activations, output_activation='softmax')

    # Training the network
    network.train(train_data, mini_batch_size=10, learningRate=0.1, epochs=100)

    # Evaluate the network on test data
    accuracy = network.score(test_data)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

def main():
    train_data, test_data = load_and_prepare_data()
    create_and_train_network(train_data, test_data)

if __name__ == '__main__':
    main()