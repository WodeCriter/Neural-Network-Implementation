import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from network import Network
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

def MB_main():
    data_csv_path = "MB_data_train.csv"
    data, true_labels = load_and_scale_MB_data(data_csv_path)
    network = create_network_for_MB(data, true_labels)
    k_fold = KFold(n_splits=20, shuffle=True, random_state=1)
    accuracies = []

    for train_index, validation_index in k_fold.split(data):
        X_train, X_validation = data[train_index], data[validation_index]
        y_train, y_validation = true_labels[train_index], true_labels[validation_index]

        accuracy = train_and_test_network_for_MB(network, X_train, y_train, X_validation, y_validation)
        accuracies.append(accuracy)

    print(f"Average Accuracy: {np.mean(accuracies) * 100:.2f}%")
    print(f"Accuracies for each fold: {accuracies}")

def load_and_scale_MB_data(data_csv_path):
    # Load the data from the CSV file
    df = pd.read_csv(data_csv_path)
    true_labels = np.array(df.iloc[:, 0].tolist())
    true_labels = np.where(np.char.find(true_labels, 'Fibro') >= 0, 0, 1)
    data = df.drop(df.columns[0], axis=1)
    pca = PCA(n_components=100)
    data = pca.fit_transform(data)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    data = np.array(data)

    return data, true_labels

def train_and_test_network_for_MB(network, X_train, y_train, X_validation, y_validation):
    # Training the network
    network.fit(X_train, y_train, X_validation, y_validation)
    # Evaluate the network on validation data
    accuracy = network.score(X_validation, y_validation)

    return accuracy

def create_network_for_MB(X_train, y_train):
    num_of_input_features = X_train.shape[1]
    num_of_possible_outputs = len(np.unique(y_train))

    network_sizes = [num_of_input_features, 64, 32, 16, 8, 4, num_of_possible_outputs]  # configuration
    activations = ['leaky_relu', 'leaky_relu', 'leaky_relu', 'leaky_relu', 'leaky_relu', 'softmax']

    network = Network(sizes=network_sizes, activations_functions_names=activations, train_learning_rate=0.001,
                      train_mini_batch_size=10, train_epochs=100, cost_function_name='cross_entropy',
                      regularization_lambda=0.0001, show_logs=False)

    return network
