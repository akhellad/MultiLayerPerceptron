import numpy as np
import pandas as pd
import argparse
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

# Classe pour les couches denses
class DenseLayer:
    def __init__(self, input_size, output_size, activation='sigmoid'):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.biases = np.zeros((1, output_size))
        self.activation = activation

    def forward(self, input_data):
        self.input_data = input_data
        self.z = np.dot(input_data, self.weights) + self.biases
        
        if self.activation == 'sigmoid':
            self.a = sigmoid(self.z)
        elif self.activation == 'softmax':
            self.a = softmax(self.z)
        
        return self.a

    def to_dict(self):
        return {
            'weights': self.weights.tolist(),
            'biases': self.biases.tolist(),
            'activation': self.activation
        }

    @staticmethod
    def from_dict(layer_dict):
        input_size, output_size = np.array(layer_dict['weights']).shape
        layer = DenseLayer(input_size, output_size, activation=layer_dict['activation'])
        layer.weights = np.array(layer_dict['weights'])
        layer.biases = np.array(layer_dict['biases'])
        return layer

# Classe pour le réseau de neurones
class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def save(self, file_path):
        model = {
            'layers': [layer.to_dict() for layer in self.layers]
        }
        with open(file_path, 'w') as f:
            json.dump(model, f)

    @staticmethod
    def load(file_path):
        with open(file_path, 'r') as f:
            model = json.load(f)
        network = NeuralNetwork()
        for layer_dict in model['layers']:
            network.add_layer(DenseLayer.from_dict(layer_dict))
        return network

def binary_crossentropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def main(args):
    network = NeuralNetwork.load(args.model)

    prediction_data = pd.read_csv(args.prediction_data)
    X_pred = prediction_data.drop(columns=['Diagnosis']).values
    y_pred = prediction_data['Diagnosis'].values

    le = LabelEncoder()
    y_pred = le.fit_transform(y_pred)

    scaler = StandardScaler()
    X_pred = scaler.fit_transform(X_pred)

    y_pred = np.eye(2)[y_pred]

    predictions = network.forward(X_pred)

    loss = binary_crossentropy(y_pred, predictions)
    print(f'Binary Cross-Entropy Loss: {loss:.4f}')

    accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y_pred, axis=1))
    print(f'Accuracy: {accuracy * 100:.2f}%')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict using a trained neural network model.')
    parser.add_argument('--model', type=str, required=True, help='Path to the saved model.')
    parser.add_argument('--prediction_data', type=str, required=True, help='Path to the prediction data.')
    args = parser.parse_args()
    
    main(args)
