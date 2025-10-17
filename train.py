import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import sys
import json
import os
from utils import DataPreprocessor

# Fonctions d'activation
def sigmoid(x):
    """Sigmoid activation function."""
    # Clip to prevent overflow
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of sigmoid function (expects sigmoid output)."""
    return x * (1 - x)

def softmax(x):
    """Softmax activation function for multi-class classification."""
    exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

def he_initialization(input_size, output_size):
    """He initialization for weights (good for ReLU/sigmoid)."""
    return np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)

def xavier_initialization(input_size, output_size):
    """Xavier initialization for weights (good for sigmoid/tanh)."""
    limit = np.sqrt(6.0 / (input_size + output_size))
    return np.random.uniform(-limit, limit, (input_size, output_size))

# Optimizers
class Optimizer:
    def update(self, weights, biases, grad_w, grad_b):
        raise NotImplementedError

class AdamOptimizer(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_w = None
        self.v_w = None
        self.m_b = None
        self.v_b = None
        self.t = 0

    def update(self, weights, biases, grad_w, grad_b):
        if self.m_w is None:
            self.m_w = np.zeros_like(weights)
            self.v_w = np.zeros_like(weights)
            self.m_b = np.zeros_like(biases)
            self.v_b = np.zeros_like(biases)
        
        self.t += 1
        self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * grad_w
        self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * np.square(grad_w)
        m_w_hat = self.m_w / (1 - self.beta1 ** self.t)
        v_w_hat = self.v_w / (1 - self.beta2 ** self.t)
        weights -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)

        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * grad_b
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * np.square(grad_b)
        m_b_hat = self.m_b / (1 - self.beta1 ** self.t)
        v_b_hat = self.v_b / (1 - self.beta2 ** self.t)
        biases -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

        return weights, biases

class DenseLayer:
    """Dense (fully connected) layer with configurable activation and initialization."""

    def __init__(self, input_size, output_size, activation='sigmoid',
                 optimizer=None, weight_init='xavier'):
        """
        Initialize a dense layer.

        Args:
            input_size: Number of input features
            output_size: Number of output neurons
            activation: Activation function ('sigmoid' or 'softmax')
            optimizer: Optimizer instance (defaults to AdamOptimizer)
            weight_init: Weight initialization method ('xavier', 'he', or 'random')
        """
        # Initialize weights based on the chosen method
        if weight_init == 'he':
            self.weights = he_initialization(input_size, output_size)
        elif weight_init == 'xavier':
            self.weights = xavier_initialization(input_size, output_size)
        else:
            self.weights = np.random.randn(input_size, output_size) * 0.1

        self.biases = np.zeros((1, output_size))
        self.activation = activation
        self.optimizer = optimizer if optimizer is not None else AdamOptimizer()

    def forward(self, input_data):
        self.input_data = input_data
        self.z = np.dot(input_data, self.weights) + self.biases
        
        if self.activation == 'sigmoid':
            self.a = sigmoid(self.z)
        elif self.activation == 'softmax':
            self.a = softmax(self.z)
        
        return self.a

    def backward(self, output_error):
        if self.activation == 'sigmoid':
            activation_derivative = sigmoid_derivative(self.a)
        elif self.activation == 'softmax':
            activation_derivative = self.a  # Derivative for softmax combined with cross-entropy is just the output
        
        self.error = output_error * activation_derivative
        self.input_error = np.dot(self.error, self.weights.T)
        
        # Update weights and biases using the optimizer
        grad_w = np.dot(self.input_data.T, self.error)
        grad_b = np.sum(self.error, axis=0, keepdims=True)
        self.weights, self.biases = self.optimizer.update(self.weights, self.biases, grad_w, grad_b)
        
        return self.input_error

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

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.train_losses = []
        self.valid_losses = []
        self.train_accuracies = []
        self.valid_accuracies = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, y, output):
        output_error = output - y
        for layer in reversed(self.layers):
            output_error = layer.backward(output_error)

    def compute_loss(self, y_true, y_pred):
        return np.mean(-y_true * np.log(y_pred))

    def compute_accuracy(self, y_true, y_pred):
        return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))

    def train(self, X_train, y_train, X_valid, y_valid, epochs, patience=10):
        best_valid_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            output_train = self.forward(X_train)
            self.backward(y_train, output_train)
            
            output_valid = self.forward(X_valid)

            train_loss = self.compute_loss(y_train, output_train)
            valid_loss = self.compute_loss(y_valid, output_valid)

            train_accuracy = self.compute_accuracy(y_train, output_train)
            valid_accuracy = self.compute_accuracy(y_valid, output_valid)

            self.train_losses.append(train_loss)
            self.valid_losses.append(valid_loss)
            self.train_accuracies.append(train_accuracy)
            self.valid_accuracies.append(valid_accuracy)

            print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}, Accuracy: {train_accuracy:.4f}, Validation Accuracy: {valid_accuracy:.4f}')
            sys.stdout.flush()

            # Arrêt anticipé
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Arrêt anticipé...")
                    break

    def save_metrics(self, metrics_file):
        """Save training metrics to a JSON file."""
        metrics = {
            'train_losses': self.train_losses,
            'valid_losses': self.valid_losses,
            'train_accuracies': self.train_accuracies,
            'valid_accuracies': self.valid_accuracies
        }
        dir_path = os.path.dirname(metrics_file)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

    def plot_metrics(self, metrics_files=None):
        epochs = range(1, len(self.train_losses) + 1)

        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, 'b', label='Perte (entraînement)')
        plt.plot(epochs, self.valid_losses, 'r', label='Perte (validation)')
        plt.title('Perte pendant l\'entraînement')
        plt.xlabel('Époques')
        plt.ylabel('Perte')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accuracies, 'b', label='Précision (entraînement)')
        plt.plot(epochs, self.valid_accuracies, 'r', label='Précision (validation)')
        plt.title('Précision pendant l\'entraînement')
        plt.xlabel('Époques')
        plt.ylabel('Précision')
        plt.legend()

        if metrics_files:
            for metrics_file in metrics_files:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                plt.subplot(1, 2, 1)
                plt.plot(range(1, len(metrics['train_losses']) + 1), metrics['train_losses'], label=f"{metrics_file} - Training loss")
                plt.plot(range(1, len(metrics['valid_losses']) + 1), metrics['valid_losses'], label=f"{metrics_file} - Validation loss")
                plt.subplot(1, 2, 2)
                plt.plot(range(1, len(metrics['train_accuracies']) + 1), metrics['train_accuracies'], label=f"{metrics_file} - Training accuracy")
                plt.plot(range(1, len(metrics['valid_accuracies']) + 1), metrics['valid_accuracies'], label=f"{metrics_file} - Validation accuracy")

        plt.show()

    def save(self, file_path):
        """Save the network architecture and weights to a JSON file."""
        model = {
            'layers': [layer.to_dict() for layer in self.layers]
        }
        dir_path = os.path.dirname(file_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with open(file_path, 'w') as f:
            json.dump(model, f, indent=2)

    @staticmethod
    def load(file_path):
        with open(file_path, 'r') as f:
            model = json.load(f)
        network = NeuralNetwork()
        for layer_dict in model['layers']:
            network.add_layer(DenseLayer.from_dict(layer_dict))
        return network

def evaluate(network, X, y):
    """Évaluer le réseau sur un jeu de données."""
    predictions = network.forward(X)
    accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y, axis=1))
    print(f'Précision : {accuracy * 100:.2f}%')
    sys.stdout.flush()

def main(args):
    """Fonction principale d'entraînement."""
    print("=" * 60)
    print("ENTRAÎNEMENT DU PERCEPTRON MULTICOUCHE")
    print("=" * 60)

    # Chargement des données
    print("\n[1/5] Chargement des données...")
    train_data = pd.read_csv(args.train_data)
    validation_data = pd.read_csv(args.validation_data)
    print(f"  Échantillons d'entraînement : {len(train_data)}")
    print(f"  Échantillons de validation : {len(validation_data)}")

    # Prétraitement des données
    print("\n[2/5] Prétraitement des données...")
    X_train = train_data.drop(columns=['Diagnosis']).values
    y_train = train_data['Diagnosis'].values
    X_valid = validation_data.drop(columns=['Diagnosis']).values
    y_valid = validation_data['Diagnosis'].values

    print(f"  Caractéristiques : {X_train.shape[1]}")
    print(f"  Classes : {len(np.unique(y_train))}")

    # Créer et ajuster le preprocesseur
    preprocessor = DataPreprocessor()
    X_train, y_train = preprocessor.fit_transform(X_train, y_train)
    X_valid, y_valid = preprocessor.transform(X_valid, y_valid)

    # Sauvegarder le preprocesseur pour la prédiction
    preprocessor_path = args.model_output.replace('.json', '_preprocessor.json')
    preprocessor.save(preprocessor_path)
    print(f"  Preprocesseur sauvegardé dans : {preprocessor_path}")

    # Construire le modèle
    print("\n[3/5] Construction de l'architecture du réseau...")
    network = NeuralNetwork()
    input_size = X_train.shape[1]
    output_size = 2

    if args.config_file:
        print(f"  Chargement de l'architecture depuis : {args.config_file}")
        with open(args.config_file, 'r') as f:
            layers_config = f.readlines()
        layer_count = 0
        for layer in layers_config:
            layer = layer.strip()
            if layer:
                layer_params = layer.split()
                units = int(layer_params[1])
                activation = layer_params[2]
                network.add_layer(DenseLayer(input_size, units, activation,
                                            weight_init=args.weight_init))
                print(f"    Couche {layer_count + 1}: {input_size} -> {units} ({activation})")
                input_size = units
                layer_count += 1
        network.add_layer(DenseLayer(input_size, output_size, activation='softmax',
                                     weight_init=args.weight_init))
        print(f"    Couche {layer_count + 1}: {input_size} -> {output_size} (softmax)")
    else:
        for idx, units in enumerate(args.layers):
            network.add_layer(DenseLayer(input_size, units, activation='sigmoid',
                                        weight_init=args.weight_init))
            print(f"    Couche {idx + 1}: {input_size} -> {units} (sigmoid)")
            input_size = units
        network.add_layer(DenseLayer(input_size, output_size, activation='softmax',
                                     weight_init=args.weight_init))
        print(f"    Couche {len(args.layers) + 1}: {input_size} -> {output_size} (softmax)")

    # Entraîner le modèle
    print("\n[4/5] Entraînement du réseau...")
    print(f"  Époques : {args.epochs}")
    print(f"  Patience (arrêt anticipé) : {args.patience}")
    print(f"  Taux d'apprentissage : {args.learning_rate}")
    print("-" * 60)
    network.train(X_train, y_train, X_valid, y_valid, epochs=args.epochs, patience=args.patience)
    print("-" * 60)

    # Évaluer le modèle
    print("\n[5/5] Évaluation et sauvegarde du modèle...")
    evaluate(network, X_valid, y_valid)

    # Sauvegarder les métriques
    metrics_file = args.model_output.replace('.json', '_metrics.json')
    network.save_metrics(metrics_file)
    print(f"  Métriques sauvegardées dans : {metrics_file}")

    # Sauvegarder le modèle
    network.save(args.model_output)
    print(f"  Modèle sauvegardé dans : {args.model_output}")

    # Afficher les courbes d'apprentissage
    if args.plot_metrics:
        metrics_files = [metrics_file] + args.plot_metrics.split(',')
        network.plot_metrics(metrics_files)
    else:
        network.plot_metrics()

    print("\n" + "=" * 60)
    print("ENTRAÎNEMENT TERMINÉ !")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train a multilayer perceptron for breast cancer classification.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--train_data', type=str, default='data/train_data.csv',
                       help='Path to the training data CSV file.')
    parser.add_argument('--validation_data', type=str, default='data/validation_data.csv',
                       help='Path to the validation data CSV file.')
    parser.add_argument('--layers', type=int, nargs='+', default=[24, 24],
                       help='List of hidden layer sizes (e.g., 24 24 for two layers of 24 neurons).')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Maximum number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for the Adam optimizer.')
    parser.add_argument('--config_file', type=str,
                       help='Path to a configuration file for network architecture.')
    parser.add_argument('--optimizer', type=str, choices=['adam'], default='adam',
                       help='Optimizer to use (currently only Adam is supported).')
    parser.add_argument('--patience', type=int, default=15,
                       help='Number of epochs with no improvement before early stopping.')
    parser.add_argument('--weight_init', type=str, choices=['xavier', 'he', 'random'],
                       default='xavier', help='Weight initialization method.')
    parser.add_argument('--model_output', type=str, default='model.json',
                       help='Path to save the trained model.')
    parser.add_argument('--plot_metrics', type=str,
                       help='Comma-separated list of metric files to compare on plots.')
    args = parser.parse_args()

    main(args)
