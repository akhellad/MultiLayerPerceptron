"""
Prediction script for the multilayer perceptron.
Loads a trained model and makes predictions on test data.
"""
import numpy as np
import pandas as pd
import argparse
import json
from utils import DataPreprocessor, binary_crossentropy, compute_accuracy


def sigmoid(x):
    """Sigmoid activation function."""
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def softmax(x):
    """Softmax activation function."""
    exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


class DenseLayer:
    """Dense layer for inference only (no training functionality)."""

    def __init__(self, input_size, output_size, activation='sigmoid'):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.biases = np.zeros((1, output_size))
        self.activation = activation

    def forward(self, input_data):
        """Forward pass through the layer."""
        self.z = np.dot(input_data, self.weights) + self.biases

        if self.activation == 'sigmoid':
            self.a = sigmoid(self.z)
        elif self.activation == 'softmax':
            self.a = softmax(self.z)

        return self.a

    @staticmethod
    def from_dict(layer_dict):
        """Load layer from dictionary."""
        input_size, output_size = np.array(layer_dict['weights']).shape
        layer = DenseLayer(input_size, output_size, activation=layer_dict['activation'])
        layer.weights = np.array(layer_dict['weights'])
        layer.biases = np.array(layer_dict['biases'])
        return layer


class NeuralNetwork:
    """Neural network for inference."""

    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        """Add a layer to the network."""
        self.layers.append(layer)

    def forward(self, X):
        """Forward pass through the entire network."""
        for layer in self.layers:
            X = layer.forward(X)
        return X

    @staticmethod
    def load(file_path):
        """Load a trained model from file."""
        with open(file_path, 'r') as f:
            model = json.load(f)
        network = NeuralNetwork()
        for layer_dict in model['layers']:
            network.add_layer(DenseLayer.from_dict(layer_dict))
        return network


def main(args):
    """Fonction principale de prédiction."""
    print("=" * 60)
    print("PRÉDICTION DU PERCEPTRON MULTICOUCHE")
    print("=" * 60)

    # Charger le modèle entraîné
    print(f"\n[1/4] Chargement du modèle depuis : {args.model}")
    network = NeuralNetwork.load(args.model)
    print(f"  Modèle chargé avec succès !")
    print(f"  Nombre de couches : {len(network.layers)}")

    # Charger le preprocesseur
    preprocessor_path = args.model.replace('.json', '_preprocessor.json')
    print(f"\n[2/4] Chargement du preprocesseur depuis : {preprocessor_path}")
    try:
        preprocessor = DataPreprocessor.load(preprocessor_path)
        print("  Preprocesseur chargé avec succès !")
    except FileNotFoundError:
        print(f"  ERREUR : Fichier preprocesseur introuvable : {preprocessor_path}")
        print("  Veuillez vous assurer d'avoir entraîné le modèle avec le script train.py mis à jour.")
        return

    # Charger les données de prédiction
    print(f"\n[3/4] Chargement des données de prédiction depuis : {args.prediction_data}")
    prediction_data = pd.read_csv(args.prediction_data)
    print(f"  Échantillons : {len(prediction_data)}")

    X_pred = prediction_data.drop(columns=['Diagnosis']).values
    y_true = prediction_data['Diagnosis'].values

    # Prétraiter les données avec le preprocesseur sauvegardé
    print("\n[4/4] Calcul des prédictions...")
    X_pred, y_true_encoded = preprocessor.transform(X_pred, y_true)

    # Faire les prédictions
    predictions = network.forward(X_pred)

    # Calculer les métriques
    loss = binary_crossentropy(y_true_encoded, predictions)
    accuracy = compute_accuracy(y_true_encoded, predictions)

    print("\n" + "=" * 60)
    print("RÉSULTATS")
    print("=" * 60)
    print(f"Perte (entropie croisée binaire) : {loss:.4f}")
    print(f"Précision : {accuracy * 100:.2f}%")
    print("=" * 60)

    # Afficher les prédictions détaillées si demandé
    if args.show_predictions:
        print("\nPrédictions détaillées (10 premiers échantillons) :")
        print("-" * 60)
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_true_encoded, axis=1)
        class_names = ['Bénin', 'Malin']

        for i in range(min(10, len(predictions))):
            true_label = class_names[true_classes[i]]
            pred_label = class_names[pred_classes[i]]
            confidence = predictions[i][pred_classes[i]] * 100
            match = "✓" if pred_classes[i] == true_classes[i] else "✗"
            print(f"  Échantillon {i+1}: Réel={true_label:9s} | Prédit={pred_label:9s} "
                  f"({confidence:.1f}%) {match}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Make predictions using a trained neural network.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model', type=str, required=True,
                       help='Path to the saved model JSON file.')
    parser.add_argument('--prediction_data', type=str, required=True,
                       help='Path to the CSV file containing data for prediction.')
    parser.add_argument('--show_predictions', action='store_true',
                       help='Show detailed predictions for the first 10 samples.')
    args = parser.parse_args()

    main(args)
