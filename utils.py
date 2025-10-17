"""
Utility functions for data preprocessing and model evaluation.
"""
import numpy as np
import json
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataPreprocessor:
    """Handles data preprocessing including scaling and label encoding."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.fitted = False

    def fit_transform(self, X, y):
        """
        Fit the scaler and label encoder, then transform the data.

        Args:
            X: Feature matrix
            y: Labels

        Returns:
            X_scaled: Scaled features
            y_encoded: One-hot encoded labels
        """
        X_scaled = self.scaler.fit_transform(X)
        y_encoded_int = self.label_encoder.fit_transform(y)
        y_encoded = np.eye(len(self.label_encoder.classes_))[y_encoded_int]
        self.fitted = True
        return X_scaled, y_encoded

    def transform(self, X, y=None):
        """
        Transform data using fitted scaler and label encoder.

        Args:
            X: Feature matrix
            y: Labels (optional)

        Returns:
            X_scaled: Scaled features
            y_encoded: One-hot encoded labels (if y is provided)
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform. Use fit_transform first.")

        X_scaled = self.scaler.transform(X)

        if y is not None:
            y_encoded_int = self.label_encoder.transform(y)
            y_encoded = np.eye(len(self.label_encoder.classes_))[y_encoded_int]
            return X_scaled, y_encoded

        return X_scaled

    def save(self, file_path):
        """Save the preprocessor state to a file."""
        if not self.fitted:
            raise ValueError("Cannot save unfitted preprocessor.")

        state = {
            'scaler_mean': self.scaler.mean_.tolist(),
            'scaler_scale': self.scaler.scale_.tolist(),
            'label_classes': self.label_encoder.classes_.tolist()
        }

        dir_path = os.path.dirname(file_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)

        with open(file_path, 'w') as f:
            json.dump(state, f, indent=2)

    @staticmethod
    def load(file_path):
        """Load a preprocessor from a file."""
        with open(file_path, 'r') as f:
            state = json.load(f)

        preprocessor = DataPreprocessor()
        preprocessor.scaler.mean_ = np.array(state['scaler_mean'])
        preprocessor.scaler.scale_ = np.array(state['scaler_scale'])
        preprocessor.scaler.n_features_in_ = len(state['scaler_mean'])
        preprocessor.label_encoder.classes_ = np.array(state['label_classes'])
        preprocessor.fitted = True

        return preprocessor


def binary_crossentropy(y_true, y_pred, epsilon=1e-15):
    """
    Compute binary cross-entropy loss.

    Args:
        y_true: True labels (one-hot encoded)
        y_pred: Predicted probabilities
        epsilon: Small value to avoid log(0)

    Returns:
        Binary cross-entropy loss
    """
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def compute_accuracy(y_true, y_pred):
    """
    Compute classification accuracy.

    Args:
        y_true: True labels (one-hot encoded)
        y_pred: Predicted probabilities

    Returns:
        Accuracy as a float
    """
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))
