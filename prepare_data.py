"""
Data preparation script for the breast cancer classification project.
Loads raw data, assigns proper column names, and splits into training and validation sets.
"""
import pandas as pd
import numpy as np
import os
import argparse


def prepare_data(input_file, output_dir, train_ratio=0.8, random_seed=42, verbose=True):
    """
    Prepare and split the breast cancer dataset.

    Args:
        input_file: Path to the raw CSV data file
        output_dir: Directory to save the split datasets
        train_ratio: Ratio of data to use for training (default: 0.8)
        random_seed: Random seed for reproducibility (default: 42)
        verbose: Print progress information (default: True)
    """
    if verbose:
        print("=" * 60)
        print("PRÉPARATION DES DONNÉES")
        print("=" * 60)

    # Charger les données brutes
    if verbose:
        print(f"\n[1/4] Chargement des données depuis : {input_file}")
    data = pd.read_csv(input_file)
    if verbose:
        print(f"  Échantillons totaux : {len(data)}")
        print(f"  Caractéristiques totales : {len(data.columns)}")

    # Define column names based on the Wisconsin Breast Cancer Dataset
    column_names = [
        'ID', 'Diagnosis',
        # Mean values
        'Radius_mean', 'Texture_mean', 'Perimeter_mean', 'Area_mean',
        'Smoothness_mean', 'Compactness_mean', 'Concavity_mean',
        'Concave_points_mean', 'Symmetry_mean', 'Fractal_dimension_mean',
        # Standard error
        'Radius_se', 'Texture_se', 'Perimeter_se', 'Area_se',
        'Smoothness_se', 'Compactness_se', 'Concavity_se',
        'Concave_points_se', 'Symmetry_se', 'Fractal_dimension_se',
        # Worst values
        'Radius_worst', 'Texture_worst', 'Perimeter_worst', 'Area_worst',
        'Smoothness_worst', 'Compactness_worst', 'Concavity_worst',
        'Concave_points_worst', 'Symmetry_worst', 'Fractal_dimension_worst'
    ]

    if verbose:
        print(f"\n[2/4] Application des noms de colonnes et prétraitement")
    data.columns = column_names

    # Supprimer la colonne ID car elle n'est pas utile pour la prédiction
    data = data.drop(columns=['ID'])
    if verbose:
        print(f"  Colonne ID supprimée")
        print(f"  Caractéristiques restantes : {len(data.columns) - 1}")
        print(f"  Distribution des diagnostics :")
        print(f"    Malin (M) : {(data['Diagnosis'] == 'M').sum()}")
        print(f"    Bénin (B) : {(data['Diagnosis'] == 'B').sum()}")

    # Mélanger les données
    if verbose:
        print(f"\n[3/4] Mélange des données (seed={random_seed})")
    data = data.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # Diviser en ensembles d'entraînement et de validation
    train_size = int(train_ratio * len(data))
    train_data = data.iloc[:train_size]
    validation_data = data.iloc[train_size:]

    if verbose:
        print(f"  Échantillons d'entraînement : {len(train_data)} ({train_ratio*100:.0f}%)")
        print(f"  Échantillons de validation : {len(validation_data)} ({(1-train_ratio)*100:.0f}%)")

    # Créer le répertoire de sortie s'il n'existe pas
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Sauvegarder les ensembles de données
    train_path = os.path.join(output_dir, 'train_data.csv')
    validation_path = os.path.join(output_dir, 'validation_data.csv')

    if verbose:
        print(f"\n[4/4] Sauvegarde des ensembles de données")
    train_data.to_csv(train_path, index=False)
    validation_data.to_csv(validation_path, index=False)

    if verbose:
        print(f"  Données d'entraînement sauvegardées dans : {train_path}")
        print(f"  Données de validation sauvegardées dans : {validation_path}")
        print("\n" + "=" * 60)
        print("PRÉPARATION DES DONNÉES TERMINÉE !")
        print("=" * 60)

    return train_path, validation_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Prepare and split the breast cancer dataset for training.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input', type=str, default='data.csv',
                       help='Path to the input CSV file.')
    parser.add_argument('--output_dir', type=str, default='data',
                       help='Directory to save the split datasets.')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Ratio of data to use for training (0.0-1.0).')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility.')
    args = parser.parse_args()

    # Valider le ratio d'entraînement
    if not 0 < args.train_ratio < 1:
        print("Erreur : train_ratio doit être entre 0 et 1")
        exit(1)

    prepare_data(
        input_file=args.input,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        random_seed=args.seed,
        verbose=True
    )
