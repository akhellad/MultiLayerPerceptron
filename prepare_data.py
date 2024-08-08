import pandas as pd
import numpy as np
import os

file_path = 'data.csv'
data = pd.read_csv(file_path)

column_names = [
    'ID', 'Diagnosis', 'Radius_mean', 'Texture_mean', 'Perimeter_mean', 'Area_mean', 'Smoothness_mean', 'Compactness_mean', 
    'Concavity_mean', 'Concave_points_mean', 'Symmetry_mean', 'Fractal_dimension_mean', 'Radius_se', 'Texture_se', 
    'Perimeter_se', 'Area_se', 'Smoothness_se', 'Compactness_se', 'Concavity_se', 'Concave_points_se', 'Symmetry_se', 
    'Fractal_dimension_se', 'Radius_worst', 'Texture_worst', 'Perimeter_worst', 'Area_worst', 'Smoothness_worst', 
    'Compactness_worst', 'Concavity_worst', 'Concave_points_worst', 'Symmetry_worst', 'Fractal_dimension_worst'
]
data.columns = column_names

data = data.drop(columns=['ID'])

data = data.sample(frac=1, random_state=42).reset_index(drop=True)

train_size = int(0.8 * len(data))

train_data = data.iloc[:train_size]
validation_data = data.iloc[train_size:]

output_dir = 'data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

train_data.to_csv(os.path.join(output_dir, 'train_data.csv'), index=False)
validation_data.to_csv(os.path.join(output_dir, 'validation_data.csv'), index=False)

print("Les données ont été divisées avec succès.")
