#!/bin/bash

echo "========================================"
echo "Multilayer Perceptron - Pipeline complète"
echo "========================================"

# Step 1: Prepare data
echo ""
echo "Étape 1 : Préparation des données..."
python prepare_data.py --input data.csv --output_dir data --train_ratio 0.8 --seed 42

# Check if data preparation was successful
if [ $? -ne 0 ]; then
    echo "Erreur : La préparation des données a échoué !"
    exit 1
fi

# Step 2: Train the model
echo ""
echo "Étape 2 : Entraînement du modèle..."
python train.py \
    --train_data data/train_data.csv \
    --validation_data data/validation_data.csv \
    --layers 24 24 \
    --epochs 100 \
    --patience 15 \
    --learning_rate 0.001 \
    --weight_init xavier \
    --model_output model.json

# Check if training was successful
if [ $? -ne 0 ]; then
    echo "Erreur : L'entraînement du modèle a échoué !"
    exit 1
fi

# Step 3: Make predictions
echo ""
echo "Étape 3 : Prédictions..."
python predict.py \
    --model model.json \
    --prediction_data data/validation_data.csv \
    --show_predictions

echo ""
echo "========================================"
echo "Pipeline terminée avec succès !"
echo "========================================"
