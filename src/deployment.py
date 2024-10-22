# src/deployment.py

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from feature_extraction import extract_features
import os

app = Flask(__name__)

# Charger le modèle
model = joblib.load('models/rf_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    # Vérifier si un fichier a été envoyé
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier fourni'}), 400

    file = request.files['file']

    # Charger le signal depuis le fichier envoyé
    signal = np.load(file)

    # Sauvegarder temporairement le signal pour extraire les caractéristiques
    temp_input_file = 'temp_signal.npy'
    np.save(temp_input_file, signal)

    # Extraire les caractéristiques du signal
    temp_features_file = 'temp_features.csv'
    extract_features(input_file=temp_input_file, output_file=temp_features_file, sample_rate=2.4e6)

    # Charger les caractéristiques
    df_features = pd.read_csv(temp_features_file)
    df_features = df_features.drop('label', axis=1, errors='ignore')

    # Prédire avec le modèle
    prediction = model.predict(df_features)

    # Nettoyer les fichiers temporaires
    os.remove(temp_input_file)
    os.remove(temp_features_file)

    # Retourner la prédiction
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
