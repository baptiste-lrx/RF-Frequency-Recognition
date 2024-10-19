# src/deployment.py

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from feature_extraction import extract_features

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

    # Extraire les caractéristiques du signal
    features = {
        'mean': np.mean(signal),
        'std': np.std(signal),
        'max': np.max(signal),
        'min': np.min(signal),
        'median': np.median(signal)
    }

    # Convertir en DataFrame
    df_features = pd.DataFrame([features])

    # Prédire avec le modèle
    prediction = model.predict(df_features)

    # Retourner la prédiction
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
