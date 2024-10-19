# src/evaluation.py

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
import argparse

def evaluate_model(features_file='data/features/features.csv', model_file='models/rf_model.joblib'):
    """
    Évalue le modèle de classification des signaux RF.

    Parameters:
    - features_file (str): Chemin du fichier des caractéristiques.
    - model_file (str): Chemin du fichier du modèle entraîné.
    """
    # Charger les caractéristiques
    df = pd.read_csv(features_file)
    print(f"Caractéristiques chargées depuis {features_file}")

    # Pour cet exemple, nous allons générer des labels aléatoires
    df['label'] = 0  # Remplacez par vos labels réels

    # Séparer les features et les labels
    X = df.drop('label', axis=1)
    y = df['label']

    # Diviser en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Charger le modèle
    clf = joblib.load(model_file)
    print(f"Modèle chargé depuis {model_file}")

    # Prédire sur l'ensemble de test
    y_pred = clf.predict(X_test)

    # Afficher le rapport de classification
    print("Rapport de classification :")
    print(classification_report(y_test, y_pred))

    # Afficher la matrice de confusion
    print("Matrice de confusion :")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Évaluation du modèle de classification.')
    parser.add_argument('--features', type=str, default='data/features/features.csv', help='Fichier des caractéristiques')
    parser.add_argument('--model', type=str, default='models/rf_model.joblib', help='Fichier du modèle entraîné')

    args = parser.parse_args()

    evaluate_model(features_file=args.features, model_file=args.model)
