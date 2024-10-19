# src/model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import argparse

def train_model(features_file='data/features/features.csv', model_file='models/rf_model.joblib'):
    """
    Entraîne un modèle de classification des signaux RF.

    Parameters:
    - features_file (str): Chemin du fichier des caractéristiques.
    - model_file (str): Chemin du fichier du modèle entraîné.
    """
    # Charger les caractéristiques
    df = pd.read_csv(features_file)
    print(f"Caractéristiques chargées depuis {features_file}")

    # Pour cet exemple, nous allons générer des labels aléatoires
    # Dans un vrai cas, vous auriez des labels correspondant aux types de signaux
    df['label'] = 0  # Remplacez par vos labels réels

    # Séparer les features et les labels
    X = df.drop('label', axis=1)
    y = df['label']

    # Diviser en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Entraîner un modèle Random Forest
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    print("Entraînement du modèle terminé.")

    # Sauvegarder le modèle
    joblib.dump(clf, model_file)
    print(f"Modèle sauvegardé dans {model_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Entraînement du modèle de classification.')
    parser.add_argument('--features', type=str, default='data/features/features.csv', help='Fichier des caractéristiques')
    parser.add_argument('--model', type=str, default='models/rf_model.joblib', help='Fichier du modèle entraîné')

    args = parser.parse_args()

    train_model(features_file=args.features, model_file=args.model)
