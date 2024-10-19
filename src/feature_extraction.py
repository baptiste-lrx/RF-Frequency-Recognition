# src/feature_extraction.py

import numpy as np
import pandas as pd
from scipy.fftpack import fft
import argparse

def extract_features(input_file='data/processed/signal_filtered.npy', output_file='data/features/features.csv', sample_rate=2.4e6):
    """
    Extrait des caractéristiques du signal RF.

    Parameters:
    - input_file (str): Chemin du fichier d'entrée.
    - output_file (str): Chemin du fichier de sortie.
    - sample_rate (float): Taux d'échantillonnage en Hz.
    """
    # Charger les données
    data = np.load(input_file)
    print(f"Données chargées depuis {input_file}")

    # Calculer la FFT
    N = len(data)
    yf = fft(data)
    xf = np.linspace(0.0, sample_rate/2.0, N//2)

    # Calculer le spectre de puissance
    power_spectrum = 2.0/N * np.abs(yf[:N//2])

    # Extraire des caractéristiques statistiques
    features = {
        'mean': np.mean(power_spectrum),
        'std': np.std(power_spectrum),
        'max': np.max(power_spectrum),
        'min': np.min(power_spectrum),
        'median': np.median(power_spectrum)
    }

    # Convertir en DataFrame
    df_features = pd.DataFrame([features])

    # Sauvegarder les caractéristiques
    df_features.to_csv(output_file, index=False)
    print(f"Caractéristiques sauvegardées dans {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extraction des caractéristiques du signal RF.')
    parser.add_argument('--input', type=str, default='data/processed/signal_filtered.npy', help='Fichier d\'entrée')
    parser.add_argument('--output', type=str, default='data/features/features.csv', help='Fichier de sortie')
    parser.add_argument('--sample_rate', type=float, default=2.4e6, help='Taux d\'échantillonnage en Hz')

    args = parser.parse_args()

    extract_features(input_file=args.input, output_file=args.output, sample_rate=args.sample_rate)
