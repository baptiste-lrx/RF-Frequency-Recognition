# src/preprocessing.py

import numpy as np
from scipy.signal import butter, lfilter
import argparse

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    y = lfilter(b, a, data)
    return y

def preprocess_signal(input_file='data/raw/signal.npy', output_file='data/processed/signal_filtered.npy', sample_rate=2.4e6, lowcut=85e6, highcut=115e6):
    """
    Filtre le signal RF pour réduire le bruit.

    Parameters:
    - input_file (str): Chemin du fichier d'entrée.
    - output_file (str): Chemin du fichier de sortie.
    - sample_rate (float): Taux d'échantillonnage en Hz.
    - lowcut (float): Fréquence basse du filtre passe-bande en Hz.
    - highcut (float): Fréquence haute du filtre passe-bande en Hz.
    """
    # Charger les données
    data = np.load(input_file)
    print(f"Données chargées depuis {input_file}")

    # Appliquer le filtre passe-bande
    filtered_data = bandpass_filter(data, lowcut, highcut, sample_rate)
    print("Filtrage terminé.")

    # Sauvegarder les données filtrées
    np.save(output_file, filtered_data)
    print(f"Données filtrées sauvegardées dans {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prétraitement du signal RF.')
    parser.add_argument('--input', type=str, default='data/raw/signal.npy', help='Fichier d\'entrée')
    parser.add_argument('--output', type=str, default='data/processed/signal_filtered.npy', help='Fichier de sortie')
    parser.add_argument('--sample_rate', type=float, default=2.4e6, help='Taux d\'échantillonnage en Hz')
    parser.add_argument('--lowcut', type=float, default=85e6, help='Fréquence basse du filtre en Hz')
    parser.add_argument('--highcut', type=float, default=115e6, help='Fréquence haute du filtre en Hz')

    args = parser.parse_args()

    preprocess_signal(
        input_file=args.input,
        output_file=args.output,
        sample_rate=args.sample_rate,
        lowcut=args.lowcut,
        highcut=args.highcut
    )
