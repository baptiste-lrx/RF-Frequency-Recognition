# src/data_acquisition.py

import numpy as np
from rtlsdr import RtlSdr
import argparse

def capture_signal(output_file='data/raw/signal.npy', duration=5, sample_rate=2.4e6, center_freq=100e6, direct_sampling=0, offset_tuning=False):
    """
    Capture des signaux RF à l'aide d'un RTL-SDR.

    Parameters:
    - output_file (str): Chemin du fichier de sortie.
    - duration (int): Durée de la capture en secondes.
    - sample_rate (float): Taux d'échantillonnage en Hz.
    - center_freq (float): Fréquence centrale en Hz.
    - direct_sampling (int): Mode de réception directe (0: désactivé, 1: voie I, 2: voie Q).
    - offset_tuning (bool): Activer le décalage de fréquence (True/False).
    """
    sdr = RtlSdr()

    # Configuration du SDR
    sdr.sample_rate = sample_rate
    sdr.center_freq = center_freq
    sdr.gain = 'auto'
    sdr.direct_sampling = direct_sampling
    sdr.offset_tuning = offset_tuning

    num_samples = int(duration * sample_rate)

    print("Début de la capture...")
    samples = sdr.read_samples(num_samples)
    print("Capture terminée.")

    # Sauvegarder les échantillons
    np.save(output_file, samples)
    print(f"Signaux sauvegardés dans {output_file}")

    # Fermer le SDR
    sdr.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Capture des signaux RF avec RTL-SDR.')
    parser.add_argument('--output', type=str, default='data/raw/signal.npy', help='Fichier de sortie')
    parser.add_argument('--duration', type=int, default=5, help='Durée de la capture en secondes')
    parser.add_argument('--sample_rate', type=float, default=2.4e6, help='Taux d\'échantillonnage en Hz')
    parser.add_argument('--freq', type=float, default=100e6, help='Fréquence centrale en Hz')
    parser.add_argument('--direct_sampling', type=int, default=0, choices=[0,1,2], help='Mode de réception directe (0: désactivé, 1: voie I, 2: voie Q)')
    parser.add_argument('--offset_tuning', action='store_true', help='Activer le décalage de fréquence')

    args = parser.parse_args()

    capture_signal(
        output_file=args.output,
        duration=args.duration,
        sample_rate=args.sample_rate,
        center_freq=args.freq,
        direct_sampling=args.direct_sampling,
        offset_tuning=args.offset_tuning
    )
