import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rd
from pathlib import Path


AUDIO_DIR = Path("Ressources")

n_fft = 1024
hop_length = 64
filename = "Ressources/ValabFr-20221128_133055.735.flac"

detections_csv = "Ressources/detections.csv"

df = pd.read_csv(detections_csv)

target = str(Path(filename))
df["filename"] = df["filename"].apply(lambda f: str(Path("Ressources") / Path(f).name))

result = df[df["filename"] == target]

if result.empty:
    raise ValueError(f"Aucune détection pour {target}")

row = result.iloc[0]


#Question 1 

def load_signal(filename):
    signal, frequence_echantillonage = librosa.load(filename, sr=None)
    # frequence_echantillonage = librosa.get_samplerate(filename)
    # print("Fréquence d'échantillonage : ",frequence_echantillonage, "Hertz")
    return signal, frequence_echantillonage

def get_spectro(signal, type):
    return np.abs(librosa.stft(signal, window=type))

def show_spectro(sr, spectre):
    spectre_db = librosa.amplitude_to_db(spectre)

    plt.figure(figsize=(12, 6))

    librosa.display.specshow(
        spectre_db,
        sr=sr,
        y_axis="log",
        x_axis="time",
    )

    plt.title("Spectrogramme STFT")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.show()


signal, frequence_echantillonage = load_signal(filename)
spectre = get_spectro(signal, "hann")


def load_signal(filename, offset=0, duration=10):
    signal, frequence_echantillonage = librosa.load(filename, offset=offset, duration=duration, sr=None)
    return signal, frequence_echantillonage

def get_spectro(signal, frequence_echantillonage, fmin, fmax, n_fft, hop_length, type):
    signal = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, window=type))
    tab_freq = librosa.fft_frequencies(sr=frequence_echantillonage, n_fft=n_fft)

    indice_min = np.searchsorted(tab_freq, fmin, 'left')
    indice_max = np.searchsorted(tab_freq, fmax, 'right')

    spectre = signal[indice_min:indice_max, :]

    return spectre

def show_spectro(sr, spectre):
    spectre_db = librosa.amplitude_to_db(spectre)

    plt.figure(figsize=(12, 6))

    librosa.display.specshow(
        spectre_db,
        sr=sr,
        x_axis="time",
        y_axis="hz"
    )

    plt.title("Spectrogramme STFT")
    plt.tight_layout()
    plt.show()

nv_signal, frequence_echantillonage = load_signal(filename)
nv_spectre = get_spectro(nv_signal,frequence_echantillonage, 100, 150, n_fft, hop_length, 'hann')


offset = row.iloc[2]
start = row.iloc[3]
fmin = row.iloc[4] - 5
fmax = row.iloc[5] + 5
duration = start - offset

nv_signal, frequence_echantillonage = load_signal(
    filename=target,
    offset=offset,
    duration=duration
)

nv_spectre = get_spectro(
    nv_signal,
    frequence_echantillonage,
    fmin,
    fmax,
    1024,
    hop_length,
    "hann"
)

N_FFT = 1024
HOP_LENGTH = 64
WINDOW_DURATION = 2.0
FMIN = 120
FMAX = 130
WINDOW = "hann"

def shift_signal(signal, pulse_offset, pulse_duration, window_duration, fs):
    pulse_samples = int(pulse_duration * fs)
    window_samples = int(window_duration * fs)
    signal_samples = len(signal)

    max_start_in_window = window_samples - pulse_samples
    pulse_pos_in_window = rd.randint(0, max_start_in_window)

    window_start = int(pulse_offset * fs) - pulse_pos_in_window
    window_start = max(0, min(signal_samples - window_samples, window_start))

    return signal[window_start : window_start + window_samples]
    

def get_pulses(target, n_fft, hop_length, window_duration):
    df = pd.read_csv(detections_csv)
    result = df[df["filename"] == target]

    all_pulses = []

    signal_complet, freq_complet = load_signal(filename=target, offset=0, duration=None)

    for _, row in result.iterrows():
        offset = row["Begin Time (s)"]
        start = row["End Time (s)"]
        fmin = row["Low Freq (Hz)"] - 5
        fmax = row["High Freq (Hz)"] + 5
        duration = start - offset

        nv_signal = shift_signal(
            signal=signal_complet,
            pulse_offset=offset,
            pulse_duration=duration,
            window_duration=window_duration,
            fs=freq_complet
        )

        nv_spectre = get_spectro(
            nv_signal,
            freq_complet,
            fmin,
            fmax,
            n_fft,
            hop_length,
            "hann"
        )

        all_pulses.append(nv_spectre)
    return all_pulses

def get_spectros_pos_one_file(filename, df, window_duration=WINDOW_DURATION, n_fft=N_FFT, hop_length=HOP_LENGTH):
    result = df[df["filename"] == filename]
    if result.empty:
        return np.empty((0, 0, 0))

    signal, sr = load_signal(filename, offset=0, duration=None)
    spectros = []

    for _, row in result.iterrows():
        offset = row["Begin Time (s)"]
        duration = row["End Time (s)"] - offset

        nv_signal = shift_signal(signal, offset, duration, window_duration, sr)
        nv_spectre = get_spectro(nv_signal, sr, FMIN, FMAX, n_fft, hop_length, WINDOW)

        spectros.append(nv_spectre)

    return np.stack(spectros, axis=0)


def get_all_spectros_pos(df, window_duration=WINDOW_DURATION, n_fft=N_FFT, hop_length=HOP_LENGTH):
    all_spectros = []

    for filename in df["filename"].unique():
        spectros = get_spectros_pos_one_file(
            filename, df, window_duration, n_fft, hop_length
        )
        if spectros.size > 0:
            all_spectros.append(spectros)

    return np.concatenate(all_spectros, axis=0)


def generate_negative(filename, df, window_duration=WINDOW_DURATION, n_fft=N_FFT, hop_length=HOP_LENGTH):
    signal, sr = load_signal(filename, offset=0, duration=None)

    window_samples = int(window_duration * sr)
    signal_samples = len(signal)

    max_start = signal_samples - window_samples
    if max_start <= 0:
        raise ValueError("Signal trop court")

    annotations = df[df["filename"] == filename][["Begin Time (s)", "End Time (s)"]].to_numpy()
    
    while True:
        start_sample = np.random.randint(0, max_start + 1)
        start_time = start_sample / sr
        end_time = start_time + window_duration

        valid = True
        for begin, end in annotations:
            if not (end_time <= begin or start_time >= end):
                valid = False
                break

        if not valid:
            continue

        window = signal[start_sample:start_sample + window_samples]

        spectro = get_spectro(window,sr,FMIN,FMAX, n_fft,hop_length, WINDOW)
        
        return spectro

def get_spectros_neg_one_file(filename, df, N,
                              window_duration=WINDOW_DURATION,
                              fmin=FMIN, fmax=FMAX,
                              n_fft=N_FFT, hop_length=HOP_LENGTH):
    spectros = []
    for _ in range(N):
        spectro = generate_negative(filename, df,
                                    window_duration=window_duration,
                                    n_fft=n_fft,
                                    hop_length=hop_length)
        if spectro is not None:
            spectros.append(spectro)

    if spectros:
        return np.stack(spectros, axis=0)
    else:
        return np.empty((0,0,0))


def get_all_spectros_neg(df,
                         window_duration=WINDOW_DURATION,
                         fmin=FMIN, fmax=FMAX,
                         n_fft=N_FFT, hop_length=HOP_LENGTH):
    all_spectros_neg = []

    for filename in df["filename"].unique():
        n_pos = len(df[df["filename"] == filename])
        spectros_neg = get_spectros_neg_one_file(filename, df, n_pos,
                                                 window_duration,
                                                 fmin, fmax,
                                                 n_fft, hop_length)
        if spectros_neg.size > 0:
            all_spectros_neg.append(spectros_neg)

    if all_spectros_neg:
        return np.concatenate(all_spectros_neg, axis=0)
    else:
        return np.empty((0,0,0))

def generate_dataset(spectros_pos, spectros_neg, output_dir='./'):
    if spectros_pos.size == 0 or spectros_neg.size == 0:
        raise ValueError("Empty positive or negative spectrograms")
    
    X = np.concatenate([spectros_pos, spectros_neg], axis=0)
    Y = np.concatenate([
        np.ones(len(spectros_pos), dtype=np.int32),
        np.zeros(len(spectros_neg), dtype=np.int32)
    ])
    
    permutation = np.random.permutation(len(X))
    X = X[permutation]
    Y = Y[permutation]
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    np.save(output_path / 'x.npy', X)
    np.save(output_path / 'y.npy', Y)
    
    print(f"Dataset généré : X={X.shape}, Y={Y.shape}")
    print(f"Positifs: {Y.sum()}, Négatifs: {len(Y) - Y.sum()}")
    
    return X, Y

# spectros_pos = get_all_spectros_pos(df)
# spectros_neg = get_all_spectros_neg(df)

# X, Y = generate_dataset(spectros_pos, spectros_neg)

def get_spectros_neg_one_file(filename, df, N,
                              window_duration=WINDOW_DURATION,
                              fmin=FMIN, fmax=FMAX,
                              n_fft=N_FFT, hop_length=HOP_LENGTH):
    spectros = []
    for _ in range(N):
        spectro = generate_negative(filename, df,
                                    window_duration=window_duration,
                                    n_fft=n_fft,
                                    hop_length=hop_length)
        if spectro is not None:
            spectros.append(spectro)

    if spectros:
        return np.stack(spectros, axis=0)
    else:
        return np.empty((0,0,0))

def get_all_spectros_neg(df, K_shifts,
                         window_duration=WINDOW_DURATION,
                         fmin=FMIN, fmax=FMAX,
                         n_fft=N_FFT, hop_length=HOP_LENGTH):
    all_spectros_neg = []

    for filename in df["filename"].unique():
        n_pos = len(df[df["filename"] == filename])
        
        N_samples = n_pos * K_shifts
        
        spectros_neg = get_spectros_neg_one_file(
            filename, df, N_samples,
            window_duration,
            fmin, fmax,
            n_fft, hop_length
        )
        if spectros_neg.size > 0:
            all_spectros_neg.append(spectros_neg)

    if all_spectros_neg:
        return np.concatenate(all_spectros_neg, axis=0)
    else:
        return np.empty((0,0,0))


def get_spectros_pos_one_file(filename, df, K_shifts, window_duration=WINDOW_DURATION, n_fft=N_FFT, hop_length=HOP_LENGTH):
    result = df[df["filename"] == filename]
    if result.empty:
        return np.empty((0, 0, 0))

    signal, sr = load_signal(filename, offset=0, duration=None)
    spectros = []

    for _, row in result.iterrows():
        offset = row["Begin Time (s)"]
        duration = row["End Time (s)"] - offset


        for _ in range(K_shifts):
            nv_signal = shift_signal(signal, offset, duration, window_duration, sr)
            
            nv_spectre = get_spectro(nv_signal, sr, FMIN, FMAX, n_fft, hop_length, WINDOW)
            spectros.append(nv_spectre)

    return np.stack(spectros, axis=0)


def get_all_spectros_pos(df, K_shifts, window_duration=WINDOW_DURATION, n_fft=N_FFT, hop_length=HOP_LENGTH):
    all_spectros = []

    for filename in df["filename"].unique():
        spectros = get_spectros_pos_one_file(
            filename, df, K_shifts, window_duration, n_fft, hop_length
        )
        if spectros.size > 0:
            all_spectros.append(spectros)

    return np.concatenate(all_spectros, axis=0)

K_SHIFTS = 3

spectros_pos = get_all_spectros_pos(df,K_SHIFTS)
spectros_neg = get_all_spectros_neg(df,K_SHIFTS)

X, Y = generate_dataset(spectros_pos, spectros_neg)