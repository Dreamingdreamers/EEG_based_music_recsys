import numpy as np
from scipy.signal import welch
import pickle
import os

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))),
    'data', 'deap-dataset', 'data_preprocessed_python'
)

FS = 128
BANDS = {
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
    'gamma': (30, 45)
}

FRONTAL_LEFT  = [0, 1, 3]
FRONTAL_RIGHT = [16, 26, 19]

def get_band_power(signal, fs=FS):
    freqs, psd = welch(signal, fs=fs, nperseg=256)
    powers = []
    for band, (low, high) in BANDS.items():
        idx = np.where((freqs >= low) & (freqs <= high))
        powers.append(float(np.mean(psd[idx])))
    return powers

def get_differential_entropy(signal, fs=FS):
    freqs, psd = welch(signal, fs=fs, nperseg=256)
    de = []
    for band, (low, high) in BANDS.items():
        idx = np.where((freqs >= low) & (freqs <= high))
        power = float(np.mean(psd[idx]))
        power = max(power, 1e-10)
        de.append(np.log(power))
    return de

def get_statistical_features(signal):
    return [
        float(np.mean(signal)),
        float(np.var(signal)),
        float(np.std(signal))
    ]

def get_frontal_asymmetry(trial_data):
    asymmetry = []
    for l_ch, r_ch in zip(FRONTAL_LEFT, FRONTAL_RIGHT):
        left_alpha  = max(get_band_power(trial_data[l_ch])[1], 1e-10)
        right_alpha = max(get_band_power(trial_data[r_ch])[1], 1e-10)
        faa = np.log(right_alpha) - np.log(left_alpha)
        asymmetry.append(faa)
    return asymmetry

def extract_features(trial_data):
    features = []
    for ch in range(32):
        signal = trial_data[ch]
        bp = get_band_power(signal)           
        de = get_differential_entropy(signal) 
        st = get_statistical_features(signal) 
        features.extend(bp + de + st)         

    faa = get_frontal_asymmetry(trial_data)
    features.extend(faa)                      

    return features  


def get_emotion_label(valence, arousal,
                      v_threshold=4.5, a_threshold=4.5):
    v = 1 if valence > v_threshold else 0
    a = 1 if arousal > a_threshold else 0
    emotion_map = {
        (1, 1): "Happy",
        (1, 0): "Calm",
        (0, 0): "Sad",
        (0, 1): "Stressed"
    }
    return emotion_map[(v, a)]


def create_windows(signal, fs=FS,
                   window_sec=8, overlap=0.5):
    window_samples = int(window_sec * fs)  
    step = int(window_samples * (1 - overlap))  
    windows = []
    start = 0
    while start + window_samples <= signal.shape[1]:
        window = signal[:, start:start + window_samples]
        windows.append(window)
        start += step
    return windows

def balance_undersample(X, y):
    """
    Randomly remove samples from majority classes
    until all classes have equal samples.
    Fastest method — no synthetic data.
    """
    from collections import Counter
    
    counts    = Counter(y)
    min_count = min(counts.values())
    
    print(f"\nUndersampling to {min_count} per class...")
    
    X_balanced, y_balanced = [], []
    
    for emotion in counts.keys():
        # Get indices of this emotion
        indices = np.where(y == emotion)[0]
        # Randomly select min_count of them
        selected = np.random.choice(
            indices, size=min_count, replace=False)
        X_balanced.append(X[selected])
        y_balanced.extend([emotion] * min_count)
    
    X_balanced = np.vstack(X_balanced)
    y_balanced = np.array(y_balanced)
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(y_balanced))
    X_balanced  = X_balanced[shuffle_idx]
    y_balanced  = y_balanced[shuffle_idx]
    
    print(f"Balanced distribution: "
          f"{dict(zip(*np.unique(y_balanced, return_counts=True)))}")
    print(f"Total samples: {len(y_balanced)}")
    
    return X_balanced, y_balanced

def load_all_subjects(balance=True):
    X, y = [], []
    print("Loading DEAP dataset with windowing...")

    for i in range(1, 33):
        filepath = os.path.join(DATA_PATH, f's{i:02d}.dat')
        with open(filepath, 'rb') as f:
            subject = pickle.load(f, encoding='latin1')

        data   = subject['data'][:, :32, :]
        labels = subject['labels']

        v_threshold = np.median(labels[:, 0])
        a_threshold = np.median(labels[:, 1])

        for trial in range(40):
            trial_data = data[trial] 
            emotion = get_emotion_label(
                labels[trial, 0],
                labels[trial, 1],
                v_threshold,
                a_threshold
            )

            windows = create_windows(trial_data)
            for window in windows:
                features = extract_features(window)
                X.append(features)
                y.append(emotion)

        print(f"  Loaded subject {i:02d}/32", end='\r')

    X = np.array(X)
    y = np.array(y)
    print(f"\nDone! X shape: {X.shape}, y shape: {y.shape}")
    print(f"Emotion distribution: "
          f"{dict(zip(*np.unique(y, return_counts=True)))}")
    X, y = balance_undersample(X, y)
    return X, y


if __name__ == "__main__":
    X, y = load_all_subjects()
    print(f"Feature vector length: {X.shape[1]}")
    print(f"Total samples: {X.shape[0]}")
