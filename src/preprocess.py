import numpy as np
from scipy.signal import welch
from collections import Counter
import pickle
import os

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))),
    'data', 'deap-dataset', 'data_preprocessed_python'
)

FS = 128

BANDS = {
    'theta': (4,  8),
    'alpha': (8,  13),
    'beta':  (13, 30),
    'gamma': (30, 45)
}

FRONTAL_LEFT  = [0, 1, 3]    # Fp1, AF3, F3
FRONTAL_RIGHT = [16, 26, 19] # Fp2, AF4, F4

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
        idx   = np.where((freqs >= low) & (freqs <= high))
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
        left_alpha  = max(
            get_band_power(trial_data[l_ch])[1], 1e-10)
        right_alpha = max(
            get_band_power(trial_data[r_ch])[1], 1e-10)
        faa = np.log(right_alpha) - np.log(left_alpha)
        asymmetry.append(faa)
    return asymmetry


def extract_features(trial_data):

    features = []
    for ch in range(32):
        signal = trial_data[ch]
        bp  = get_band_power(signal)           
        de  = get_differential_entropy(signal) 
        st  = get_statistical_features(signal)  
        features.extend(bp + de + st)         
    faa = get_frontal_asymmetry(trial_data)
    features.extend(faa)                      
    return features  

def get_emotion_label(valence, arousal,
                      v_threshold=4.5,
                      a_threshold=4.5):
    
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
    step           = int(window_samples * (1 - overlap))
    windows = []
    start   = 0
    while start + window_samples <= signal.shape[1]:
        window = signal[:, start:start + window_samples]
        windows.append(window)
        start += step
    return windows

def balance_undersample(X, y):
    
    counts    = Counter(y)
    min_count = min(counts.values())

    X_bal, y_bal = [], []
    for emotion in counts.keys():
        idx = np.where(y == emotion)[0]
        sel = np.random.choice(
            idx, size=min_count, replace=False)
        X_bal.append(X[sel])
        y_bal.extend([emotion] * min_count)

    X_bal = np.vstack(X_bal)
    y_bal = np.array(y_bal)
    shuf  = np.random.permutation(len(y_bal))
    return X_bal[shuf], y_bal[shuf]

def load_all_subjects(balance=None):
    X, y = [], []
    print("Loading DEAP dataset with windowing...")
    for i in range(1, 33):
        filepath = os.path.join(
            DATA_PATH, f's{i:02d}.dat')
        with open(filepath, 'rb') as f:
            subject = pickle.load(f, encoding='latin1')
        data   = subject['data'][:, :32, :]
        labels = subject['labels']
        v_threshold = np.median(labels[:, 0])
        a_threshold = np.median(labels[:, 1])
        for trial in range(40):
            trial_data = data[trial]
            emotion    = get_emotion_label(
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

    print(f"\nLoaded raw dataset:")
    print(f"  X shape:      {X.shape}")
    print(f"  Distribution: "
          f"{dict(zip(*np.unique(y, return_counts=True)))}")

    if balance == 'undersample':
        X, y = balance_undersample(X, y)
        print(f"\nAfter undersampling:")
        print(f"  X shape: {X.shape}")
        print(f"  Distribution: "
              f"{dict(zip(*np.unique(y, return_counts=True)))}")

    return X, y

CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))),
    'data', 'raw_features_cache.pkl'
)

def load_all_subjects_cached(force_reload=False):
    if os.path.exists(CACHE_PATH) and not force_reload:
        print("Loading from cache (instant)...")
        with open(CACHE_PATH, 'rb') as f:
            cache = pickle.load(f)
        X, y = cache['X'], cache['y']
        print(f"Loaded from cache: {X.shape}")
        return X, y

    print("No cache found, extracting features "
          "(~5 minutes, one-time only)...")
    X, y = load_all_subjects(balance=None)
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump({'X': X, 'y': y}, f)
    print(f"Cache saved → {CACHE_PATH}")
    print("Future runs will load instantly from this cache.")

    return X, y
RAW_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))),
    'data', 'raw_signal_cache.pkl'
)

def load_all_subjects_raw_signal(force_reload=False):
    if os.path.exists(RAW_CACHE_PATH) and not force_reload:
        print("Loading raw signal cache (instant)...")
        with open(RAW_CACHE_PATH, 'rb') as f:
            cache = pickle.load(f)
        X, y = cache['X'], cache['y']
        print(f"Loaded: {X.shape}")
        return X, y

    print("Extracting raw signal windows "
          "(~2 minutes, one-time only)...")
    X, y = [], []

    for i in range(1, 33):
        filepath = os.path.join(
            DATA_PATH, f's{i:02d}.dat')
        with open(filepath, 'rb') as f:
            subject = pickle.load(f, encoding='latin1')

        data   = subject['data'][:, :32, :]
        labels = subject['labels']

        v_threshold = np.median(labels[:, 0])
        a_threshold = np.median(labels[:, 1])

        for trial in range(40):
            trial_data = data[trial]
            emotion    = get_emotion_label(
                labels[trial, 0], labels[trial, 1],
                v_threshold, a_threshold
            )

            windows = create_windows(trial_data)
            for window in windows:
                X.append(window)   # raw (32, 1024)
                y.append(emotion)

        print(f"  Subject {i:02d}/32", end='\r')

    X = np.array(X)  # (N, 32, 1024)
    y = np.array(y)

    print(f"\nRaw signal dataset: {X.shape}")
    print(f"Distribution: "
          f"{dict(zip(*np.unique(y, return_counts=True)))}")

    with open(RAW_CACHE_PATH, 'wb') as f:
        pickle.dump({'X': X, 'y': y}, f)
    print(f"Cache saved → {RAW_CACHE_PATH}")

    return X, y

if __name__ == "__main__":
    print("=" * 55)
    print("ThetaPlay — Preprocessing Pipeline Test")
    print("=" * 55)

    print("\nTest 1: Feature extraction...")
    fake     = np.random.randn(32, 1024)
    features = extract_features(fake)
    assert len(features) == 355, \
        f"Expected 355, got {len(features)}"
    print(f"  Feature vector: {len(features)} ✅")

    print("\nTest 2: Emotion labels...")
    assert get_emotion_label(7, 7) == "Happy"
    assert get_emotion_label(7, 3) == "Calm"
    assert get_emotion_label(3, 3) == "Sad"
    assert get_emotion_label(3, 7) == "Stressed"
    print("  All labels correct ✅")

    print("\nTest 3: Windowing...")
    fake_trial = np.random.randn(32, 8064)
    windows    = create_windows(fake_trial)
    print(f"  Windows created: {len(windows)}")
    print(f"  Window shape:    {windows[0].shape} ✅")

    # Test 4 — Full pipeline with caching
    print("\nTest 4: Full pipeline (cached)...")
    X, y = load_all_subjects_cached()

    print("\nTest 5: Raw signal pipeline (for CNN/LSTM)...")
    X_raw_sig, y_raw_sig = load_all_subjects_raw_signal()

    print(f"  Raw signal shape: {X_raw_sig.shape}")
    print(f"  (samples, channels, timepoints)")
    print(f"\n{'─'*45}")
    print(f"Pipeline Summary:")
    print(f"  Feature vector length: {X.shape[1]}")
    print(f"  Total samples:         {X.shape[0]}")
    print(f"  Sample emotions:       {y[:5]}")
    print(f"{'─'*45}")
    print(f"\nPreprocessing pipeline complete ✅")
