import os
import pandas as pd
import numpy as np
from scipy.fft import fft

RAW_DIR   = "dataset/raw"
OUT_FILE  = "dataset/processed/processed_dataset.csv"
WINDOW    = 100   # samples per window


def time_features(sig):
    return {
        "mean": np.mean(sig),
        "std":  np.std(sig),
        "rms":  np.sqrt(np.mean(sig**2)),
        "max":  np.max(sig),
        "min":  np.min(sig)
    }

def freq_features(sig, n=5):
    fft_vals = np.abs(fft(sig))
    return {f"fft_{i+1}": fft_vals[i+1] for i in range(n)}

def extract_window_features(df_win, numeric_cols):
    feats = {}
    for col in numeric_cols:
        sig = df_win[col].to_numpy()
        tf = time_features(sig)
        ff = freq_features(sig)
        feats.update({f"{col}_{k}": v for k, v in tf.items()})
        feats.update({f"{col}_{k}": v for k, v in ff.items()})
    return feats

def process_file(path):
    df = pd.read_csv(path)
    if 'fault' not in df.columns:
        print(f"⚠️ Skipping {path}: no 'fault' column found.")
        return None

    # Identify numeric sensor columns
    numeric_cols = [c for c in df.columns 
                    if c.lower().startswith('encoder') 
                    or c.lower().startswith('angular') 
                    or c.lower().startswith('linear')]
    if not numeric_cols:
        print(f"⚠️ Skipping {path}: no sensor columns found.")
        return None

    records = []
    label = df['fault'].iloc[0]
    for start in range(0, len(df) - WINDOW + 1, WINDOW):
        win = df.iloc[start : start + WINDOW]
        feats = extract_window_features(win, numeric_cols)
        feats['label'] = label
        records.append(feats)

    return pd.DataFrame(records)

def main():
    all_feats = []
    for fname in os.listdir(RAW_DIR):
        if not fname.lower().endswith('.csv'):
            continue
        path = os.path.join(RAW_DIR, fname)
        print("Processing", fname)
        df_feats = process_file(path)
        if df_feats is not None:
            all_feats.append(df_feats)

    if not all_feats:
        print("❌ No valid files processed. Check your RAW_DIR contents.")
        return

    result = pd.concat(all_feats, ignore_index=True)
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    result.to_csv(OUT_FILE, index=False)
    print(f"\n✅ Saved processed features to {OUT_FILE}")

if __name__ == "__main__":
    main()
