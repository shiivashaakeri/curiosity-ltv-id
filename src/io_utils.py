import os
import numpy as np
import pandas as pd
import ast

def load_channel_data(chan: str, split: str = "test", 
                      base_dir: str = "data/raw/data/data") -> np.ndarray:
    """
    Load the raw telemetry array for a given channel and split (train/test).
    """
    fname = f"{chan}.npy"
    path = os.path.join(base_dir, split, fname)
    return np.load(path)

def get_anomaly_windows(chan: str, spacecraft: str = "MSL",
                        labels_csv: str = "data/raw/labeled_anomalies.csv") -> list:
    """
    Return a list of [start, end] windows for anomalies on a given channel.
    """
    df = pd.read_csv(labels_csv)
    entry = df[(df["spacecraft"] == spacecraft) & (df["chan_id"] == chan)]
    if entry.empty:
        return []
    # parse the string into a Python list
    return ast.literal_eval(entry["anomaly_sequences"].iloc[0])
