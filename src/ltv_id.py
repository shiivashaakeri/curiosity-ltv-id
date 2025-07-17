import numpy as np

def fit_arx(u: np.ndarray,
            y: np.ndarray,
            na: int,
            nb: int,
            nk: int) -> np.ndarray:
    """
    Fit an ARX(na,nb,nk) model y[k] + a1 y[k-1] + ... + ana y[k-na]
                         = b1 u[k-nk] + ... + bnb u[k-nk-nb+1]
    Returns θ = [–a1, …, –ana, b1, …, bnb].
    """
    N = len(y)
    max_lag = max(na, nb + nk - 1)
    Phi, Y = [], []
    for k in range(max_lag, N):
        # output lags
        row = [-y[k - i] for i in range(1, na + 1)]
        # input lags
        row += [u[k - j] for j in range(nk, nb + nk)]
        Phi.append(row)
        Y.append(y[k])
    Φ = np.vstack(Phi)
    Y = np.array(Y)
    θ, *_ = np.linalg.lstsq(Φ, Y, rcond=None)
    return θ

def sliding_arx(u: np.ndarray,
                y: np.ndarray,
                na: int,
                nb: int,
                nk: int,
                win_size: int,
                hop: int):
    """
    Slide an ARX(na,nb,nk) fit across u,y.
    Returns:
      thetas: array of shape (n_windows, na+nb)
      rmses:  list of length n_windows
    """
    max_lag = max(na, nb + nk - 1)
    thetas, rmses = [], []

    for start in range(0, len(u) - win_size + 1, hop):
        u_win = u[start : start + win_size]
        y_win = y[start : start + win_size]
        θ    = fit_arx(u_win, y_win, na, nb, nk)
        thetas.append(θ)

        # one‐step‐ahead predictions
        y_pred = np.zeros_like(y_win)
        for k in range(max_lag, len(y_win)):
            y_pred[k] = (
                -θ[0] * y_win[k-1]
                -θ[1] * y_win[k-2]
                +θ[2] * u_win[k-1]
                +θ[3] * u_win[k-2]
            )
        err = y_win[max_lag:] - y_pred[max_lag:]
        rmses.append(np.sqrt(np.mean(err**2)))

    return np.vstack(thetas), rmses

import os
import pandas as pd
from src.io_utils import load_channel_data, get_anomaly_windows

def process_channel(
    chan: str,
    y_chan: int,
    spacecraft: str = "MSL",
    split: str = "test",
    na: int = 2,
    nb: int = 2,
    nk: int = 1,
    win_size: int = 50,
    hop: int = 10,
    raw_base: str = "data/raw/data/data",
    labels_csv: str = "data/raw/labeled_anomalies.csv",
    out_dir: str = "data/processed"
):
    """
    Load the full-channel data, run sliding ARX, and save both θ and RMSE CSVs.
    """
    # 1) load
    data = load_channel_data(chan, split=split, base_dir=raw_base)
    u_full = data[:, 0]
    y_full = data[:, y_chan]

    # 2) compute sliding‐window ARX
    thetas, rmses = sliding_arx(u_full, y_full, na, nb, nk, win_size, hop)

    # 3) save outputs
    os.makedirs(out_dir, exist_ok=True)
    theta_df = pd.DataFrame(
        thetas,
        columns=[f"-a{i+1}" for i in range(na)] + [f"b{j+1}" for j in range(nb)]
    )
    theta_df.to_csv(os.path.join(out_dir, f"{chan}_thetas.csv"), index=False)
    rmse_df = pd.DataFrame({'rmse': rmses})
    rmse_df.to_csv(os.path.join(out_dir, f"{chan}_rmses.csv"), index=False)

    return theta_df, rmse_df