import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_channel_results(
    chan: str,
    win_size: int = 50,
    hop: int    = 10,
    out_dir: str= "data/processed"
):
    """
    Load θ‐ and rmse‐CSVs for `chan` and plot:
      • the four ARX coefficients over time
      • the windowed RMSE over time
    """
    # 1) load
    theta_df = pd.read_csv(os.path.join(out_dir, f"{chan}_thetas.csv"))
    rmse     = pd.read_csv(os.path.join(out_dir, f"{chan}_rmses.csv"))["rmse"].values

    # 2) compute window centers
    centers = np.arange(len(theta_df)) * hop + win_size/2

    # 3) plot coefficients
    plt.figure(figsize=(9,5))
    for i, col in enumerate(theta_df.columns):
        plt.plot(centers, theta_df[col], marker='o', label=col)
    plt.title(f"{chan} ARX Coefficients")
    plt.xlabel("Sample Index")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 4) plot RMSE
    plt.figure(figsize=(9,3))
    plt.plot(centers, rmse, '-o')
    plt.title(f"{chan} Prediction RMSE")
    plt.xlabel("Sample Index")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.show()