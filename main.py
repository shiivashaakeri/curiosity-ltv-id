#!/usr/bin/env python3
import argparse
import os
from src.ltv_id import process_channel

def main():
    parser = argparse.ArgumentParser(
        description="Run sliding‐window ARX on one Curiosity channel."
    )
    parser.add_argument("chan",    help="Channel ID (e.g. M-6)")
    parser.add_argument("y_chan",  type=int, help="Output channel index (0–54)")
    parser.add_argument("--split", default="test", help="train or test split")
    args = parser.parse_args()

    # project root
    ROOT = os.path.dirname(__file__)
    RAW_BASE = os.path.join(ROOT, "data", "raw", "data", "data")
    LABELS  = os.path.join(ROOT, "data", "raw", "labeled_anomalies.csv")
    OUT_DIR = os.path.join(ROOT, "data", "processed")

    # run the pipeline
    theta_df, rmse_df = process_channel(
        chan=args.chan,
        y_chan=args.y_chan,
        spacecraft="MSL",
        split=args.split,
        raw_base=RAW_BASE,
        labels_csv=LABELS,
        out_dir=OUT_DIR
    )

    print(f"Done: θ shape {theta_df.shape}, RMSE shape {rmse_df.shape}")

if __name__ == "__main__":
    main()