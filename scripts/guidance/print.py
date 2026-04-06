import os
import sys
import pandas as pd
import numpy as np


if __name__ == "__main__":
    # csv_path = sys.argv[1]
    csv_dir = "csvs/cdr+fr"
    
    for csv_path in sorted(os.listdir(csv_dir)):
        if not csv_path.endswith(".csv"):
            continue
        csv_path = os.path.join(csv_dir, csv_path)
        try:
            print(f"Analyzing {csv_path}...")
            df = pd.read_csv(csv_path)

            if "foldx_energy" not in df.columns or "delta_fitness" not in df.columns:
                continue

            # ddg = df["ddG_pred"].values
            ddg = df["foldx_energy"].values
            ddf = df["delta_fitness"].values
            num_ddg_lt_0 = np.sum(ddg < 0)
            min_ddg = np.min(ddg)
            median_ddg = np.median(ddg)
            median_ddf = np.median(ddf)
            print()
            print(f"Number of variants with ddG_pred < 0: {num_ddg_lt_0}")
            print(f"Minimum ddG_pred: {min_ddg}")
            print(f"Median delta_fitness: {median_ddf}")
            print(f"Median ddG_pred: {median_ddg}")
            # print(df[["ddG_pred", "delta_fitness"]].describe())
            print()
            corr = np.corrcoef(ddg, ddf)[0, 1]
            # print(f"Correlation between ddG_pred and delta_fitness: {corr:.4f}")
        except Exception as e:
            print(f"Error reading CSV file: {e}")
        
        print("\n" + "="*50 + "\n")