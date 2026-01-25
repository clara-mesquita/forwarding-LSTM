import os
import shutil
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import STL

warnings.filterwarnings("ignore")

# --- Configuration ---
CUBIC_DIR = "datasets/treated/cubic/"
BBR_DIR = "datasets/treated/bbr/"
OUTPUT_DIR = "datasets/validated-set/"
CUBIC_OUT_DIR = os.path.join(OUTPUT_DIR, "cubic")
BBR_OUT_DIR = os.path.join(OUTPUT_DIR, "bbr")
SUMMARY_FILE = "selection_summary.csv"
N_SAMPLES = 10
RANDOM_STATE = 42
CADENCE = pd.Timedelta(minutes=10)
ACF_LAG = 144  # 24 hours * 6 points per hour

# --- Create output directories ---
os.makedirs(CUBIC_OUT_DIR, exist_ok=True)
os.makedirs(BBR_OUT_DIR, exist_ok=True)


def safe_qcut(series: pd.Series, q: int = 3, labels=None, fallback_label="All"):
    """
    Robust qcut:
    - Returns a categorical series with given labels if possible
    - If not possible (too few unique values / empty / all nan), returns constant label
    """
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().nunique() < 2:
        return pd.Series([fallback_label] * len(series), index=series.index)

    try:
        # If duplicates='drop' reduces bins, labels may mismatch -> use numeric bins then map
        bins = pd.qcut(s, q=q, labels=False, duplicates="drop")
        if bins.isnull().all():
            return pd.Series([fallback_label] * len(series), index=series.index)

        n_bins = int(bins.max()) + 1
        if labels is None:
            labels = [f"Q{i + 1}" for i in range(n_bins)]
        else:
            # truncate labels if fewer bins than expected
            labels = list(labels)[:n_bins]

        return bins.map({i: labels[i] for i in range(n_bins)}).astype("category")
    except Exception:
        return pd.Series([fallback_label] * len(series), index=series.index)


def profile_dataset(file_path):
    """Calculates profiling metrics for a given time series dataset."""
    try:
        df = pd.read_csv(file_path)

        if "Timestamp" not in df.columns:
            print(f"Warning: 'Timestamp' column not found in {file_path}. Skipping.")
            return None

        if "Vazao" not in df.columns:
            print(f"Warning: 'Vazao' column not found in {file_path}. Skipping.")
            return None

        # Parse
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df["Vazao"] = pd.to_numeric(df["Vazao"], errors="coerce")

        # Drop invalid timestamp rows
        df = df.dropna(subset=["Timestamp"]).copy()
        if df.empty:
            print(f"Warning: No valid timestamps in {file_path}. Skipping.")
            return None

        # Set index
        df = df.set_index("Timestamp").sort_index()

        # If vazao all nan, skip
        if df["Vazao"].isna().all():
            print(
                f"Warning: 'Vazao' column has no valid numeric data in {file_path}. Skipping."
            )
            return None

        # --- Coverage and Missing Rate ---
        time_diffs = df.index.to_series().diff().dropna()
        total_duration = df.index.max() - df.index.min()
        if total_duration <= pd.Timedelta(0):
            print(f"Warning: Non-positive duration in {file_path}. Skipping.")
            return None

        expected_points = int(total_duration / CADENCE) + 1
        observed_points = len(df)
        coverage = (
            min(observed_points / expected_points, 1.0) if expected_points > 0 else 0.0
        )
        missing_rate = 1.0 - coverage

        # --- Gap Analysis ---
        gaps = time_diffs[time_diffs > CADENCE]
        max_gap = gaps.max() if not gaps.empty else pd.Timedelta(0)

        # --- Burstiness ---
        if len(gaps) > 1:
            gaps_in_minutes = gaps.dt.total_seconds() / 60.0
            mu = float(gaps_in_minutes.mean())
            sigma = float(gaps_in_minutes.std())
            burstiness = (sigma - mu) / (sigma + mu) if (sigma + mu) > 0 else 0.0
        else:
            burstiness = 0.0

        # --- Seasonality and Variance ---
        # IMPORTANT: resample only 'Vazao' to avoid dtype object mean errors
        s = df["Vazao"].resample(CADENCE).mean()
        s = s.interpolate(method="linear")  # regular series for STL

        variance = float(s.var()) if s.notna().sum() > 1 else 0.0

        if len(s) > 2 * ACF_LAG and s.notna().sum() > 2 * ACF_LAG:
            stl = STL(s, period=ACF_LAG, robust=True)
            res = stl.fit()
            denom = np.var(res.trend + res.resid)
            seasonality_strength = (
                float(1 - np.var(res.resid) / denom) if denom > 0 else 0.0
            )
        else:
            seasonality_strength = 0.0  # Not enough data for STL

        # --- Metadata ---
        filename = os.path.basename(file_path)
        protocol = "cubic" if "cubic" in file_path.lower() else "bbr"
        region = filename.split("_")[0] if "_" in filename else "unknown"

        return {
            "filename": filename,
            "filepath": file_path,
            "protocol": protocol,
            "region": region,
            "coverage": coverage,
            "missing_rate": missing_rate,
            "max_gap_hours": max_gap.total_seconds() / 3600.0,
            "burstiness": burstiness,
            "variance": variance,
            "seasonality_strength": seasonality_strength,
        }

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def select_and_copy(df: pd.DataFrame, protocol: str, out_dir: str):
    """Stratified selection with robust fallbacks."""
    if df is None or df.empty:
        print(f"Warning: No valid profiles for {protocol.upper()}. Skipping selection.")
        return None

    # --- Stratification bins (robust) ---
    df = df.copy()
    df["missing_bin"] = safe_qcut(
        df["missing_rate"], q=3, labels=["Low", "Medium", "High"]
    )
    df["max_gap_bin"] = safe_qcut(
        df["max_gap_hours"], q=3, labels=["Low", "Medium", "High"]
    )

    df["stratify_key"] = (
        df["missing_bin"].astype(str)
        + "_"
        + df["max_gap_bin"].astype(str)
        + "_"
        + df["region"].astype(str)
    )

    # --- Selection ---
    if len(df) <= N_SAMPLES:
        print(f"Warning: Not enough samples for {protocol} ({len(df)}), selecting all.")
        selected_df = df
    else:
        # train_test_split stratify may fail if some classes have 1 element
        try:
            _, selected_df = train_test_split(
                df,
                test_size=N_SAMPLES,
                stratify=df["stratify_key"],
                random_state=RANDOM_STATE,
            )
        except Exception:
            print(
                f"Warning: Stratification failed for {protocol}. Performing random sampling."
            )
            selected_df = df.sample(n=N_SAMPLES, random_state=RANDOM_STATE)

    print(f"Selected {len(selected_df)} files for {protocol.upper()}.")

    # --- Copy Files ---
    os.makedirs(out_dir, exist_ok=True)
    for _, row in selected_df.iterrows():
        dst = os.path.join(out_dir, row["filename"])
        shutil.copy(row["filepath"], dst)
        print(f"  Copied {row['filename']} to {out_dir}")

    return selected_df


def main():
    all_files_cubic = [
        os.path.join(CUBIC_DIR, f) for f in os.listdir(CUBIC_DIR) if f.endswith(".csv")
    ]
    all_files_bbr = [
        os.path.join(BBR_DIR, f) for f in os.listdir(BBR_DIR) if f.endswith(".csv")
    ]

    print("Profiling CUBIC datasets...")
    cubic_profiles = [profile_dataset(f) for f in all_files_cubic]
    print("Profiling BBR datasets...")
    bbr_profiles = [profile_dataset(f) for f in all_files_bbr]

    df_cubic = pd.DataFrame([p for p in cubic_profiles if p is not None])
    df_bbr = pd.DataFrame([p for p in bbr_profiles if p is not None])

    selected_dfs = []

    print("\nPerforming stratified selection for CUBIC...")
    sel_cubic = select_and_copy(df_cubic, "cubic", CUBIC_OUT_DIR)
    if sel_cubic is not None and not sel_cubic.empty:
        selected_dfs.append(sel_cubic)

    print("\nPerforming stratified selection for BBR...")
    sel_bbr = select_and_copy(df_bbr, "bbr", BBR_OUT_DIR)
    if sel_bbr is not None and not sel_bbr.empty:
        selected_dfs.append(sel_bbr)

    # --- Generate Summary CSV ---
    if not selected_dfs:
        print("No datasets selected. Summary file not generated.")
        return

    summary_df = pd.concat(selected_dfs, ignore_index=True)
    if "filepath" in summary_df.columns:
        summary_df.drop(columns=["filepath"], inplace=True, errors="ignore")
    if "stratify_key" in summary_df.columns:
        summary_df.drop(columns=["stratify_key"], inplace=True, errors="ignore")

    summary_df.to_csv(SUMMARY_FILE, index=False)
    print(f"\nSelection summary saved to {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
