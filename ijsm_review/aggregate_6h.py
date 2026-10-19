"""Average selected BBR and GEANT datasets into 6-hour bins; missing bins are -1.

Run with: python ijsm_review/aggregate_6h.py
Requires pandas. Paths are relative to this script, independent of the working directory.
"""

from pathlib import Path

import pandas as pd


REVIEW_DIR = Path(__file__).resolve().parent
SOURCE_DIR = REVIEW_DIR.parent / "sbrc_final_results" / "choosen_treated_datasets"
GEANT_DIR = REVIEW_DIR.parent / "sbrc_final_results" / "geant"
OUTPUT_DIR = REVIEW_DIR / "data"
LINKS = ("ac-to", "es-ce", "es-pr", "go-sp", "ms-to", "sc-to")


def aggregate_6h(df):
    """Use calendar-aligned, left-closed bins labeled by their start time."""
    df = df[["Data", "Vazao"]].copy()
    df["Data"] = pd.to_datetime(df["Data"], errors="raise")
    if df.empty or df["Data"].isna().any():
        raise ValueError("Input must contain observations with valid Data timestamps.")
    df["Vazao"] = pd.to_numeric(df["Vazao"], errors="raise")
    # Existing missing-value markers must not contribute to the mean.
    df["Vazao"] = df["Vazao"].replace(-1, float("nan"))
    return (
        df.set_index("Data")[["Vazao"]]
        .resample("6h", origin="start_day", closed="left", label="left")
        .mean()
        .fillna(-1)
        .reset_index()
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for link in LINKS:
        source = SOURCE_DIR / f"{link}_bbr_interval_0.csv"
        result = aggregate_6h(pd.read_csv(source))
        output = OUTPUT_DIR / f"{source.stem}_6h.csv"
        result.to_csv(output, index=False)
        missing = result["Vazao"].eq(-1).sum()
        print(f"{output.name}: {len(result)} intervals, {missing} missing")
    geant_files = sorted(GEANT_DIR.glob("*.csv"))
    if not geant_files:
        raise FileNotFoundError(f"No GEANT CSV files found in {GEANT_DIR}")
    geant_output = OUTPUT_DIR / "geant"
    geant_output.mkdir(parents=True, exist_ok=True)
    for source in geant_files:
        result = aggregate_6h(pd.read_csv(source))
        name = source.name if source.stem.endswith("_6h") else f"{source.stem}_6h.csv"
        result.to_csv(geant_output / name, index=False)
        missing = result["Vazao"].eq(-1).sum()
        print(f"geant/{name}: {len(result)} intervals, {missing} missing")


if __name__ == "__main__":
    main()
