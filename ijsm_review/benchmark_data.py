"""Synchronized multivariate panels and chronological, missing-aware samples."""

from pathlib import Path

import numpy as np
import pandas as pd

from aggregate_6h import aggregate_6h


def load_panel(directory):
    paths = sorted(Path(directory).glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files in {directory}")
    series = []
    for path in paths:
        frame = aggregate_6h(pd.read_csv(path)).set_index("Data")
        series.append(frame["Vazao"].rename(path.stem).replace(-1, np.nan) / 1e6)
    # Use the common measurement period, never the union of unrelated timestamps.
    panel = pd.concat(series, axis=1, join="inner").sort_index()
    if panel.empty:
        raise ValueError("Datasets have no common measurement period.")
    if np.isinf(panel.to_numpy()).any():
        raise ValueError("Infinite throughput values are invalid.")
    return panel


def prepare_training(raw, seq_len, pred_len, mean=None, std=None):
    """Call separately on selection-training and final-training portions."""
    filled = pd.DataFrame(raw).interpolate(limit_direction="both").to_numpy()
    if not np.isfinite(filled).all():
        raise ValueError("Every link needs at least one observed training value.")
    if mean is None:
        mean, std = filled.mean(axis=0), filled.std(axis=0)
        std = np.where(std > 1e-8, std, 1.)
    inputs = (filled - mean) / std
    targets = (raw - mean) / std
    x, y, mask = make_windows(inputs, targets, seq_len, pred_len, seq_len, len(raw))
    return x, y, mask, mean, std, inputs


def make_windows(inputs, targets, seq_len, pred_len, start, end):
    x, y, masks = [], [], []
    for i in range(max(seq_len, start), end - pred_len + 1):
        target = targets[i:i + pred_len]
        valid = np.isfinite(target)
        if valid.any():
            x.append(inputs[i - seq_len:i])
            y.append(np.nan_to_num(target, nan=0.))
            masks.append(valid)
    if not x:
        raise ValueError("No observed forecast targets; reduce the context/horizon or supply more data.")
    return (np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32),
            np.asarray(masks, dtype=bool))


def validation_windows(raw, cutoff, seq_len, pred_len, mean, std, training_inputs):
    # Validation histories use only past observations, never backward interpolation.
    tail = pd.DataFrame(raw[cutoff:])
    seed = training_inputs[-1] * std + mean
    causal = pd.concat([pd.DataFrame([seed]), tail], ignore_index=True).ffill().iloc[1:].to_numpy()
    inputs = np.concatenate([training_inputs, (causal - mean) / std])
    targets = (raw - mean) / std
    return make_windows(inputs, targets, seq_len, pred_len, cutoff, len(raw))
