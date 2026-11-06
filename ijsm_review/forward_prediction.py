"""Train and evaluate the proposed hybrid recursive LSTM on review datasets.

Run: python ijsm_review/forward_prediction.py
Dependencies: pandas, numpy, scikit-learn, tensorflow.
"""

import json
import argparse
import hashlib
import importlib.util
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from aggregate_6h import GEANT_DIR, OUTPUT_DIR, REVIEW_DIR, aggregate_6h


LOOK_BACK = 3
WINDOW_SIZE = 28


def prepare_data(df):
    """Split chronologically; interpolate training data only, in Mbit/s."""
    df = aggregate_6h(df)
    values = df["Vazao"].replace(-1, np.nan).to_numpy(dtype=float) / 1_000_000
    if np.isinf(values).any():
        raise ValueError("Throughput contains infinite values.")
    split = int(len(values) * 0.8)
    if split <= LOOK_BACK + 6 or split == len(values):
        raise ValueError("Not enough intervals for training and evaluation.")
    train = pd.Series(values[:split]).interpolate(limit_direction="both").to_numpy()
    if not np.isfinite(train).all():
        raise ValueError("Training data has no valid throughput observations.")
    return df, train, values[split:], split


def create_dataset(values):
    x = np.array([values[i - LOOK_BACK:i] for i in range(LOOK_BACK, len(values))])
    return x.reshape(-1, LOOK_BACK, 1), values[LOOK_BACK:]


def walk_forward(model, train_scaled, test_scaled):
    """Use observed values when available and predictions across missing runs."""
    state = np.asarray(train_scaled[-LOOK_BACK:]).reshape(1, LOOK_BACK, 1).copy()
    predictions = []
    for start in range(0, len(test_scaled), WINDOW_SIZE):
        for actual in test_scaled[start:start + WINDOW_SIZE]:
            prediction = float(np.asarray(model.predict(state, verbose=0)).item())
            if not np.isfinite(prediction):
                raise ValueError("Model produced a non-finite prediction.")
            predictions.append(prediction)
            state = np.roll(state, -1, axis=1)
            state[0, -1, 0] = actual if np.isfinite(actual) else prediction
    return np.asarray(predictions)


def evaluate(actual, predicted):
    mask = np.isfinite(actual)
    if not mask.any():
        raise ValueError("No observed test values available for evaluation.")
    actual, predicted = actual[mask], predicted[mask]
    error = predicted - actual
    rmse = float(np.sqrt(np.mean(error ** 2)))
    fuzzy_path = REVIEW_DIR.parent / "code/forwarding-LSTM/grid/fuzzy_analysis.py"
    spec = importlib.util.spec_from_file_location("fuzzy_analysis", fuzzy_path)
    fuzzy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fuzzy)
    # The existing helper accepts bit/s, whereas this pipeline evaluates Mbit/s.
    fuzzy_accuracy, fuzzy_count = fuzzy.compute_fuzzy_accuracy(actual * 1e6, predicted * 1e6)
    return {
        "observed_test_intervals": int(mask.sum()),
        "RMSE": rmse,
        "MAE": float(np.mean(np.abs(error))),
        "Fuzzy_Accuracy": None if fuzzy_accuracy is None else float(fuzzy_accuracy),
        "fuzzy_evaluated_intervals": int(fuzzy_count),
        "NRMSE_mean": rmse / (abs(float(actual.mean())) + 1e-12),
        "SMAPE": float(np.mean(200 * np.abs(error) /
                               (np.abs(actual) + np.abs(predicted) + 1e-12))),
    }


def train_model(train):
    import tensorflow as tf
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import MinMaxScaler

    def build_model():
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(LOOK_BACK, 1)),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(1),
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                      loss=tf.keras.losses.Huber(delta=0.25))
        return model

    def fit(model, x, y, validation, epochs, batch_size):
        return model.fit(
            x, y, validation_data=validation, epochs=epochs, batch_size=batch_size,
            shuffle=False, verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True)],
        )

    best_loss, best_params = float("inf"), None
    for epochs, batch_size in product((100, 300, 500), (32, 64, 128)):
        losses = []
        # Fit a fresh model and scaler in each fold to avoid validation leakage.
        for train_idx, val_idx in TimeSeriesSplit(n_splits=5).split(train):
            scaler = MinMaxScaler().fit(train[train_idx].reshape(-1, 1))
            fold_train = scaler.transform(train[train_idx].reshape(-1, 1)).ravel()
            fold_val = scaler.transform(train[val_idx].reshape(-1, 1)).ravel()
            x, y = create_dataset(fold_train)
            xv, yv = create_dataset(np.concatenate([fold_train[-LOOK_BACK:], fold_val]))
            tf.keras.backend.clear_session()
            model = build_model()
            history = fit(model, x, y, (xv, yv), epochs, batch_size)
            losses.append(min(history.history["val_loss"]))
        loss = float(np.mean(losses))
        print(f"epochs={epochs}, batch_size={batch_size}, validation_loss={loss:.6g}", flush=True)
        if loss < best_loss:
            best_loss = loss
            best_params = {"epochs": epochs, "batch_size": batch_size}

    # Reserve the final 20% of training for early stopping, then fit all training data.
    cutoff = int(len(train) * 0.8)
    scaler = MinMaxScaler().fit(train[:cutoff].reshape(-1, 1))
    scaled = scaler.transform(train.reshape(-1, 1)).ravel()
    x, y = create_dataset(scaled[:cutoff])
    xv, yv = create_dataset(scaled[cutoff - LOOK_BACK:])
    tf.keras.backend.clear_session()
    model = build_model()
    history = fit(model, x, y, (xv, yv), **best_params)
    selected_epochs = int(np.argmin(history.history["val_loss"])) + 1
    scaler = MinMaxScaler().fit(train.reshape(-1, 1))
    scaled = scaler.transform(train.reshape(-1, 1)).ravel()
    x, y = create_dataset(scaled)
    tf.keras.backend.clear_session()
    model = build_model()
    model.fit(x, y, epochs=selected_epochs, batch_size=best_params["batch_size"],
              shuffle=False, verbose=0)
    return model, scaler, {**best_params, "fitted_epochs": selected_epochs}


def main():
    import tensorflow as tf

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("geant", "bbr", "sbrc"), default="geant")
    parser.add_argument("--input-dir", type=Path, help="Override the directory of 6-hour CSV inputs")
    parser.add_argument("--output-dir", type=Path, default=REVIEW_DIR / "results" / "forwarding_lstm")
    parser.add_argument("--source", help="Run only this input CSV filename")
    parser.add_argument("--skip-existing", action="store_true", help="Resume completed inputs with matching source hash")
    args = parser.parse_args()
    tf.keras.utils.set_random_seed(42)
    directory = args.input_dir or (GEANT_DIR if args.dataset == "geant" else
                                   OUTPUT_DIR if args.dataset == "bbr" else
                                   REVIEW_DIR.parent / "sbrc_final_results" / "choosen_treated_datasets_6h")
    sources = sorted(directory.glob("*.csv"))
    if args.source:
        sources = [source for source in sources if source.name == args.source]
    if not sources:
        raise FileNotFoundError(f"No CSV files found in {directory}")
    from benchmark_data import load_panel
    panel = load_panel(directory)
    result_dir = args.output_dir / args.dataset
    result_dir.mkdir(parents=True, exist_ok=True)
    for source in sources:
        name = source.stem if source.stem.endswith("_6h") else f"{source.stem}_6h"
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        metrics_path = result_dir / f"{name}_metrics.json"
        if args.skip_existing and metrics_path.exists() and (result_dir / f"{name}.keras").exists() and (result_dir / f"{name}_predictions.csv").exists():
            saved = json.loads(metrics_path.read_text(encoding="utf-8"))
            if saved.get("source_sha256") == source_hash and saved.get("shared_intervals") == len(panel) and saved.get("test_start") == str(panel.index[int(len(panel) * 0.8)]):
                print(f"Skipping completed {source.name}", flush=True)
                continue
        print(f"Processing {source.name}", flush=True)
        aligned = pd.DataFrame({"Data": panel.index,
                                "Vazao": panel[source.stem].to_numpy() * 1e6})
        aligned["Vazao"] = aligned["Vazao"].fillna(-1)
        df, train, test, split = prepare_data(aligned)
        model, scaler, params = train_model(train)
        train_scaled = scaler.transform(train.reshape(-1, 1)).ravel()
        test_scaled = scaler.transform(test.reshape(-1, 1)).ravel()
        predictions = scaler.inverse_transform(
            walk_forward(model, train_scaled, test_scaled).reshape(-1, 1)).ravel()
        metrics = (evaluate(test, predictions) if np.isfinite(test).any() else
                   {"observed_test_intervals": 0, "RMSE": None, "MAE": None,
                    "Fuzzy_Accuracy": None, "fuzzy_evaluated_intervals": 0,
                    "NRMSE_mean": None, "SMAPE": None})
        model.save(result_dir / f"{name}.keras")
        pd.DataFrame({
            "Data": df["Data"].iloc[split:].to_numpy(),
            "Actual_Mbps": np.where(np.isfinite(test), test, -1),
            "Observed": np.isfinite(test), "Prediction_Mbps": predictions,
        }).to_csv(result_dir / f"{name}_predictions.csv", index=False)
        metadata = {"parameters": params, "metrics": metrics,
                    "source": str(source.resolve()),
                    "source_sha256": source_hash,
                    "shared_intervals": len(panel),
                    "test_start": str(panel.index[split]),
                    "look_back": LOOK_BACK, "window_size": WINDOW_SIZE,
                    "scaler_min": scaler.data_min_.tolist(),
                    "scaler_max": scaler.data_max_.tolist(), "units": "Mbit/s"}
        metrics_path.write_text(
            json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
        print(metrics, flush=True)


if __name__ == "__main__":
    main()
