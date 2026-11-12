"""Replay the archived grid evaluators and verify errors before computing kappa."""
import csv
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import os
import warnings

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import MinMaxScaler

from plot_latest_journal import ROOT, OUT, NETWORKS


def replay(row):
    model, link = row["model"], row["link"]
    network = next(key for key, label in NETWORKS.items() if label == row["network"])
    series = pd.read_csv(ROOT / "datasets/data" / network / f"{link}.csv")["Vazao"] / 1e6
    series = series.replace(-1, np.nan)
    split = int(len(series) * .8)
    train = series.iloc[:split].interpolate(limit_direction="both").to_numpy().reshape(-1, 1)
    test = series.iloc[split:].to_numpy().reshape(-1, 1)
    scaler = MinMaxScaler().fit(train)
    train, scaled = scaler.transform(train), scaler.transform(test)
    if model == "SARIMA":
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            best = None
            for cfg in product((0, 1), repeat=6):
                try:
                    fit = SARIMAX(train[1:, 0], order=cfg[:3], seasonal_order=(*cfg[3:], 12)).fit(disp=False)
                except (ValueError, np.linalg.LinAlgError):
                    continue
                if np.isfinite(fit.aic) and (best is None or fit.aic < best.aic):
                    best = fit
        assert best is not None, link
        pred = scaler.inverse_transform(best.forecast(len(test) - 1).reshape(-1, 1))
        actual = test[1:]
    else:
        from tensorflow.keras.models import load_model
        forward = model == "Forwarding LSTM"
        filename = f"{link}.keras" if forward else f"{link}_{model}.keras"
        fitted = load_model(ROOT / ("results" if forward else "results2") / network / "models" / filename,
                            compile=False)
        if forward:
            state = train[-3:].reshape(1, 3, 1)
            estimates = []
            observed_test = np.isfinite(test[:, 0]) & (test[:, 0] != -1)
            for point, observed in zip(scaled[:, 0], observed_test):
                estimate = float(fitted(state, training=False).numpy()[0, 0])
                estimates.append(estimate)
                state = np.roll(state, -1, axis=1)
                state[0, -1, 0] = point if observed else estimate
            # Match the archived evaluator, which offsets targets by three intervals.
            actual = test[3:]
            observed = observed_test[3:]
            actual = actual[observed]
            pred = scaler.inverse_transform(np.array(estimates[:len(observed)]).reshape(-1, 1))[observed]
        else:
            offset = 24 if model == "Transformer" else 1
            windows = np.lib.stride_tricks.sliding_window_view(scaled[:, 0], offset)[:-1, :, None]
            pred = scaler.inverse_transform(fitted.predict(windows, verbose=0))
            actual = test[offset:]
    actual, pred = actual.ravel(), pred.ravel()
    assert len(actual) and np.isfinite(actual).all() and np.isfinite(pred).all(), (model, link)
    delta = pred - actual
    rmse = np.sqrt(np.mean(delta ** 2))
    computed = [rmse, np.mean(np.abs(delta)), rmse / (actual.mean() + 1e-12)]
    reported = [float(row[m]) for m in ("RMSE", "MAE", "NRMSE")]
    np.testing.assert_allclose(computed, reported, rtol=1e-4, atol=.001, err_msg=f"{model} {link}")
    truth, forecast = (np.floor(values / 100).astype(int) for values in (actual, pred))
    kappa = (cohen_kappa_score(truth, forecast)
             if np.unique(np.r_[truth, forecast]).size > 1 else None)
    return row["network"], link, model, kappa, len(actual)


def main():
    with (OUT / "plotted_metrics.csv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    sarima = [row for row in rows if row["model"] == "SARIMA"]
    with ProcessPoolExecutor(max_workers=4) as pool:
        pending = pool.map(replay, sarima)
        values = [replay(row) for row in rows if row["model"] != "SARIMA"]
        values.extend(pending)
    assert len(values) == 80
    with (OUT / "replayed_kappa.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["network", "link", "model", "Cohen_Kappa", "pairs"])
        writer.writerows(values)


if __name__ == "__main__":
    main()
