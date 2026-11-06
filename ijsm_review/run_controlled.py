"""Per-link, missing-aware comparison; outputs never replace earlier experiments."""
import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
from itertools import product
import json
import os
from pathlib import Path
import random
import time
import warnings

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from statsmodels.tsa.statespace.sarimax import SARIMAX
import torch

from aggregate_6h import REVIEW_DIR
from benchmark_data import load_panel
from controlled_models import Network, online
from forward_prediction import evaluate

MODELS = ("Persistence", "Ridge", "RandomForest", "MLP", "Transformer",
          "LSTM", "GRU", "SARIMA", "ARIMA", "ForwardingLSTM")
NEURAL = ("MLP", "Transformer", "LSTM", "GRU", "ForwardingLSTM")
STOCHASTIC = (*NEURAL, "RandomForest")
VERSION = 1
# This prior implementation differs only in lacking a retry for FAILED final
# statistical fits. All completed artifacts have identical fitting semantics.
COMPATIBLE_RUNNER_HASHES = ("e00f50c523eef0f10fa6e126cd01b560c5068cb3ce92a9db4f88b8ee2aec23ac",)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def preparation(raw):
    """Fill inputs and fit bounds using this training partition alone."""
    filled = pd.Series(raw).interpolate(limit_direction="both").to_numpy()
    if not np.isfinite(filled).all():
        raise ValueError("No observed values in training")
    lower = float(np.min(filled))
    scale = max(float(np.max(filled) - lower), 1e-8)
    return (filled - lower) / scale, lower, scale


def windows(inputs, targets, context, start=None):
    x, y = [], []
    for index in range(context if start is None else max(context, start), len(targets)):
        if np.isfinite(targets[index]):
            x.append(inputs[index-context:index])
            y.append(targets[index])
    if not x:
        raise ValueError("No observed targets for this partition/context")
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)


def arrays(raw, context, lower=None, scale=None):
    inputs, fit_lower, fit_scale = preparation(raw)
    if lower is not None:
        inputs = (inputs * fit_scale + fit_lower - lower) / scale
    else:
        lower, scale = fit_lower, fit_scale
    return (*windows(inputs, (raw-lower)/scale, context), inputs, lower, scale)


def validation_arrays(train_inputs, validation, context):
    # Seed forward filling only with the last training input; never future values.
    history = pd.Series(np.r_[train_inputs[-1], validation]).ffill().to_numpy()[1:]
    inputs = np.r_[train_inputs, history]
    targets = np.r_[np.full(len(train_inputs), np.nan), validation]
    return windows(inputs, targets, context, start=len(train_inputs))


def score(actual, forecast):
    mask = np.isfinite(actual)
    if not mask.any():
        raise ValueError("Validation has no observed targets")
    return float(np.mean((actual[mask] - forecast[mask]) ** 2))


def configurations(name):
    if name == "Ridge":
        return [{"alpha": value} for value in (.1, 1., 10., 100.)]
    if name == "RandomForest":
        return [dict(n_estimators=n, max_depth=d, min_samples_leaf=l)
                for n, d, l in product((100, 300), (5, 10), (2, 5))]
    if name == "MLP":
        return [dict(hidden=h, learning_rate=.001, batch_size=32) for h in ((32,), (64, 32))]
    if name == "Transformer":
        return [dict(layers=n, learning_rate=.001, batch_size=32) for n in (1, 2)]
    if name in ("LSTM", "GRU"):
        return [dict(units=u, learning_rate=lr, batch_size=32)
                for u, lr in product((32, 64, 128), (.01, .001, .0001))]
    if name == "ForwardingLSTM":
        return [dict(units=64, learning_rate=.001, batch_size=b) for b in (32, 64, 128)]
    if name in ("SARIMA", "ARIMA"):
        orders = ((1, 0, 0), (1, 1, 0), (1, 0, 1))
        seasonal = ((0, 0, 0, 0),) if name == "ARIMA" else ((1, 0, 0, 4), (1, 0, 0, 12), (1, 0, 0, 28))
        return [dict(order=o, seasonal_order=s) for o, s in product(orders, seasonal)]
    return [{}]


def fit_network(name, config, context, x, y, seed, epochs, patience, validation=None):
    seed_all(seed)
    model = Network(name, context, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    loss_function = (torch.nn.HuberLoss(delta=.25)
                     if name in ("LSTM", "GRU", "ForwardingLSTM") else torch.nn.MSELoss())
    tx, ty = torch.from_numpy(x), torch.from_numpy(y)
    best, best_epoch, stale, state = float("inf"), 1, 0, None
    trace = []
    for epoch in range(1, epochs+1):
        model.train()
        for start in range(0, len(x), config["batch_size"]):
            optimizer.zero_grad()
            prediction = model(tx[start:start+config["batch_size"]])
            loss = loss_function(prediction, ty[start:start+config["batch_size"]])
            if not torch.isfinite(loss):
                raise ValueError("Non-finite neural loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        if validation is None:
            continue
        model.eval()
        with torch.no_grad():
            vx, vy = validation
            val_loss = float(loss_function(model(torch.from_numpy(vx)), torch.from_numpy(vy)))
        trace.append(val_loss)
        if val_loss < best:
            best, best_epoch, stale, state = val_loss, epoch, 0, deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break
    if state is not None:
        model.load_state_dict(state)
    model.eval()
    return model, best_epoch if validation is not None else epochs, trace


def classical(name, config, x, y, seed):
    if name == "Ridge":
        model = Ridge(**config)
    elif name == "RandomForest":
        model = RandomForestRegressor(**config, random_state=seed, n_jobs=1)
    else:
        return None
    return model.fit(x, y)


def fit_statistical(raw, config, retry=False):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fitted = SARIMAX(raw, **config, trend="n", enforce_stationarity=True,
                         enforce_invertibility=True).fit(disp=False, maxiter=150)
        if retry and not fitted.mle_retvals.get("converged", False):
            fitted = fitted.model.fit(start_params=fitted.params, method="powell", disp=False, maxiter=500)
            messages = ["Initial final-fit LBFGS did not converge; retried with Powell (500 iterations)"]
        else:
            messages = []
    return fitted, messages + [str(item.message) for item in caught]


def statistical_online(fitted, actual):
    predictions = []
    for value in actual:
        forecast = float(np.asarray(fitted.forecast(1)).item())
        if not np.isfinite(forecast):
            raise ValueError("Non-finite SARIMAX forecast")
        predictions.append(forecast)
        fitted = fitted.extend([value if np.isfinite(value) else forecast])
    return np.asarray(predictions)


def select(name, raw, context, args):
    cutoff = int(len(raw)*.8)
    # A link can stop reporting long before the common train/test boundary.
    # In that case reserve the last 20% of observed training targets for validation.
    if not np.isfinite(raw[cutoff:]).any():
        observed = np.flatnonzero(np.isfinite(raw))
        cutoff = int(observed[int(len(observed)*.8)])
    if cutoff <= context:
        raise ValueError("Insufficient observed training history for validation")
    x, y, inputs, lower, scale = arrays(raw[:cutoff], context)
    validation = (raw[cutoff:]-lower)/scale
    val_arrays = validation_arrays(inputs, validation, context)
    candidates = []
    winner = None
    for config in configurations(name):
        record = {"config": config, "selection_cutoff": cutoff}
        started = time.perf_counter()
        try:
            if name in NEURAL:
                model, epochs, trace = fit_network(name, config, context, x, y, args.seeds[0],
                                                   args.epochs, args.patience, val_arrays)
                predicted = online(model, inputs[-context:], validation)
                record.update(selected_epochs=epochs, validation_loss=trace)
            elif name in ("ARIMA", "SARIMA"):
                model, messages = fit_statistical((raw[:cutoff]-lower)/scale, config)
                predicted = statistical_online(model, validation)
                record.update(warnings=messages, converged=bool(model.mle_retvals.get("converged", False)))
                if not record["converged"]:
                    raise ValueError("SARIMAX optimizer did not converge")
            else:
                model = classical(name, config, x, y, args.seeds[0])
                predicted = online(model, inputs[-context:], validation)
            record["validation_mse_mbps"] = score(raw[cutoff:], predicted*scale+lower)
            if winner is None or record["validation_mse_mbps"] < winner["validation_mse_mbps"]:
                winner = record
        except (ValueError, np.linalg.LinAlgError, RuntimeError) as error:
            record["error"] = str(error)
        record["seconds"] = time.perf_counter()-started
        candidates.append(record)
    if winner is None:
        raise RuntimeError(f"All configurations failed: {candidates}")
    return winner, candidates


def fingerprint(args, panel, dataset, link, context, name, runner_hash=None):
    payload = {"version": VERSION, "dataset": dataset, "link": link, "context": context,
               "name": name, "epochs": args.epochs, "patience": args.patience,
               "seeds": args.seeds,
               "panel_hash": hashlib.sha256(panel.to_csv().encode()).hexdigest(),
               "implementation": {file: hashlib.sha256((REVIEW_DIR/file).read_bytes()).hexdigest()
                                  for file in ("controlled_models.py", "run_controlled.py")}}
    if runner_hash is not None:
        payload["implementation"]["run_controlled.py"] = runner_hash
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def experiment(args, dataset, panel, link, context, name):
    raw = panel[link].to_numpy()
    split = int(len(raw)*.8)
    train, test = raw[:split], raw[split:]
    seeds = args.seeds if name in STOCHASTIC else [args.seeds[0]]
    destination = args.output / dataset / f"context_{context}" / name / link
    signature = fingerprint(args, panel, dataset, link, context, name)
    compatible = {signature, *(fingerprint(args, panel, dataset, link, context, name, old)
                               for old in COMPATIBLE_RUNNER_HASHES)}
    completed = []
    for seed in seeds:
        directory = destination / f"seed_{seed}"
        path = directory / "run.json"
        if path.exists() and all((directory/file).exists() for file in ("metrics.csv", "predictions.csv", "model.joblib" if name not in NEURAL else "model.pt")):
            metadata = json.loads(path.read_text(encoding="utf-8"))
            if metadata.get("fingerprint") in compatible:
                completed.append(seed)
    if len(completed) == len(seeds):
        print(f"RESUME {dataset} context={context} {name} {link}", flush=True)
        return
    print(f"RUN {dataset} context={context} {name} {link}", flush=True)
    started = time.perf_counter()
    winner, candidates = select(name, train, context, args)
    selection_seconds = time.perf_counter()-started
    x, y, inputs, lower, scale = arrays(train, context)
    for seed in seeds:
        if seed in completed:
            continue
        directory = destination / f"seed_{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        if name in NEURAL:
            model, _, _ = fit_network(name, winner["config"], context, x, y, seed,
                                       winner["selected_epochs"], args.patience)
        elif name in ("SARIMA", "ARIMA"):
            model, messages = fit_statistical((train-lower)/scale, winner["config"], retry=True)
            if not model.mle_retvals.get("converged", False):
                raise RuntimeError(f"Final {name} fit did not converge: {link}")
        else:
            model = classical(name, winner["config"], x, y, seed)
        train_seconds = time.perf_counter()-started
        started = time.perf_counter()
        scaled_test = (test-lower)/scale
        predictions = (statistical_online(model, scaled_test) if name in ("SARIMA", "ARIMA")
                       else online(model, inputs[-context:], scaled_test))*scale+lower
        inference_seconds = time.perf_counter()-started
        metrics = (evaluate(test, predictions) if np.isfinite(test).any() else
                   dict(observed_test_intervals=0, RMSE=None, MAE=None,
                        NRMSE_mean=None, SMAPE=None, Fuzzy_Accuracy=None, fuzzy_evaluated_intervals=0))
        metrics.update(dataset=dataset, link=link, model=name, context=context, seed=seed)
        pd.DataFrame([metrics]).to_csv(directory/"metrics.csv", index=False)
        pd.DataFrame({"Data": panel.index[split:], "link": link, "Observed": np.isfinite(test),
                      "Actual_Mbps": np.where(np.isfinite(test), test, -1),
                      "Prediction_Mbps": predictions, "horizon_step": 1}).to_csv(directory/"predictions.csv", index=False)
        if name in NEURAL:
            torch.save(dict(state_dict=model.state_dict(), model=name, context=context,
                            config=winner["config"], lower=lower, scale=scale), directory/"model.pt")
        else:
            joblib.dump(dict(model=model, lower=lower, scale=scale, context=context), directory/"model.joblib")
        metadata = dict(fingerprint=signature, protocol="univariate online hybrid one-step", units="Mbit/s",
                        dataset=dataset, link=link, context=context, model=name, seed=seed,
                        chosen=winner, candidates=candidates, selection_seconds=selection_seconds,
                        training_seconds=train_seconds, inference_seconds=inference_seconds,
                        lower=lower, scale=scale, training_targets=len(y),
                        selection_training_end=str(panel.index[winner["selection_cutoff"]-1]),
                        validation_start=str(panel.index[winner["selection_cutoff"]]),
                        outer_training_end=str(panel.index[split-1]), test_start=str(panel.index[split]),
                        selection_seed=args.seeds[0], selection_policy="select once; refit same configuration and epoch count for all seeds",
                        statistical_history="all past training data, not limited to lag context" if name in ("ARIMA", "SARIMA") else None,
                        implementation_note="PyTorch port preserving recurrent architecture and Huber loss; controlled tuning differs from legacy runs" if name in ("LSTM", "GRU", "ForwardingLSTM") else None,
                        final_fit_warnings=messages if name in ("ARIMA", "SARIMA") else [],
                        software={p: importlib.metadata.version(p) for p in ("torch", "numpy", "pandas", "scikit-learn", "statsmodels")})
        (directory/"run.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
        print(f"  SAVED seed={seed} RMSE={metrics['RMSE']}", flush=True)


def summarize(output):
    paths = sorted(output.glob("*/context_*/*/*/seed_*/metrics.csv"))
    if not paths:
        return
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    frame.to_csv(output/"metrics_all.csv", index=False)
    columns = ["RMSE", "MAE", "NRMSE_mean", "SMAPE"]
    per_link = frame.groupby(["dataset", "context", "model", "link"])[columns].mean()
    per_link.to_csv(output/"metrics_mean_per_link.csv")
    per_link.groupby(["dataset", "context", "model"])[columns].agg(["mean", "median", "count"]).to_csv(output/"summary.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=("geant", "sbrc"), default=["geant", "sbrc"])
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--contexts", nargs="+", type=int, default=[3, 28])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--link", help="Restrict to one exact link stem")
    parser.add_argument("--output", type=Path, default=REVIEW_DIR/"results"/"controlled")
    args = parser.parse_args()
    if min(*args.contexts, args.epochs, args.patience) < 1:
        parser.error("Contexts, epochs, and patience must be positive")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        source = (REVIEW_DIR/"data"/"geant" if dataset == "geant" else
                  REVIEW_DIR.parent/"sbrc_final_results"/"choosen_treated_datasets_6h")
        panel = load_panel(source)
        panel.to_csv(args.output/f"{dataset}_aligned_panel_mbps.csv", na_rep="-1")
        links = [args.link] if args.link else list(panel.columns)
        for context, link, name in product(args.contexts, links, args.models):
            experiment(args, dataset, panel, link, context, name)
            summarize(args.output)
    print(f"COMPLETE {args.output}", flush=True)


if __name__ == "__main__":
    main()
