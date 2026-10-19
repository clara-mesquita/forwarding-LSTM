"""Train pinned multivariate models with a common, missing-aware protocol.

python ijsm_review/run_benchmarks.py --model all --dataset geant
"""

import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from aggregate_6h import OUTPUT_DIR, REVIEW_DIR
from benchmark_data import load_panel, prepare_training, validation_windows
from benchmark_models import MODELS, build_model
from forward_prediction import evaluate


def masked_mse(prediction, target, mask):
    return ((prediction[mask] - target[mask]) ** 2).mean()


def seed_all(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loader(arrays, batch_size, shuffle):
    import torch

    dataset = torch.utils.data.TensorDataset(*(torch.from_numpy(x) for x in arrays))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def fit(model, train_arrays, args, device, validation=None, epochs=None):
    import torch

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    train_loader = loader(train_arrays, args.batch_size, True)
    val_loader = loader(validation, args.batch_size, False) if validation is not None else None
    best_loss, best_epoch, stale, best_state = float("inf"), 0, 0, None
    trace = []
    for epoch in range(epochs or args.epochs):
        model.train()
        for x, target, mask in train_loader:
            x, target, mask = x.to(device), target.to(device), mask.to(device)
            optimizer.zero_grad()
            predicted = model(x, None, None, None)
            loss = masked_mse(predicted, target, mask)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss.")
            loss.backward()
            optimizer.step()
        if val_loader is None:
            continue
        model.eval()
        total, count = 0., 0
        with torch.no_grad():
            for x, target, mask in val_loader:
                x, target, mask = x.to(device), target.to(device), mask.to(device)
                predicted = model(x, None, None, None)
                total += float(((predicted[mask] - target[mask]) ** 2).sum())
                count += int(mask.sum())
        val_loss = total / count
        if not np.isfinite(val_loss):
            raise ValueError("Non-finite validation loss.")
        trace.append(val_loss)
        print(f"epoch={epoch + 1}, validation_mse={val_loss:.6g}", flush=True)
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch + 1, 0
            best_state = deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_epoch, trace


def predict_online(model, train_inputs, test_raw, mean, std, pred_len, device):
    """Assimilate observations only after their forecast block has been issued."""
    import torch

    state = train_inputs.copy()
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(test_raw), pred_len):
            x = torch.tensor(state[None], dtype=torch.float32, device=device)
            output = model(x, None, None, None).cpu().numpy()[0]
            if output.shape != (pred_len, test_raw.shape[1]) or not np.isfinite(output).all():
                raise ValueError("Invalid model forecast shape or non-finite predictions.")
            size = min(pred_len, len(test_raw) - start)
            predictions.extend(output[:size] * std + mean)
            observed = (test_raw[start:start + size] - mean) / std
            feedback = np.where(np.isfinite(observed), observed, output[:size])
            state = np.concatenate([state, feedback])[-len(state):]
    return np.asarray(predictions)


def run(args):
    import torch

    torch.set_num_threads(min(args.threads, os.cpu_count() or 1))
    seed_all(args.seed)
    directory = OUTPUT_DIR / "geant" if args.dataset == "geant" else OUTPUT_DIR
    panel = load_panel(directory)
    raw = panel.to_numpy()
    split = int(len(raw) * 0.8)
    cutoff = int(split * 0.8)
    if cutoff < args.seq_len + args.pred_len or split - cutoff < args.pred_len:
        raise ValueError("Common measurement period is too short for the requested context/horizon.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    print(f"{args.model}: {len(panel)} synchronized intervals, {len(panel.columns)} links, device={device}", flush=True)
    training = prepare_training(raw[:cutoff], args.seq_len, args.pred_len)
    x, y, mask, mean, std, inputs = training
    validation = validation_windows(raw[:split], cutoff, args.seq_len, args.pred_len, mean, std, inputs)
    model, config, source = build_model(args.model, args.seq_len, args.pred_len, raw.shape[1])
    model.to(device)
    started = time.perf_counter()
    selected_epochs, trace = fit(model, (x, y, mask), args, device, validation=validation)

    # Refit on all outer training data; the outer test period never selects epochs/scaling.
    seed_all(args.seed)
    x, y, mask, mean, std, inputs = prepare_training(raw[:split], args.seq_len, args.pred_len)
    model, config, source = build_model(args.model, args.seq_len, args.pred_len, raw.shape[1])
    model.to(device)
    fit(model, (x, y, mask), args, device, epochs=selected_epochs)
    train_seconds = time.perf_counter() - started
    started = time.perf_counter()
    predictions = predict_online(model, inputs[-args.seq_len:], raw[split:], mean, std, args.pred_len, device)
    inference_seconds = time.perf_counter() - started

    result_dir = Path(args.output_dir) / args.dataset / args.model
    result_dir.mkdir(parents=True, exist_ok=True)
    rows, metrics = [], []
    for c, name in enumerate(panel.columns):
        actual, predicted = raw[split:, c], predictions[:, c]
        observed = np.isfinite(actual)
        score = evaluate(actual, predicted) if observed.any() else {"observed_test_intervals": 0}
        metrics.append({"link": name, **score})
        rows.append(pd.DataFrame({
            "Data": panel.index[split:], "link": name, "Observed": observed,
            "Actual_Mbps": np.where(observed, actual, -1), "Prediction_Mbps": predicted,
            "horizon_step": np.arange(len(actual)) % args.pred_len + 1,
        }))
    pd.concat(rows, ignore_index=True).to_csv(result_dir / "predictions.csv", index=False)
    pd.DataFrame(metrics).to_csv(result_dir / "metrics.csv", index=False)
    panel.to_csv(result_dir / "aligned_panel_mbps.csv", na_rep="-1")
    torch.save({"state_dict": model.state_dict(), "config": config,
                "mean": mean.tolist(), "std": std.tolist(), "columns": list(panel.columns)},
               result_dir / "model.pt")
    versions = {package: importlib.metadata.version(package)
                for package in ("torch", "numpy", "pandas", "einops", "reformer-pytorch", "matplotlib")}
    metadata = {
        "arguments": vars(args), "architecture": config, "source": source,
        "software": versions, "python": sys.version, "device": str(device),
        "columns": list(panel.columns), "intervals": len(panel),
        "selection_training_end": str(panel.index[cutoff - 1]),
        "outer_training_end": str(panel.index[split - 1]), "test_start": str(panel.index[split]),
        "selected_epochs": selected_epochs, "validation_mse": trace,
        "mean": mean.tolist(), "std": std.tolist(), "units": "Mbit/s",
        "training_seconds": train_seconds, "inference_seconds": inference_seconds,
        "protocol": "online hybrid" if args.pred_len == 1 else "direct non-overlapping forecast blocks",
        "implementation_sha256": {name: hashlib.sha256((REVIEW_DIR / name).read_bytes()).hexdigest()
                                  for name in ("run_benchmarks.py", "benchmark_models.py", "benchmark_data.py",
                                               "aggregate_6h.py", "forward_prediction.py", "benchmark_sources.json")},
        "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.glob("*.csv"))},
    }
    (result_dir / "run.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved {result_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*MODELS, "all"), default="all")
    parser.add_argument("--dataset", choices=("geant", "bbr"), default="geant")
    parser.add_argument("--seq-len", type=int, default=28)
    parser.add_argument("--pred-len", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", default=str(REVIEW_DIR / "results" / "benchmarks"))
    args = parser.parse_args()
    if min(args.seq_len, args.pred_len, args.epochs, args.patience, args.batch_size, args.threads) < 1 or args.learning_rate <= 0:
        parser.error("Lengths, epochs, patience, batch size, threads and learning rate must be positive.")
    if args.model == "all":
        # Both upstream repositories use top-level 'model', 'layers' and 'utils' packages.
        for name in MODELS:
            arguments = vars(args).copy()
            arguments["model"] = name
            command = [sys.executable, str(Path(__file__).resolve())]
            for key, value in arguments.items():
                command.extend([f"--{key.replace('_', '-')}", str(value)])
            subprocess.run(command, check=True)
    else:
        run(args)


if __name__ == "__main__":
    main()
