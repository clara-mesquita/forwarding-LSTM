"""Audit coverage, observed masks, timestamps and saved error metrics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import torch
from benchmark_data import load_panel
from controlled_models import Network, online
from run_controlled import MODELS, STOCHASTIC, preparation

ROOT = Path(__file__).resolve().parent
OUT = ROOT/"results"/"controlled"


def main():
    torch.set_num_threads(1)
    verified = 0
    unscorable = 0
    missing = []
    restored = set()
    for dataset in ("geant", "sbrc"):
        directory = (ROOT/"data"/"geant" if dataset == "geant" else
                     ROOT.parent/"sbrc_final_results"/"choosen_treated_datasets_6h")
        panel = load_panel(directory)
        split = int(len(panel)*.8)
        for context in (3, 28):
            for name in MODELS:
                for link in panel.columns:
                    for seed in (range(42, 47) if name in STOCHASTIC else [42]):
                        path = OUT/dataset/f"context_{context}"/name/link/f"seed_{seed}"
                        required = ["run.json", "metrics.csv", "predictions.csv",
                                    "model.pt" if name in ("MLP", "Transformer", "LSTM", "GRU", "ForwardingLSTM") else "model.joblib"]
                        if not all((path/file).exists() for file in required):
                            missing.append(str(path))
                            continue
                        meta = json.loads((path/"run.json").read_text(encoding="utf-8"))
                        pred = pd.read_csv(path/"predictions.csv")
                        metrics = pd.read_csv(path/"metrics.csv").iloc[0]
                        actual = panel[link].to_numpy()[split:]
                        mask = np.isfinite(actual)
                        observed = pred.Observed.astype(str).str.lower().eq("true").to_numpy()
                        np.testing.assert_array_equal(observed, mask)
                        np.testing.assert_array_equal(pd.to_datetime(pred.Data).to_numpy(), panel.index[split:].to_numpy())
                        np.testing.assert_allclose(pred.Actual_Mbps.to_numpy()[mask], actual[mask])
                        assert np.isfinite(pred.Prediction_Mbps.to_numpy()).all(), path
                        assert int(metrics.observed_test_intervals) == mask.sum(), path
                        assert meta["test_start"] == str(panel.index[split]), path
                        assert meta["context"] == context and meta["seed"] == seed, path
                        if (name, context) not in restored:
                            if name in ("MLP", "Transformer", "LSTM", "GRU", "ForwardingLSTM"):
                                saved = torch.load(path/"model.pt", map_location="cpu", weights_only=True)
                                model = Network(saved["model"], saved["context"], saved["config"])
                                model.load_state_dict(saved["state_dict"])
                            else:
                                saved = joblib.load(path/"model.joblib")
                                model = saved["model"]
                            if name in ("SARIMA", "ARIMA"):
                                first = float(np.asarray(model.forecast(1)).item())
                            else:
                                history, _, _ = preparation(panel[link].to_numpy()[:split])
                                first = online(model, history[-context:], [np.nan])[0]
                            first = first*saved["scale"] + saved["lower"]
                            np.testing.assert_allclose(first, pred.Prediction_Mbps.iloc[0], rtol=1e-6, atol=1e-6)
                            restored.add((name, context))
                        if mask.any():
                            predicted = pred.Prediction_Mbps.to_numpy()[mask]
                            truth = actual[mask]
                            rmse = np.sqrt(np.mean((truth-predicted)**2))
                            values = dict(RMSE=rmse, MAE=np.mean(np.abs(truth-predicted)),
                                          NRMSE_mean=rmse/(abs(truth.mean())+1e-12),
                                          SMAPE=np.mean(200*np.abs(truth-predicted)/(np.abs(truth)+np.abs(predicted)+1e-12)))
                            for key, value in values.items():
                                np.testing.assert_allclose(metrics[key], value, rtol=1e-7, atol=1e-8, err_msg=str(path))
                        else:
                            assert all(pd.isna(metrics[key]) for key in ("RMSE", "MAE", "NRMSE_mean", "SMAPE")), path
                            unscorable += 1
                        verified += 1
    report = dict(verified_runs=verified, restored_model_context_pairs=len(restored), unscorable_runs=unscorable,
                  missing_runs=missing, complete=not missing)
    (OUT/"audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: len(value) if key == "missing_runs" else value for key, value in report.items()}, indent=2))
    if missing:
        raise RuntimeError(f"Missing {len(missing)} runs; see audit.json")


if __name__ == "__main__":
    main()
