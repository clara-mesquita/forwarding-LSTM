import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from benchmark_data import load_panel, prepare_training, validation_windows


ROOT = Path(__file__).resolve().parent
HAS_TORCH = importlib.util.find_spec("torch") is not None


class BenchmarkDataTest(unittest.TestCase):
    def test_panel_aligns_timestamps_not_row_numbers(self):
        with tempfile.TemporaryDirectory() as folder:
            for name, start, values in [("a", "2024-01-01", [1e6, 2e6, -1, 4e6]),
                                        ("b", "2024-01-01 06:00", [5e6, 6e6, 7e6, 8e6])]:
                pd.DataFrame({"Data": pd.date_range(start, periods=4, freq="6h"),
                              "Vazao": values}).to_csv(Path(folder) / f"{name}.csv", index=False)
            panel = load_panel(folder)
        self.assertEqual(len(panel), 3)
        self.assertEqual(panel.iloc[0].tolist(), [2., 5.])
        self.assertTrue(np.isnan(panel.iloc[1, 0]))

    def test_missing_targets_are_masked_and_validation_is_causal(self):
        raw = np.repeat(np.arange(20, dtype=float).reshape(-1, 1), 2, axis=1)
        raw[4, 0] = np.nan
        raw[12, 0] = np.nan
        x, y, mask, mean, std, filled = prepare_training(raw[:12], 4, 1)
        self.assertFalse(mask[0, 0, 0])
        self.assertTrue(np.isfinite(x).all())
        val_x, _, _ = validation_windows(raw, 12, 4, 1, mean, std, filled)
        # At target t=13, missing t=12 must be filled with t=11, not t=13.
        self.assertAlmostEqual(float(val_x[1, -1, 0]), float((11 - mean[0]) / std[0]), places=5)
        self.assertLess(float(mean[0]), 12.)


@unittest.skipUnless(HAS_TORCH, "Run with .venv-benchmarks for PyTorch checks")
class BenchmarkTorchTest(unittest.TestCase):
    def test_online_feedback_does_not_use_current_or_future_targets(self):
        import torch
        from run_benchmarks import predict_online

        class Model(torch.nn.Module):
            def forward(self, x, *_):
                return x[:, -1:, :] + 1

        actual = np.array([[np.nan], [10.], [np.nan]])
        result = predict_online(Model(), np.array([[1.], [2.], [3.]]), actual,
                                np.zeros(1), np.ones(1), 1, "cpu")
        np.testing.assert_allclose(result[:, 0], [4., 5., 11.])

    def test_direct_forecast_does_not_assimilate_within_block(self):
        import torch
        from run_benchmarks import predict_online

        class Model(torch.nn.Module):
            def forward(self, x, *_):
                return torch.cat([x[:, -1:, :] + 1, x[:, -1:, :] + 2], dim=1)

        actual = np.array([[100.], [np.nan], [np.nan]])
        result = predict_online(Model(), np.array([[1.], [2.], [3.]]), actual,
                                np.zeros(1), np.ones(1), 2, "cpu")
        # t=1 is forecast before observing 100 at t=0; then feedback uses [100, 5].
        np.testing.assert_allclose(result[:, 0], [4., 5., 6.])

    def test_upstream_models_forward_and_backward(self):
        for name in ("iTransformer", "TimeBridge", "TimeMixerPP"):
            with self.subTest(model=name):
                code = f"""
import torch
import warnings
from benchmark_models import build_model
warnings.simplefilter('error', RuntimeWarning)
torch.set_num_threads(2)
torch.manual_seed(42)
model, _, _ = build_model({name!r}, 16, 3, 3, d_model=16)
x = torch.randn(2, 16, 3)
output = model(x, None, None, None)
assert output.shape == (2, 3, 3), output.shape
assert torch.isfinite(output).all()
output.square().mean().backward()
assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
model.eval()
with torch.no_grad():
    assert torch.isfinite(model(torch.ones_like(x), None, None, None)).all()
if {name!r} == 'TimeMixerPP':
    assert all(head.weight.grad is not None and head.weight.grad.abs().sum() > 0 for head in model.predict_layers)
"""
                subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, timeout=120)


if __name__ == "__main__":
    unittest.main()
