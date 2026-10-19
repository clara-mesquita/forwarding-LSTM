# Multivariate throughput benchmarks

Implemented models:

| CLI name | Implementation |
| --- | --- |
| `iTransformer` | [Official iTransformer](https://github.com/thuml/iTransformer), pinned in `benchmark_sources.json` |
| `TimeBridge` | [Official forecasting TimeBridge](https://github.com/Hank0626/TimeBridge), pinned in `benchmark_sources.json` |
| `TimeMixerPP` | [PyPOTS TimeMixer++ blocks](https://github.com/WenjieDu/PyPOTS/tree/main/pypots/nn/modules/timemixerpp), with a local all-scale averaging adapter |

TimeMixer++ is different from TimeMixer. The author's anonymous release returned
HTTP 403 during setup. The PyPOTS code explicitly identifies itself as an
implementation inspired by that release. Our adapter averages all prediction
heads as permitted by [paper Eq. 3](https://arxiv.org/html/2410.16032v3#S3.E3),
instead of returning only the last scale as the pinned PyPOTS forecast does.
This is an adapted implementation, not an exact author-code reproduction.
The adapter also excludes the zero-frequency FFT bin structurally, preventing
division by zero when a constant input produces tied FFT magnitudes. Learned
stride-2 convolutions perform downsampling.
KEST-Trans is excluded as requested.

## Setup and running

An isolated CPU environment was created at `ijsm_review/.venv-benchmarks`.
From the project root on Windows:

```powershell
python -m venv ijsm_review/.venv-benchmarks
ijsm_review/.venv-benchmarks/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
ijsm_review/.venv-benchmarks/Scripts/python -m pip install -r ijsm_review/requirements-benchmarks.txt
python ijsm_review/setup_benchmarks.py
python ijsm_review/aggregate_6h.py
ijsm_review/.venv-benchmarks/Scripts/python ijsm_review/run_benchmarks.py --model all --dataset geant
ijsm_review/.venv-benchmarks/Scripts/python ijsm_review/run_benchmarks.py --model all --dataset bbr
```

Use the PyTorch installation appropriate to your hardware for GPU experiments.
`requirements-benchmarks.lock.txt` records the complete environment used for
verification on Windows/Python 3.13; the CPU PyTorch index above is needed to
resolve its `+cpu` build. Upstream commit hashes and licenses stay in the fetched
repositories; `setup_benchmarks.py` verifies clean, pinned checkouts. Vendor code,
the environment, and generated results are ignored by Git.

Select one model with `--model iTransformer`, `--model TimeBridge`, or
`--model TimeMixerPP`. The default context is 28 intervals (7 days), with one
6-hour prediction per step. `--pred-len 28` issues direct, non-overlapping
7-day forecasts; it does not assimilate observations within a forecast block.
Keep the same context, horizon, and splits for all compared methods.

## Evaluation protocol

- Each CSV represents one directional throughput link, not an inferred physical
  topology. GEANT and BBR are separate experiments. No adjacency or routing
  information is invented.
- Links are joined by timestamp over their common measurement period. Source
  units are bit/s; training and outputs use Mbit/s. `-1` and blanks are missing.
  Every missing interval inside the common period remains on the shared grid.
- The last 20% is the test set. Within the first 80%, the final 20% selects
  the epoch count using early stopping. Selection-training and final-training
  interpolation/scaling are performed separately; test values never set either.
- Missing training inputs are interpolated within the training period. Missing
  targets are excluded from MSE. Validation histories are filled forward using
  only past observations. Each training split must contain observations for
  every included link.
- The models use common Adam (0.0001), masked MSE, batch size 32, maximum 100
  epochs and patience 10. Model width is 32, with two iTransformer/Mixer layers;
  TimeBridge uses integrated attention, patch downsampling and cointegrated
  attention. TimeBridge patches span 4 intervals (one day). These compact,
  shared experiment settings are not claims of reproducing published scores.
- At inference, each forecast is issued before its target is read. Observed
  values then update the next context; missing values use the model prediction.
  With horizon >1, the complete block is issued before feedback is applied.
- Metrics use only observed test values. The original repository's
  RMSE/MAE/NRMSE/SMAPE/fuzzy evaluation is reused. Fuzzy accuracy retains its
  existing 0–2000 Mbit/s range and records its evaluated count.
- Older per-link LSTM results use different measurement periods and a three-step
  context. They must be rerun on the exported common panel/split with matched
  context before being presented as a controlled comparison. Do not compare
  one-step online scores with direct 28-step scores.

## Outputs and verification

`ijsm_review/results/benchmarks/<dataset>/<model>/` contains:

- `model.pt`: weights, architecture settings, scaling, and channel order.
- `predictions.csv`: timestamps, link names, actual values, predictions,
  observation masks, and forecast horizon steps.
- `metrics.csv`: observed-only metrics per link.
- `aligned_panel_mbps.csv`: the exact synchronized input panel.
- `run.json`: source revision, input SHA-256 hashes, software versions, seed,
  device, chronological boundaries, scaling, selected epochs and timings.

Tests cover timestamp alignment, missing-target masks, causal validation inputs,
online feedback, and forward/backward passes through all three architectures.
The TimeMixer++ test checks that every scale head receives a gradient.

```powershell
ijsm_review/.venv-benchmarks/Scripts/python -m unittest discover -s ijsm_review
ijsm_review/.venv-benchmarks/Scripts/python ijsm_review/run_benchmarks.py --model all --dataset geant --epochs 1 --output-dir ijsm_review/results/smoke_benchmarks
```

One-epoch runs verify execution and output generation. They are not converged
benchmark results. Full training and repeated-seed comparisons remain to run.
