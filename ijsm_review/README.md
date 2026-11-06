# GEANT experiments

For the controlled per-link comparison (Persistence, Ridge, Random Forest,
MLP, standard Transformer, LSTM, GRU, ARIMA, SARIMA and Forwarding LSTM;
contexts 3/28 and five stochastic seeds), see [CONTROLLED.md](CONTROLLED.md).
Run it with `& .\ijsm_review\run_controlled.ps1`.

To prepare and run the forwarding LSTM and all three benchmarks on the GEANT
datasets in `ijsm_review/data/geant` and the six selected BBR datasets in
`sbrc_final_results/choosen_treated_datasets_6h`, open PowerShell in the
project root and run:

```powershell
& .\ijsm_review\run_all.ps1
```

The script installs an isolated Python 3.13 environment, fetches the pinned
benchmark implementations, runs the checks, and then trains all four models.
It launches each LSTM link in a separate process and resumes outputs whose
saved source hash and test period match, so rerunning the command keeps the
completed GEANT links.
Use `& .\ijsm_review\run_all.ps1 -SetupOnly` to prepare dependencies without
training. The BBR files in `ijsm_review/data` currently match the selected
SBRC files byte for byte, so they are run once under `sbrc`. If they differ,
the script also runs the `bbr` group. Full training includes extensive LSTM
parameter search and may take a long time.

Outputs are written under `ijsm_review/results/forwarding_lstm/<dataset>` and
`ijsm_review/results/benchmarks/<dataset>/<model>`. All four methods use the
same shared timestamp period and 80/20 split for a given dataset group.
The forwarding LSTM retains its original 3-step context; benchmarks use 28
steps. Their architecture, training, and tuning protocols also differ, so
the scores compare these configured methods rather than isolate architecture
alone.
The selected BBR `es-pr` link has no observed values in the common test period;
it receives predictions but has no error score.

Generate comparison charts from completed runs with:

```powershell
& .\ijsm_review\.venv-experiments\Scripts\python.exe .\ijsm_review\plot_results.py
```

The charts and their source tables are in `ijsm_review/results/plots`.

For iTransformer, TimeBridge, and TimeMixer++ benchmarks, see [BENCHMARKS.md](BENCHMARKS.md).

Prepare the six selected BBR datasets and all ten GEANT datasets:

```powershell
python ijsm_review/aggregate_6h.py
```

Outputs are in `ijsm_review/data` (BBR) and `ijsm_review/data/geant`.
Six-hour intervals start at 00:00, 06:00, 12:00 and 18:00. Empty or
all-missing intervals are `-1`; missing markers are excluded from averages.

Train and evaluate the proposed algorithm on GEANT:

```powershell
python -m pip install tensorflow scikit-learn pandas numpy
python ijsm_review/forward_prediction.py
```

The pipeline uses an 80/20 chronological split, training-only interpolation
and scaling, two 64-unit LSTM layers, Huber loss, Adam (0.001), look-back 3,
and the existing 28-step hybrid loop. Observations feed the next prediction
when available; predictions feed it across missing intervals. This is online
hybrid prediction: 28 is a processing window, not a fixed 28-step forecast
without observations.

Parameter selection uses five chronological folds over epochs 100/300/500
and batch sizes 32/64/128, with early stopping (patience 10). Each fold starts
with a fresh model. Final training selects an epoch count using the last 20%
of training, then refits on the whole training portion. This search trains
45 candidate models per dataset, plus final fits.

Models, timestamp-aligned predictions, observed-only RMSE/MAE/NRMSE/SMAPE,
selected parameters, and scaling bounds are saved in
`ijsm_review/results/forwarding_lstm/geant`.
The existing fuzzy-accuracy helper is also reused, with its original 0–2000
Mbit/s bins; out-of-range pairs are excluded and their evaluated count is saved.
Predictions and metrics use Mbit/s; prepared datasets retain the source bit/s.
Missing values are masked before unit conversion. Evaluation starts at the
first test interval, correcting the older implementation's look-back shift.
These corrections mean results need not reproduce earlier experiment metrics.

Run lightweight checks without TensorFlow:

```powershell
python -m unittest discover -s ijsm_review
```
