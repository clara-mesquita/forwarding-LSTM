# GEANT experiments

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
selected parameters, and scaling bounds are saved in `ijsm_review/results/geant`.
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
