# Figures styled after the supplied templates

The MAE reference supplies the error-figure palette; the Cohen Kappa reference supplies the kappa palette. The first four colors were sampled exactly from the source images. A complementary fifth color was added for Transformer, which is absent from the templates.

The references' horizontal grouping, upward category ordering, black frame, dashed vertical grid, sans-serif type, internal legend and English title pattern are reproduced. Actual current data and model names are retained. The kappa reference omits category text; the recreated figures include link labels for identification.

All figures show SARIMA, GRU, LSTM, Forwarding LSTM and Transformer. GEANT omits Ankara–Lisboa, LUT–Viena, Lisboa–RESTENA and Anella–Poznań as requested. RNP omits ES–PR. Five stochastic seeds are averaged per link; SARIMA is deterministic. Error metrics use observed test intervals only. Cohen's kappa is unweighted with 100-Mbit/s classes extended beyond 0–2000; × denotes an undefined value, not zero.

Context 3 means 18 hours of history; context 28 means 7 days. Forecast horizon is 6 hours. SARIMA uses its full fitted historical state. Context is identified by filenames/captions, not an extra subtitle. Each dataset/metric uses matching axis limits across contexts.

## geant_context_3_cohen_kappa

COHEN KAPPA for GEANT, context 3; one-step online prediction, mean across seeds. See the shared methodology and exclusions above.

## geant_context_28_cohen_kappa

COHEN KAPPA for GEANT, context 28; one-step online prediction, mean across seeds. See the shared methodology and exclusions above.

## geant_context_3_smape

SMAPE for GEANT, context 3; one-step online prediction, mean across seeds. See the shared methodology and exclusions above.

## geant_context_28_smape

SMAPE for GEANT, context 28; one-step online prediction, mean across seeds. See the shared methodology and exclusions above.

## geant_context_3_rmse

RMSE for GEANT, context 3; one-step online prediction, mean across seeds. See the shared methodology and exclusions above.

## geant_context_28_rmse

RMSE for GEANT, context 28; one-step online prediction, mean across seeds. See the shared methodology and exclusions above.

## geant_context_3_mae

MAE for GEANT, context 3; one-step online prediction, mean across seeds. See the shared methodology and exclusions above.

## geant_context_28_mae

MAE for GEANT, context 28; one-step online prediction, mean across seeds. See the shared methodology and exclusions above.

## rnp_context_3_cohen_kappa

COHEN KAPPA for RNP, context 3; one-step online prediction, mean across seeds. See the shared methodology and exclusions above.

## rnp_context_28_cohen_kappa

COHEN KAPPA for RNP, context 28; one-step online prediction, mean across seeds. See the shared methodology and exclusions above.

## rnp_context_3_smape

SMAPE for RNP, context 3; one-step online prediction, mean across seeds. See the shared methodology and exclusions above.

## rnp_context_28_smape

SMAPE for RNP, context 28; one-step online prediction, mean across seeds. See the shared methodology and exclusions above.

## rnp_context_3_rmse

RMSE for RNP, context 3; one-step online prediction, mean across seeds. See the shared methodology and exclusions above.

## rnp_context_28_rmse

RMSE for RNP, context 28; one-step online prediction, mean across seeds. See the shared methodology and exclusions above.

## rnp_context_3_mae

MAE for RNP, context 3; one-step online prediction, mean across seeds. See the shared methodology and exclusions above.

## rnp_context_28_mae

MAE for RNP, context 28; one-step online prediction, mean across seeds. See the shared methodology and exclusions above.
