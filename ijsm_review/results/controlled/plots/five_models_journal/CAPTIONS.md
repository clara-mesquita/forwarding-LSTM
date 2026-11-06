# Figure captions

Each file is a standalone figure with one metric, dataset and context. No titles, subtitles or footnotes are embedded in the artwork.

Figures use a compact 7.2 × 7.4-inch canvas, cropped to artwork with minimal padding; PDF/SVG are vector, PNG is 600 dpi. Axis labels are 13 pt, link labels 12 pt, numeric ticks 11 pt and legend 10.5 pt before resizing.

Values are means across stochastic seeds per link, calculated on observed test intervals. These are descriptive means, without confidence intervals. Smaller errors are better; larger kappa is better.

Cohen’s kappa is unweighted with fixed classes floor(throughput in Mbit/s / 100), extended below zero and above 2000 Mbit/s. A × at the left edge denotes an undefined value, not zero. ES–PR is omitted from the RNP figures at the user's request; it has no observed test values.

The displayed GEANT links exclude the four user-selected directions: Ankara → Lisboa, LUT → Viena, Lisboa → RESTENA and Anella → Poznań. This is a selected subset, not the full GEANT evaluation.

## geant_context_3_cohen_kappa

Cohen’s κ by link for GEANT, using a 3-interval history (18 h) to predict the next 6-h interval. Bars indicate means across seeds. SARIMA uses its full fitted history irrespective of context. See the general notes above for omitted GEANT directions and undefined values.

## geant_context_28_cohen_kappa

Cohen’s κ by link for GEANT, using a 28-interval history (168 h) to predict the next 6-h interval. Bars indicate means across seeds. SARIMA uses its full fitted history irrespective of context. See the general notes above for omitted GEANT directions and undefined values.

## geant_context_3_smape

SMAPE (%) by link for GEANT, using a 3-interval history (18 h) to predict the next 6-h interval. Bars indicate means across seeds. SARIMA uses its full fitted history irrespective of context. See the general notes above for omitted GEANT directions and undefined values.

## geant_context_28_smape

SMAPE (%) by link for GEANT, using a 28-interval history (168 h) to predict the next 6-h interval. Bars indicate means across seeds. SARIMA uses its full fitted history irrespective of context. See the general notes above for omitted GEANT directions and undefined values.

## geant_context_3_rmse

RMSE (Mbit/s) by link for GEANT, using a 3-interval history (18 h) to predict the next 6-h interval. Bars indicate means across seeds. SARIMA uses its full fitted history irrespective of context. See the general notes above for omitted GEANT directions and undefined values.

## geant_context_28_rmse

RMSE (Mbit/s) by link for GEANT, using a 28-interval history (168 h) to predict the next 6-h interval. Bars indicate means across seeds. SARIMA uses its full fitted history irrespective of context. See the general notes above for omitted GEANT directions and undefined values.

## geant_context_3_mae

MAE (Mbit/s) by link for GEANT, using a 3-interval history (18 h) to predict the next 6-h interval. Bars indicate means across seeds. SARIMA uses its full fitted history irrespective of context. See the general notes above for omitted GEANT directions and undefined values.

## geant_context_28_mae

MAE (Mbit/s) by link for GEANT, using a 28-interval history (168 h) to predict the next 6-h interval. Bars indicate means across seeds. SARIMA uses its full fitted history irrespective of context. See the general notes above for omitted GEANT directions and undefined values.

## rnp_context_3_cohen_kappa

Cohen’s κ by link for RNP, using a 3-interval history (18 h) to predict the next 6-h interval. Bars indicate means across seeds. SARIMA uses its full fitted history irrespective of context. See the general notes above for omitted GEANT directions and undefined values.

## rnp_context_28_cohen_kappa

Cohen’s κ by link for RNP, using a 28-interval history (168 h) to predict the next 6-h interval. Bars indicate means across seeds. SARIMA uses its full fitted history irrespective of context. See the general notes above for omitted GEANT directions and undefined values.

## rnp_context_3_smape

SMAPE (%) by link for RNP, using a 3-interval history (18 h) to predict the next 6-h interval. Bars indicate means across seeds. SARIMA uses its full fitted history irrespective of context. See the general notes above for omitted GEANT directions and undefined values.

## rnp_context_28_smape

SMAPE (%) by link for RNP, using a 28-interval history (168 h) to predict the next 6-h interval. Bars indicate means across seeds. SARIMA uses its full fitted history irrespective of context. See the general notes above for omitted GEANT directions and undefined values.

## rnp_context_3_rmse

RMSE (Mbit/s) by link for RNP, using a 3-interval history (18 h) to predict the next 6-h interval. Bars indicate means across seeds. SARIMA uses its full fitted history irrespective of context. See the general notes above for omitted GEANT directions and undefined values.

## rnp_context_28_rmse

RMSE (Mbit/s) by link for RNP, using a 28-interval history (168 h) to predict the next 6-h interval. Bars indicate means across seeds. SARIMA uses its full fitted history irrespective of context. See the general notes above for omitted GEANT directions and undefined values.

## rnp_context_3_mae

MAE (Mbit/s) by link for RNP, using a 3-interval history (18 h) to predict the next 6-h interval. Bars indicate means across seeds. SARIMA uses its full fitted history irrespective of context. See the general notes above for omitted GEANT directions and undefined values.

## rnp_context_28_mae

MAE (Mbit/s) by link for RNP, using a 28-interval history (168 h) to predict the next 6-h interval. Bars indicate means across seeds. SARIMA uses its full fitted history irrespective of context. See the general notes above for omitted GEANT directions and undefined values.
