# Figure captions

**Figure 1. Reported throughput forecast errors by model and network.** Panels (a–c) show the six RNP links and panels (d–f) show the ten GÉANT links. Open circles represent individual links and diamonds mark the across-link median. Vertical offsets separate overlapping points. NRMSE is RMSE divided by the mean of the evaluated target values. Lower values indicate smaller reported errors.

**Figure 2. Reported NRMSE by directed link and model.** Panels (a) and (b) show RNP and GÉANT, respectively. Both panels use the same color scale. The G01–G10 link codes are mapped to source and destination names in the accompanying CSV.

**Figure 3. Reported wall-clock experiment runtime by model and network.** Open circles represent individual links and diamonds mark the across-link median. The horizontal axis is logarithmic. The recorded duration includes preprocessing and model fitting or tuning, as applicable, through evaluation.

**Evaluation scope.** These figures use the November 9, 2026 grid evaluation JSONs listed in the accompanying CSV. All five models have error and runtime values for the same 16 links. Test offsets, treatment of missing values, and forecast procedures differ by model, so values describe the respective runs and do not establish a controlled ranking. Fuzzy accuracy is omitted from the figures because all 79 reported values equal 1.0 and one GÉANT SARIMA value is missing.
