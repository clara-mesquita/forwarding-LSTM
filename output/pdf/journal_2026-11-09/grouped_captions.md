# Grouped-by-link figure captions

**MAE, RMSE, and NRMSE.** Each figure shows one metric, with RNP (six directed links) on the left and GÉANT (ten directed links) on the right. Each group represents a link, and the five bars represent the five models. Both panels use the same horizontal scale. NRMSE is RMSE divided by the mean of the evaluated target values. The full directed GÉANT link names are in the accompanying plotted metrics CSV.

**Cohen’s κ.** Unweighted agreement between observed and predicted throughput after assigning both to fixed 100 Mbit/s classes using floor(throughput / 100). Dots on the zero line mark exact zeros. A colored × at the right edge marks an undefined value when both series occupy the same single class. The accompanying replay CSV gives the number of evaluated pairs.

**Evaluation scope.** The error figures use the November 9, 2026 grid exports. For κ, the saved neural models were replayed and SARIMA was refit with its original AIC grid. Each replay had to reproduce the exported MAE, RMSE, and NRMSE within numerical tolerance before κ was accepted. The replay preserves each run’s test procedure, including the original three-interval (18-hour) target offset in the Forwarding LSTM evaluator. These figures describe the archived runs under their respective evaluation procedures.
