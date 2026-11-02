# Pre-specified success criteria (reviewer comment 20)

These criteria were written on 27 October 2026, after the experiment code was finalised and **before any experiment results were viewed**. They are scored by `scripts/summarise_results.py`, and the thresholds below must not be changed after results are seen.

All criteria apply to the proposed missing-aware model in the default scenario: 300-patient development cohort, 2,000-patient independent test set and 20 replicates, unless stated otherwise. "Strongest conventional baseline" means L2 logistic regression on all interpretable variables.

| # | Criterion | Question | Pass threshold |
|---|---|---|---|
| 1 | Predictive performance | Does it discriminate? | For both tasks, the 2.5th percentile of AUROC across replicates is above 0.55. For durable clinical benefit (DCB), mean AUROC is at least the strongest baseline's minus 0.01. |
| 2 | Calibration | Are probabilities reliable? | For both tasks, mean ECE ≤ 0.05 and mean calibration slope in 0.8–1.2. |
| 3 | Robustness | Does performance survive missing modalities? | DCB AUROC headroom retained at 50% missingness, (AUROC₅₀ − 0.5)/(AUROC₁₀ − 0.5), is at least 0.80. It is also at least the zero-imputation model's retention. |
| 4 | Stability | Does the model behave consistently? | Standard deviation of DCB AUROC across 20 replicates at n = 300 is ≤ 0.03. |
| 5 | Uncertainty | Does it know when it is uncertain? | Mondrian conformal coverage (target 0.90) has mean 0.87–0.95 for both tasks in the default scenario. The ensemble's uncertainty flags errors better than chance: AUROC of uncertainty for misclassification above 0.55. |
| 6 | Explainability | Are explanations stable and coherent? | Median pairwise rank correlation of global feature importance across 20 bootstrap retrainings is ≥ 0.80 for DCB. Attribution share on no-effect variables is below that variables' share of the feature count. |
| 7 | Generalisation | Does performance survive distribution shift? | Worst-case DCB AUROC drop across the covariate shifts tested is ≤ 0.05. Conformal coverage stays ≥ 0.85 under every covariate shift. |
| 8 | Reproducibility | Can another researcher reproduce it? | Re-running the generator reproduces the analysis cohort byte for byte, and the full code, configuration and seeds are packaged. |

A criterion that fails is reported as a failure and discussed. It is not removed and its threshold is not adjusted.
