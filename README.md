# Rare Outbreak-Like Signal Flagging: Balanced Calibrated LightGBM vs. Logistic Regression

Social-web mining for disease outbreak detection has to flag rare, weak signals in noisy text-derived data, where a classifier that ignores class imbalance can look accurate while missing most of the events of interest. The survey "A survey of social web mining applications for disease outbreak detection" (2015) reviews this setting. This experiment asks whether a class-balanced, calibrated gradient-boosting model with per-alert feature attributions handles such rare positives better than a standard logistic regression.

**Question:** Can a class-balanced, calibrated gradient-boosting model with per-alert feature attributions flag rare outbreak-like signals more reliably than a standard logistic regression on a public social-text-derived dataset?

**Dataset:** UCI Adult (income >50K), loaded via the `ucimlrepo` package with an OpenML fallback. This is a stand-in tabular dataset with a rare positive class. It is not the paper's own data and not social-media outbreak data.

**Baseline:** Logistic regression on standardised and one-hot features, trained on the raw imbalanced labels.

**Improved model:** Class-balanced LightGBM with cross-validated hyperparameters (on the training set only), sigmoid calibration, and per-alert top feature contributions for a few flagged cases.

**How to run**
```
pip install -r requirements.txt
python analysis.py
```

**Results:** Results are written to `results.json` (and a comparison chart to `figures/comparison.png`).

## Results (latest run)

On the UCI Adult income dataset (a stand-in for rare-positive detection, not outbreak social-text data), a standard logistic-regression alert classifier reached 66% F1 on the rare >50K class, while a class-balanced, calibrated LightGBM model reached 71% F1 on the same held-out test split (single split, so variance is unknown).

| Metric | Value |
|---|---|
| baseline_score | 0.6597 |
| improved_score | 0.715 |
| improvement | 0.05521 |
| baseline_score_std | 0 |
| improved_score_std | 0 |
| baseline_f1 | 0.6597 |
| improved_f1 | 0.715 |
| baseline_recall | 0.5982 |
| improved_recall | 0.6554 |
| baseline_brier | 0.1033 |
| improved_brier | 0.08792 |

Run on 2026-10-23. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Adult is a tabular income dataset, not social-media outbreak data; it is only a stand-in for a rare-positive-class task. Threshold fixed at 0.5 for both models. Best LightGBM grid params (last seed): {'learning_rate': 0.1, 'n_estimators': 200, 'num_leaves': 31}. Test set used once per model; tuning used only train CV. Baseline is deterministic, so its std is 0. Single split, so results have split-level variance.

## Reference

This project is motivated by the following paper:

> G Bello-Orgaz, J Hernandez-Castro… - … Distributed Computing VIII, 2015 - Springer (2015). *A survey of social web mining applications for disease outbreak detection*. … Distributed Computing VIII, 2015 - Springer. https://link.springer.com/chapter/10.1007/978-3-319-10422-5_36

Paper author: Prof. Julio Hernandez-Castro, University of Kent (J.C.Hernandez-Castro@kent.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
