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

## Reference

This project is motivated by the following paper:

> G Bello-Orgaz, J Hernandez-Castro… - … Distributed Computing VIII, 2015 - Springer (2015). *A survey of social web mining applications for disease outbreak detection*. … Distributed Computing VIII, 2015 - Springer. https://link.springer.com/chapter/10.1007/978-3-319-10422-5_36

Paper author: Prof. Julio Hernandez-Castro, University of Kent (J.C.Hernandez-Castro@kent.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
