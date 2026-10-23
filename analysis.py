"""Rare-signal flagging: class-balanced calibrated LightGBM vs. plain logistic regression.

Stand-in experiment: UCI Adult is not outbreak data; it is used only as a public
tabular dataset with a rare positive class (income >50K, ~24%) to mimic
rare-signal detection. Both models share data loading, split, and evaluation.
"""
import json, os, sys, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, brier_score_loss
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import requests

SEED = 42
SEEDS = [42, 43, 44]
MAX_ROWS = 200_000
OUT = "results.json"
DATA_URL = "https://archive.ics.uci.edu/dataset/2/adult"


def fetch_with_retry(fn, tries=4):
    """Call a download function, retrying with exponential backoff on HTTP 429/5xx."""
    for i in range(tries):
        try:
            return fn()
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            if code == 429 or code >= 500:
                wait = 2 ** i
                print(f"HTTP {code}, retrying in {wait}s")
                time.sleep(wait)
                continue
            raise
        except requests.ConnectionError:
            time.sleep(2 ** i)
    raise RuntimeError("download failed after retries")


def load_data():
    """Load UCI Adult via ucimlrepo (fallback: OpenML) and subsample if needed."""
    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_with_retry(lambda: fetch_ucirepo(id=2))
        X = ds.data.features.copy()
        y = ds.data.targets.iloc[:, 0]
    except Exception as e:  # fall back to OpenML mirror of the same dataset
        print(f"ucimlrepo failed ({e}); trying OpenML")
        import openml
        d = fetch_with_retry(lambda: openml.datasets.get_dataset(1590))
        X, y, _, _ = d.get_data(target=d.default_target_attribute)
        X = X.copy()
    y = y.astype(str).str.strip().str.rstrip(".").map(lambda v: 1 if v == ">50K" else 0)
    if len(X) > MAX_ROWS:
        idx = X.sample(MAX_ROWS, random_state=SEED).index
        X, y = X.loc[idx], y.loc[idx]
    print(f"Loaded {len(X)} rows, positive rate {y.mean():.3f}")
    return X.reset_index(drop=True), y.reset_index(drop=True)


def evaluate(name, model, X_te, y_te):
    """Shared evaluation: F1 on the rare class (main), recall and Brier (secondary)."""
    p = model.predict_proba(X_te)[:, 1]
    pred = (p >= 0.5).astype(int)
    return {
        f"{name}_f1": f1_score(y_te, pred, pos_label=1),
        f"{name}_recall": recall_score(y_te, pred),
        f"{name}_brier": brier_score_loss(y_te, p),
    }


def main():
    t0 = time.time()
    X, y = load_data()
    if y.nunique() < 2:
        print("Only one class present; cannot compute metrics"); sys.exit(1)
    # Single stratified split, fixed seed, 25% test.
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    print(f"Train {len(X_tr)}, test {len(X_te)}")

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    # Shared preprocessing: impute; scale+one-hot for LR, ordinal codes for LightGBM.
    pre_lr = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_cols),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore")), cat_cols),
    ])
    X_tr_l, X_te_l = X_tr.copy(), X_te.copy()
    for c in cat_cols:
        X_tr_l[c] = X_tr_l[c].astype("category")
        X_te_l[c] = pd.Categorical(X_te_l[c], categories=X_tr_l[c].cat.categories)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    # BASELINE: standard logistic regression on raw imbalanced labels (deterministic).
    print("Fitting baseline logistic regression")
    base = make_pipeline(pre_lr, LogisticRegression(max_iter=2000))
    base.fit(X_tr, y_tr)
    res = evaluate("baseline", base, X_te, y_te)

    # IMPROVED: class-balanced LightGBM, CV-tuned on train only, sigmoid-calibrated.
    # Each seed gives a separate model; means and stds are reported.
    grid = {"num_leaves": [15, 31], "learning_rate": [0.05, 0.1], "n_estimators": [200]}
    imp_scores, imp_f1s, imp_recalls, imp_briers = [], [], [], []
    best_params = None
    for s in SEEDS:
        print(f"Tuning + fitting improved LightGBM, seed {s}")
        search = GridSearchCV(
            lgb.LGBMClassifier(class_weight="balanced", random_state=s, verbose=-1, n_jobs=2),
            grid, scoring="f1", cv=StratifiedKFold(3, shuffle=True, random_state=s), n_jobs=1,
        )
        search.fit(X_tr_l, y_tr)
        best_params = search.best_params_
        cal = CalibratedClassifierCV(search.best_estimator_, method="sigmoid", cv=3)
        cal.fit(X_tr_l, y_tr)
        r = evaluate("improved", cal, X_te_l, y_te)
        imp_f1s.append(r["improved_f1"]); imp_recalls.append(r["improved_recall"])
        imp_briers.append(r["improved_brier"])

        # Per-alert feature contributions (LightGBM pred_contrib) for a few flagged cases, seed 42 only.
        if s == SEEDS[0]:
            booster = search.best_estimator_.booster_
            flagged = np.where(cal.predict_proba(X_te_l)[:, 1] >= 0.5)[0][:3]
            contrib = booster.predict(X_te_l.iloc[flagged], pred_contrib=True)
            for row, c in zip(flagged, contrib):
                top = np.argsort(-np.abs(c[:-1]))[:3]
                desc = ", ".join(f"{X_te.columns[j]}={c[j]:+.2f}" for j in top)
                print(f"Alert test#{row}: top contributions (log-odds) {desc}")

    # Baseline is deterministic: its std across seeds is 0 by construction.
    base_f1 = res["baseline_f1"]
    metrics = {
        "baseline_score": base_f1,
        "improved_score": float(np.mean(imp_f1s)),
        "improvement": float(np.mean(imp_f1s) - base_f1),
        "baseline_score_std": 0.0,
        "improved_score_std": float(np.std(imp_f1s)),
        "baseline_f1": base_f1,
        "improved_f1": float(np.mean(imp_f1s)),
        "baseline_recall": res["baseline_recall"],
        "improved_recall": float(np.mean(imp_recalls)),
        "baseline_brier": res["baseline_brier"],
        "improved_brier": float(np.mean(imp_briers)),
    }
    out = {
        "question": "Can a class-balanced, calibrated gradient-boosting model with per-alert feature attributions flag rare outbreak-like signals more reliably than a standard logistic regression on a public social-text-derived dataset?",
        "method": "Stand-in experiment on UCI Adult (rare >50K class). Single stratified 75/25 split (seed 42). Baseline: logistic regression on raw labels. Improved: class-balanced LightGBM with CV grid search and sigmoid calibration, with per-alert contributions. Metric: F1 on rare class.",
        "dataset": {
            "name": "UCI Adult (income >50K)",
            "source_url": DATA_URL,
            "n_rows": int(len(X)),
            "n_train": int(len(X_tr)),
            "n_test": int(len(X_te)),
            "same_as_paper": False,
            "baseline_is_papers_method": False,
        },
        "metric_name": "F1 on rare positive class",
        "higher_is_better": True,
        "baseline": "Standard logistic regression on standardised/one-hot features, raw imbalanced labels.",
        "improved": "Class-balanced LightGBM (CV-tuned on train), sigmoid-calibrated probabilities, per-alert top feature contributions.",
        "metrics": {k: float(v) for k, v in metrics.items()},
        "notes": f"Adult is a tabular income dataset, not social-media outbreak data; it is only a stand-in for a rare-positive-class task. Threshold fixed at 0.5 for both models. Best LightGBM grid params (last seed): {best_params}. Test set used once per model; tuning used only train CV. Baseline is deterministic, so its std is 0. Single split, so results have split-level variance.",
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # Optional chart: baseline vs improved for main and secondary metrics.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs("figures", exist_ok=True)
        labels = ["F1", "Recall", "1-Brier"]
        b = [metrics["baseline_f1"], metrics["baseline_recall"], 1 - metrics["baseline_brier"]]
        i = [metrics["improved_f1"], metrics["improved_recall"], 1 - metrics["improved_brier"]]
        xs = np.arange(len(labels))
        plt.figure(figsize=(6, 4))
        plt.bar(xs - 0.2, b, 0.4, label="baseline (LR)")
        plt.bar(xs + 0.2, i, 0.4, label="improved (balanced LGBM)")
        plt.xticks(xs, labels); plt.ylim(0, 1); plt.legend(); plt.title("Test-set metrics")
        plt.tight_layout(); plt.savefig("figures/comparison.png", dpi=120); plt.close()
    except Exception as e:
        print(f"Skipping chart: {e}")

    print(f"Done in {time.time() - t0:.1f}s. Wrote {OUT}")


if __name__ == "__main__":
    main()
