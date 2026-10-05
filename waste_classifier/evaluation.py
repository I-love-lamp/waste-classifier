"""Scoring helpers for the evaluation notebook.

Convention: the "positive" class for binary metrics (ROC, PR, thresholds) is R (recyclable),
so p_pos is the predicted probability of R. Swap with `positive="O"` if needed.
"""
import numpy as np
import pandas as pd
from sklearn import metrics


def predict_dataset(model, ds):
    """Run a model over a (non-shuffled) dataset. Returns (y_true, probs) as class-index and probability arrays."""
    y_true, probs = [], []
    for images, labels in ds:
        probs.append(model.predict_on_batch(images))
        y_true.append(np.argmax(labels, axis=1))
    return np.concatenate(y_true), np.concatenate(probs)


def extract_features(extractor, ds):
    """Like predict_dataset but returns (features, y_true) for cached-feature training."""
    feats, y_true = [], []
    for images, labels in ds:
        feats.append(extractor.predict_on_batch(images))
        y_true.append(np.argmax(labels, axis=1))
    return np.concatenate(feats), np.concatenate(y_true)


def summary(y_true, probs, class_names, positive="R") -> pd.Series:
    """Headline metrics at the default argmax decision."""
    pos = class_names.index(positive)
    y_pred = probs.argmax(1)
    p_pos = probs[:, pos]
    y_bin = (y_true == pos).astype(int)
    return pd.Series({
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "balanced_accuracy": metrics.balanced_accuracy_score(y_true, y_pred),
        "macro_f1": metrics.f1_score(y_true, y_pred, average="macro"),
        "mcc": metrics.matthews_corrcoef(y_true, y_pred),
        "roc_auc": metrics.roc_auc_score(y_bin, p_pos),
        "average_precision": metrics.average_precision_score(y_bin, p_pos),
        "log_loss": metrics.log_loss(y_true, probs, labels=list(range(len(class_names)))),
        "brier": metrics.brier_score_loss(y_bin, p_pos),
        "ece": expected_calibration_error(y_true, probs),
        "n": len(y_true),
    })


def per_class_report(y_true, probs, class_names) -> pd.DataFrame:
    report = metrics.classification_report(y_true, probs.argmax(1), target_names=class_names, output_dict=True)
    return pd.DataFrame(report).T


def expected_calibration_error(y_true, probs, n_bins: int = 10) -> float:
    """Weighted gap between confidence and accuracy across confidence bins (lower is better)."""
    conf = probs.max(1)
    correct = probs.argmax(1) == y_true
    bins = np.linspace(0.5, 1.0, n_bins + 1)  # binary: confidence is never below 0.5
    idx = np.clip(np.digitize(conf, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def threshold_sweep(y_true, p_pos, pos_index: int, thresholds=None) -> pd.DataFrame:
    """Metrics when predicting positive iff p_pos >= threshold."""
    thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2) if thresholds is None else thresholds
    y_bin = (y_true == pos_index).astype(int)
    rows = []
    for t in thresholds:
        pred = (p_pos >= t).astype(int)
        rows.append({
            "threshold": t,
            "accuracy": metrics.accuracy_score(y_bin, pred),
            "precision": metrics.precision_score(y_bin, pred, zero_division=0),
            "recall": metrics.recall_score(y_bin, pred, zero_division=0),
            "f1": metrics.f1_score(y_bin, pred, zero_division=0),
            "balanced_accuracy": metrics.balanced_accuracy_score(y_bin, pred),
        })
    return pd.DataFrame(rows)


def coverage_curve(y_true, probs, thresholds=None) -> pd.DataFrame:
    """For each "unsure" cutoff: share of items answered and accuracy on those vs the ones flagged unsure.

    This is the trade-off behind the app's UNSURE_BELOW setting.
    """
    thresholds = np.round(np.arange(0.5, 1.0, 0.025), 3) if thresholds is None else thresholds
    conf = probs.max(1)
    correct = probs.argmax(1) == y_true
    rows = []
    for t in thresholds:
        sure = conf >= t
        rows.append({
            "unsure_below": t,
            "coverage": sure.mean(),
            "accuracy_when_sure": correct[sure].mean() if sure.any() else np.nan,
            "accuracy_when_unsure": correct[~sure].mean() if (~sure).any() else np.nan,
            "errors_caught": (~correct & ~sure).sum() / max((~correct).sum(), 1),
        })
    return pd.DataFrame(rows)


def metrics_path(model_path):
    from pathlib import Path
    return Path(model_path).with_suffix(".metrics.json")


def score_model(model, test_ds, class_names, positive="R") -> dict:
    """Headline TEST metrics for the app's About panel, saved next to the model by train.py."""
    from datetime import date

    y_true, probs = predict_dataset(model, test_ds)
    s = summary(y_true, probs, class_names, positive)
    return {
        "accuracy": float(s.accuracy),
        "balanced_accuracy": float(s.balanced_accuracy),
        "macro_f1": float(s.macro_f1),
        "roc_auc": float(s.roc_auc),
        "n_test": int(s.n),
        "dataset": "Kaggle techsash/waste-classification-data TEST split",
        "evaluated": date.today().isoformat(),
    }


def main():
    """Re-score an existing model on TEST and write its .metrics.json: python -m waste_classifier.evaluation [model]"""
    import argparse
    import json
    from pathlib import Path

    import keras

    from .predictor import DEFAULT_MODEL_PATH
    from .train import load_data

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("model", type=Path, nargs="?", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data", type=Path, help="dataset root (default: download from Kaggle)")
    args = parser.parse_args()

    _, _, test_ds = load_data(args.data)
    scores = score_model(keras.models.load_model(args.model), test_ds, test_ds.class_names)
    metrics_path(args.model).write_text(json.dumps(scores, indent=2))
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
