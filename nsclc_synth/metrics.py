"""Evaluation metrics (comments 9-12): discrimination, threshold metrics, calibration,
decision-curve net benefit, Mondrian split-conformal prediction sets, uncertainty quality."""
from __future__ import annotations

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score


def ece(y, p, bins=10):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(p, edges[1:-1])
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def calibration_slope_intercept(y, p):
    lp = logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    slope = LogisticRegression(C=1e6).fit(lp, y).coef_[0, 0]
    # calibration-in-the-large: intercept with slope fixed at 1 (offset model), via Newton steps
    a = 0.0
    for _ in range(50):
        q = 1 / (1 + np.exp(-(lp[:, 0] + a)))
        a += (y - q).sum() / max((q * (1 - q)).sum(), 1e-9)
    return float(slope), float(a)


def net_benefit(y, p, thresholds=np.arange(0.05, 0.61, 0.05)):
    n, prev = len(y), y.mean()
    nb, nb_all = [], []
    for t in thresholds:
        pos = p >= t
        tp, fp = (pos & (y == 1)).sum(), (pos & (y == 0)).sum()
        nb.append(tp / n - fp / n * t / (1 - t))
        nb_all.append(prev - (1 - prev) * t / (1 - t))
    return thresholds, np.array(nb), np.array(nb_all)


def binary_metrics(y, p, threshold):
    y = np.asarray(y).astype(int)
    pred = p >= threshold
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    sd = lambda a, b: a / b if b else np.nan
    slope, intercept = calibration_slope_intercept(y, p)
    return {"auroc": roc_auc_score(y, p), "auprc": average_precision_score(y, p),
            "sensitivity": sd(tp, tp + fn), "specificity": sd(tn, tn + fp),
            "ppv": sd(tp, tp + fp), "npv": sd(tn, tn + fn),
            "brier": float(np.mean((p - y) ** 2)), "ece": ece(y, p),
            "cal_slope": slope, "cal_intercept": intercept, "prevalence": float(y.mean())}


class MondrianConformal:
    """Class-conditional (Mondrian) split conformal classifier for a binary outcome.
    Guarantees >= 1 - alpha coverage within each true class under exchangeability."""

    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def fit(self, y_cal, p_cal):
        self.q = {}
        for c in (0, 1):
            s = 1 - (p_cal[y_cal == c] if c == 1 else 1 - p_cal[y_cal == c])   # nonconformity
            n = len(s)
            k = int(np.ceil((n + 1) * (1 - self.alpha)))
            self.q[c] = np.inf if k > n else np.sort(s)[k - 1]
        return self

    def sets(self, p):
        inc1 = (1 - p) <= self.q[1]
        inc0 = p <= self.q[0]
        return inc0, inc1

    def evaluate(self, y, p):
        inc0, inc1 = self.sets(p)
        covered = np.where(y == 1, inc1, inc0)
        size = inc0.astype(int) + inc1.astype(int)
        return {"coverage": float(covered.mean()),
                "coverage_class1": float(covered[y == 1].mean()) if (y == 1).any() else np.nan,
                "coverage_class0": float(covered[y == 0].mean()) if (y == 0).any() else np.nan,
                "mean_set_size": float(size.mean()), "singleton_rate": float((size == 1).mean()),
                "empty_rate": float((size == 0).mean())}


def uncertainty_quality(y, p_mean, u, threshold):
    """Does uncertainty flag errors? AUROC of uncertainty for misclassification, and AUROC
    of the 50% most-certain predictions (selective prediction)."""
    err = ((p_mean >= threshold).astype(int) != y).astype(int)
    out = {"u_auroc_for_error": roc_auc_score(err, u) if 0 < err.mean() < 1 else np.nan}
    keep = u <= np.quantile(u, 0.5)
    out["auroc_most_certain_50pct"] = roc_auc_score(y[keep], p_mean[keep]) if 0 < y[keep].mean() < 1 else np.nan
    out["auroc_all"] = roc_auc_score(y, p_mean)
    return out
