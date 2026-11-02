"""Shared machinery for the experiment programme: cohorts, splits, baselines, evaluation."""
from __future__ import annotations

import copy
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from .config import DGPConfig
from .data import MODALITIES, Preparer, interpretable_frame, subset
from .generator import generate
from .metrics import MondrianConformal, binary_metrics, net_benefit
from .model import FittedModel, ModelSpec

TASKS = {0: "dcb", 1: "irae_g2", 2: "irae_g3"}
YCOLS = ["dcb", "irae_g2", "irae_g3"]
TEST_N = 2000
_TEST_CACHE: Dict[str, dict] = {}


def test_cohort(cfg: DGPConfig, seed: int = 900001) -> dict:
    c = copy.deepcopy(cfg)
    c.n, c.seed = TEST_N, seed
    key = repr(c.to_dict())
    if key not in _TEST_CACHE:
        _TEST_CACHE[key] = generate(c)
    return _TEST_CACHE[key]


def dev_split(cohort: dict, seed: int, calib_frac: float = 0.25):
    """Development cohort -> training rows and a calibration split (Platt + conformal)."""
    y = cohort["patients"]["dcb"].values
    idx = np.arange(len(y))
    strat = y if 0 < y.sum() < len(y) - 1 and min(y.sum(), len(y) - y.sum()) >= 4 else None
    tr, ca = train_test_split(idx, test_size=calib_frac, random_state=seed, stratify=strat)
    return tr, ca


class LogisticBaseline:
    """Conventional baseline: L2 logistic regression on interpretable variables with
    training-median imputation and modality-availability indicators; Platt-recalibrated."""

    def __init__(self, cols: Optional[List[str]] = None, tasks=(0, 1), C: float = 0.5):
        self.cols, self.tasks, self.C = cols, tasks, C

    def _frame(self, cohort, idx=None, drop=()):
        f = interpretable_frame(cohort)
        p = cohort["patients"]
        for k, col in zip(MODALITIES, ["has_pathology", "has_radiology", "has_systemic"]):
            av = p[col].values.astype(bool) & (k not in drop)
            f[f"avail_{k}"] = av.astype(float)
            if k in drop:
                from .data import INTERP
                for c, lg in INTERP[k]:
                    f[("log_" if lg else "") + c] = np.nan
        if self.cols is not None:
            f = f[self.cols]
        return f if idx is None else f.iloc[idx]

    def fit(self, cohort, tr, ca):
        F = self._frame(cohort, tr)
        self.med = F.median()
        Z = F.fillna(self.med)
        self.mu, self.sd = Z.mean(), Z.std().replace(0, 1.0)
        X = ((Z - self.mu) / self.sd).values
        y = cohort["patients"][YCOLS].values
        self.m = [LogisticRegression(C=self.C, max_iter=2000).fit(X, y[tr, t]) for t in self.tasks]
        Lc = self._logits(cohort, ca)
        self.platt = [LogisticRegression(C=1e4).fit(Lc[:, [j]], y[ca, t]) for j, t in enumerate(self.tasks)]
        return self

    def _logits(self, cohort, idx=None, drop=()):
        X = ((self._frame(cohort, idx, drop).fillna(self.med) - self.mu) / self.sd).values
        return np.column_stack([m.decision_function(X) for m in self.m])

    def predict_raw(self, cohort, idx=None, drop=()):
        return expit(self._logits(cohort, idx, drop))

    def predict(self, cohort, idx=None, drop=()):
        L = self._logits(cohort, idx, drop)
        return np.column_stack([pl.predict_proba(L[:, [j]])[:, 1] for j, pl in enumerate(self.platt)])


def evaluate(y: np.ndarray, p: np.ndarray, thr: float, conf: Optional[MondrianConformal] = None,
             prefix: str = "") -> dict:
    out = {f"{prefix}{k}": v for k, v in binary_metrics(y, p, thr).items()}
    if conf is not None:
        out.update({f"{prefix}{k}": v for k, v in conf.evaluate(y, p).items()})
    return out


def run_deep(spec: ModelSpec, cohort: dict, tr, ca, seed: int):
    prep = Preparer().fit({**cohort, "patients": cohort["patients"].iloc[tr],
                           **{k: cohort[k][tr] for k in ["z_pathology", "z_radiology", "z_systemic"]}})
    full = prep.transform(cohort)
    model = FittedModel(spec).fit(subset(full, tr), subset(full, ca), seed=seed)
    return model, prep, full


def eval_model_on(model_kind: str, model, prep, test: dict, train_prev: np.ndarray,
                  confs: Optional[list], drop=(), tasks=(0, 1)) -> dict:
    if model_kind == "deep":
        d = prep.transform(test, drop=drop)
        P, R = model.predict(d), model.predict_raw(d)
    else:
        P, R = model.predict(test, drop=drop), model.predict_raw(test, drop=drop)
    y = test["patients"][YCOLS].values
    row = {}
    for j, t in enumerate(tasks):
        row.update(evaluate(y[:, t], P[:, j], train_prev[t], None, prefix=f"{TASKS[t]}_"))
        if confs:
            row.update({f"{TASKS[t]}_{k}": v for k, v in confs[j].evaluate(y[:, t], R[:, j]).items()})
    return row


def conformal_for(model_kind, model, full_or_cohort, ca, cohort, tasks=(0, 1), alpha=0.1):
    if model_kind == "deep":
        Pc = model.predict_raw(subset(full_or_cohort, ca))
    else:
        Pc = model.predict_raw(cohort, ca)
    y = cohort["patients"][YCOLS].values[ca]
    return [MondrianConformal(alpha).fit(y[:, t], Pc[:, j]) for j, t in enumerate(tasks)]


def append_rows(path: str, rows: List[dict]):
    """Append rows, always aligned to the existing header; widen the file if new columns appear."""
    df = pd.DataFrame(rows)
    if not os.path.exists(path):
        df.to_csv(path, index=False)
        return
    header = pd.read_csv(path, nrows=0).columns.tolist()
    new = [c for c in df.columns if c not in header]
    if new:
        old = pd.read_csv(path)
        pd.concat([old, df], ignore_index=True)[header + new].to_csv(path, index=False)
    else:
        df.reindex(columns=header).to_csv(path, mode="a", header=False, index=False)


def done_keys(path: str, cols: List[str]) -> set:
    if not os.path.exists(path):
        return set()
    d = pd.read_csv(path)
    return set(map(tuple, d[cols].astype(str).values))
