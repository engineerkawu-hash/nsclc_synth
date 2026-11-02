"""Systemic molecular and clinical stream: preprocessing (thesis Section 3.4.3).

One code path serves both the synthetic cohort and real data. Every statistic used in a
transform (medians, means, SDs, category levels) is learned in `fit` on the training
partition only, then applied unchanged in `transform` (no leakage).

Steps:
  1. unify missing codes ("not reported", "Unknown", "--", "") -> NaN
  2. continuous variables: optional log transform, median imputation + missing indicator,
     z-standardisation (Eq. 3.24)
  3. ordinal variables: integer coding, median imputation + missing indicator, standardisation
  4. categorical variables: one-hot with an explicit "missing" level; unseen levels -> all zeros
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

MISSING_CODES = {"not reported", "unknown", "--", "", "nan", "none", "not applicable"}


def unify_missing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            s = out[c].astype(str).str.strip()
            out[c] = out[c].where(~s.str.lower().isin(MISSING_CODES), np.nan)
    return out


@dataclass
class SystemicPreprocessor:
    continuous: List[str]
    ordinal: List[str] = field(default_factory=list)
    categorical: List[str] = field(default_factory=list)
    log_vars: List[str] = field(default_factory=list)
    min_missing_for_indicator: float = 0.0  # add indicator if any missingness seen in training

    def fit(self, df: pd.DataFrame) -> "SystemicPreprocessor":
        d = unify_missing(df)
        self.stats_: Dict[str, Dict] = {}
        for c in self.continuous + self.ordinal:
            x = self._num(d, c)
            med = float(np.nanmedian(x)) if np.isfinite(x).any() else 0.0
            filled = np.where(np.isnan(x), med, x)
            sd = float(filled.std()) or 1.0
            self.stats_[c] = dict(median=med, mean=float(filled.mean()), sd=sd,
                                  indicator=bool(np.isnan(x).mean() > self.min_missing_for_indicator))
        self.levels_ = {c: sorted(d[c].dropna().astype(str).unique().tolist()) + ["__missing__"]
                        for c in self.categorical}
        self.feature_names_ = self._names()
        return self

    def _num(self, d, c):
        x = pd.to_numeric(d[c], errors="coerce").astype(float).values
        if c in self.log_vars:
            x = np.log(np.clip(x, 1e-6, None))
        return x

    def _names(self):
        names = []
        for c in self.continuous + self.ordinal:
            names.append(c)
            if self.stats_[c]["indicator"]:
                names.append(f"{c}__missing")
        for c in self.categorical:
            names += [f"{c}={lv}" for lv in self.levels_[c]]
        return names

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        d = unify_missing(df)
        cols = {}
        for c in self.continuous + self.ordinal:
            st = self.stats_[c]
            x = self._num(d, c)
            miss = np.isnan(x)
            cols[c] = (np.where(miss, st["median"], x) - st["mean"]) / st["sd"]
            if st["indicator"]:
                cols[f"{c}__missing"] = miss.astype(float)
        for c in self.categorical:
            v = d[c].astype(object).where(d[c].notna(), "__missing__").astype(str)
            for lv in self.levels_[c]:
                cols[f"{c}={lv}"] = (v == lv).astype(float).values
        return pd.DataFrame(cols, index=df.index)[self.feature_names_]

    def fit_transform(self, df):
        return self.fit(df).transform(df)
