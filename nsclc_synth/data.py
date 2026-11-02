"""Turn a generated cohort into model-ready modality blocks (fit on training data only).

Blocks
  C  clinical (always available): age, sex, histology, smoking, ECOG, stage, line, chemo-IO,
     autoimmune history
  P  pathology: PD-L1 TPS, log TIL density, stromal TIL %, necrosis %  + 512-d embedding
  R  radiology: log volume, sphericity, entropy, log contrast         + 128-d embedding
  S  systemic : log TMB, log NLR, log LDH, albumin, log CRP, log ctDNA, EGFR, STK11 + 64-d embedding
Masks m[:, k] = 1 if modality k (P, R, S) is available.
Within an available modality, sporadic missing values are median-imputed (training medians).
For an unavailable modality the block is filled with training medians (i.e. 0 after
standardisation); how the model then treats it is decided by the missing-modality strategy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

MODALITIES = ["P", "R", "S"]

INTERP = {
    "P": [("pdl1_tps", False), ("til_density", True), ("stromal_til_pct", False), ("necrosis_pct", False)],
    "R": [("tumour_volume_cm3", True), ("sphericity", False), ("glcm_entropy", False), ("glcm_contrast", True)],
    "S": [("tmb", True), ("nlr", True), ("ldh", True), ("albumin", False), ("crp", True), ("ctdna_vaf", True),
          ("egfr_mut", False), ("stk11_mut", False)],
}
EMB = {"P": "z_pathology", "R": "z_radiology", "S": "z_systemic"}
AVAIL = {"P": "has_pathology", "R": "has_radiology", "S": "has_systemic"}


def clinical_matrix(p: pd.DataFrame) -> pd.DataFrame:
    stage = p.stage.map({"IIIB": 0, "IIIC": 1, "IVA": 2, "IVB": 3}).astype(float)
    return pd.DataFrame({
        "age": p.age.astype(float), "male": (p.sex == "Male").astype(float),
        "lusc": (p.histology == "LUSC").astype(float),
        "smoke_current": (p.smoking == "Current").astype(float),
        "smoke_never": (p.smoking == "Never").astype(float),
        "ecog": p.ecog.astype(float), "stage": stage,
        "first_line": (p.line == "1L").astype(float), "chemo_io": p.chemo_io.astype(float),
        "autoimmune": p.autoimmune_history.astype(float)}, index=p.index)


def interp_matrix(p: pd.DataFrame, k: str) -> pd.DataFrame:
    out = {}
    for col, lg in INTERP[k]:
        x = p[col].astype(float)
        out[("log_" if lg else "") + col] = np.log(np.clip(x, 1e-3, None)) if lg else x
    return pd.DataFrame(out, index=p.index)


@dataclass
class Prepared:
    C: np.ndarray
    X: Dict[str, np.ndarray]          # per modality: [interpretable | embedding], standardised
    m: np.ndarray                     # n x 3 availability
    y: np.ndarray                     # n x 3 (dcb, irae_g2, irae_g3)
    names: Dict[str, list]            # feature names per block
    n_interp: Dict[str, int]


class Preparer:
    """Fit standardisation/medians on training rows only; apply unchanged elsewhere."""

    def fit(self, cohort: dict) -> "Preparer":
        p = cohort["patients"]
        Cm = clinical_matrix(p)
        self.c_med, self.c_mu = Cm.median(), None
        Cf = Cm.fillna(self.c_med)
        self.c_mu, self.c_sd = Cf.mean(), Cf.std().replace(0, 1.0)
        self.stats = {}
        for k in MODALITIES:
            av = p[AVAIL[k]].values.astype(bool)
            I = interp_matrix(p, k)[av]
            E = cohort[EMB[k]][av]
            med_i = I.median()
            If = I.fillna(med_i)
            self.stats[k] = dict(med_i=med_i, mu_i=If.mean(), sd_i=If.std().replace(0, 1.0),
                                 mu_e=np.nanmean(E, 0), sd_e=np.nanstd(E, 0) + 1e-6,
                                 names=list(I.columns) + [f"{k}_emb{j}" for j in range(E.shape[1])])
        return self

    def transform(self, cohort: dict, drop: tuple = ()) -> Prepared:
        """`drop`: modalities to remove at test time (counterfactual / Scenario C)."""
        p = cohort["patients"]
        C = ((clinical_matrix(p).fillna(self.c_med) - self.c_mu) / self.c_sd).values.astype(np.float32)
        X, m = {}, []
        for k in MODALITIES:
            st = self.stats[k]
            av = p[AVAIL[k]].values.astype(bool) & (k not in drop)
            I = ((interp_matrix(p, k).fillna(st["med_i"]) - st["mu_i"]) / st["sd_i"]).values
            E = (np.nan_to_num(cohort[EMB[k]], nan=0.0) - st["mu_e"]) / st["sd_e"]
            B = np.concatenate([I, E], axis=1).astype(np.float32)
            B[~av] = 0.0                                  # training-median fill (0 after scaling)
            X[k] = B
            m.append(av)
        y = p[["dcb", "irae_g2", "irae_g3"]].values.astype(np.float32)
        return Prepared(C=C, X=X, m=np.column_stack(m).astype(np.float32), y=y,
                        names={k: self.stats[k]["names"] for k in MODALITIES},
                        n_interp={k: len(INTERP[k]) for k in MODALITIES})


def subset(d: Prepared, idx) -> Prepared:
    return Prepared(C=d.C[idx], X={k: v[idx] for k, v in d.X.items()}, m=d.m[idx], y=d.y[idx],
                    names=d.names, n_interp=d.n_interp)


def interpretable_frame(cohort: dict) -> pd.DataFrame:
    """Conventional tabular design (no embeddings) for logistic baselines."""
    p = cohort["patients"]
    parts = [clinical_matrix(p)] + [interp_matrix(p, k) for k in MODALITIES]
    return pd.concat(parts, axis=1)
