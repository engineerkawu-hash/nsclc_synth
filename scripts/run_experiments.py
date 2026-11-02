"""Experiment programme for reviewer comments 4-12 (and inputs to 17-19).

  main      default scenario; fusion strategies; baselines; ensembles; MC dropout; conformal;
            decision curves; test-time modality removal (Scenario C); distribution shift (D)
  missing   7 missing-modality strategies + 3 other fusion strategies x 6 missingness levels (comments 6, 8)
  ablation  component ablation (comment 7)
  prevalence  Scenarios A and B
  noise     Scenario E (retraining under noise)
  samplesize  Scenario F
Each experiment appends one row per (replicate, model, condition) to outputs/experiments/<exp>.csv
and resumes where it stopped.

Usage: python scripts/run_experiments.py --exp main --reps 20
"""
import argparse
import copy
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nsclc_synth import DGPConfig, generate  # noqa: E402
from nsclc_synth.config import ShiftConfig  # noqa: E402
from nsclc_synth.data import INTERP  # noqa: E402
from nsclc_synth.experiment import (LogisticBaseline, append_rows, conformal_for, dev_split,  # noqa: E402
                                    done_keys, eval_model_on, run_deep, test_cohort)
from nsclc_synth.metrics import net_benefit  # noqa: E402
from nsclc_synth.model import ModelSpec  # noqa: E402

OUT = "outputs/experiments"
CLIN = ["age", "male", "lusc", "smoke_current", "smoke_never", "ecog", "stage", "first_line", "chemo_io", "autoimmune"]

BASELINES = {
    "LR: PD-L1 only": ["pdl1_tps", "avail_P"],
    "LR: PD-L1 + TMB": ["pdl1_tps", "log_tmb", "avail_P", "avail_S"],
    "LR: clinical only": CLIN,
    "LR: all interpretable variables": None,
}
FUSIONS = {
    "Proposed (missing-aware attention)": ModelSpec(fusion="missing_aware", missing="proposed"),
    "Intermediate fusion (zero fill)": ModelSpec(fusion="intermediate", missing="zero"),
    "Early fusion": ModelSpec(fusion="early", missing="median"),
    "Late fusion": ModelSpec(fusion="late", missing="mask"),
    "Attention fusion (not missing-aware)": ModelSpec(fusion="attention", missing="median"),
}
MISSING = {
    "Complete-case": ModelSpec(fusion="intermediate", missing="complete_case"),
    "Median imputation": ModelSpec(fusion="intermediate", missing="median"),
    "Zero imputation": ModelSpec(fusion="intermediate", missing="zero"),
    "Modality masking": ModelSpec(fusion="intermediate", missing="mask"),
    "Learned modality embeddings": ModelSpec(fusion="intermediate", missing="learned_emb"),
    "Modality dropout": ModelSpec(fusion="intermediate", missing="mod_dropout"),
    "Proposed (missing-aware attention)": ModelSpec(fusion="missing_aware", missing="proposed"),
}
SHIFTS = {
    "none": ShiftConfig(),
    "scanner offset 0.5 SD": ShiftConfig(scanner_offset=0.5),
    "scanner offset 1 SD": ShiftConfig(scanner_offset=1.0),
    "scanner offset 2 SD": ShiftConfig(scanner_offset=2.0, scanner_scale=1.2),
    "TMB assay x0.6": ShiftConfig(tmb_assay_scale=0.6),
    "TMB assay x1.6": ShiftConfig(tmb_assay_scale=1.6),
    "PD-L1 assay -0.5 SD": ShiftConfig(pdl1_assay_shift=-0.5),
    "NLR +0.3 log": ShiftConfig(nlr_shift=0.3),
    "older, more female (age +8, 45% male)": ShiftConfig(age_shift=8, p_male_override=0.45),
}
PREV_SHIFT = {"DCB prevalence 15% (label shift)": 0.15, "DCB prevalence 45% (label shift)": 0.45}
REMOVALS = ["", "P", "R", "S", "PR", "PS", "RS", "PRS"]


def fit_baseline(cols, cohort, tr, ca):
    return LogisticBaseline(cols=cols).fit(cohort, tr, ca)


def train_prev(cohort, tr):
    return cohort["patients"][["dcb", "irae_g2"]].values[tr].mean(0)


def exp_main(reps):
    path = f"{OUT}/main.csv"
    done = done_keys(path, ["rep"])
    base = DGPConfig()
    tests = {name: test_cohort(DGPConfig(shift=s)) for name, s in SHIFTS.items()}
    for name, prev in PREV_SHIFT.items():
        c = DGPConfig()
        c.outcome.dcb_prevalence = prev
        tests[name] = test_cohort(c)
    dca_rows = []
    for r in range(reps):
        if (str(r),) in done:
            continue
        t0 = time.time()
        cfg = copy.deepcopy(base)
        cfg.seed = 5000 + r
        dev = generate(cfg)
        tr, ca = dev_split(dev, seed=r)
        prev = train_prev(dev, tr)
        rows = []
        fitted = {}
        for mname, spec in FUSIONS.items():
            model, prep, full = run_deep(spec, dev, tr, ca, seed=r)
            fitted[mname] = ("deep", model, prep, conformal_for("deep", model, full, ca, dev))
        for bname, cols in BASELINES.items():
            b = fit_baseline(cols, dev, tr, ca)
            fitted[bname] = ("lr", b, None, conformal_for("lr", b, None, ca, dev))
        # deep ensemble of the proposed model (5 members) and MC dropout
        members = [fitted["Proposed (missing-aware attention)"][1]]
        for s in range(1, 5):
            members.append(run_deep(FUSIONS["Proposed (missing-aware attention)"], dev, tr, ca, seed=1000 * s + r)[0])
        prep0 = fitted["Proposed (missing-aware attention)"][2]
        for tname, test in tests.items():
            removal_list = REMOVALS if tname == "none" else [""]
            for rem in removal_list:
                for mname, (kind, model, prep, confs) in fitted.items():
                    if rem and mname not in ("Proposed (missing-aware attention)", "Intermediate fusion (zero fill)",
                                             "Late fusion", "LR: all interpretable variables"):
                        continue
                    row = eval_model_on(kind, model, prep, test, prev, confs, drop=tuple(rem))
                    rows.append({"rep": r, "model": mname, "test": tname, "removed": rem or "none", **row})
            # ensemble + MC-dropout uncertainty on this test set (no removal)
            from nsclc_synth.metrics import binary_metrics, uncertainty_quality
            d = prep0.transform(test)
            y = test["patients"][["dcb", "irae_g2"]].values
            Pe = np.stack([m.predict(d) for m in members], 0)
            Pm = members[0].mc_predict(d, T=30, seed=r)
            for label, S in [("Deep ensemble (5)", Pe), ("MC dropout (30)", Pm)]:
                row = {"rep": r, "model": label, "test": tname, "removed": "none"}
                for t, tn in [(0, "dcb"), (1, "irae_g2")]:
                    mean, sd = S[:, :, t].mean(0), S[:, :, t].std(0)
                    row.update({f"{tn}_{k}": v for k, v in binary_metrics(y[:, t], mean, prev[t]).items()})
                    row.update({f"{tn}_{k}": v for k, v in uncertainty_quality(y[:, t], mean, sd, prev[t]).items()})
                    row[f"{tn}_mean_pred_sd"] = float(sd.mean())
                rows.append(row)
            if tname == "none":
                for mname in ["Proposed (missing-aware attention)", "LR: all interpretable variables", "LR: PD-L1 only"]:
                    kind, model, prep, _ = fitted[mname]
                    P = model.predict(prep.transform(test)) if kind == "deep" else model.predict(test)
                    for t, tn in [(0, "dcb"), (1, "irae_g2")]:
                        th, nb, nb_all = net_benefit(y[:, t], P[:, t])
                        for a_, b_, c_ in zip(th, nb, nb_all):
                            dca_rows.append({"rep": r, "model": mname, "task": tn, "threshold": round(a_, 2),
                                             "net_benefit": b_, "treat_all": c_})
        append_rows(path, rows)
        append_rows(f"{OUT}/main_dca.csv", dca_rows)
        dca_rows = []
        print(f"main rep {r} done in {time.time()-t0:.0f}s", flush=True)


def exp_missing(reps, levels=(0.1, 0.2, 0.3, 0.4, 0.5, 0.7)):
    path = f"{OUT}/missing.csv"
    done = done_keys(path, ["rep", "level", "model"])
    specs = {**MISSING, **{k: v for k, v in FUSIONS.items() if k in ("Early fusion", "Late fusion",
                                                                      "Attention fusion (not missing-aware)")}}
    for lv in levels:
        cfg = DGPConfig()
        cfg.missingness.p_pathology = cfg.missingness.p_radiology = cfg.missingness.p_systemic = 1 - lv
        cfg.missingness.force_at_least_one = False
        test = test_cohort(cfg)
        for r in range(reps):
            t0 = time.time()
            c = copy.deepcopy(cfg)
            c.seed = 6000 + r
            dev = generate(c)
            tr, ca = dev_split(dev, seed=r)
            prev = train_prev(dev, tr)
            rows = []
            n_complete = int((dev["patients"].n_modalities.values[tr] == 3).sum())
            for mname, spec in specs.items():
                if (str(r), str(lv), mname) in done:
                    continue
                if spec.missing == "complete_case":
                    yc = dev["patients"][["dcb", "irae_g2"]].values[tr][dev["patients"].n_modalities.values[tr] == 3]
                    if n_complete < 20 or yc.sum(0).min() < 3 or (len(yc) - yc.sum(0)).min() < 3:
                        rows.append({"rep": r, "level": lv, "model": mname, "n_train_used": n_complete,
                                     "note": "not estimable: too few complete cases"})
                        continue
                model, prep, full = run_deep(spec, dev, tr, ca, seed=r)
                confs = conformal_for("deep", model, full, ca, dev)
                row = eval_model_on("deep", model, prep, test, prev, confs)
                rows.append({"rep": r, "level": lv, "model": mname,
                             "n_train_used": n_complete if spec.missing == "complete_case" else len(tr), **row})
            b = fit_baseline(None, dev, tr, ca)
            if (str(r), str(lv), "LR: all interpretable variables") not in done:
                rows.append({"rep": r, "level": lv, "model": "LR: all interpretable variables", "n_train_used": len(tr),
                             **eval_model_on("lr", b, None, test, prev, conformal_for("lr", b, None, ca, dev))})
            append_rows(path, rows)
            print(f"missing level {lv} rep {r} done in {time.time()-t0:.0f}s", flush=True)


def exp_ablation(reps):
    path = f"{OUT}/ablation.csv"
    done = done_keys(path, ["rep", "model"])
    P = dict(fusion="missing_aware", missing="proposed")
    variants = {
        "Full model": ModelSpec(**P),
        "- pathology": ModelSpec(**P, modalities=("R", "S")),
        "- radiology": ModelSpec(**P, modalities=("P", "S")),
        "- systemic": ModelSpec(**P, modalities=("P", "R")),
        "- missing-modality mechanism (zero fill)": ModelSpec(fusion="intermediate", missing="zero"),
        "- foundation-model embeddings (interpretable only)": ModelSpec(**P, use_embeddings=False),
        "- dual-task learning (two single-task models)": None,
        "early fusion instead of intermediate": ModelSpec(fusion="early", missing="median"),
        "late fusion instead of intermediate": ModelSpec(fusion="late", missing="mask"),
    }
    test = test_cohort(DGPConfig())
    for r in range(reps):
        t0 = time.time()
        cfg = DGPConfig()
        cfg.seed = 7000 + r
        dev = generate(cfg)
        tr, ca = dev_split(dev, seed=r)
        prev = train_prev(dev, tr)
        rows = []
        for name, spec in variants.items():
            if (str(r), name) in done:
                continue
            if spec is None:
                row = {}
                for t in (0, 1):
                    sp = ModelSpec(**P, tasks=(t,))
                    m, prep, full = run_deep(sp, dev, tr, ca, seed=r)
                    confs = conformal_for("deep", m, full, ca, dev, tasks=(t,))
                    row.update(eval_model_on("deep", m, prep, test, prev, confs, tasks=(t,)))
            else:
                m, prep, full = run_deep(spec, dev, tr, ca, seed=r)
                row = eval_model_on("deep", m, prep, test, prev, conformal_for("deep", m, full, ca, dev))
            rows.append({"rep": r, "model": name, **row})
        append_rows(path, rows)
        print(f"ablation rep {r} done in {time.time()-t0:.0f}s", flush=True)


def _two_model_scenario(path, cells, reps, seed0, train_only=None):
    """Proposed model vs the strongest conventional baseline across scenario cells."""
    done = done_keys(path, ["rep", "cell", "model"])
    for cell, cfg in cells.items():
        test_cfg = copy.deepcopy(cfg)
        if train_only:
            train_only(test_cfg)          # undo train-only perturbations (e.g. label noise) at test
        test = test_cohort(test_cfg)
        for r in range(reps):
            if (str(r), cell, "LR: all interpretable variables") in done:
                continue
            t0 = time.time()
            c = copy.deepcopy(cfg)
            c.seed = seed0 + r
            dev = generate(c)
            tr, ca = dev_split(dev, seed=r)
            prev = train_prev(dev, tr)
            rows = []
            m, prep, full = run_deep(FUSIONS["Proposed (missing-aware attention)"], dev, tr, ca, seed=r)
            rows.append({"rep": r, "cell": cell, "model": "Proposed (missing-aware attention)", "n_dev": c.n,
                         **eval_model_on("deep", m, prep, test, prev, conformal_for("deep", m, full, ca, dev))})
            b = fit_baseline(None, dev, tr, ca)
            rows.append({"rep": r, "cell": cell, "model": "LR: all interpretable variables", "n_dev": c.n,
                         **eval_model_on("lr", b, None, test, prev, conformal_for("lr", b, None, ca, dev))})
            append_rows(path, rows)
            print(f"{os.path.basename(path)} {cell} rep {r} done in {time.time()-t0:.0f}s", flush=True)


def exp_prevalence(reps):
    cells = {}
    for p in (0.10, 0.20, 0.30, 0.50):
        c = DGPConfig(); c.outcome.dcb_prevalence = p
        cells[f"B: DCB {int(p*100)}%"] = c
    for p in (0.10, 0.25, 0.40):
        c = DGPConfig(); c.outcome.tox2_prevalence = p
        cells[f"A: irAE>=2 {int(p*100)}%"] = c
    c = DGPConfig(); c.outcome.dcb_prevalence = 0.5; c.outcome.tox2_prevalence = 0.5
    cells["A: balanced (both 50%)"] = c
    _two_model_scenario(f"{OUT}/prevalence.csv", cells, reps, 8000)


def exp_noise(reps):
    cells = {"none": DGPConfig()}
    for v in (2.0, 3.0):
        c = DGPConfig(); c.noise.measurement_noise = v; cells[f"measurement noise x{v:g}"] = c
        c = DGPConfig(); c.noise.embedding_noise = v; cells[f"embedding noise x{v:g}"] = c
    for v in (0.1, 0.2):
        c = DGPConfig(); c.noise.label_noise = v; cells[f"label noise {int(v*100)}% (training only)"] = c
    for v in (0.1, 0.3):
        c = DGPConfig(); c.noise.feature_corruption = v; cells[f"feature corruption {int(v*100)}%"] = c
    for v in (0.2, 0.4):
        c = DGPConfig(); c.missingness.within_modality_missing = v; cells[f"within-modality missing {int(v*100)}%"] = c

    def clean_labels(cfg):
        cfg.noise.label_noise = 0.0
    _two_model_scenario(f"{OUT}/noise.csv", cells, reps, 9000, train_only=clean_labels)


def exp_samplesize(reps):
    cells = {}
    for n in (100, 200, 300, 500, 1000):
        c = DGPConfig(n=n); cells[f"n={n}"] = c
    _two_model_scenario(f"{OUT}/samplesize.csv", cells, reps, 10000)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    {"main": exp_main, "missing": exp_missing, "ablation": exp_ablation, "prevalence": exp_prevalence,
     "noise": exp_noise, "samplesize": exp_samplesize}[a.exp](a.reps)
