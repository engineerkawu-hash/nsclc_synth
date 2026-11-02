"""Comments 17-19: structured explainability, explanation stability and counterfactuals.

Levels of explanation (comment 17)
  feature  : Integrated Gradients (IG) per input, interpretable variables reported individually,
             embedding dimensions summed per modality
  modality : exact Shapley values over the available modalities (3 players, 8 coalitions),
             a coalition's value = calibrated logit with the other modalities masked
  patient  : worked examples (feature + modality attributions for single synthetic patients)
  counterfactual : see comment 19 below

Because the data are synthetic, explanations can be checked against the known generating
process: the attribution mass placed on variables that have NO effect on an outcome by design
("null variables") measures implausible reliance.

Stability (comment 18): rank agreement of global importance across 20 bootstrap retrainings;
per-patient agreement under input perturbation, under a removed modality, and between models
trained on clean vs noisy measurements.

Counterfactuals (comment 19): the same patient with all modalities, without pathology, without
radiology, without systemic data; and a systemic-profile change holding pathology and radiology fixed.
"""
import copy
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nsclc_synth import DGPConfig, generate  # noqa: E402
from nsclc_synth.data import MODALITIES, Preparer, Prepared, subset  # noqa: E402
from nsclc_synth.experiment import dev_split, test_cohort  # noqa: E402
from nsclc_synth.model import FittedModel, ModelSpec, integrated_gradients  # noqa: E402

OUT = "outputs/explainability"
SPEC = ModelSpec(fusion="missing_aware", missing="proposed")
CLIN = ["age", "male", "lusc", "smoke_current", "smoke_never", "ecog", "stage", "first_line", "chemo_io", "autoimmune"]
# variables with no effect on the outcome in the data-generating process (see generator.py)
NULL = {0: ["age", "male", "autoimmune"],
        1: ["age", "ecog", "stage", "smoke_current", "smoke_never", "log_tmb", "log_nlr", "log_ldh",
            "albumin", "log_crp", "log_ctdna_vaf", "egfr_mut", "stk11_mut", "log_tumour_volume_cm3"]}
TASK = {0: "DCB", 1: "irAE>=2"}


def fit_one(dev, tr, ca, seed, boot=False):
    rng = np.random.default_rng(seed)
    tr_b = rng.choice(tr, len(tr), replace=True) if boot else tr
    prep = Preparer().fit({"patients": dev["patients"].iloc[tr_b],
                           **{k: dev[k][tr_b] for k in ["z_pathology", "z_radiology", "z_systemic"]}})
    full = prep.transform(dev)
    model = FittedModel(SPEC).fit(subset(full, tr_b), subset(full, ca), seed=seed)
    return model, prep


def feature_attr(model, d: Prepared, task):
    """IG -> DataFrame (patients x named features); embeddings summed per modality."""
    ig = integrated_gradients(model, d, task)
    cols = {}
    for j, c in enumerate(CLIN):
        cols[c] = ig["C"][:, j]
    for k in MODALITIES:
        ni = d.n_interp[k]
        for j in range(ni):
            cols[d.names[k][j]] = ig[k][:, j]
        cols[f"{k} embedding"] = ig[k][:, ni:].sum(1)
    return pd.DataFrame(cols)


def modality_shapley(model, prep, cohort, task):
    """Exact Shapley over P, R, S on the calibrated logit scale."""
    a, b = model.platt[task]
    v = {}
    for r in range(4):
        for S in itertools.combinations(MODALITIES, r):
            drop = tuple(k for k in MODALITIES if k not in S)
            L = model.logits(prep.transform(cohort, drop=drop))[:, task]
            v[S] = a * L + b
    phi = {k: np.zeros(len(cohort["patients"])) for k in MODALITIES}
    n = 3
    from math import factorial
    for k in MODALITIES:
        others = [o for o in MODALITIES if o != k]
        for r in range(3):
            for S in itertools.combinations(others, r):
                w = factorial(len(S)) * factorial(n - len(S) - 1) / factorial(n)
                Sk = tuple(sorted(S + (k,), key=MODALITIES.index))
                phi[k] += w * (v[Sk] - v[tuple(sorted(S, key=MODALITIES.index))])
    return pd.DataFrame(phi), v[()]


def perturb(d: Prepared, sd, seed):
    rng = np.random.default_rng(seed)
    X = {k: (v + rng.normal(scale=sd, size=v.shape).astype(np.float32)) * (d.m[:, [MODALITIES.index(k)]])
         for k, v in d.X.items()}
    return Prepared(C=d.C + rng.normal(scale=sd, size=d.C.shape).astype(np.float32), X=X, m=d.m, y=d.y,
                    names=d.names, n_interp=d.n_interp)


def rowwise_spearman(A: pd.DataFrame, B: pd.DataFrame):
    r = [spearmanr(a, b)[0] for a, b in zip(A.values, B.values) if np.std(a) > 0 and np.std(b) > 0]
    return float(np.nanmedian(r))


def main(n_boot=20, n_explain=600):
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    dev = generate(DGPConfig(n=300, seed=42))
    tr, ca = dev_split(dev, seed=0)
    test = test_cohort(DGPConfig())
    ex = {"patients": test["patients"].iloc[:n_explain].reset_index(drop=True),
          **{k: test[k][:n_explain] for k in ["z_pathology", "z_radiology", "z_systemic"]}}

    glob_rows, mod_rows, stab_rows = [], [], []
    globals_ = {0: [], 1: []}
    for b in range(n_boot):
        model, prep = fit_one(dev, tr, ca, seed=100 + b, boot=True)
        d = prep.transform(ex)
        for t in (0, 1):
            A = feature_attr(model, d, t)
            g = A.abs().mean()
            globals_[t].append(g)
            for f, v in g.items():
                glob_rows.append({"boot": b, "task": TASK[t], "feature": f, "mean_abs_IG": v})
            phi, _ = modality_shapley(model, prep, ex, t)
            for k in MODALITIES:
                mod_rows.append({"boot": b, "task": TASK[t], "modality": k,
                                 "mean_abs_shapley": float(phi[k].abs().mean()),
                                 "mean_shapley": float(phi[k].mean())})
            if b < 5:  # per-patient stability checks on 5 of the bootstrap models
                Ap = feature_attr(model, perturb(d, 0.1, seed=b), t)
                d_noP = prep.transform(ex, drop=("P",))
                AnoP = feature_attr(model, d_noP, t)
                keep = [c for c in A.columns if not c.startswith(("pdl1", "log_til", "stromal", "necrosis", "P "))]
                hasP = ex["patients"].has_pathology.values == 1
                stab_rows += [
                    {"boot": b, "task": TASK[t], "check": "input perturbation (0.1 SD)",
                     "median_patient_spearman": rowwise_spearman(A, Ap)},
                    {"boot": b, "task": TASK[t], "check": "pathology removed (shared features)",
                     "median_patient_spearman": rowwise_spearman(A.loc[hasP, keep], AnoP.loc[hasP, keep])}]
        if b == 0:
            model0, prep0 = model, prep
        print(f"bootstrap {b} done ({time.time()-t0:.0f}s)", flush=True)

    # cross-retraining stability of global rankings
    for t in (0, 1):
        G = pd.DataFrame(globals_[t])
        rhos = [spearmanr(G.iloc[i], G.iloc[j])[0] for i, j in itertools.combinations(range(len(G)), 2)]
        top = [set(G.iloc[i].nlargest(5).index) for i in range(len(G))]
        jac = [len(a & b) / len(a | b) for a, b in itertools.combinations(top, 2)]
        stab_rows.append({"boot": "all", "task": TASK[t], "check": "global ranking across 20 bootstrap retrainings",
                          "median_patient_spearman": np.nan, "pairwise_rank_rho_median": float(np.median(rhos)),
                          "pairwise_rank_rho_min": float(np.min(rhos)), "top5_jaccard_median": float(np.median(jac))})
        null_share = (G[NULL[t]].sum(1) / G.sum(1))
        stab_rows.append({"boot": "all", "task": TASK[t], "check": "attribution share on null (no-effect) variables",
                          "null_share_median": float(null_share.median()),
                          "null_share_range": f"{null_share.min():.3f} to {null_share.max():.3f}",
                          "null_variables": ", ".join(NULL[t]), "null_features_share_of_count": len(NULL[t]) / G.shape[1]})

    # clean vs noisy-measurement training (same seeds)
    cfg_noisy = DGPConfig(n=300, seed=42)
    cfg_noisy.noise.measurement_noise = 2.0
    cfg_noisy.noise.embedding_noise = 2.0
    dev_noisy = generate(cfg_noisy)
    for t in (0, 1):
        consensus = pd.DataFrame(globals_[t]).median()
        rh = []
        for b in range(5):
            m_n, p_n = fit_one(dev_noisy, tr, ca, seed=100 + b, boot=True)
            rh.append(spearmanr(consensus, feature_attr(m_n, p_n.transform(ex), t).abs().mean())[0])
        stab_rows.append({"boot": "all", "task": TASK[t], "check": "noisy-measurement training vs clean consensus ranking",
                          "pairwise_rank_rho_median": float(np.median(rh)), "pairwise_rank_rho_min": float(np.min(rh))})

    pd.DataFrame(glob_rows).to_csv(f"{OUT}/global_feature_importance_by_bootstrap.csv", index=False)
    pd.DataFrame(mod_rows).to_csv(f"{OUT}/modality_shapley_by_bootstrap.csv", index=False)
    pd.DataFrame(stab_rows).to_csv(f"{OUT}/explanation_stability.csv", index=False)

    # ---- counterfactuals (comment 19) on model 0 ---------------------------------------
    p0 = ex["patients"]
    allm = (p0.n_modalities == 3).values
    preds = {lab: model0.predict(prep0.transform(ex, drop=drop))
             for lab, drop in [("A: all modalities", ()), ("B: pathology removed", ("P",)),
                               ("C: radiology removed", ("R",)), ("D: systemic removed", ("S",))]}
    thr = dev["patients"][["dcb", "irae_g2"]].values[tr].mean(0)
    cf_rows = []
    for lab, P in preds.items():
        for t in (0, 1):
            dlt = P[allm, t] - preds["A: all modalities"][allm, t]
            flip = ((P[allm, t] >= thr[t]) != (preds["A: all modalities"][allm, t] >= thr[t])).mean()
            cf_rows.append({"scenario": lab, "task": TASK[t], "mean_prediction": float(P[allm, t].mean()),
                            "mean_change_vs_A": float(dlt.mean()), "mean_abs_change_vs_A": float(np.abs(dlt).mean()),
                            "pct_decision_flips_vs_A": 100 * float(flip), "n_patients_all_modalities": int(allm.sum())})
    pd.DataFrame(cf_rows).to_csv(f"{OUT}/counterfactual_modality_removal.csv", index=False)

    # worked examples: three patients with all modalities (low / middle / high predicted DCB)
    idx_all = np.where(allm)[0]
    order = idx_all[np.argsort(preds["A: all modalities"][idx_all, 0])]
    picks = {"Patient 1 (low predicted benefit)": order[int(0.1 * len(order))],
             "Patient 2 (middle)": order[len(order) // 2],
             "Patient 3 (high predicted benefit)": order[int(0.9 * len(order))]}
    d0 = prep0.transform(ex)
    ex_rows = []
    phis = {t: modality_shapley(model0, prep0, ex, t)[0] for t in (0, 1)}
    attrs = {t: feature_attr(model0, d0, t) for t in (0, 1)}
    for lab, i in picks.items():
        row = {"patient": lab, "pdl1_tps": p0.pdl1_tps[i], "tmb": p0.tmb[i], "nlr": p0.nlr[i], "ecog": p0.ecog[i],
               "regimen": p0.regimen[i], "true_dcb": int(p0.dcb[i]), "true_irae_g2": int(p0.irae_g2[i])}
        for t in (0, 1):
            for s_lab, P in preds.items():
                row[f"{TASK[t]} p ({s_lab[:1]})"] = round(float(P[i, t]), 3)
            for k in MODALITIES:
                row[f"{TASK[t]} Shapley {k}"] = round(float(phis[t].loc[i, k]), 3)
            top = attrs[t].loc[i].abs().nlargest(3).index.tolist()
            row[f"{TASK[t]} top-3 features"] = "; ".join(f"{f} ({attrs[t].loc[i, f]:+.2f})" for f in top)
        ex_rows.append(row)
    pd.DataFrame(ex_rows).to_csv(f"{OUT}/worked_examples.csv", index=False)

    # systemic-profile counterfactual: favourable vs unfavourable inflammatory markers,
    # pathology and radiology (and the systemic embedding) held fixed
    names_S = d0.names["S"]
    j = {n: names_S.index(n) for n in ["log_nlr", "log_crp", "albumin", "log_ldh"]}
    hasS = d0.m[:, 2] == 1
    out_cf = []
    for lab, q, alb_q in [("favourable (NLR, CRP, LDH at 25th pct; albumin at 75th)", 25, 75),
                          ("unfavourable (NLR, CRP, LDH at 75th pct; albumin at 25th)", 75, 25)]:
        dd = copy.deepcopy(d0)
        for n_, col in j.items():
            qq = alb_q if n_ == "albumin" else q
            dd.X["S"][hasS, col] = np.percentile(d0.X["S"][hasS, col], qq)
        P = model0.predict(dd)
        for t in (0, 1):
            out_cf.append({"systemic profile": lab, "task": TASK[t],
                           "mean_prediction": float(P[hasS, t].mean()),
                           "mean_change_vs_observed": float((P[hasS, t] - preds["A: all modalities"][hasS, t]).mean())})
    pd.DataFrame(out_cf).to_csv(f"{OUT}/counterfactual_systemic_profile.csv", index=False)

    # "what would need to change": NLR reduction needed to move a predicted non-responder over threshold
    near = hasS & (preds["A: all modalities"][:, 0] < thr[0]) & (preds["A: all modalities"][:, 0] > thr[0] - 0.10)
    need = []
    for i in np.where(near)[0]:
        found = np.nan
        for dz in np.arange(0.1, 3.01, 0.1):
            dd = subset(d0, [i])
            dd = Prepared(C=dd.C, X={k: v.copy() for k, v in dd.X.items()}, m=dd.m, y=dd.y, names=dd.names, n_interp=dd.n_interp)
            dd.X["S"][0, j["log_nlr"]] -= dz
            if model0.predict(dd)[0, 0] >= thr[0]:
                found = dz
                break
        need.append(found)
    need = np.array(need)
    pd.DataFrame([{"patients just below DCB threshold (within 10 points)": int(near.sum()),
                   "flipped by lowering NLR within 3 SD (%)": 100 * float(np.isfinite(need).mean()) if len(need) else np.nan,
                   "median NLR reduction needed (SD of log NLR)": float(np.nanmedian(need)) if np.isfinite(need).any() else np.nan}]
                 ).to_csv(f"{OUT}/counterfactual_nlr_threshold.csv", index=False)

    json.dump({"n_boot": n_boot, "n_explain": n_explain, "seconds": round(time.time() - t0)},
              open(f"{OUT}/run_info.json", "w"), indent=2)
    print("explainability done", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
