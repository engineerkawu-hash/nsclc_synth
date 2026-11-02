"""Additional analyses requested in the review of the revised thesis.

  g3          grade >= 3 irAE prediction (20 replicates; proposed single-task model vs LR baseline)
  uncertainty patient-level uncertainty in the reviewer's format: probability, 90% bootstrap
              interval over 20 bootstrap-retrained models, uncertainty band, conformal prediction set;
              checked against the true generating probability (available because data are synthetic)
  shap        Shapley values (permutation sampling, grouped inputs) for the proposed model, compared
              with Integrated Gradients on the same model and patients
  conflict    modality conflict: patients whose pathology-only and systemic-only predictions disagree
  distribution distributional comparison of synthetic data with published proportions (SMD) and
              with real TCGA-LUSC stage (SMD, KS, Wasserstein)
Outputs: outputs/additional/*.csv
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import ks_2samp, spearmanr, wasserstein_distance
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from nsclc_synth import DGPConfig, generate  # noqa: E402
from nsclc_synth.data import MODALITIES, Preparer, subset  # noqa: E402
from nsclc_synth.experiment import (LogisticBaseline, YCOLS, conformal_for, dev_split,  # noqa: E402
                                    eval_model_on, run_deep, test_cohort)
from nsclc_synth.metrics import MondrianConformal  # noqa: E402
from nsclc_synth.model import FittedModel, ModelSpec  # noqa: E402
import run_explainability as RX  # noqa: E402

OUT = "outputs/additional"
SPEC = ModelSpec(fusion="missing_aware", missing="proposed")


def prev3(cohort, tr):
    return cohort["patients"][YCOLS].values[tr].mean(0)


# ----------------------------------------------------------------------------- grade >= 3
def run_g3(reps=20):
    test = test_cohort(DGPConfig())
    rows = []
    for r in range(reps):
        dev = generate(DGPConfig(seed=5000 + r))
        tr, ca = dev_split(dev, seed=r)
        pv = prev3(dev, tr)
        sp = ModelSpec(fusion="missing_aware", missing="proposed", tasks=(2,))
        m, prep, full = run_deep(sp, dev, tr, ca, seed=r)
        rows.append({"rep": r, "model": "Proposed (missing-aware attention), single-task",
                     "events_train": int(dev["patients"].irae_g3.values[tr].sum()),
                     **eval_model_on("deep", m, prep, test, pv, conformal_for("deep", m, full, ca, dev, tasks=(2,)), tasks=(2,))})
        b = LogisticBaseline(tasks=(2,)).fit(dev, tr, ca)
        rows.append({"rep": r, "model": "LR: all interpretable variables",
                     "events_train": int(dev["patients"].irae_g3.values[tr].sum()),
                     **eval_model_on("lr", b, None, test, pv, conformal_for("lr", b, None, ca, dev, tasks=(2,)), tasks=(2,))})
        print("g3 rep", r, flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/g3.csv", index=False)


# ----------------------------------------------------------------------------- uncertainty
def run_uncertainty(n_boot=20):
    dev = generate(DGPConfig(n=300, seed=42))
    tr, ca = dev_split(dev, seed=0)
    test = test_cohort(DGPConfig())
    models = [RX.fit_one(dev, tr, ca, seed=200 + b, boot=True) for b in range(n_boot)]
    P = np.stack([m.predict(p.transform(test))[:, 0] for m, p in models], 0)          # boot x n
    m0, p0 = RX.fit_one(dev, tr, ca, seed=200, boot=False)
    d0 = p0.transform(test)
    point = m0.predict(d0)[:, 0]
    raw = m0.predict_raw(d0)[:, 0]
    raw_ca = m0.predict_raw(subset(p0.transform(dev), ca))[:, 0]
    conf = MondrianConformal(0.10).fit(dev["patients"].dcb.values[ca], raw_ca)
    inc0, inc1 = conf.sets(raw)
    lo, hi = np.percentile(P, 5, 0), np.percentile(P, 95, 0)
    width = hi - lo
    band = np.where(width < 0.10, "low", np.where(width < 0.20, "moderate", "high"))
    truth = test["latent"].p_dcb_true.values
    y = test["patients"].dcb.values
    sset = np.where(inc0 & inc1, "{benefit, no benefit}", np.where(inc1, "{benefit}", np.where(inc0, "{no benefit}", "{}")))
    df = pd.DataFrame({"patient_id": test["patients"].patient_id, "p_benefit": point, "lo90": lo, "hi90": hi,
                       "width": width, "band": band, "set": sset, "true_probability": truth, "outcome": y,
                       "n_modalities": test["patients"].n_modalities})
    df.to_csv(f"{OUT}/uncertainty_patients.csv", index=False)
    summ = []
    for bnd in ["low", "moderate", "high", "all"]:
        s = df if bnd == "all" else df[df.band == bnd]
        summ.append({"band": bnd, "patients (%)": round(100 * len(s) / len(df), 1), "mean interval width": round(s.width.mean(), 3),
                     "true probability inside 90% interval (%)": round(100 * ((s.true_probability >= s.lo90) & (s.true_probability <= s.hi90)).mean(), 1),
                     "mean |prediction - true probability|": round((s.p_benefit - s.true_probability).abs().mean(), 3),
                     "AUROC within band": round(roc_auc_score(s.outcome, s.p_benefit), 3) if 0 < s.outcome.mean() < 1 else np.nan,
                     "single-outcome conformal sets (%)": round(100 * (~s.set.str.contains(",")).mean(), 1),
                     "mean number of modalities": round(s.n_modalities.mean(), 2)})
    pd.DataFrame(summ).to_csv(f"{OUT}/uncertainty_bands.csv", index=False)
    # five illustrative patients in the reviewer's format
    picks = []
    for bnd, st in [("low", "{benefit}"), ("low", "{no benefit}"), ("moderate", "{benefit, no benefit}"), ("high", "{benefit, no benefit}"), ("high", "{no benefit}")]:
        c = df[(df.band == bnd) & (df.set == st)]
        if len(c):
            picks.append(c.iloc[len(c) // 2])
    ex = pd.DataFrame(picks)
    ex["display"] = [f"Prediction: {100*r.p_benefit:.0f}% benefit (90% interval {100*r.lo90:.0f}–{100*r.hi90:.0f}%); uncertainty: {r.band}; prediction set: {r.set}"
                     for _, r in ex.iterrows()]
    ex.to_csv(f"{OUT}/uncertainty_examples.csv", index=False)
    print(pd.DataFrame(summ).to_string(index=False))
    return m0, p0, test


# ----------------------------------------------------------------------------- SHAP
def run_shap(m0, p0, test, n_pat=300, n_perm=100, seed=0):
    ex = {"patients": test["patients"].iloc[:n_pat].reset_index(drop=True),
          **{k: test[k][:n_pat] for k in ["z_pathology", "z_radiology", "z_systemic"]}}
    d = p0.transform(ex)
    groups = [("C", [j], name) for j, name in enumerate(RX.CLIN)]
    for k in MODALITIES:
        ni = d.n_interp[k]
        groups += [(k, [j], d.names[k][j]) for j in range(ni)]
        groups.append((k, list(range(ni, d.X[k].shape[1])), f"{k} embedding"))
    G = len(groups)
    rng = np.random.default_rng(seed)
    out = {}
    for task in (0, 1):
        a = m0.platt[task][0]
        phi = np.zeros((n_pat, G))

        def value(mask):  # mask: boolean array over groups (True = present)
            C = d.C.copy() * 0
            X = {k: v.copy() * 0 for k, v in d.X.items()}
            for on, (blk, cols, _) in zip(mask, groups):
                if on:
                    if blk == "C":
                        C[:, cols] = d.C[:, cols]
                    else:
                        X[blk][:, cols] = d.X[blk][:, cols]
            from nsclc_synth.data import Prepared
            dd = Prepared(C=C, X=X, m=d.m, y=d.y, names=d.names, n_interp=d.n_interp)
            return a * m0.logits(dd)[:, task]
        for _ in range(n_perm):
            order = rng.permutation(G)
            mask = np.zeros(G, bool)
            prev = value(mask)
            for gi in order:
                mask[gi] = True
                cur = value(mask)
                phi[:, gi] += cur - prev
                prev = cur
        phi /= n_perm
        names = [g[2] for g in groups]
        S = pd.DataFrame(phi, columns=names)
        ig = RX.feature_attr(m0, d, task)[names]
        gs, gi_ = S.abs().mean(), ig.abs().mean()
        out[task] = (gs, gi_)
        pd.DataFrame({"feature": names, "mean_abs_shap": gs.values, "mean_abs_ig": gi_.values}).sort_values(
            "mean_abs_shap", ascending=False).to_csv(f"{OUT}/shap_vs_ig_{['dcb', 'irae'][task]}.csv", index=False)
    rows = []
    for task in (0, 1):
        gs, gi_ = out[task]
        null = RX.NULL[task]
        rows.append({"task": ["DCB", "irAE>=2"][task],
                     "rank agreement SHAP vs IG (Spearman)": round(spearmanr(gs, gi_)[0], 3),
                     "top-5 overlap (Jaccard)": round(len(set(gs.nlargest(5).index) & set(gi_.nlargest(5).index)) /
                                                      len(set(gs.nlargest(5).index) | set(gi_.nlargest(5).index)), 2),
                     "SHAP top-5": "; ".join(gs.nlargest(5).index),
                     "SHAP share on null variables (%)": round(100 * gs[null].sum() / gs.sum(), 1),
                     "IG share on null variables (%)": round(100 * gi_[null].sum() / gi_.sum(), 1)})
    pd.DataFrame(rows).to_csv(f"{OUT}/shap_summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


# ----------------------------------------------------------------------------- modality conflict
def run_conflict(reps=20):
    test = test_cohort(DGPConfig())
    y = test["patients"].dcb.values
    rows = []
    for r in range(reps):
        dev = generate(DGPConfig(seed=5000 + r))
        tr, ca = dev_split(dev, seed=r)
        thr = dev["patients"].dcb.values[tr].mean()
        m, prep, full = run_deep(SPEC, dev, tr, ca, seed=r)
        pf = m.predict(prep.transform(test))[:, 0]
        pP = m.predict(prep.transform(test, drop=("R", "S")))[:, 0]
        pS = m.predict(prep.transform(test, drop=("P", "R")))[:, 0]
        pR = m.predict(prep.transform(test, drop=("P", "S")))[:, 0]
        raw_ca = m.predict_raw(subset(full, ca))[:, 0]
        conf = MondrianConformal(0.10).fit(dev["patients"].dcb.values[ca], raw_ca)
        i0, i1 = conf.sets(m.predict_raw(prep.transform(test))[:, 0])
        size = i0.astype(int) + i1.astype(int)
        both = (test["patients"].has_pathology.values == 1) & (test["patients"].has_systemic.values == 1)
        conf_ps = both & ((pP >= thr) != (pS >= thr))
        conc_ps = both & ~conf_ps
        fused_follows_P = ((pf >= thr) == (pP >= thr))[conf_ps].mean()
        rows.append({"rep": r, "patients with P and S": int(both.sum()),
                     "conflict rate P vs S (%)": 100 * conf_ps.sum() / both.sum(),
                     "fused follows pathology in conflict (%)": 100 * fused_follows_P,
                     "AUROC fused, conflicting": roc_auc_score(y[conf_ps], pf[conf_ps]),
                     "AUROC fused, concordant": roc_auc_score(y[conc_ps], pf[conc_ps]),
                     "AUROC pathology-only, conflicting": roc_auc_score(y[conf_ps], pP[conf_ps]),
                     "AUROC systemic-only, conflicting": roc_auc_score(y[conf_ps], pS[conf_ps]),
                     "benefit rate, conflicting (%)": 100 * y[conf_ps].mean(),
                     "single-outcome sets, conflicting (%)": 100 * (size[conf_ps] == 1).mean(),
                     "single-outcome sets, concordant (%)": 100 * (size[conc_ps] == 1).mean(),
                     "mean |p_fused - 0.5*(pP+pS)|, conflicting": float(np.abs(pf - 0.5 * (pP + pS))[conf_ps].mean())})
        print("conflict rep", r, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(f"{OUT}/conflict.csv", index=False)
    print(d.drop(columns="rep").agg(["mean", "std"]).round(3).T.to_string())


# ----------------------------------------------------------------------------- distributions
def run_distribution(tcga_path="real_data/TCGA_LUSC_clinical_clean.xlsx"):
    ref = generate(DGPConfig(n=50_000, seed=1))["patients"]

    def smd_p(p1, p2):
        return (p1 - p2) / np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
    t = ref.pdl1_tps.dropna()
    items = [  # name, synthetic proportion, published proportion, source
        ("Male", (ref.sex == "Male").mean(), 0.708, "Mok et al. (2019): 902/1,274"),
        ("Male", (ref.sex == "Male").mean(), 0.493, "Hellmann et al. (2018): 37/75"),
        ("Non-squamous histology", (ref.histology == "LUAD").mean(), 0.787, "Hellmann et al. (2018): 59/75"),
        ("Never smoker", (ref.smoking == "Never").mean(), 0.200, "Hellmann et al. (2018): 15/75"),
        ("PD-L1 TPS < 1%", (t < 1).mean(), 0.482, "X. Yang et al. (2021): 424/879"),
        ("PD-L1 TPS 1–49%", ((t >= 1) & (t < 50)).mean(), 0.303, "X. Yang et al. (2021): 266/879"),
        ("PD-L1 TPS ≥ 50%", (t >= 50).mean(), 0.280, "X. Yang et al. (2021), EGFR/ALK-negative"),
        ("Durable clinical benefit", ref.dcb.mean(), 0.31, "Rizvi et al. (2018): midpoint of 25.1–38.6%"),
        ("Any-grade irAE", ref.irae_any.mean(), 0.428, "Ksienski et al. (2019): 116/271"),
        ("Treatment interruption / irAE grade ≥ 2", ref.irae_g2.mean(), 0.207, "Ksienski et al. (2019): 56/271"),
    ]
    rows = [{"Characteristic": n, "Synthetic (%)": round(100 * a, 1), "Published (%)": round(100 * b, 1),
             "SMD": round(smd_p(a, b), 3), "Within |SMD| < 0.1": "Yes" if abs(smd_p(a, b)) < 0.1 else "No", "Source": s}
            for n, a, b, s in items]
    # PD-L1 category distribution distance vs Yang (all-comer)
    syn = np.array([(t < 1).mean(), ((t >= 1) & (t < 50)).mean(), (t >= 50).mean()])
    yang = np.array([0.482, 0.303, 0.215])
    tv = 0.5 * np.abs(syn - yang).sum()
    wd = wasserstein_distance([0, 1, 2], [0, 1, 2], syn, yang)
    # stage vs real TCGA-LUSC (expected to differ: different populations)
    tc = pd.read_excel(tcga_path, sheet_name="Key_Clinical")
    st = tc["Ajcc Pathologic Stage"].map(lambda s: 1 if str(s).startswith("Stage I") and not str(s).startswith(("Stage II", "Stage III")) else
                                         2 if str(s).startswith("Stage II") and not str(s).startswith("Stage III") else
                                         3 if str(s).startswith("Stage III") else 4 if str(s).startswith("Stage IV") else np.nan).dropna()
    ss = ref.stage.map({"IIIB": 3, "IIIC": 3, "IVA": 4, "IVB": 4}).values
    ks = ks_2samp(ss, st.values)
    extra = pd.DataFrame([
        {"Comparison": "PD-L1 category distribution vs X. Yang et al. (2021), all-comers", "Statistic": f"total variation {tv:.3f}; Wasserstein (category units) {wd:.3f}"},
        {"Comparison": "Stage (I–IV) vs real TCGA-LUSC", "Statistic": f"SMD {(ss.mean() - st.mean()) / np.sqrt((ss.var() + st.var()) / 2):.2f}; KS D {ks.statistic:.3f}; Wasserstein {wasserstein_distance(ss, st.values):.2f} stages — expected, because TCGA-LUSC is predominantly early-stage and the synthetic population is advanced by design"},
    ])
    pd.DataFrame(rows).to_csv(f"{OUT}/distribution_smd.csv", index=False)
    extra.to_csv(f"{OUT}/distribution_extra.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False)); print(extra.to_string(index=False))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    what = sys.argv[1:] or ["distribution", "g3", "uncertainty", "conflict"]
    if "distribution" in what:
        run_distribution()
    if "g3" in what:
        run_g3()
    if "uncertainty" in what or "shap" in what:
        m0, p0, test = run_uncertainty()
        run_shap(m0, p0, test)
    if "conflict" in what:
        run_conflict()
