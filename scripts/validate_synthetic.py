"""Validate the synthetic NSCLC cohort (reviewer comment 14).

Four checks:
  1. Marginal fidelity: synthetic summaries vs published targets for ICI-treated advanced NSCLC.
  2. Real-data comparison: synthetic LUSC subgroup vs real TCGA-LUSC (age, sex) using
     Kolmogorov-Smirnov, Wasserstein distance and standardised mean differences (SMD).
  3. Dependence structure: designed associations and correlation signs are reproduced,
     including published effect contrasts (Rizvi 2018 TMB split; TMB-PD-L1 independence).
  4. Sampling variability: spread of key summaries across 200 independent n=300 cohorts.

Usage: python scripts/validate_synthetic.py --tcga <TCGA_LUSC_clinical_clean.xlsx> --out outputs/validation
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance, spearmanr
import statsmodels.api as sm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nsclc_synth import DGPConfig, generate, save_cohort, relationships_table  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Previous (v1) synthetic cohort, as reported in thesis Tables 4.1 and 4.3 (n = 300)
V1 = {"Median age (years)": None, "Mean age (years)": 65.7, "Male (%)": 100 * 170 / 300,
      "LUAD / non-squamous (%)": 100 * 188 / 300, "Never smoker (%)": 100 * 38 / 300,
      "PD-L1 TPS <1% (%)": None, "PD-L1 TPS >=50% (%)": None,
      "Durable clinical benefit (%)": 24.3, "irAE any grade (%)": 64.3,
      "irAE grade >=2 (%)": 35.7, "irAE grade >=3 (%)": 13.7, "TMB median (mut/Mb)": 7.5}


def smd_continuous(a, b):
    return (np.mean(a) - np.mean(b)) / np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)


def smd_binary(p1, p2):
    return (p1 - p2) / np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)


def summaries(p: pd.DataFrame) -> dict:
    t = p.pdl1_tps.dropna()
    tmb = p.tmb
    med = tmb.median()
    m = p[["tmb", "pdl1_tps"]].dropna()
    out = {
        "Median age (years)": p.age.median(), "Mean age (years)": p.age.mean(),
        "Male (%)": 100 * (p.sex == "Male").mean(),
        "LUAD / non-squamous (%)": 100 * (p.histology == "LUAD").mean(),
        "Never smoker (%)": 100 * (p.smoking == "Never").mean(),
        "PD-L1 TPS <1% (%)": 100 * (t < 1).mean(), "PD-L1 TPS >=50% (%)": 100 * (t >= 50).mean(),
        "Durable clinical benefit (%)": 100 * p.dcb.mean(),
        "DCB, TMB >= median (%)": 100 * p.loc[tmb >= med, "dcb"].mean(),
        "DCB, TMB < median (%)": 100 * p.loc[tmb < med, "dcb"].mean(),
        "Spearman rho, TMB vs PD-L1": spearmanr(m.tmb, m.pdl1_tps)[0],
        "irAE any grade (%)": 100 * p.irae_any.mean(),
        "irAE grade >=2 (%)": 100 * p.irae_g2.mean(),
        "irAE grade >=3 (%)": 100 * p.irae_g3.mean(),
        "TMB median (mut/Mb)": med,
    }
    # association contrasts
    ct = pd.crosstab(p.irae_g2, p.dcb)
    out["OR, DCB given irAE grade >=2"] = (ct.loc[1, 1] * ct.loc[0, 0]) / (ct.loc[1, 0] * ct.loc[0, 1])
    nl = p.nlr
    hi = nl >= nl.median()
    out["OR, DCB given NLR >= median"] = _or(p.dcb[hi], p.dcb[~hi & nl.notna()])
    out["RR, irAE >=2 chemo-IO vs mono"] = p.loc[p.chemo_io == 1, "irae_g2"].mean() / p.loc[p.chemo_io == 0, "irae_g2"].mean()
    return out


def _or(a, b):
    a1, a0, b1, b0 = a.sum(), (1 - a).sum(), b.sum(), (1 - b).sum()
    return (a1 * b0) / (a0 * b1)


# Published targets: (low, high, source). Ranges span the cited values; bounds marked
# "assumption" are not taken from a single source.
TARGETS = {
    "Median age (years)": (63, 66, "Mok et al. 2019 (median 63); Hellmann et al. 2018 (median 66)"),
    "Male (%)": (49, 71, "Hellmann et al. 2018 (49%); Mok et al. 2019 (70.8%)"),
    "LUAD / non-squamous (%)": (65, 80, "Hellmann et al. 2018 (79% non-squamous); lower bound assumption"),
    "Never smoker (%)": (10, 20, "Hellmann et al. 2018 (20%); lower bound assumption"),
    "PD-L1 TPS <1% (%)": (33, 50, "Hellmann et al. 2018 (33%); Yang et al. 2021 (48-50%)"),
    "PD-L1 TPS >=50% (%)": (21.5, 30, "Yang et al. 2021 (21.5% all; 28.0% EGFR/ALK-negative)"),
    "Durable clinical benefit (%)": (25, 39, "Rizvi et al. 2018 (25.1-38.6% by TMB group)"),
    "DCB, TMB >= median (%)": (33.6, 43.6, "Rizvi et al. 2018: 38.6% (+/-5 points)"),
    "DCB, TMB < median (%)": (20.1, 30.1, "Rizvi et al. 2018: 25.1% (+/-5 points)"),
    "Spearman rho, TMB vs PD-L1": (-0.15, 0.15, "Hellmann et al. 2018 (rho = -0.087); Rizvi et al. 2018 (independent)"),
    "irAE any grade (%)": (35, 50, "Ksienski et al. 2019 (42.8%); range assumption"),
    "irAE grade >=2 (%)": (18, 30, "Ksienski et al. 2019 (20.7% interrupted ICI for irAE); range assumption"),
    "irAE grade >=3 (%)": (8, 18, "Mok et al. 2019 (grade >=3 treatment-related AE 18%, upper bound)"),
    "OR, DCB given irAE grade >=2": (1.05, 3.0, "Kopanos et al. 2025; Zhao et al. 2026 (irAEs associate with benefit)"),
    "OR, DCB given NLR >= median": (0.3, 0.95, "Xie et al. 2019; Yang et al. 2021 (high NLR worse outcome)"),
    "RR, irAE >=2 chemo-IO vs mono": (1.4, 2.6, "Chen et al. 2025 (RR 2.02 for irAE with chemo-IO)"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tcga", required=True)
    ap.add_argument("--out", default="outputs/validation")
    ap.add_argument("--n-ref", type=int, default=50_000)
    ap.add_argument("--n-rep", type=int, default=200)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # ---- cohorts -------------------------------------------------------------------
    ref_cfg = DGPConfig(n=a.n_ref, seed=1)
    ref = generate(ref_cfg)["patients"]
    cfg300 = DGPConfig(n=300, seed=42)
    cohort = generate(cfg300)
    save_cohort(cohort, os.path.join(os.path.dirname(a.out), "cohort_n300_seed42"), cfg300)
    p300 = cohort["patients"]
    relationships_table().to_csv(os.path.join(a.out, "table_simulated_relationships.csv"), index=False)

    # ---- 4. sampling variability across 200 cohorts of n=300 ------------------------
    reps = [summaries(generate(DGPConfig(n=300, seed=1000 + i))["patients"]) for i in range(a.n_rep)]
    reps = pd.DataFrame(reps)

    # ---- 1. marginal fidelity ---------------------------------------------------------
    s_ref, s300 = summaries(ref), summaries(p300)
    rows = []
    for k, (lo, hi, src) in TARGETS.items():
        q = reps[k].quantile([0.025, 0.975]).values
        v1 = V1.get(k)
        rows.append({
            "Quantity": k, "Target range": f"{lo:g} to {hi:g}", "Source": src,
            f"Synthetic v2 (n={a.n_ref:,})": round(s_ref[k], 2),
            "Within target": "Yes" if lo <= s_ref[k] <= hi else "No",
            "v2 cohort n=300 (seed 42)": round(s300[k], 2),
            "95% range over 200 cohorts of n=300": f"{q[0]:.2f} to {q[1]:.2f}",
            "Previous v1 cohort (thesis Tables 4.1, 4.3)": ("" if v1 is None else round(v1, 1)),
            "v1 within target": ("" if v1 is None else ("Yes" if lo <= v1 <= hi else "No")),
        })
    fid = pd.DataFrame(rows)
    fid.to_csv(os.path.join(a.out, "table_marginal_fidelity.csv"), index=False)

    # ---- 2. real-data comparison: TCGA-LUSC -------------------------------------------
    tc = pd.read_excel(a.tcga, sheet_name="Key_Clinical")
    t_age = tc["Age At Index"].dropna().values.astype(float)
    t_male = (tc["Sex At Birth"].str.lower() == "male").mean()
    syn_lusc = ref[ref.histology == "LUSC"]
    s_age = syn_lusc.age.values.astype(float)
    s_male = (syn_lusc.sex == "Male").mean()
    ks = ks_2samp(s_age, t_age)
    real = pd.DataFrame([
        {"Variable": "Age (years)", "TCGA-LUSC (real)": f"mean {t_age.mean():.1f} (SD {t_age.std(ddof=1):.1f}), n={len(t_age)}",
         "Synthetic LUSC subgroup": f"mean {s_age.mean():.1f} (SD {s_age.std(ddof=1):.1f}), n={len(s_age)}",
         "SMD": round(smd_continuous(s_age, t_age), 3), "KS D": round(ks.statistic, 3),
         "KS p": f"{ks.pvalue:.3g}", "Wasserstein (years)": round(wasserstein_distance(s_age, t_age), 2),
         "Independent check?": "Yes: age was set from trial data, not from TCGA"},
        {"Variable": "Male sex", "TCGA-LUSC (real)": f"{100*t_male:.1f}%", "Synthetic LUSC subgroup": f"{100*s_male:.1f}%",
         "SMD": round(smd_binary(s_male, t_male), 3), "KS D": "", "KS p": "", "Wasserstein (years)": "",
         "Independent check?": "No: the LUSC sex ratio was set with TCGA-LUSC in view"},
    ])
    real.to_csv(os.path.join(a.out, "table_tcga_lusc_comparison.csv"), index=False)
    # same comparison at n=300 scale (LUSC subgroup of the analysis cohort)
    l300 = p300[p300.histology == "LUSC"].age.values.astype(float)
    ks300 = ks_2samp(l300, t_age)

    # ---- 3. dependence structure ------------------------------------------------------
    pairs = [("pdl1_tps", "til_density", "+"), ("pdl1_tps", "stromal_til_pct", "+"),
             ("tumour_volume_cm3", "ldh", "+"), ("tumour_volume_cm3", "ctdna_vaf", "+"),
             ("nlr", "crp", "+"), ("nlr", "albumin", "-"), ("pdl1_tps", "glcm_entropy", "+"),
             ("tmb", "pdl1_tps", "0"), ("tmb", "nlr", "0")]
    dep = []
    for x, y, sign in pairs:
        m = ref[[x, y]].dropna()
        rho = spearmanr(m[x], m[y])[0]
        ok = (rho > 0.05) if sign == "+" else (rho < -0.05) if sign == "-" else (abs(rho) < 0.1)
        dep.append({"Variable pair": f"{x} vs {y}", "Designed direction": sign,
                    "Spearman rho": round(rho, 3), "Reproduced": "Yes" if ok else "No"})
    dep = pd.DataFrame(dep)
    dep.to_csv(os.path.join(a.out, "table_dependence_structure.csv"), index=False)

    # multivariable recovery of designed outcome effects (observed variables only)
    d = ref.copy()
    d["log_tmb"] = np.log(d.tmb)
    d["log_nlr"] = np.log(d.nlr)
    d["log_til"] = np.log(d.til_density)
    d["log_vol"] = np.log(d.tumour_volume_cm3)
    X = ["pdl1_tps", "log_til", "log_tmb", "log_nlr", "log_vol", "ecog", "egfr_mut", "stk11_mut", "chemo_io"]
    cc = d.dropna(subset=X)
    Z = cc[X].astype(float)
    Z = (Z - Z.mean()) / Z.std()
    fit = sm.Logit(cc.dcb, sm.add_constant(Z)).fit(disp=0)
    designed = {"pdl1_tps": "+", "log_til": "+", "log_tmb": "+", "log_nlr": "-", "log_vol": "-",
                "ecog": "-", "egfr_mut": "-", "stk11_mut": "-", "chemo_io": "+"}
    rec = pd.DataFrame({"Predictor (standardised)": X,
                        "Designed sign": [designed[k] for k in X],
                        "Odds ratio per SD": np.round(np.exp(fit.params[X].values), 3),
                        "95% CI": [f"{np.exp(lo):.2f} to {np.exp(hi):.2f}" for lo, hi in fit.conf_int().loc[X].values],
                        "Sign recovered": ["Yes" if np.sign(fit.params[k]) == (1 if designed[k] == "+" else -1) else "No" for k in X]})
    rec.to_csv(os.path.join(a.out, "table_effect_recovery_dcb.csv"), index=False)

    Xt = ["chemo_io", "autoimmune_history", "pdl1_tps", "log_til", "age"]
    cct = d.dropna(subset=Xt)
    Zt = cct[Xt].astype(float)
    Zt = (Zt - Zt.mean()) / Zt.std()
    fitt = sm.Logit(cct.irae_g2, sm.add_constant(Zt)).fit(disp=0)
    designed_t = {"chemo_io": "+", "autoimmune_history": "+", "pdl1_tps": "+ (via shared U_imm)",
                  "log_til": "+ (via shared U_imm)", "age": "0"}
    rect = pd.DataFrame({"Predictor (standardised)": Xt, "Designed sign": [designed_t[k] for k in Xt],
                         "Odds ratio per SD": np.round(np.exp(fitt.params[Xt].values), 3),
                         "95% CI": [f"{np.exp(lo):.2f} to {np.exp(hi):.2f}" for lo, hi in fitt.conf_int().loc[Xt].values]})
    rect.to_csv(os.path.join(a.out, "table_effect_recovery_irae.csv"), index=False)

    # modality availability
    avail = pd.DataFrame({
        "Quantity": ["Pathology available (%)", "Radiology available (%)", "Systemic available (%)",
                     "All three modalities (%)", "Exactly two (%)", "Exactly one (%)"],
        f"Synthetic v2 (n={a.n_ref:,})": [round(100 * ref.has_pathology.mean(), 1), round(100 * ref.has_radiology.mean(), 1),
                                          round(100 * ref.has_systemic.mean(), 1), round(100 * (ref.n_modalities == 3).mean(), 1),
                                          round(100 * (ref.n_modalities == 2).mean(), 1), round(100 * (ref.n_modalities == 1).mean(), 1)],
        "v2 cohort n=300": [round(100 * p300.has_pathology.mean(), 1), round(100 * p300.has_radiology.mean(), 1),
                            round(100 * p300.has_systemic.mean(), 1), round(100 * (p300.n_modalities == 3).mean(), 1),
                            round(100 * (p300.n_modalities == 2).mean(), 1), round(100 * (p300.n_modalities == 1).mean(), 1)],
        "Previous v1 cohort": [76.3, 66.3, 94.7, 48.0, 41.3, 10.7]})
    avail.to_csv(os.path.join(a.out, "table_modality_availability.csv"), index=False)

    # ---- figures -------------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    bins = np.arange(35, 92, 3)
    ax[0].hist(t_age, bins=bins, density=True, alpha=0.55, label=f"TCGA-LUSC, real (n={len(t_age)})", color="#3b6ea5")
    ax[0].hist(s_age, bins=bins, density=True, alpha=0.55, label="Synthetic LUSC subgroup", color="#d9822b")
    ax[0].set_xlabel("Age (years)"); ax[0].set_ylabel("Density"); ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title(f"(a) Age: SMD {smd_continuous(s_age, t_age):.2f}, Wasserstein {wasserstein_distance(s_age, t_age):.1f} y", fontsize=10)

    cats = ["<1%", "1-49%", ">=50%"]
    tt = ref.pdl1_tps.dropna()
    syn = [100 * (tt < 1).mean(), 100 * ((tt >= 1) & (tt < 50)).mean(), 100 * (tt >= 50).mean()]
    yang = [48.2, 30.3, 21.5]
    xx = np.arange(3)
    ax[1].bar(xx - 0.2, syn, 0.4, label="Synthetic v2", color="#d9822b")
    ax[1].bar(xx + 0.2, yang, 0.4, label="Yang et al. 2021 (n=879, real)", color="#3b6ea5")
    ax[1].set_xticks(xx, cats); ax[1].set_ylabel("% of patients"); ax[1].legend(frameon=False, fontsize=8)
    ax[1].set_title("(b) PD-L1 TPS categories", fontsize=10)

    groups = ["TMB >= median", "TMB < median"]
    synd = [s_ref["DCB, TMB >= median (%)"], s_ref["DCB, TMB < median (%)"]]
    riz = [38.6, 25.1]
    ax[2].bar(np.arange(2) - 0.2, synd, 0.4, label="Synthetic v2", color="#d9822b")
    ax[2].bar(np.arange(2) + 0.2, riz, 0.4, label="Rizvi et al. 2018 (n=240, real)", color="#3b6ea5")
    ax[2].set_xticks(np.arange(2), groups); ax[2].set_ylabel("Durable clinical benefit (%)")
    ax[2].legend(frameon=False, fontsize=8); ax[2].set_title("(c) Benefit by TMB group", fontsize=10)
    for x_ in ax:
        x_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "figure_synthetic_vs_real.png"), dpi=200)

    num = ["age", "pdl1_tps", "til_density", "stromal_til_pct", "tumour_volume_cm3", "glcm_entropy",
           "tmb", "nlr", "crp", "albumin", "ldh", "ctdna_vaf", "dcb", "irae_g2"]
    cm = ref[num].corr(method="spearman")
    fig2, axh = plt.subplots(figsize=(8.5, 7))
    im = axh.imshow(cm.values, cmap="RdBu_r", vmin=-1, vmax=1)
    axh.set_xticks(range(len(num)), num, rotation=60, ha="right", fontsize=8)
    axh.set_yticks(range(len(num)), num, fontsize=8)
    for i in range(len(num)):
        for j in range(len(num)):
            axh.text(j, i, f"{cm.values[i, j]:.2f}", ha="center", va="center", fontsize=6,
                     color="white" if abs(cm.values[i, j]) > 0.5 else "black")
    fig2.colorbar(im, ax=axh, shrink=0.8, label="Spearman rho")
    axh.set_title("Correlation structure of the synthetic v2 cohort", fontsize=10)
    fig2.tight_layout()
    fig2.savefig(os.path.join(a.out, "figure_correlation_structure.png"), dpi=200)

    summary = {
        "n_reference": a.n_ref, "n_replicates": a.n_rep,
        "targets_met_v2": int((fid["Within target"] == "Yes").sum()), "targets_total": len(fid),
        "v1_targets_checked": int((fid["v1 within target"] != "").sum()),
        "v1_targets_met": int((fid["v1 within target"] == "Yes").sum()),
        "tcga_age_ks_D_ref": round(ks.statistic, 3), "tcga_age_ks_p_ref": float(ks.pvalue),
        "tcga_age_ks_D_n300_lusc": round(ks300.statistic, 3), "tcga_age_ks_p_n300_lusc": float(ks300.pvalue),
        "n300_lusc": int(len(l300)),
        "intercepts": cohort["intercepts"],
    }
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    pd.set_option("display.width", 250, "display.max_columns", 20, "display.max_colwidth", 60)
    print(fid.drop(columns=["Source"]).to_string(index=False))
    print(real.to_string(index=False))
    print(dep.to_string(index=False))
    print(rec.to_string(index=False))
    print(rect.to_string(index=False))
    print(avail.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
