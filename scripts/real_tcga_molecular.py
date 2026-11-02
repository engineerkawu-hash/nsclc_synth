"""Reviewer comments 2 and 14: real TCGA-LUSC molecular and smoking variables in the systemic stream.

Inputs come from colab/TCGA_LUSC_real_data.ipynb (open-access GDC masked somatic MAFs and exposure
records) and from scripts/real_tcga_systemic.py (patient-level clinical table).

It (1) merges TMB, EGFR/STK11/KRAS/KEAP1/TP53 status and smoking into the real patient table,
(2) runs the same SystemicPreprocessor used for the synthetic cohort on the extended real vector,
(3) repeats the overall-survival proxy check with and without the new variables on the same patients,
(4) compares the synthetic cohort with the real data for TMB, EGFR, STK11 and smoking, overall and
    for the synthetic squamous (LUSC) subgroup, which is the like-for-like comparison.
OS is not an immunotherapy outcome; it is used only to check that the processed vector keeps real signal.

Usage: python scripts/real_tcga_molecular.py --patients outputs/real_tcga/tcga_lusc_patient_level.csv \
        --gdc outputs/real_tcga_gdc --synthetic outputs/cohort_n300_seed42 --out outputs/real_tcga_molecular
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nsclc_synth.systemic_pipeline import SystemicPreprocessor  # noqa: E402
from real_tcga_systemic import cv_cindex  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SMOKING = {"Lifelong Non-Smoker": "Never", "Current Smoker": "Current",
           "Current Reformed Smoker for < or = 15 yrs": "Former",
           "Current Reformed Smoker for > 15 yrs": "Former",
           "Current Reformed Smoker, Duration Not Specified": "Former"}
GENES = ["egfr_mut", "stk11_mut", "kras_mut", "keap1_mut", "tp53_mut"]


def smd_prop(p1, p2):
    pooled = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
    return float((p1 - p2) / pooled) if pooled > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", default="outputs/real_tcga/tcga_lusc_patient_level.csv")
    ap.add_argument("--gdc", default="outputs/real_tcga_gdc")
    ap.add_argument("--synthetic", default="outputs/cohort_n300_seed42")
    ap.add_argument("--out", default="outputs/real_tcga_molecular")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = {}

    p = pd.read_csv(a.patients).rename(columns={"cases.case_id": "case_id"})
    mut = pd.read_csv(os.path.join(a.gdc, "tcga_lusc_mutation_features.csv"))
    ex = pd.read_csv(os.path.join(a.gdc, "tcga_lusc_exposure.csv"))
    log["clinical_patients"] = int(len(p))
    log["mutation_cases"] = int(len(mut))
    log["mutation_cases_primary_tumour_sample"] = int(mut["is_primary"].sum())
    log["mutation_cases_matched_to_clinical"] = int(mut["case_id"].isin(p["case_id"]).sum())

    ex["smoking"] = ex["tobacco_smoking_status"].map(SMOKING)
    ex["pack_years"] = pd.to_numeric(ex["pack_years_smoked"], errors="coerce")
    d = (p.merge(mut[["case_id", "tmb_per_mb", "n_nonsynonymous"] + GENES], on="case_id", how="left")
          .merge(ex[["case_id", "smoking", "pack_years"]], on="case_id", how="left"))
    d = d.rename(columns={"tmb_per_mb": "tmb"})
    d.to_csv(os.path.join(a.out, "tcga_lusc_patient_level_with_molecular.csv"), index=False)

    # ---- coverage -----------------------------------------------------------------------
    cov = pd.DataFrame([
        ("TMB (nonsynonymous per Mb)", d["tmb"].notna().sum()),
        ("EGFR status", d["egfr_mut"].notna().sum()),
        ("STK11 status", d["stk11_mut"].notna().sum()),
        ("KRAS / KEAP1 / TP53 status", d["tp53_mut"].notna().sum()),
        ("Smoking status (never/former/current)", d["smoking"].notna().sum()),
        ("Pack-years", d["pack_years"].notna().sum()),
    ], columns=["Variable", "Patients with value (n)"])
    cov["Patients with value (%)"] = (100 * cov["Patients with value (n)"] / len(d)).round(1)
    cov.to_csv(os.path.join(a.out, "table_molecular_coverage.csv"), index=False)

    # ---- descriptive summary of real molecular data ---------------------------------------
    m = d[d["tmb"].notna()]
    desc = [("TMB, median (IQR), mut/Mb",
             f"{m.tmb.median():.2f} ({m.tmb.quantile(.25):.2f}-{m.tmb.quantile(.75):.2f})"),
            ("TMB >= 10 mut/Mb, n (%)", f"{int((m.tmb >= 10).sum())} ({100*(m.tmb >= 10).mean():.1f})")]
    for g in GENES:
        desc.append((f"{g.split('_')[0].upper()} mutated, n (%)", f"{int(m[g].sum())} ({100*m[g].mean():.1f})"))
    s = d["smoking"].dropna()
    for lv in ["Never", "Former", "Current"]:
        desc.append((f"Smoking {lv.lower()}, n (%)", f"{int((s == lv).sum())} ({100*(s == lv).mean():.1f})"))
    pd.DataFrame(desc, columns=["Characteristic", "TCGA-LUSC (real)"]).to_csv(
        os.path.join(a.out, "table_molecular_descriptives.csv"), index=False)

    # ---- the systemic pipeline on the extended real vector --------------------------------
    usable = d[d["os_days"].notna() & (d["os_days"] >= 0)].copy()
    usable.loc[usable["os_days"] == 0, "os_days"] = 0.5
    clin = dict(continuous=["age"], ordinal=["stage", "t_num", "n_num"],
                categorical=["sex", "race", "m_cat", "laterality", "residual_disease", "prior_malignancy", "site", "subtype"])
    ext = dict(continuous=["age", "tmb", "pack_years"], ordinal=["stage", "t_num", "n_num"],
               categorical=clin["categorical"] + ["smoking", "egfr_mut", "stk11_mut", "kras_mut", "keap1_mut", "tp53_mut"],
               log_vars=["tmb"])
    cols_ext = ext["continuous"] + ext["ordinal"] + ext["categorical"]
    X = SystemicPreprocessor(**ext).fit_transform(usable[cols_ext])
    assert not X.isna().any().any(), "pipeline left missing values"
    log["extended_real_vector_dimension"] = int(X.shape[1])
    log["extended_real_vector_features"] = list(X.columns)

    # OS proxy on the SAME patients (those with mutation data), with and without the new variables
    same = usable[usable["tmb"].notna()].reset_index(drop=True)
    log["patients_in_survival_comparison"] = int(len(same))
    log["events_in_survival_comparison"] = int(same["os_event"].sum())
    res = []
    for label, kw in [("Clinical vector (as before)", clin),
                      ("Clinical + smoking", dict(clin, continuous=clin["continuous"] + ["pack_years"],
                                                 categorical=clin["categorical"] + ["smoking"])),
                      ("Clinical + smoking + molecular", ext)]:
        cols = kw["continuous"] + kw["ordinal"] + kw["categorical"]
        mu, sd, k = cv_cindex(same, cols, prep_kwargs=kw)
        res.append({"Model (penalised Cox, 5x5-fold CV)": label, "C-index mean": round(mu, 3),
                    "C-index SD": round(sd, 3), "Folds": k})
    pd.DataFrame(res).to_csv(os.path.join(a.out, "table_proxy_survival_cindex_molecular.csv"), index=False)

    # ---- synthetic vs real (comment 14) ------------------------------------------------------
    syn = pd.read_csv(os.path.join(a.synthetic, "patients.csv"))
    rows = []
    for grp, sy in [("All synthetic (LUAD + LUSC)", syn), ("Synthetic LUSC subgroup", syn[syn.histology == "LUSC"])]:
        st, rt = sy["tmb"].dropna().values, m["tmb"].values
        ks = ks_2samp(st, rt)
        rows.append({"Comparison": grp, "Variable": "TMB (mut/Mb)",
                     "Synthetic": f"median {np.median(st):.2f} (n={len(st)})",
                     "Real TCGA-LUSC": f"median {np.median(rt):.2f} (n={len(rt)})",
                     "SMD": round(float((np.log1p(st).mean() - np.log1p(rt).mean()) /
                                        np.sqrt((np.log1p(st).var() + np.log1p(rt).var()) / 2)), 3),
                     "KS D (p)": f"{ks.statistic:.3f} ({ks.pvalue:.3g})",
                     "Wasserstein (log 1+TMB)": round(wasserstein_distance(np.log1p(st), np.log1p(rt)), 3)})
        for g in ["egfr_mut", "stk11_mut"]:
            ps, pr = sy[g].dropna().mean(), m[g].mean()
            rows.append({"Comparison": grp, "Variable": f"{g.split('_')[0].upper()} mutated",
                         "Synthetic": f"{100*ps:.1f}% (n={sy[g].notna().sum()})",
                         "Real TCGA-LUSC": f"{100*pr:.1f}% (n={len(m)})",
                         "SMD": round(smd_prop(ps, pr), 3), "KS D (p)": "", "Wasserstein (log 1+TMB)": ""})
        for lv in ["Never", "Current"]:
            ps, pr = (sy["smoking"] == lv).mean(), (s == lv).mean()
            rows.append({"Comparison": grp, "Variable": f"Smoking {lv.lower()}",
                         "Synthetic": f"{100*ps:.1f}% (n={len(sy)})", "Real TCGA-LUSC": f"{100*pr:.1f}% (n={len(s)})",
                         "SMD": round(smd_prop(ps, pr), 3), "KS D (p)": "", "Wasserstein (log 1+TMB)": ""})
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "table_synthetic_vs_real_molecular.csv"), index=False)

    # figure: TMB distributions
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    bins = np.linspace(np.log(0.2), np.log(80), 40)
    ax.hist(np.log(m["tmb"].clip(lower=0.2)), bins=bins, density=True, alpha=.55, color="#3b6ea5",
            label=f"Real TCGA-LUSC (n={len(m)})")
    sl = syn.loc[syn.histology == "LUSC", "tmb"].dropna()
    ax.hist(np.log(sl), bins=bins, density=True, alpha=.55, color="#d9822b", label=f"Synthetic LUSC (n={len(sl)})")
    ticks = [0.5, 1, 2, 5, 10, 20, 50]
    ax.set_xticks(np.log(ticks)); ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlabel("TMB (nonsynonymous mutations per Mb, log scale)"); ax.set_ylabel("Density")
    ax.legend(frameon=False, fontsize=9); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "figure_tmb_real_vs_synthetic.png"), dpi=200)

    json.dump(log, open(os.path.join(a.out, "molecular_log.json"), "w"), indent=2)
    pd.set_option("display.width", 220)
    print(json.dumps({k: v for k, v in log.items() if k != "extended_real_vector_features"}, indent=2))
    for f in ["table_molecular_coverage", "table_molecular_descriptives", "table_proxy_survival_cindex_molecular",
              "table_synthetic_vs_real_molecular"]:
        print("\n", f); print(pd.read_csv(os.path.join(a.out, f + ".csv")).to_string(index=False))


if __name__ == "__main__":
    main()
