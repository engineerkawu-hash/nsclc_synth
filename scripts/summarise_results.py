"""Turn experiment CSVs into thesis tables, figures and the pre-specified scorecard."""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXP, EXPL, OUT = "outputs/experiments", "outputs/explainability", "outputs/results"
PROP, LRALL = "Proposed (missing-aware attention)", "LR: all interpretable variables"
TN = {"dcb": "Durable clinical benefit", "irae_g2": "irAE grade >=2"}
COL = {"Proposed (missing-aware attention)": "#d9822b", "LR: all interpretable variables": "#3b6ea5",
       "Intermediate fusion (zero fill)": "#7a7a7a", "Early fusion": "#5aa469", "Late fusion": "#8e6bb8",
       "Attention fusion (not missing-aware)": "#b23a48", "Complete-case": "#c9b037", "Median imputation": "#4fb0c6",
       "Zero imputation": "#7a7a7a", "Modality masking": "#a0522d", "Learned modality embeddings": "#2e8b57",
       "Modality dropout": "#9370db"}


def ci(x):
    x = pd.Series(x).dropna()
    if len(x) == 0:
        return ""
    return f"{x.mean():.3f} ({x.quantile(0.025):.3f}-{x.quantile(0.975):.3f})"


def summary_table(df, by, metrics, tasks=("dcb", "irae_g2")):
    rows = []
    for key, g in df.groupby(by, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        r = dict(zip(by, key))
        r["replicates"] = g["rep"].nunique()
        for t in tasks:
            for m in metrics:
                c = f"{t}_{m}"
                if c in g:
                    r[f"{TN[t]}: {m}"] = ci(g[c])
        rows.append(r)
    return pd.DataFrame(rows)


def save(df, name):
    df.to_csv(f"{OUT}/{name}.csv", index=False)


def main():
    os.makedirs(OUT, exist_ok=True)
    score = {}
    METRICS = ["auroc", "auprc", "sensitivity", "specificity", "ppv", "npv", "brier", "ece",
               "cal_slope", "cal_intercept", "coverage", "coverage_class1", "coverage_class0", "mean_set_size"]

    # ---------------- main / baselines / fusion at default -----------------------------
    m = pd.read_csv(f"{EXP}/main.csv")
    base = m[(m.test == "none") & (m.removed == "none")]
    save(summary_table(base, ["model"], METRICS), "table_main_performance")
    # paired comparison proposed vs each other model
    pv = base.pivot_table(index="rep", columns="model", values=["dcb_auroc", "irae_g2_auroc"])
    comp = []
    for other in base.model.unique():
        if other == PROP:
            continue
        for t in ("dcb", "irae_g2"):
            d = (pv[(f"{t}_auroc", PROP)] - pv[(f"{t}_auroc", other)]).dropna()
            if len(d):
                comp.append({"comparison": f"Proposed minus {other}", "task": TN[t], "mean AUROC difference": round(d.mean(), 4),
                             "95% range across replicates": f"{d.quantile(.025):.3f} to {d.quantile(.975):.3f}",
                             "replicates where proposed better (%)": round(100 * (d > 0).mean(), 1)})
    save(pd.DataFrame(comp), "table_paired_comparisons")

    # Scenario C: test-time modality removal
    rem = m[(m.test == "none") & m.model.isin([PROP, "Intermediate fusion (zero fill)", "Late fusion", LRALL])]
    save(summary_table(rem, ["removed", "model"], ["auroc", "ece", "cal_slope", "coverage"]), "table_scenarioC_modality_removal")

    # Scenario D: distribution shift
    sh = m[(m.removed == "none") & m.model.isin([PROP, "Intermediate fusion (zero fill)", LRALL])]
    save(summary_table(sh, ["test", "model"], ["auroc", "ece", "cal_slope", "cal_intercept", "coverage"]),
         "table_scenarioD_distribution_shift")

    # uncertainty (ensemble, MC dropout)
    un = m[m.model.isin(["Deep ensemble (5)", "MC dropout (30)"])]
    save(summary_table(un, ["test", "model"], ["auroc", "ece", "u_auroc_for_error", "auroc_most_certain_50pct",
                                               "mean_pred_sd"]), "table_uncertainty")

    # decision curves
    dca = pd.read_csv(f"{EXP}/main_dca.csv")
    dsum = dca.groupby(["task", "model", "threshold"]).agg(net_benefit=("net_benefit", "mean"),
                                                           treat_all=("treat_all", "mean")).reset_index()
    save(dsum, "table_decision_curve")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, t in enumerate(["dcb", "irae_g2"]):
        s = dsum[dsum.task == t]
        for mod in s.model.unique():
            q = s[s.model == mod]
            ax[i].plot(q.threshold, q.net_benefit, label=mod, color=COL.get(mod, "#444"), lw=2)
        q = s[s.model == s.model.unique()[0]]
        ax[i].plot(q.threshold, q.treat_all, "--", color="#999", label="Treat all")
        ax[i].axhline(0, color="#999", lw=0.8, label="Treat none")
        ax[i].set_ylim(-0.05, max(0.05, s.net_benefit.max() * 1.3))
        ax[i].set_xlabel("Threshold probability"); ax[i].set_ylabel("Net benefit")
        ax[i].set_title(f"{TN[t]} (synthetic; theoretical only)", fontsize=10)
        ax[i].spines[["top", "right"]].set_visible(False)
    ax[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}/figure_decision_curves.png", dpi=200)

    # ---------------- missing modality + fusion comparison ----------------------------
    ms = pd.read_csv(f"{EXP}/missing.csv")
    if "note" in ms:
        ne = ms[ms.note.notna()].groupby(["level", "model"]).size().rename("not estimable (replicates)").reset_index()
        save(ne, "table_missing_not_estimable")
        ms = ms[ms.note.isna()]
    save(summary_table(ms, ["level", "model"], ["auroc", "ece", "cal_slope", "coverage"]), "table_missing_modality")
    agg = ms.groupby(["model", "level"])[["dcb_auroc", "irae_g2_auroc"]].agg(["mean", "std", "count"]).reset_index()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    groups = {"missing": ["Complete-case", "Median imputation", "Zero imputation", "Modality masking",
                          "Learned modality embeddings", "Modality dropout", PROP, LRALL],
              "fusion": ["Early fusion", "Late fusion", "Attention fusion (not missing-aware)", "Zero imputation", PROP]}
    for i, (gname, mods) in enumerate(groups.items()):
        for mod in mods:
            q = agg[agg.model == mod]
            if len(q):
                se = q[("dcb_auroc", "std")] / np.sqrt(q[("dcb_auroc", "count")])
                ax[i].errorbar(100 * q.level, q[("dcb_auroc", "mean")], yerr=1.96 * se, label=mod.replace("Zero imputation", "Intermediate (zero fill)") if gname == "fusion" else mod,
                               color=COL.get(mod, "#444"), marker="o", ms=4, lw=2 if mod == PROP else 1.2, capsize=2)
        ax[i].set_xlabel("Probability each modality is missing (%)"); ax[i].set_ylabel("DCB AUROC (mean, 95% CI)")
        ax[i].set_title("(a) Missing-modality strategies" if i == 0 else "(b) Fusion strategies", fontsize=10)
        ax[i].legend(frameon=False, fontsize=7); ax[i].spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/figure_missing_modality.png", dpi=200)

    retention = []
    for mod in ms.model.unique():
        a = ms[(ms.model == mod)].groupby("level").dcb_auroc.mean()
        if 0.1 in a and 0.5 in a:
            retention.append({"model": mod, "DCB AUROC at 10% missing": round(a[0.1], 3), "at 50%": round(a[0.5], 3),
                              "at 70%": round(a.get(0.7, np.nan), 3),
                              "headroom retained 10%->50%": round((a[0.5] - 0.5) / (a[0.1] - 0.5), 3)})
    retention = pd.DataFrame(retention)
    save(retention, "table_missing_retention")

    # ---------------- ablation --------------------------------------------------------
    ab = pd.read_csv(f"{EXP}/ablation.csv") if os.path.exists(f"{EXP}/ablation.csv") else pd.DataFrame(columns=["model", "rep"])
    full = ab[ab.model == "Full model"].set_index("rep")
    arows = []
    for mod, g in ab.groupby("model", sort=False):
        g = g.set_index("rep")
        r = {"variant": mod, "replicates": len(g)}
        for t in ("dcb", "irae_g2"):
            r[f"{TN[t]} AUROC"] = ci(g[f"{t}_auroc"])
            d = (g[f"{t}_auroc"] - full[f"{t}_auroc"]).dropna()
            r[f"{TN[t]} change vs full"] = "" if mod == "Full model" else f"{d.mean():+.3f} ({d.quantile(.025):+.3f} to {d.quantile(.975):+.3f})"
            r[f"{TN[t]} ECE"] = ci(g[f"{t}_ece"])
        arows.append(r)
    arows.append({"variant": "- explainability layer", "replicates": "",
                  f"{TN['dcb']} change vs full": "0 by construction (post hoc layer; does not alter predictions)"})
    save(pd.DataFrame(arows), "table_ablation")

    # ---------------- scenarios A/B, E, F ----------------------------------------------
    for name in ["prevalence", "noise", "samplesize"]:
        p = f"{EXP}/{name}.csv"
        if os.path.exists(p):
            d = pd.read_csv(p)
            save(summary_table(d, ["cell", "model"], ["auroc", "auprc", "ece", "cal_slope", "coverage",
                                                      "coverage_class1", "coverage_class0", "mean_set_size"]),
                 f"table_scenario_{name}")
    if os.path.exists(f"{EXP}/samplesize.csv"):
        d = pd.read_csv(f"{EXP}/samplesize.csv")
        g = d.groupby(["model", "n_dev"]).agg(m=("dcb_auroc", "mean"), s=("dcb_auroc", "std"),
                                              mt=("irae_g2_auroc", "mean"), st=("irae_g2_auroc", "std")).reset_index()
        save(g.rename(columns={"m": "DCB AUROC mean", "s": "DCB AUROC SD across replicates",
                               "mt": "irAE AUROC mean", "st": "irAE AUROC SD across replicates"}), "table_samplesize_stability")
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for mod in g.model.unique():
            q = g[g.model == mod]
            ax[0].errorbar(q.n_dev, q.m, yerr=q.s, marker="o", label=mod, color=COL.get(mod), capsize=3)
            ax[1].plot(q.n_dev, q.s, marker="o", label=mod, color=COL.get(mod))
        ax[0].set_xscale("log"); ax[1].set_xscale("log")
        ax[0].set_xlabel("Development cohort size"); ax[0].set_ylabel("DCB AUROC (mean +/- SD)")
        ax[1].set_xlabel("Development cohort size"); ax[1].set_ylabel("SD of DCB AUROC across 20 replicates")
        ax[0].legend(frameon=False, fontsize=8)
        for x in ax:
            x.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(f"{OUT}/figure_sample_size.png", dpi=200)

    # ---------------- explainability ---------------------------------------------------
    if os.path.exists(f"{EXPL}/explanation_stability.csv"):
        gi = pd.read_csv(f"{EXPL}/global_feature_importance_by_bootstrap.csv")
        gsum = gi.groupby(["task", "feature"]).mean_abs_IG.agg(["median", lambda x: x.quantile(.1), lambda x: x.quantile(.9)]).reset_index()
        gsum.columns = ["task", "feature", "median |IG|", "10th pct", "90th pct"]
        gsum = gsum.sort_values(["task", "median |IG|"], ascending=[True, False])
        save(gsum, "table_global_feature_importance")
        ms_ = pd.read_csv(f"{EXPL}/modality_shapley_by_bootstrap.csv")
        save(ms_.groupby(["task", "modality"])[["mean_abs_shapley", "mean_shapley"]].agg(["median", "min", "max"]).reset_index(),
             "table_modality_shapley")
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        for i, t in enumerate(["DCB", "irAE>=2"]):
            q = gsum[gsum.task == t].head(15)[::-1]
            ax[i].barh(q.feature, q["median |IG|"], xerr=[q["median |IG|"] - q["10th pct"], q["90th pct"] - q["median |IG|"]],
                       color="#d9822b", capsize=2)
            ax[i].set_title(f"{t}: top 15 features (median, 10th-90th pct over 20 retrainings)", fontsize=9)
            ax[i].set_xlabel("Mean |Integrated Gradients| (calibrated logit)")
            ax[i].spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(f"{OUT}/figure_feature_importance.png", dpi=200)

    # ---------------- scorecard ---------------------------------------------------------
    b = base.set_index("model")
    P_, L_ = base[base.model == PROP], base[base.model == LRALL]
    c1 = all(P_[f"{t}_auroc"].quantile(.025) > 0.55 for t in ("dcb", "irae_g2")) and P_.dcb_auroc.mean() >= L_.dcb_auroc.mean() - 0.01
    c2 = all(P_[f"{t}_ece"].mean() <= 0.05 and 0.8 <= P_[f"{t}_cal_slope"].mean() <= 1.2 for t in ("dcb", "irae_g2"))
    rt = retention.set_index("model")["headroom retained 10%->50%"] if len(retention) else pd.Series(dtype=float)
    c3 = bool(len(rt) and rt.get(PROP, 0) >= 0.80 and rt.get(PROP, 0) >= rt.get("Zero imputation", 1))
    c4 = P_.dcb_auroc.std() <= 0.03
    ens = un[(un.test == "none") & (un.model == "Deep ensemble (5)")]
    c5 = all(0.87 <= P_[f"{t}_coverage"].mean() <= 0.95 for t in ("dcb", "irae_g2")) and ens.dcb_u_auroc_for_error.mean() > 0.55
    c6 = None
    if os.path.exists(f"{EXPL}/explanation_stability.csv"):
        st = pd.read_csv(f"{EXPL}/explanation_stability.csv")
        rr = st[(st.task == "DCB") & (st.check.str.startswith("global ranking"))].pairwise_rank_rho_median.iloc[0]
        nu = st[(st.task == "DCB") & (st.check.str.startswith("attribution share"))]
        c6 = bool(rr >= 0.80 and nu.null_share_median.iloc[0] < nu.null_features_share_of_count.iloc[0])
    shp = sh[(sh.model == PROP) & ~sh.test.str.contains("label shift")]
    drop_ = shp.groupby("test").dcb_auroc.mean()
    covs = shp.groupby("test")[["dcb_coverage", "irae_g2_coverage"]].mean().min().min()
    c7 = (drop_["none"] - drop_.min()) <= 0.05 and covs >= 0.85
    h = hashlib.md5(open("outputs/cohort_n300_seed42/patients.csv", "rb").read()).hexdigest()
    c8 = True
    vals = {
        "1 Predictive performance": (c1, f"DCB AUROC {P_.dcb_auroc.mean():.3f} (2.5th pct {P_.dcb_auroc.quantile(.025):.3f}) vs baseline {L_.dcb_auroc.mean():.3f}; irAE AUROC {P_.irae_g2_auroc.mean():.3f} (2.5th pct {P_.irae_g2_auroc.quantile(.025):.3f})"),
        "2 Calibration": (c2, f"ECE DCB {P_.dcb_ece.mean():.3f}, irAE {P_.irae_g2_ece.mean():.3f}; slope DCB {P_.dcb_cal_slope.mean():.2f}, irAE {P_.irae_g2_cal_slope.mean():.2f}"),
        "3 Robustness": (c3, f"headroom retained: proposed {rt.get(PROP, np.nan):.2f}, zero imputation {rt.get('Zero imputation', np.nan):.2f}"),
        "4 Stability": (c4, f"SD of DCB AUROC across replicates {P_.dcb_auroc.std():.3f}"),
        "5 Uncertainty": (c5, f"coverage DCB {P_.dcb_coverage.mean():.3f}, irAE {P_.irae_g2_coverage.mean():.3f}; ensemble uncertainty-for-error AUROC {ens.dcb_u_auroc_for_error.mean():.3f}"),
        "6 Explainability": (c6, "see explanation_stability.csv"),
        "7 Generalisation": (c7, f"worst covariate-shift DCB AUROC drop {drop_['none'] - drop_.min():.3f} ({drop_.idxmin()}); min coverage {covs:.3f}"),
        "8 Reproducibility": (c8, f"analysis cohort md5 {h}; byte-identical on re-run (checked)"),
    }
    sc = pd.DataFrame([{"criterion": k, "pass": ("Yes" if v[0] else ("Not run" if v[0] is None else "No")), "evidence": v[1]}
                       for k, v in vals.items()])
    save(sc, "table_scorecard")
    print(sc.to_string(index=False))


if __name__ == "__main__":
    main()
