"""Reviewer comment 2: run the systemic/clinical preprocessing pipeline on REAL data (TCGA-LUSC).

What this does and does not show
  - It shows the preprocessing stage of the systemic stream runs on genuine, messy clinical
    records: multi-row patients, conflicting codes, informative missingness, absent variables.
  - It reports which framework variables real public clinical data can and cannot supply.
  - As a proxy check that the pipeline preserves real clinical signal, it relates the
    processed vector to overall survival (OS). OS is NOT an immunotherapy outcome; TCGA-LUSC
    patients were largely treated before routine checkpoint inhibition.

Usage: python scripts/real_tcga_systemic.py --tcga <xlsx> --synthetic outputs/cohort_n300_seed42 --out outputs/real_tcga
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.model_selection import StratifiedKFold
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nsclc_synth.systemic_pipeline import SystemicPreprocessor, unify_missing  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STAGE_MAIN = {"Stage I": 1, "Stage IA": 1, "Stage IB": 1, "Stage II": 2, "Stage IIA": 2, "Stage IIB": 2,
              "Stage III": 3, "Stage IIIA": 3, "Stage IIIB": 3, "Stage IIIC": 3, "Stage IV": 4,
              "Stage IVA": 4, "Stage IVB": 4}
SYN_STAGE = {"IIIB": 3, "IIIC": 3, "IVA": 4, "IVB": 4}


def first_valid(s):
    s = s.dropna()
    return s.iloc[0] if len(s) else np.nan


def collapse(full: pd.DataFrame):
    """One row per patient. Returns (patients, data-quality log)."""
    f = unify_missing(full)
    g = f.groupby("cases.case_id")
    qlog = {"rows_in_file": int(len(f)), "patients": int(g.ngroups),
            "rows_per_patient": g.size().value_counts().sort_index().to_dict()}

    prim = f[f["diagnoses.diagnosis_is_primary_disease"].astype(str).str.lower() == "true"]
    # among primary-disease rows keep those carrying a pathological stage, if any
    prim = prim.assign(_has_stage=prim["diagnoses.ajcc_pathologic_stage"].notna())
    prim = prim.sort_values(["cases.case_id", "_has_stage"], ascending=[True, False])
    n_stage_values = prim.groupby("cases.case_id")["diagnoses.ajcc_pathologic_stage"].nunique()
    qlog["patients_with_conflicting_stage_codes"] = int((n_stage_values > 1).sum())
    pd1 = prim.groupby("cases.case_id").first()
    qlog["patients_without_primary_disease_row"] = int(g.ngroups - pd1.shape[0])
    other_dx = f.loc[f["diagnoses.diagnosis_is_primary_disease"].astype(str).str.lower() == "false",
                     "diagnoses.classification_of_tumor"].value_counts().to_dict()
    qlog["non_primary_diagnosis_rows_by_type"] = {str(k): int(v) for k, v in other_dx.items()}

    tr = f[f["treatments.treatment_or_therapy"].astype(str).str.lower() == "yes"]
    tt = tr["treatments.treatment_type"].astype(str)
    agg_tr = pd.DataFrame({
        "any_systemic_therapy": tr.assign(x=tt.str.contains("Chemotherapy|Pharmaceutical")).groupby("cases.case_id")["x"].any(),
        "any_radiotherapy": tr.assign(x=tt.str.contains("Radiation")).groupby("cases.case_id")["x"].any(),
        "any_immunotherapy": tr.assign(x=tt.str.contains("Immunotherapy")).groupby("cases.case_id")["x"].any(),
    })

    p = pd.DataFrame({
        "age": g["demographic.age_at_index"].agg(first_valid),
        "sex": g["demographic.sex_at_birth"].agg(first_valid),
        "race": g["demographic.race"].agg(first_valid),
        "vital_status": g["demographic.vital_status"].agg(first_valid),
        "days_to_death": g["demographic.days_to_death"].agg(lambda s: pd.to_numeric(s, errors="coerce").max()),
        "days_to_last_follow_up": g["diagnoses.days_to_last_follow_up"].agg(lambda s: pd.to_numeric(s, errors="coerce").max()),
    })
    for src, dst in [("diagnoses.ajcc_pathologic_stage", "stage_raw"), ("diagnoses.ajcc_pathologic_t", "t_cat"),
                     ("diagnoses.ajcc_pathologic_n", "n_cat"), ("diagnoses.ajcc_pathologic_m", "m_cat"),
                     ("diagnoses.laterality", "laterality"), ("diagnoses.residual_disease", "residual_disease"),
                     ("diagnoses.prior_malignancy", "prior_malignancy"), ("diagnoses.primary_diagnosis", "subtype"),
                     ("diagnoses.sites_of_involvement", "site")]:
        p[dst] = pd1[src]
    p = p.join(agg_tr.astype(float))
    p[["any_systemic_therapy", "any_radiotherapy", "any_immunotherapy"]] = p[["any_systemic_therapy", "any_radiotherapy", "any_immunotherapy"]].fillna(0.0)
    p["stage"] = p["stage_raw"].map(STAGE_MAIN)
    p["t_num"] = pd.to_numeric(p["t_cat"].astype(str).str.extract(r"T(\d)")[0], errors="coerce")
    p["n_num"] = pd.to_numeric(p["n_cat"].astype(str).str.extract(r"N(\d)")[0], errors="coerce")
    p["subtype"] = p["subtype"].replace({"Squamous cell carcinoma, NOS": "SCC NOS"})
    p.loc[~p["subtype"].isin(["SCC NOS", np.nan]), "subtype"] = "SCC variant/other"

    # overall survival
    dead = p["vital_status"].astype(str).str.lower() == "dead"
    p["os_event"] = dead.astype(int)
    p["os_days"] = np.where(dead, p["days_to_death"], p["days_to_last_follow_up"])
    bad = p["os_days"].isna() | (p["os_days"] < 0)
    qlog["patients_without_usable_survival_time"] = int(bad.sum())
    qlog["deaths"] = int(dead.sum())
    for c in ["any_systemic_therapy", "any_radiotherapy", "any_immunotherapy"]:
        p[c] = p[c].astype(int)
    return p, qlog


SCHEMA = [
    # framework variable, stream, TCGA-LUSC clinical file?, note
    ("Age", "clinical", "Yes", "age_at_index"),
    ("Sex", "clinical", "Yes", "sex_at_birth"),
    ("Histology", "clinical", "Yes (constant)", "all LUSC by project definition"),
    ("Stage", "clinical", "Yes", "pathological AJCC stage; mostly early stage"),
    ("Smoking history", "clinical", "No", "exposure records not in this download; GDC exposure file"),
    ("ECOG performance status", "clinical", "No", "not recorded"),
    ("Line / ICI regimen", "clinical", "No", "pre-immunotherapy era; 4 patients any immunotherapy"),
    ("Autoimmune history", "clinical", "No", "not recorded"),
    ("TMB", "molecular", "No (obtainable)", "derivable from GDC open-access masked somatic MAF"),
    ("EGFR / STK11 status", "molecular", "No (obtainable)", "derivable from GDC open-access masked somatic MAF"),
    ("NLR, CRP, albumin, LDH", "blood", "No", "no laboratory values in TCGA clinical data"),
    ("ctDNA", "blood (liquid biopsy)", "No", "no blood-based sequencing in TCGA"),
    ("Response (RECIST/iRECIST)", "outcome", "No", "TCGA predates routine ICI use"),
    ("irAE (CTCAE grade)", "outcome", "No", "TCGA predates routine ICI use"),
]


def cv_cindex(df, cols, n_splits=5, seeds=(1, 2, 3, 4, 5), penalizer=0.1, prep_kwargs=None):
    """Repeated stratified CV; preprocessing is fitted inside each training fold."""
    scores = []
    for sd in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=sd)
        for tr_idx, te_idx in skf.split(df, df["os_event"]):
            tr, te = df.iloc[tr_idx], df.iloc[te_idx]
            prep = SystemicPreprocessor(**prep_kwargs).fit(tr[cols])
            Xtr, Xte = prep.transform(tr[cols]), prep.transform(te[cols])
            keep = Xtr.columns[Xtr.std() > 0]
            fit_df = Xtr[keep].assign(T=tr["os_days"].values, E=tr["os_event"].values)
            cph = CoxPHFitter(penalizer=penalizer).fit(fit_df, "T", "E")
            risk = cph.predict_partial_hazard(Xte[keep])
            scores.append(concordance_index(te["os_days"], -risk, te["os_event"]))
    return float(np.mean(scores)), float(np.std(scores)), len(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tcga", required=True)
    ap.add_argument("--synthetic", default="outputs/cohort_n300_seed42")
    ap.add_argument("--out", default="outputs/real_tcga")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    full = pd.read_excel(a.tcga, sheet_name="Full_Data")
    p, qlog = collapse(full)
    p.to_csv(os.path.join(a.out, "tcga_lusc_patient_level.csv"))

    # cross-check against the Key_Clinical sheet
    key = pd.read_excel(a.tcga, sheet_name="Key_Clinical")
    qlog["check_patients_match_key_sheet"] = bool(len(p) == len(key))
    qlog["check_deaths_match_key_sheet"] = bool(qlog["deaths"] == int((key["Vital Status"] == "Dead").sum()))

    # ---- missingness report --------------------------------------------------------
    report_vars = ["age", "sex", "race", "stage", "t_num", "n_num", "m_cat", "laterality",
                   "residual_disease", "prior_malignancy", "site", "os_days"]
    miss = pd.DataFrame({"Variable": report_vars,
                         "Missing (n)": [int(p[v].isna().sum()) for v in report_vars],
                         "Missing (%)": [round(100 * p[v].isna().mean(), 1) for v in report_vars]})
    # informative missingness: is a missing value associated with death?
    inf = []
    for v in ["stage", "residual_disease", "site", "race"]:
        m = p[v].isna()
        tab = pd.crosstab(m, p["os_event"]).reindex(index=[False, True], columns=[0, 1], fill_value=0)
        orr, pv = fisher_exact(tab.values)
        inf.append({"Variable": v, "Deaths when missing (%)": round(100 * p.loc[m, "os_event"].mean(), 1) if m.any() else np.nan,
                    "Deaths when recorded (%)": round(100 * p.loc[~m, "os_event"].mean(), 1),
                    "Fisher p": round(pv, 4)})
    miss.to_csv(os.path.join(a.out, "table_missingness.csv"), index=False)
    pd.DataFrame(inf).to_csv(os.path.join(a.out, "table_informative_missingness.csv"), index=False)
    pd.DataFrame(SCHEMA, columns=["Framework variable", "Stream", "Available in TCGA-LUSC clinical", "Note"]).to_csv(
        os.path.join(a.out, "table_schema_coverage.csv"), index=False)

    # ---- the preprocessing pipeline on real data ---------------------------------------
    usable = p[p["os_days"].notna() & (p["os_days"] >= 0)].copy()
    usable.loc[usable["os_days"] == 0, "os_days"] = 0.5
    full_cols = dict(continuous=["age"], ordinal=["stage", "t_num", "n_num"],
                     categorical=["sex", "race", "m_cat", "laterality", "residual_disease", "prior_malignancy",
                                  "site", "subtype"])
    cols_all = full_cols["continuous"] + full_cols["ordinal"] + full_cols["categorical"]
    prep = SystemicPreprocessor(**full_cols).fit(usable[cols_all])
    X = prep.transform(usable[cols_all])
    assert not X.isna().any().any(), "pipeline left missing values"
    qlog["real_vector_dimension"] = int(X.shape[1])
    qlog["real_vector_features"] = list(X.columns)
    qlog["patients_in_survival_analysis"] = int(len(usable))
    qlog["events_in_survival_analysis"] = int(usable["os_event"].sum())

    # same code path on the synthetic cohort (shared variables only)
    syn = pd.read_csv(os.path.join(a.synthetic, "patients.csv"))
    syn["stage_num"] = syn["stage"].map(SYN_STAGE)
    syn_prep = SystemicPreprocessor(continuous=["age", "tmb", "nlr", "ldh", "albumin", "crp", "ctdna_vaf"],
                                    ordinal=["ecog", "stage_num"],
                                    categorical=["sex", "histology", "smoking", "egfr_mut", "stk11_mut"],
                                    log_vars=["tmb", "nlr", "ldh", "crp", "ctdna_vaf"])
    Xs = syn_prep.fit_transform(syn[syn_prep.continuous + syn_prep.ordinal + syn_prep.categorical])
    assert not Xs.isna().any().any()
    qlog["synthetic_vector_dimension"] = int(Xs.shape[1])

    # ---- proxy-signal check with overall survival --------------------------------------
    res = []
    for label, kw in [("Stage only", dict(continuous=[], ordinal=["stage"], categorical=[])),
                      ("Age + sex + stage", dict(continuous=["age"], ordinal=["stage"], categorical=["sex"])),
                      ("Full processed clinical vector", full_cols)]:
        cols = kw["continuous"] + kw["ordinal"] + kw["categorical"]
        m_, s_, k_ = cv_cindex(usable, cols, prep_kwargs=kw)
        res.append({"Model (penalised Cox, 5x5-fold CV)": label, "C-index mean": round(m_, 3),
                    "C-index SD": round(s_, 3), "Folds": k_})
    res = pd.DataFrame(res)
    res.to_csv(os.path.join(a.out, "table_proxy_survival_cindex.csv"), index=False)

    # KM by stage
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    kmf = KaplanMeierFitter()
    colors = {1: "#3b6ea5", 2: "#5aa469", 3: "#d9822b", 4: "#b23a48"}
    for st in [1, 2, 3, 4]:
        sub = usable[usable["stage"] == st]
        if len(sub) >= 5:
            kmf.fit(sub["os_days"] / 30.44, sub["os_event"], label=f"Stage {'I'*st if st<4 else 'IV'} (n={len(sub)})")
            kmf.plot_survival_function(ax=ax, ci_show=False, color=colors[st])
    ax.set_xlabel("Months since diagnosis"); ax.set_ylabel("Overall survival")
    ax.set_title("TCGA-LUSC (real): overall survival by pathological stage", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "figure_tcga_km_by_stage.png"), dpi=200)

    with open(os.path.join(a.out, "data_quality_log.json"), "w") as fh:
        json.dump(qlog, fh, indent=2, default=str)
    pd.set_option("display.width", 220, "display.max_colwidth", 70)
    print(json.dumps({k: v for k, v in qlog.items() if k != "real_vector_features"}, indent=2, default=str))
    print(miss.to_string(index=False)); print(pd.DataFrame(inf).to_string(index=False)); print(res.to_string(index=False))
    print(p["stage"].value_counts(dropna=False).sort_index().to_dict(), "immunotherapy:", int(p["any_immunotherapy"].sum()))


if __name__ == "__main__":
    main()
