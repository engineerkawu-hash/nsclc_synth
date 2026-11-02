"""Documented data-generating process (DGP) for a synthetic multimodal NSCLC immunotherapy cohort.

Structure (thesis Section 3.3.3):

    hidden latent factors  -->  observed clinical / pathology / radiology / systemic variables
            |                          |
            +-------> outcomes <-------+   (durable clinical benefit; irAE grade >=2, >=3, any)

Hidden latent factors (standard normal unless stated):
    U_imm    tumour immune activation ("hot" tumour). Partly visible through PD-L1, TILs,
             the pathology embedding and weakly through radiomic texture. Shared by both
             outcomes, which produces the reported positive irAE-benefit association.
    U_burden tumour burden. Drives volume, LDH, ctDNA, stage, ECOG.
    U_infl   systemic inflammation. Drives NLR, CRP, albumin.
    U_auto   autoimmune susceptibility. Drives toxicity only.
    U_sev    toxicity-severity propensity. Drives grade >=3 given grade >=2.
    U_resp   response determinants not visible in any modality (irreducible noise).

Every relationship is listed in RELATIONSHIPS with its justification and source, and that
table is exported verbatim so the thesis documents exactly what was simulated.
"""
from __future__ import annotations

import copy
from dataclasses import replace
from functools import lru_cache
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm

from .config import DGPConfig, NoiseConfig, ShiftConfig

# ---------------------------------------------------------------------------------------
# Relationship registry: the single source of truth for what was intentionally simulated.
# ---------------------------------------------------------------------------------------
RELATIONSHIPS = [
    # (cause, effect, direction, form, strength, justification, source, evidence_status)
    ("histology=LUSC", "male sex", "+", "probability", "P(male|LUSC) = P(male)+0.12",
     "Squamous histology is more common in men.", "Assumption; consistent with TCGA-LUSC (74% male, this study)", "assumption-checked"),
    ("histology=LUSC", "never-smoker", "-", "probability", "P(never|LUSC)=0.04 vs 0.20 in LUAD",
     "Squamous NSCLC is strongly smoking-associated.", "Assumption", "assumption"),
    ("U_burden", "ECOG, stage", "+", "latent ordinal", "loading 0.45 (ECOG), 0.50 (stage)",
     "Higher tumour burden worsens performance status and stage.", "Assumption", "assumption"),
    ("U_imm", "PD-L1 TPS", "+", "latent -> thresholded TPS", "loading 0.60",
     "PD-L1 reflects adaptive immune resistance in inflamed tumours.", "Doroshow et al. (2021) [thesis ref.]", "literature"),
    ("smoking, male, ECOG 0", "PD-L1 TPS", "+", "latent shift", "0.25 / 0.10 / 0.10 SD",
     "PD-L1 expression more frequent in smokers, men and good PS.", "Yang et al. (2021), doi:10.7150/jca.63003", "literature"),
    ("EGFR mutation", "PD-L1 TPS", "-", "latent shift", "-0.40 SD",
     "PD-L1 expression less frequent in EGFR-mutant tumours.", "Yang et al. (2021), doi:10.7150/jca.63003", "literature"),
    ("U_imm", "TIL density, stromal TIL %", "+", "log-linear", "loading 0.65",
     "TILs are the histological correlate of an inflamed tumour.", "Saltz et al. (2018); AbdulJabbar et al. (2020) [thesis refs.]", "literature"),
    ("U_imm", "pathology embedding z_P", "+", "linear loadings", "SNR 1.5",
     "Foundation-model embeddings encode morphology including immune infiltrate beyond PD-L1.", "Chen et al. (2024); Rakaee et al. (2025) [thesis refs.]", "literature"),
    ("U_imm", "radiomic texture (entropy)", "+", "linear", "loading 0.30",
     "A radiomic signature tracks CD8 infiltration.", "Sun et al. (2018) [thesis ref.]", "literature"),
    ("U_burden", "tumour volume, LDH, ctDNA VAF", "+", "log-linear", "loading 0.80 / 0.50 / 0.60",
     "Tumour burden markers co-vary.", "Assumption", "assumption"),
    ("smoking", "TMB", "+", "log-linear", "+0.55 (former), +0.70 (current) log-units",
     "Tobacco exposure raises mutation frequency more than tenfold.", "Govindan et al. (2012), doi:10.1016/j.cell.2012.08.024", "literature"),
    ("EGFR mutation", "TMB", "-", "log-linear", "-0.60 log-units",
     "EGFR-mutant tumours have lower TMB (median 3.77 vs 6.12 mut/Mb).", "Offin et al. (2019), doi:10.1158/1078-0432.CCR-18-1102", "literature"),
    ("TMB", "PD-L1", "0", "independence", "no shared latent",
     "TMB and PD-L1 are independent predictors.", "Rizvi et al. (2018), doi:10.1200/JCO.2017.75.3384; Hellmann et al. (2018) rho=-0.087", "literature"),
    ("U_infl", "NLR, CRP (+), albumin (-)", "+/-", "log-linear", "loading 0.55 / 0.60 / -0.45",
     "Systemic inflammation markers co-vary.", "Assumption", "assumption"),
    ("U_imm, TMB", "durable clinical benefit", "+", "logistic", "1.00 / 0.55 per SD",
     "Inflamed, high-TMB tumours benefit more.", "Rizvi et al. (2018): DCB 38.6% vs 25.1% above vs below TMB median", "literature"),
    ("NLR", "durable clinical benefit", "-", "logistic", "-0.45 per SD log NLR",
     "High baseline NLR predicts worse ICI outcome.", "Xie et al. (2019) doi:10.1080/07357907.2019.1639057; Yang et al. (2021) doi:10.1136/bmjopen-2021-049123", "literature"),
    ("U_burden, ECOG", "durable clinical benefit", "-", "logistic", "-0.35 / -0.35",
     "Higher burden and poor PS reduce benefit.", "Ksienski et al. (2019) doi:10.1016/j.cllc.2018.09.005 (ECOG>=2 worse OS)", "literature"),
    ("STK11, EGFR", "durable clinical benefit", "-", "logistic", "-0.60 / -0.90",
     "EGFR and STK11 variants associate with lack of benefit.", "Rizvi et al. (2018), doi:10.1200/JCO.2017.75.3384", "literature"),
    ("chemo-immunotherapy", "durable clinical benefit", "+", "logistic", "+0.35",
     "Adding chemotherapy improves response.", "Pan et al. (2026) doi:10.21037/tcr-2026-1-0328", "literature"),
    ("U_imm", "irAE grade>=2", "+", "logistic (shared factor)", "0.45",
     "irAEs associate with better ICI outcomes (marker of immune activation).", "Kopanos et al. (2025) doi:10.3390/cancers17223675; Zhao et al. (2026) doi:10.1016/j.lungcan.2026.109422", "literature"),
    ("U_auto, autoimmune history", "irAE grade>=2", "+", "logistic", "0.90 / 1.00",
     "Host autoimmune predisposition drives toxicity.", "Les et al. (2023) [thesis ref.]", "literature"),
    ("chemo-immunotherapy", "irAE grade>=2", "+", "logistic", "+0.70 (OR ~2)",
     "Combination therapy roughly doubles irAE risk.", "Chen et al. (2025) doi:10.1080/14737140.2025.2522980 (RR 2.02)", "literature"),
    ("U_sev, chemo-IO, age", "irAE grade>=3 | grade>=2", "+", "logistic", "0.80 / 0.40 / 0.30",
     "Severity is partly host- and regimen-dependent.", "Assumption", "assumption"),
    ("ECOG 2, stage IVB", "modality availability", "-", "MAR logistic", "-0.6 / -0.3",
     "Sicker patients less often have complete work-up.", "Assumption", "assumption"),
]

RELATIONSHIP_COLUMNS = ["cause", "effect", "direction", "form", "strength",
                        "justification", "source", "evidence_status"]


def relationships_table() -> pd.DataFrame:
    return pd.DataFrame(RELATIONSHIPS, columns=RELATIONSHIP_COLUMNS)


# ---------------------------------------------------------------------------------------
# Modality membership of the interpretable variables
# ---------------------------------------------------------------------------------------
CLINICAL_VARS = ["age", "sex", "histology", "smoking", "ecog", "stage", "line",
                 "regimen", "chemo_io", "autoimmune_history"]
PATHOLOGY_VARS = ["pdl1_tps", "til_density", "stromal_til_pct", "necrosis_pct"]
RADIOLOGY_VARS = ["tumour_volume_cm3", "sphericity", "glcm_entropy", "glcm_contrast"]
SYSTEMIC_VARS = ["tmb", "nlr", "ldh", "albumin", "crp", "ctdna_vaf", "egfr_mut", "stk11_mut"]
OUTCOME_VARS = ["dcb", "irae_any", "irae_g2", "irae_g3", "irae_max_grade",
                "recist_best", "irae_organ"]


def _ordinal_from_latent(score: np.ndarray, probs) -> np.ndarray:
    """Cut a ~N(0,1) score into ordered categories with the requested marginal probs."""
    cuts = norm.ppf(np.cumsum(probs)[:-1])
    return np.searchsorted(cuts, score)


def _loadings(rng: np.random.Generator, d_out: int, d_in: int) -> np.ndarray:
    """Fixed random loading matrix with unit-norm columns (one per latent input)."""
    w = rng.normal(size=(d_out, d_in))
    return w / np.linalg.norm(w, axis=0, keepdims=True) * np.sqrt(d_out / d_in)


def _embed(rng, A, inputs, snr, noise_mult, corruption):
    structured = inputs @ A.T                       # n x d
    s_var = structured.var(axis=0).mean() if len(structured) > 1 else 1.0
    noise_sd = np.sqrt(max(s_var, 1e-6) / snr) * noise_mult
    z = structured + rng.normal(scale=noise_sd, size=structured.shape)
    if corruption > 0:
        mask = rng.random(z.shape) < corruption
        z[mask] = rng.normal(scale=z.std(), size=mask.sum())
    return z.astype(np.float32)


# ---------------------------------------------------------------------------------------
# Core simulation (latent + observed + linear predictors); intercepts supplied externally
# ---------------------------------------------------------------------------------------
def _simulate_core(cfg: DGPConfig, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    n, P, S, Nz = cfg.n, cfg.population, cfg.shift, cfg.noise
    mn = Nz.measurement_noise
    lat = {k: rng.standard_normal(n) for k in
           ["U_imm", "U_burden", "U_infl", "U_auto", "U_sev", "U_resp"]}

    # --- clinical -------------------------------------------------------------------
    p_male = S.p_male_override if S.p_male_override is not None else P.p_male
    lusc = rng.random(n) > P.p_luad
    p_m_lusc = min(0.95, p_male + 0.12)
    p_m_luad = np.clip((p_male - (1 - P.p_luad) * p_m_lusc) / P.p_luad, 0.05, 0.95)
    male = rng.random(n) < np.where(lusc, p_m_lusc, p_m_luad)
    age = np.clip(rng.normal(P.age_mean + S.age_shift, P.age_sd, n), P.age_min, P.age_max)

    never = rng.random(n) < np.where(lusc, P.p_never_lusc, P.p_never_luad)
    current = (~never) & (rng.random(n) < P.p_current_given_ever)
    smoking = np.where(never, "Never", np.where(current, "Current", "Former"))

    ecog = _ordinal_from_latent(0.45 * lat["U_burden"] + np.sqrt(1 - 0.45**2) * rng.standard_normal(n),
                                P.ecog_probs)
    stage_idx = _ordinal_from_latent(0.50 * lat["U_burden"] + np.sqrt(1 - 0.25) * rng.standard_normal(n),
                                     P.stage_probs)
    stage = np.array(["IIIB", "IIIC", "IVA", "IVB"])[stage_idx]
    first_line = rng.random(n) < P.p_first_line
    chemo_io = first_line & (rng.random(n) < P.p_chemo_io_first_line)
    mono = rng.choice(["Pembrolizumab", "Nivolumab", "Atezolizumab"], n, p=[0.55, 0.28, 0.17])
    regimen = np.where(chemo_io, "Chemo-IO", mono)
    line = np.where(first_line, "1L", np.where(rng.random(n) < 0.72, "2L", "3L+"))
    autoimmune = rng.random(n) < P.p_autoimmune_history

    egfr = (~lusc) & (rng.random(n) < np.where(never, P.p_egfr_luad * 3.5, P.p_egfr_luad * 0.5))
    stk11 = (~lusc) & (~egfr) & (rng.random(n) < np.where(never, 0.03, P.p_stk11_luad * 1.2))

    smoker = (~never).astype(float)
    # --- pathology --------------------------------------------------------------------
    # population-level centring constants (not cohort means) so shifted cohorts stay shifted
    pd_lat = (0.60 * lat["U_imm"] + 0.25 * (smoker - 0.86) + 0.10 * (male - 0.62)
              + 0.10 * ((ecog == 0) - 0.38) - 0.40 * (egfr - 0.03))
    pd_lat = pd_lat + 0.79 * rng.standard_normal(n)          # residual SD so Var(pd_lat) ~ 1
    pd_obs = pd_lat + 0.25 * mn * rng.standard_normal(n) + S.pdl1_assay_shift   # assay error
    # cut-points chosen for TPS<1% ~ 40%, 1-49% ~ 32%, >=50% ~ 28%
    z1, z2 = norm.ppf(0.40), norm.ppf(0.72)
    u = norm.cdf(pd_obs)
    tps = np.where(pd_obs < z1, 0.0,
          np.where(pd_obs < z2, 1 + 48 * (u - norm.cdf(z1)) / (norm.cdf(z2) - norm.cdf(z1)),
                   50 + 50 * (u - norm.cdf(z2)) / (1 - norm.cdf(z2))))
    tps = np.round(np.clip(tps, 0, 100))

    til_lat = 0.65 * lat["U_imm"] + np.sqrt(1 - 0.65**2) * rng.standard_normal(n)
    til_density = np.exp(5.6 + 0.7 * til_lat + 0.15 * mn * rng.standard_normal(n))       # cells/mm2
    stromal_til = np.clip(100 * expit(-1.4 + 0.9 * til_lat + 0.2 * mn * rng.standard_normal(n)), 0, 100)
    necrosis = np.clip(100 * expit(-2.2 + 0.6 * lat["U_burden"] + 0.4 * lusc
                                   + 0.5 * rng.standard_normal(n)), 0, 100)

    # --- radiology --------------------------------------------------------------------
    log_vol = np.log(25) + 0.80 * lat["U_burden"] + 0.55 * rng.standard_normal(n)
    vol = np.exp(log_vol + 0.05 * mn * rng.standard_normal(n))
    sphericity = np.clip(0.75 - 0.06 * lat["U_burden"] + 0.06 * rng.standard_normal(n), 0.3, 0.98)
    entropy = (4.6 + 0.30 * lat["U_imm"] + 0.25 * lat["U_burden"] + 0.40 * rng.standard_normal(n))
    contrast = np.exp(3.0 + 0.20 * lat["U_burden"] - 0.15 * lat["U_imm"] + 0.35 * rng.standard_normal(n))
    entropy = entropy * S.scanner_scale + 0.4 * S.scanner_offset + 0.1 * mn * rng.standard_normal(n)
    contrast = contrast * S.scanner_scale * np.exp(0.2 * S.scanner_offset)

    # --- systemic / molecular ---------------------------------------------------------
    log_tmb_true = (np.log(4.3) + 0.55 * (smoking == "Former") + 0.70 * (smoking == "Current")
                    + 0.20 * lusc - 0.60 * egfr + 0.55 * rng.standard_normal(n))
    tmb = np.exp(log_tmb_true + 0.15 * mn * rng.standard_normal(n)) * S.tmb_assay_scale
    log_nlr_true = np.log(3.1) + 0.55 * 0.6 * lat["U_infl"] + 0.10 * lat["U_burden"] + 0.25 * rng.standard_normal(n)
    nlr = np.exp(log_nlr_true + S.nlr_shift + 0.10 * mn * rng.standard_normal(n))
    ldh = np.exp(np.log(230) + 0.50 * 0.45 * lat["U_burden"] + 0.20 * rng.standard_normal(n))
    albumin = np.clip(3.9 - 0.45 * 0.45 * lat["U_infl"] + 0.35 * rng.standard_normal(n), 2.0, 5.3)
    crp = np.exp(np.log(12) + 0.60 * 1.0 * lat["U_infl"] + 0.6 * rng.standard_normal(n))
    ctdna = np.exp(np.log(2.5) + 0.60 * lat["U_burden"] + 0.7 * rng.standard_normal(n))

    # --- standardised drivers for the outcome model ------------------------------------
    # Fixed (population-level) standardisation constants, so a shifted cohort is not silently
    # re-centred: z_tmb, z_nlr and z_age are on the source-population scale.
    z_tmb = (log_tmb_true - (np.log(4.3) + 0.45)) / 0.62
    z_nlr = (log_nlr_true - np.log(3.1)) / 0.43
    z_age = (age - 65.0) / 8.5
    s = cfg.outcome.signal_scale
    eta_dcb = s * (1.00 * lat["U_imm"] + 0.55 * z_tmb - 0.45 * z_nlr
                   - 0.35 * lat["U_burden"] - 0.35 * (ecog - 0.72)
                   - 0.60 * stk11 - 0.90 * egfr + 0.35 * chemo_io) + 0.45 * lat["U_resp"]
    eta_tox = s * (cfg.outcome.shared_immune_weight_tox * lat["U_imm"] + 0.90 * lat["U_auto"]
                   + 1.00 * autoimmune + 0.70 * chemo_io)
    eta_sev = 0.80 * lat["U_sev"] + 0.40 * chemo_io + 0.30 * z_age
    eta_g1 = 0.6 * lat["U_auto"]

    return dict(lat=lat, age=age, male=male, lusc=lusc, smoking=smoking, ecog=ecog, stage=stage,
                line=line, regimen=regimen, chemo_io=chemo_io, autoimmune=autoimmune,
                egfr=egfr, stk11=stk11, tps=tps, pd_lat=pd_lat, til_density=til_density,
                stromal_til=stromal_til, necrosis=necrosis, vol=vol, sphericity=sphericity,
                entropy=entropy, contrast=contrast, tmb=tmb, nlr=nlr, ldh=ldh, albumin=albumin,
                crp=crp, ctdna=ctdna, eta_dcb=eta_dcb, eta_tox=eta_tox, eta_sev=eta_sev,
                eta_g1=eta_g1, log_tmb_true=log_tmb_true, log_nlr_true=log_nlr_true, til_lat=til_lat)


def _solve_intercept(eta: np.ndarray, target: float) -> float:
    lo, hi = -15.0, 15.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if expit(eta + mid).mean() < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


@lru_cache(maxsize=64)
def _intercepts_cached(cfg_key: str) -> Tuple[float, float, float, float]:
    cfg = _CFG_REGISTRY[cfg_key]
    ref = copy.deepcopy(cfg)
    ref.n, ref.shift, ref.noise = 200_000, ShiftConfig(), NoiseConfig()
    core = _simulate_core(ref, np.random.default_rng(987654321))
    O = cfg.outcome
    b_dcb = _solve_intercept(core["eta_dcb"], O.dcb_prevalence)
    b_tox = _solve_intercept(core["eta_tox"], O.tox2_prevalence)
    b_sev = _solve_intercept(core["eta_sev"], O.tox3_given_tox2)
    b_g1 = _solve_intercept(core["eta_g1"], O.tox_any_extra / (1 - O.tox2_prevalence))
    return b_dcb, b_tox, b_sev, b_g1


_CFG_REGISTRY: Dict[str, DGPConfig] = {}


def calibrated_intercepts(cfg: DGPConfig) -> Tuple[float, float, float, float]:
    """Intercepts are solved once on a 200,000-patient reference population drawn from the
    *unshifted, noise-free* version of the config, so that (i) the requested prevalences hold
    in the source population and (ii) distribution shift changes prevalence naturally."""
    key_cfg = copy.deepcopy(cfg)
    key_cfg.n, key_cfg.seed = 0, 0
    key_cfg.shift, key_cfg.noise = ShiftConfig(), NoiseConfig()
    key = repr(key_cfg.to_dict())
    _CFG_REGISTRY[key] = key_cfg
    return _intercepts_cached(key)


# ---------------------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------------------
def generate(cfg: DGPConfig) -> Dict[str, object]:
    """Generate one synthetic cohort.

    Returns a dict with:
        patients : DataFrame of clinical, interpretable modality variables, availability flags
                   and outcomes (NaN where a modality is unavailable)
        z_pathology, z_radiology, z_systemic : float32 arrays (NaN rows where unavailable)
        latent   : DataFrame of hidden factors and true linear predictors (validation only;
                   never to be used as model input)
    """
    rng = np.random.default_rng(cfg.seed)
    c = _simulate_core(cfg, rng)
    n, lat = cfg.n, c["lat"]
    b_dcb, b_tox, b_sev, b_g1 = calibrated_intercepts(cfg)

    p_dcb = expit(c["eta_dcb"] + b_dcb)
    p_tox = expit(c["eta_tox"] + b_tox)
    dcb = rng.random(n) < p_dcb
    g2 = rng.random(n) < p_tox
    g3 = g2 & (rng.random(n) < expit(c["eta_sev"] + b_sev))
    g1_only = (~g2) & (rng.random(n) < expit(c["eta_g1"] + b_g1))
    g4 = g3 & (rng.random(n) < 0.15)
    max_grade = np.select([g4, g3, g2, g1_only], [4, 3, 2, 1], 0)

    # RECIST best overall response, consistent with DCB (CR/PR/SD lasting >6 months)
    r = rng.random(n)
    recist = np.where(dcb, np.where(r < 0.08, "CR", np.where(r < 0.62, "PR", "SD")),
                      np.where(r < 0.06, "PR", np.where(r < 0.22, "SD", "PD")))
    organs = np.array(["Cutaneous", "Endocrine", "Gastrointestinal", "Pulmonary", "Hepatic"])
    organ_p = np.array([0.32, 0.24, 0.19, 0.14, 0.11])
    organ = np.where(max_grade > 0, rng.choice(organs, n, p=organ_p), "None")
    pulm_bump = (max_grade > 0) & c["lusc"] & (rng.random(n) < 0.08)
    organ = np.where(pulm_bump, "Pulmonary", organ)

    # label noise (Scenario E)
    ln = cfg.noise.label_noise
    if ln > 0:
        dcb = np.where(rng.random(n) < ln, ~dcb, dcb)
        g2 = np.where(rng.random(n) < ln, ~g2, g2)

    patients = pd.DataFrame({
        "patient_id": [f"SYN{cfg.seed:04d}-{i:05d}" for i in range(1, n + 1)],
        "age": np.round(c["age"]).astype(int),
        "sex": np.where(c["male"], "Male", "Female"),
        "histology": np.where(c["lusc"], "LUSC", "LUAD"),
        "smoking": c["smoking"], "ecog": c["ecog"], "stage": c["stage"], "line": c["line"],
        "regimen": c["regimen"], "chemo_io": c["chemo_io"].astype(int),
        "autoimmune_history": c["autoimmune"].astype(int),
        "pdl1_tps": c["tps"], "til_density": np.round(c["til_density"], 1),
        "stromal_til_pct": np.round(c["stromal_til"], 1), "necrosis_pct": np.round(c["necrosis"], 1),
        "tumour_volume_cm3": np.round(c["vol"], 2), "sphericity": np.round(c["sphericity"], 3),
        "glcm_entropy": np.round(c["entropy"], 3), "glcm_contrast": np.round(c["contrast"], 2),
        "tmb": np.round(c["tmb"], 2), "nlr": np.round(c["nlr"], 2), "ldh": np.round(c["ldh"]),
        "albumin": np.round(c["albumin"], 2), "crp": np.round(c["crp"], 1),
        "ctdna_vaf": np.round(c["ctdna"], 2),
        "egfr_mut": c["egfr"].astype(int), "stk11_mut": c["stk11"].astype(int),
        "dcb": dcb.astype(int), "irae_any": (max_grade > 0).astype(int),
        "irae_g2": g2.astype(int), "irae_g3": g3.astype(int), "irae_max_grade": max_grade,
        "recist_best": recist, "irae_organ": organ,
    })
    if ln > 0:  # keep the graded hierarchy coherent with the (possibly flipped) g2 label
        patients.loc[patients.irae_g2 == 0, "irae_g3"] = 0

    # --- embeddings (one fixed "foundation model" shared by every cohort) -------------
    E = cfg.embedding
    lrng = np.random.default_rng(cfg.loading_seed)
    A_P = _loadings(lrng, E.d_pathology, 6)
    A_R = _loadings(lrng, E.d_radiology, 5)
    A_L = _loadings(lrng, E.d_systemic, 6)
    erng = np.random.default_rng(cfg.seed + 7919)
    zs = lambda x: (x - x.mean()) / (x.std() + 1e-9)
    in_P = np.column_stack([lat["U_imm"], zs(c["pd_lat"]), c["til_lat"], zs(c["lusc"].astype(float)),
                            zs(np.log(c["necrosis"] + 1)), 0.3 * lat["U_burden"]])
    in_R = np.column_stack([zs(np.log(c["vol"])), zs(c["sphericity"]), zs(c["entropy"]),
                            zs(np.log(c["contrast"])), 0.3 * lat["U_imm"]])
    in_L = np.column_stack([zs(np.log(c["tmb"])), zs(np.log(c["nlr"])), zs(np.log(c["ldh"])),
                            zs(c["albumin"]), zs(np.log(c["crp"])), zs(np.log(c["ctdna"]))])
    Nz = cfg.noise
    z_P = _embed(erng, A_P, in_P, E.signal_to_noise, Nz.embedding_noise, Nz.feature_corruption)
    z_R = _embed(erng, A_R, in_R, E.signal_to_noise, Nz.embedding_noise, Nz.feature_corruption)
    z_R = z_R + cfg.shift.scanner_offset  # scanner shift also moves the learned representation
    z_L = _embed(erng, A_L, in_L, E.signal_to_noise, Nz.embedding_noise, Nz.feature_corruption)

    # --- modality availability (MAR) ---------------------------------------------------
    M = cfg.missingness
    mrng = np.random.default_rng(cfg.seed + 104729)
    pen = M.ecog2_penalty * (c["ecog"] == 2) + M.stage_ivb_penalty * (c["stage"] == "IVB")
    avail = {}
    for name, p in [("pathology", M.p_pathology), ("radiology", M.p_radiology), ("systemic", M.p_systemic)]:
        avail[name] = mrng.random(n) < expit(logit(np.clip(p, 1e-4, 1 - 1e-4)) - pen)
    if M.force_at_least_one:
        none = ~(avail["pathology"] | avail["radiology"] | avail["systemic"])
        avail["systemic"] = avail["systemic"] | none
    for name, cols, arr in [("pathology", PATHOLOGY_VARS, z_P), ("radiology", RADIOLOGY_VARS, z_R),
                            ("systemic", SYSTEMIC_VARS, z_L)]:
        patients[f"has_{name}"] = avail[name].astype(int)
        patients.loc[~avail[name], cols] = np.nan
        arr[~avail[name]] = np.nan
        if M.within_modality_missing > 0:  # sporadic missing values inside an available modality
            for col in cols:
                hole = avail[name] & (mrng.random(n) < M.within_modality_missing)
                patients.loc[hole, col] = np.nan
    patients["n_modalities"] = patients[["has_pathology", "has_radiology", "has_systemic"]].sum(axis=1)

    latent = pd.DataFrame({k: v for k, v in lat.items()})
    latent["patient_id"] = patients["patient_id"]
    latent["p_dcb_true"], latent["p_irae_g2_true"] = p_dcb, p_tox
    return dict(patients=patients, z_pathology=z_P, z_radiology=z_R, z_systemic=z_L,
                latent=latent, intercepts=dict(dcb=b_dcb, tox2=b_tox, sev=b_sev, g1=b_g1))


def save_cohort(cohort: Dict[str, object], outdir: str, cfg: DGPConfig) -> None:
    import os
    os.makedirs(outdir, exist_ok=True)
    cohort["patients"].to_csv(os.path.join(outdir, "patients.csv"), index=False)
    cohort["latent"].to_csv(os.path.join(outdir, "latent_truth_DO_NOT_USE_AS_INPUT.csv"), index=False)
    for k in ["z_pathology", "z_radiology", "z_systemic"]:
        np.save(os.path.join(outdir, f"{k}.npy"), cohort[k])
    cfg.to_json(os.path.join(outdir, "config.json"))
