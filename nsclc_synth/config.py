"""Configuration for the synthetic NSCLC multimodal data-generating process (DGP).

Every knob that the simulation scenarios (thesis Scenarios A-F) vary is exposed here,
so a scenario is fully described by a config object plus a random seed.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Optional
import json


@dataclass
class PopulationConfig:
    """Baseline patient characteristics (advanced NSCLC, PD-(L)1-treated)."""
    age_mean: float = 65.0          # KEYNOTE-042 median 63 (Mok 2019); Hellmann 2018 median 66
    age_sd: float = 8.5
    age_min: float = 35.0
    age_max: float = 90.0
    p_male: float = 0.62            # 49% (Hellmann 2018) to 71% (Mok 2019); midpoint-ish
    p_luad: float = 0.70            # non-squamous 79% (Hellmann 2018)
    # smoking among LUAD / LUSC (LUSC almost always ever-smokers)
    p_never_luad: float = 0.20
    p_never_lusc: float = 0.04
    p_current_given_ever: float = 0.35
    ecog_probs: tuple = (0.38, 0.52, 0.10)      # ECOG 0 / 1 / 2
    stage_probs: tuple = (0.12, 0.08, 0.42, 0.38)  # IIIB, IIIC, IVA, IVB
    p_first_line: float = 0.60
    p_chemo_io_first_line: float = 0.45  # share of first-line patients on chemo-immunotherapy
    p_egfr_luad: float = 0.08       # actionable drivers are under-represented in ICI cohorts
    p_stk11_luad: float = 0.12
    p_autoimmune_history: float = 0.06


@dataclass
class OutcomeConfig:
    """Outcome prevalences are hit by calibrating the model intercepts (see generator)."""
    dcb_prevalence: float = 0.30        # Rizvi 2018: 38.6% vs 25.1% above/below TMB median
    tox2_prevalence: float = 0.25       # grade >=2 irAE; Ksienski 2019: 20.7% interrupted ICI for irAE
    tox3_given_tox2: float = 0.42       # => grade >=3 ~ 10-11%; KEYNOTE-042 grade>=3 TRAE 18%
    tox_any_extra: float = 0.20         # additional grade-1-only events => any-grade ~45% (Ksienski 42.8%)
    # Strength of the shared immune-activation factor in both outcomes (drives irAE-benefit link)
    shared_immune_weight_dcb: float = 0.55
    shared_immune_weight_tox: float = 0.45
    # Oracle signal strength: scales all observed-predictor log-odds coefficients
    signal_scale: float = 1.0


@dataclass
class MissingnessConfig:
    """Probability that each modality is available; MAR dependence on clinical state."""
    p_pathology: float = 0.78
    p_radiology: float = 0.70
    p_systemic: float = 0.92
    # log-odds penalties: sicker patients less likely to have every modality
    ecog2_penalty: float = 0.6
    stage_ivb_penalty: float = 0.3
    within_modality_missing: float = 0.05   # missing single variables inside an available modality
    force_at_least_one: bool = True


@dataclass
class ShiftConfig:
    """Distribution-shift knobs (Scenario D). Zero = no shift."""
    scanner_offset: float = 0.0       # additive shift (SD units) on radiomic features & embedding
    scanner_scale: float = 1.0        # multiplicative shift on radiomic texture features
    tmb_assay_scale: float = 1.0      # e.g. panel-vs-WES calibration difference
    pdl1_assay_shift: float = 0.0     # additive shift on the latent PD-L1 before thresholding
    nlr_shift: float = 0.0            # additive shift on log NLR
    age_shift: float = 0.0            # years
    p_male_override: Optional[float] = None


@dataclass
class NoiseConfig:
    """Noise knobs (Scenario E)."""
    measurement_noise: float = 1.0    # multiplier on all measurement-error SDs
    embedding_noise: float = 1.0      # multiplier on isotropic noise in modality embeddings
    label_noise: float = 0.0          # symmetric flip probability on binary outcomes
    feature_corruption: float = 0.0   # fraction of embedding entries replaced by random noise


@dataclass
class EmbeddingConfig:
    d_pathology: int = 512
    d_radiology: int = 128
    d_systemic: int = 64
    signal_to_noise: float = 1.5      # ratio of structured to isotropic variance per embedding


@dataclass
class DGPConfig:
    n: int = 300
    seed: int = 42
    population: PopulationConfig = field(default_factory=PopulationConfig)
    outcome: OutcomeConfig = field(default_factory=OutcomeConfig)
    missingness: MissingnessConfig = field(default_factory=MissingnessConfig)
    shift: ShiftConfig = field(default_factory=ShiftConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    # seed for the fixed embedding loading matrices; kept separate from the patient seed so
    # every cohort (development, external, scenario) shares one "foundation model"
    loading_seed: int = 2026

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
