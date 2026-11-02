"""Multimodal fusion network with configurable fusion and missing-modality strategies.

Fusion strategies (comment 8)
  early         concatenate every block (clinical + P + R + S, median-filled) -> MLP -> heads
  intermediate  modality encoders -> concatenate encodings -> MLP trunk -> heads
  late          one network per modality (+clinical); average the logits of available modalities
  attention     modality encoders -> attention pooling over modality tokens (not missing-aware)
  missing_aware attention pooling that masks absent modalities, learned "absent" embeddings and
                modality dropout during training  (the proposed mechanism)

Missing-modality strategies (comment 6), used with intermediate fusion
  complete_case  train on patients with all three modalities only (median fill at test)
  median         fill absent block with training medians, no indicator
  zero           fill absent block with zeros (training mean), no indicator
  mask           zero fill + availability indicators fed to the trunk
  learned_emb    absent modality's encoding replaced by a learned vector
  mod_dropout    mask + random modality dropout during training
  proposed       learned_emb + mod_dropout + masked attention fusion (= fusion 'missing_aware')

Heads: dual task (durable clinical benefit, irAE grade >= 2) over a shared representation, or
single task. Probabilities are Platt-recalibrated on a held-out calibration split.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from .data import MODALITIES, Prepared

torch.set_num_threads(int(__import__("os").environ.get("NSCLC_THREADS", "2")))


@dataclass
class ModelSpec:
    fusion: str = "missing_aware"
    missing: str = "proposed"
    tasks: Sequence[int] = (0, 1)        # 0 = DCB, 1 = irAE >= 2
    modalities: Sequence[str] = ("P", "R", "S")
    use_embeddings: bool = True           # False = interpretable features only (ablation)
    hidden: int = 32
    dropout: float = 0.3
    p_mod_dropout: float = 0.25
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-3
    batch: int = 64


class Encoder(nn.Module):
    def __init__(self, d_in, h, p):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(min(p, 0.2)), nn.Linear(d_in, 2 * h), nn.GELU(),
                                 nn.Dropout(p), nn.Linear(2 * h, h), nn.GELU())

    def forward(self, x):
        return self.net(x)


class FusionNet(nn.Module):
    def __init__(self, spec: ModelSpec, dims: Dict[str, int], d_clin: int):
        super().__init__()
        self.spec, h = spec, spec.hidden
        self.mods = list(spec.modalities)
        nt = len(spec.tasks)
        if spec.fusion == "early":
            d = d_clin + sum(dims[k] for k in self.mods)
            self.trunk = nn.Sequential(nn.Dropout(0.2), nn.Linear(d, 2 * h), nn.GELU(), nn.Dropout(spec.dropout),
                                       nn.Linear(2 * h, h), nn.GELU())
            self.head = nn.Linear(h, nt)
            return
        self.clin = nn.Sequential(nn.Linear(d_clin, h), nn.GELU())
        self.enc = nn.ModuleDict({k: Encoder(dims[k], h, spec.dropout) for k in self.mods})
        self.absent = nn.ParameterDict({k: nn.Parameter(torch.zeros(h)) for k in self.mods})
        if spec.fusion == "late":
            self.heads = nn.ModuleDict({k: nn.Linear(2 * h, nt) for k in self.mods + ["C"]})
            return
        if spec.fusion in ("attention", "missing_aware"):
            self.type_emb = nn.Parameter(torch.randn(len(self.mods) + 1, h) * 0.02)
            self.score = nn.Sequential(nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))
            d_trunk = 2 * h
        else:  # intermediate
            d_trunk = h * (len(self.mods) + 1) + (len(self.mods) if spec.missing in ("mask", "mod_dropout") else 0)
        self.trunk = nn.Sequential(nn.Dropout(spec.dropout), nn.Linear(d_trunk, h), nn.GELU())
        self.head = nn.Linear(h, nt)

    def forward(self, C, X, m, return_attn=False):
        sp = self.spec
        if sp.fusion == "early":
            z = self.trunk(torch.cat([C] + [X[k] for k in self.mods], 1))
            return self.head(z)
        c = self.clin(C)
        hs = []
        for i, k in enumerate(self.mods):
            hk = self.enc[k](X[k])
            mk = m[:, MODALITIES.index(k)].unsqueeze(1)
            if sp.missing in ("learned_emb", "proposed") or sp.fusion == "missing_aware":
                hk = mk * hk + (1 - mk) * self.absent[k]
            hs.append(hk)
        if sp.fusion == "late":
            logits, wts = [], []
            for k, hk in zip(self.mods, hs):
                logits.append(self.heads[k](torch.cat([hk, c], 1)))
                wts.append(m[:, MODALITIES.index(k)].unsqueeze(1))
            logits.append(self.heads["C"](torch.cat([c, c], 1)))
            wts.append(torch.ones_like(wts[0]))
            W = torch.stack(wts, 0)
            return (torch.stack(logits, 0) * W).sum(0) / W.sum(0)
        if sp.fusion in ("attention", "missing_aware"):
            tok = torch.stack([c] + hs, 1) + self.type_emb.unsqueeze(0)          # n x (M+1) x h
            s = self.score(tok).squeeze(-1)
            if sp.fusion == "missing_aware":
                avail = torch.cat([torch.ones_like(m[:, :1]), m[:, [MODALITIES.index(k) for k in self.mods]]], 1)
                s = s.masked_fill(avail == 0, -1e4)
            a = torch.softmax(s, 1)
            pooled = (a.unsqueeze(-1) * tok).sum(1)
            out = self.head(self.trunk(torch.cat([pooled, c], 1)))
            return (out, a) if return_attn else out
        z = torch.cat([c] + hs, 1)
        if sp.missing in ("mask", "mod_dropout"):
            z = torch.cat([z, m[:, [MODALITIES.index(k) for k in self.mods]]], 1)
        return self.head(self.trunk(z))


def _block(d: Prepared, spec: ModelSpec, med: Optional[Dict[str, np.ndarray]] = None):
    X = {}
    for k in spec.modalities:
        B = d.X[k] if spec.use_embeddings else d.X[k][:, :d.n_interp[k]]
        if spec.missing in ("median", "complete_case") or spec.fusion in ("early", "attention"):
            if med is not None:
                miss = d.m[:, MODALITIES.index(k)] == 0
                B = B.copy()
                B[miss] = med[k] if spec.use_embeddings else med[k][:d.n_interp[k]]
        X[k] = torch.from_numpy(np.ascontiguousarray(B))
    return torch.from_numpy(d.C), X, torch.from_numpy(d.m)


class FittedModel:
    def __init__(self, spec: ModelSpec):
        self.spec = spec

    def fit(self, train: Prepared, calib: Prepared, seed: int = 0):
        sp = self.spec
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        self.med = {k: np.median(train.X[k][train.m[:, MODALITIES.index(k)] == 1], 0) for k in MODALITIES}
        if sp.missing == "complete_case":
            keep = train.m[:, [MODALITIES.index(k) for k in sp.modalities]].min(1) == 1
            from .data import subset
            train = subset(train, np.where(keep)[0])
        C, X, M = _block(train, sp, self.med)
        Y = torch.from_numpy(train.y[:, list(sp.tasks)])
        dims = {k: X[k].shape[1] for k in sp.modalities}
        self.net = FusionNet(sp, dims, C.shape[1])
        opt = torch.optim.AdamW(self.net.parameters(), lr=sp.lr, weight_decay=sp.weight_decay)
        pos = Y.mean(0).clamp(0.05, 0.95)
        pw = (1 - pos) / pos
        n = len(Y)
        do_moddrop = sp.missing in ("mod_dropout", "proposed") or sp.fusion == "missing_aware"
        for ep in range(sp.epochs):
            self.net.train()
            perm = rng.permutation(n)
            for i in range(0, n, sp.batch):
                b = perm[i:i + sp.batch]
                Mb = M[b].clone()
                if do_moddrop:
                    drop = torch.from_numpy((rng.random(Mb.shape) < sp.p_mod_dropout).astype(np.float32))
                    Mb = Mb * (1 - drop)
                Xb = {k: X[k][b] * Mb[:, MODALITIES.index(k)].unsqueeze(1) if do_moddrop else X[k][b]
                      for k in sp.modalities}
                out = self.net(C[b], Xb, Mb)
                loss = F.binary_cross_entropy_with_logits(out, Y[b], pos_weight=pw)
                opt.zero_grad()
                loss.backward()
                opt.step()
        # Platt scaling per task on the calibration split
        raw = self.logits(calib)
        self.platt = []
        for j, t in enumerate(sp.tasks):
            lr = LogisticRegression(C=1e4).fit(raw[:, [j]], calib.y[:, t])
            self.platt.append((float(lr.coef_[0, 0]), float(lr.intercept_[0])))
        return self

    @torch.no_grad()
    def logits(self, d: Prepared, mc_dropout: bool = False):
        self.net.train(mc_dropout)
        C, X, M = _block(d, self.spec, self.med)
        return self.net(C, X, M).numpy()

    def predict(self, d: Prepared, mc_dropout: bool = False):
        L = self.logits(d, mc_dropout)
        return np.column_stack([expit(a * L[:, j] + b) for j, (a, b) in enumerate(self.platt)])

    def predict_raw(self, d: Prepared):
        """Uncalibrated sigmoid(logit). Used as the conformal score: it does not depend on the
        calibration split, so split-conformal exchangeability holds."""
        return expit(self.logits(d))

    def mc_predict(self, d: Prepared, T: int = 30, seed: int = 0):
        torch.manual_seed(seed)
        return np.stack([self.predict(d, mc_dropout=True) for _ in range(T)], 0)

    @torch.no_grad()
    def attention(self, d: Prepared):
        self.net.eval()
        C, X, M = _block(d, self.spec, self.med)
        _, a = self.net(C, X, M, return_attn=True)
        return a.numpy()


def integrated_gradients(model: FittedModel, d: Prepared, task_idx: int, steps: int = 32):
    """Integrated Gradients (Aumann-Shapley path attribution) on the calibrated-logit scale,
    baseline = training mean (zeros after standardisation). Returns per-block attributions."""
    sp, net = model.spec, model.net
    net.eval()
    C, X, M = _block(d, sp, model.med)
    a_scale = model.platt[task_idx][0]
    inputs = {"C": C, **X}
    total = {k: torch.zeros_like(v) for k, v in inputs.items()}
    for s in np.linspace(1.0 / steps, 1.0, steps):
        scaled = {k: (v * s).requires_grad_(True) for k, v in inputs.items()}
        out = net(scaled["C"], {k: scaled[k] for k in sp.modalities}, M)[:, task_idx]
        grads = torch.autograd.grad(out.sum(), list(scaled.values()))
        for (k, _), g in zip(scaled.items(), grads):
            total[k] += g / steps
    return {k: (total[k] * inputs[k] * a_scale).detach().numpy() for k in inputs}
