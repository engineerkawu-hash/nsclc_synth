"""Reviewer comment 3: diagnostics for real pathology foundation-model embeddings (TCGA-LUSC).

Input: the folder written by colab/TCGA_LUSC_real_data.ipynb (patch_embeddings/, patch_coords/,
thumbnails/, slide_embeddings.csv, run_log.json). No outcome labels are used anywhere.

Checks
  1. Run summary: slides, patches, magnification, time per slide.
  2. Representation structure: effective dimensionality (PCA), within- vs between-slide cosine
     similarity, and slide identification from a single patch by 1-nearest-neighbour (chance = 1/30).
     High identification means the embedding carries slide-specific morphological information
     rather than a generic constant vector.
  3. Spatial coherence: for each slide, the correlation between patch-embedding similarity and
     physical distance; neighbouring tissue should look alike. A PCA-to-colour map is drawn on the
     slide thumbnail for visual inspection.
  4. Linkage: how many embedded slides have real clinical and molecular records, giving genuine
     multi-stream patient records for the systemic and pathology streams.

Usage: python scripts/real_tcga_pathology.py --fm <notebook output folder> \
        --molecular outputs/real_tcga_molecular/tcga_lusc_patient_level_with_molecular.csv --out outputs/real_tcga_pathology
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402


def l2n(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fm", required=True)
    ap.add_argument("--molecular", default="outputs/real_tcga_molecular/tcga_lusc_patient_level_with_molecular.csv")
    ap.add_argument("--out", default="outputs/real_tcga_pathology")
    ap.add_argument("--per-slide", type=int, default=200, help="patches sampled per slide for the retrieval test")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(2026)
    runlog = json.load(open(os.path.join(a.fm, "run_log.json")))
    sl = pd.DataFrame(runlog["slides"])

    E, C, lab = {}, {}, []
    for f in sorted(glob.glob(os.path.join(a.fm, "patch_embeddings", "*.npy"))):
        sid = os.path.basename(f)[:-4]
        E[sid] = np.load(f).astype(np.float32)
        C[sid] = np.load(os.path.join(a.fm, "patch_coords", sid + ".npy"))
    sids = list(E)
    allE = np.concatenate([E[s] for s in sids])
    info = {"model": runlog["model"], "device": runlog["device"], "slides": len(sids),
            "patches_total": int(len(allE)), "embedding_dim": int(allE.shape[1]),
            "failed_slides": len(runlog.get("failed", []))}

    # 1. run summary
    summ = pd.DataFrame([
        ("Foundation model", runlog["model"]),
        ("Hardware", runlog["device"].upper() + " (Google Colab)"),
        ("Diagnostic slides embedded / attempted", f"{len(sids)} / {len(sids) + info['failed_slides']}"),
        ("Patch size and magnification", f"{runlog['patch_px']} px at {runlog['target_mpp']} um/px (about 20x)"),
        ("Tissue patches available per slide, median (range)",
         f"{int(sl.tissue_patches_available.median()):,} ({int(sl.tissue_patches_available.min()):,}-{int(sl.tissue_patches_available.max()):,})"),
        ("Patches embedded", f"{info['patches_total']:,} (cap {runlog['max_patches']} per slide)"),
        ("Embedding dimension", str(info["embedding_dim"])),
        ("Slide file size, median (range), GB", f"{sl.file_gb.median():.2f} ({sl.file_gb.min():.2f}-{sl.file_gb.max():.2f})"),
        ("Time per slide incl. download, median (range), s", f"{sl.total_s.median():.0f} ({sl.total_s.min():.0f}-{sl.total_s.max():.0f})"),
    ], columns=["Item", "Value"])
    summ.to_csv(os.path.join(a.out, "table_fm_run_summary.csv"), index=False)

    # 2. representation structure
    pca = PCA().fit(allE)
    ev = pca.explained_variance_ratio_
    cum = np.cumsum(ev)
    part_ratio = float(ev.sum() ** 2 / (ev ** 2).sum())
    sub, slab = [], []
    for i, s in enumerate(sids):
        k = min(a.per_slide, len(E[s]))
        sub.append(E[s][rng.choice(len(E[s]), k, replace=False)]); slab += [i] * k
    X = l2n(np.concatenate(sub)); y = np.array(slab)
    S = X @ X.T
    same = y[:, None] == y[None, :]
    np.fill_diagonal(S, -np.inf)
    nn = y[S.argmax(1)]
    acc = float((nn == y).mean())
    np.fill_diagonal(S, np.nan)
    within = float(np.nanmean(np.where(same, S, np.nan)))
    between = float(np.nanmean(np.where(~same, S, np.nan)))
    # slide-level: mean-pooled embeddings
    M = l2n(np.stack([E[s].mean(0) for s in sids]))
    SM = M @ M.T
    slide_between = float(SM[~np.eye(len(sids), dtype=bool)].mean())
    struct = pd.DataFrame([
        ("Principal components for 50% / 80% / 95% of variance",
         f"{int(np.searchsorted(cum, .5)) + 1} / {int(np.searchsorted(cum, .8)) + 1} / {int(np.searchsorted(cum, .95)) + 1} of {allE.shape[1]}"),
        ("Effective dimensionality (participation ratio)", f"{part_ratio:.1f}"),
        ("Mean cosine similarity, patches within the same slide", f"{within:.3f}"),
        ("Mean cosine similarity, patches from different slides", f"{between:.3f}"),
        ("Slide identified from one patch by 1-nearest neighbour", f"{100 * acc:.1f}% (chance {100 / len(sids):.1f}%)"),
        ("Mean cosine similarity between slide-level (mean-pooled) embeddings", f"{slide_between:.3f}"),
    ], columns=["Diagnostic", "Value"])
    struct.to_csv(os.path.join(a.out, "table_fm_representation.csv"), index=False)
    info.update(nn_slide_accuracy=acc, within_cos=within, between_cos=between, participation_ratio=part_ratio,
                slide_level_mean_cos=slide_between)

    # 3. spatial coherence: rank correlation between embedding distance and physical distance
    rows = []
    for s in sids:
        e, c = l2n(E[s]), C[s].astype(float)
        if len(e) < 50:
            continue
        idx = rng.choice(len(e), min(400, len(e)), replace=False)
        e, c = e[idx], c[idx]
        iu = np.triu_indices(len(e), 1)
        de = 1 - (e @ e.T)[iu]
        dp = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))[iu]
        rho = spearmanr(de, dp).statistic
        near = dp <= np.quantile(dp, 0.05)
        far = dp >= np.quantile(dp, 0.95)
        rows.append({"slide_id": s, "spearman_embedding_vs_physical_distance": rho,
                     "cos_sim_nearest_5pct_pairs": float(1 - de[near].mean()),
                     "cos_sim_farthest_5pct_pairs": float(1 - de[far].mean())})
    sp = pd.DataFrame(rows)
    sp.to_csv(os.path.join(a.out, "spatial_coherence_by_slide.csv"), index=False)
    spat = pd.DataFrame([
        ("Slides analysed", f"{len(sp)}"),
        ("Spearman rho, embedding distance vs physical distance, median (range)",
         f"{sp.iloc[:, 1].median():.3f} ({sp.iloc[:, 1].min():.3f} to {sp.iloc[:, 1].max():.3f})"),
        ("Slides with rho > 0", f"{int((sp.iloc[:, 1] > 0).sum())} of {len(sp)}"),
        ("Cosine similarity, nearest 5% of patch pairs, median", f"{sp.cos_sim_nearest_5pct_pairs.median():.3f}"),
        ("Cosine similarity, farthest 5% of patch pairs, median", f"{sp.cos_sim_farthest_5pct_pairs.median():.3f}"),
    ], columns=["Diagnostic", "Value"])
    spat.to_csv(os.path.join(a.out, "table_fm_spatial_coherence.csv"), index=False)

    # PCA colour maps on thumbnails
    thumbs = sorted(glob.glob(os.path.join(a.fm, "thumbnails", "*.jpg")))[:3]
    if thumbs:
        p3 = PCA(3).fit(allE)
        fig, axes = plt.subplots(2, len(thumbs), figsize=(4.2 * len(thumbs), 7.6))
        axes = np.array(axes).reshape(2, -1)
        for j, t in enumerate(thumbs):
            s = os.path.basename(t)[:-4]
            im = Image.open(t)
            z = p3.transform(E[s])
            z = (z - np.percentile(z, 2, 0)) / (np.percentile(z, 98, 0) - np.percentile(z, 2, 0) + 1e-8)
            z = z.clip(0, 1)
            sc = 64.0  # thumbnails are level-0 dimensions / 64
            p0 = sl.loc[sl.slide_id == s, "level0_patch_px"].iloc[0] / sc
            axes[0, j].imshow(im); axes[0, j].set_title("-".join(s.split("-")[:3]), fontsize=9)
            axes[1, j].imshow(np.full((im.size[1], im.size[0], 3), 1.0))
            for (x, yv), col in zip(C[s] / sc, z):
                axes[1, j].add_patch(plt.Rectangle((x, yv), p0, p0, color=col, lw=0))
            axes[1, j].set_xlim(0, im.size[0]); axes[1, j].set_ylim(im.size[1], 0)
            for r in (0, 1):
                axes[r, j].axis("off")
        axes[0, 0].text(-0.04, 0.5, "H&E thumbnail", transform=axes[0, 0].transAxes, rotation=90, va="center", ha="right")
        axes[1, 0].text(-0.04, 0.5, "Patch embeddings (PCA 1-3 as RGB)", transform=axes[1, 0].transAxes,
                        rotation=90, va="center", ha="right")
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "figure_fm_pca_maps.png"), dpi=200)

    # 4. linkage with real clinical and molecular data
    se = pd.read_csv(os.path.join(a.fm, "slide_embeddings.csv"))
    if os.path.exists(a.molecular):
        mol = pd.read_csv(a.molecular)
        lk = se[["slide_id", "case_id"]].merge(mol, on="case_id", how="left")
        link = pd.DataFrame([
            ("Embedded slides", len(se)),
            ("With real clinical record", int(lk["age"].notna().sum())),
            ("With real TMB and driver-gene status", int(lk["tmb"].notna().sum())),
            ("With smoking status", int(lk["smoking"].notna().sum())),
            ("With overall-survival time", int(lk["os_days"].notna().sum())),
        ], columns=["Linkage", "Patients (n)"])
        link.to_csv(os.path.join(a.out, "table_fm_linkage.csv"), index=False)
        print(link.to_string(index=False))

    json.dump(info, open(os.path.join(a.out, "pathology_log.json"), "w"), indent=2)
    for t in [summ, struct, spat]:
        print(t.to_string(index=False)); print()


if __name__ == "__main__":
    main()
