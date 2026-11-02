"""Reviewer comment 3: run a REAL pretrained pathology foundation model on REAL TCGA-LUSC slides.

Run this on Google Colab with a GPU (Runtime > Change runtime type > T4 GPU), or on any CUDA machine.
It produces, for each slide:
    <out>/patch_embeddings/<slide_id>.npy   patch embeddings  (K x 1024 for UNI)
    <out>/patch_coords/<slide_id>.npy       patch (x, y) coordinates at level 0
    <out>/slide_embeddings.csv              mean-pooled slide embedding (1 row per slide)
    <out>/run_log.json                      model, magnification, patch counts, timings

Send me slide_embeddings.csv, run_log.json and the patch_embeddings folder (zip). I then
plug them into the attention-MIL pathology stream (thesis Section 3.5.1).

STEP 0 (in a Colab cell, once):
    !pip -q install openslide-python openslide-bin timm huggingface_hub pandas
    # gdc-client for slide download: on gdc.cancer.gov/access-data/gdc-data-transfer-tool copy
    # the link for the Ubuntu build, then:  !wget -q "<that link>" -O gdc.zip && unzip -o gdc.zip
    # (unzip any inner zip too) and check with  !./gdc-client --version

STEP 1  Model access.  UNI (Chen et al., 2024, Nature Medicine) is gated: create a free
        Hugging Face account, open https://huggingface.co/MahmoodLab/UNI, accept the terms,
        create a read token, then in Colab:   from huggingface_hub import login; login()
        If access is not granted in time, use --model ctranspath-free fallback below
        (a non-gated ImageNet ViT; report it honestly as NOT a pathology foundation model).

STEP 2  Slides.  On https://portal.gdc.cancer.gov/projects/TCGA-LUSC open Repository, filter
        Data Type = Slide Image, Experimental Strategy = Diagnostic Slide, add 30-50 files to
        the cart, download the MANIFEST (small text file), upload it to Colab, then:
            !./gdc-client download -m gdc_manifest.txt -d slides
        Diagnostic slides are 0.2-3 GB each, so 30 slides is roughly 30-60 GB: start with 10
        to test, then add more. Colab disk is ~100 GB.

STEP 3  Run:
        !python run_foundation_model_tcga.py --slides slides --out fm_out --model uni --max-patches 2000
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import torch


def load_model(name: str):
    import timm
    if name == "uni":
        model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True,
                                  init_values=1e-5, dynamic_img_size=True)
        mean, std, dim, label = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 1024, "UNI (ViT-L/16, Chen et al. 2024)"
    elif name == "imagenet-vit":
        model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        mean, std, dim, label = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 768, "ImageNet ViT-B/16 (NOT a pathology foundation model)"
    else:
        raise ValueError(name)
    model.eval()
    return model, mean, std, dim, label


def tissue_patches(slide, patch_px: int, target_mpp: float, max_patches: int, sat_thresh: int = 20):
    """Saturation-threshold tissue mask on a thumbnail, then a grid of non-overlapping patches."""
    from PIL import Image
    mpp = float(slide.properties.get("openslide.mpp-x", 0.25))
    level0_patch = int(round(patch_px * target_mpp / mpp))          # patch size at level 0
    W, H = slide.dimensions
    thumb = slide.get_thumbnail((W // 64, H // 64)).convert("HSV")
    sat = np.array(thumb)[:, :, 1]
    mask = sat > sat_thresh
    sx, sy = W / mask.shape[1], H / mask.shape[0]
    coords = []
    for y in range(0, H - level0_patch, level0_patch):
        for x in range(0, W - level0_patch, level0_patch):
            my0, mx0 = int(y / sy), int(x / sx)
            my1, mx1 = max(my0 + 1, int((y + level0_patch) / sy)), max(mx0 + 1, int((x + level0_patch) / sx))
            if mask[my0:my1, mx0:mx1].mean() > 0.5:
                coords.append((x, y))
    rng = np.random.default_rng(0)
    if len(coords) > max_patches:
        coords = [coords[i] for i in sorted(rng.choice(len(coords), max_patches, replace=False))]
    return coords, level0_patch, mpp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides", required=True)
    ap.add_argument("--out", default="fm_out")
    ap.add_argument("--model", default="uni", choices=["uni", "imagenet-vit"])
    ap.add_argument("--patch-px", type=int, default=224)
    ap.add_argument("--mpp", type=float, default=0.5, help="target microns per pixel (0.5 ~ 20x)")
    ap.add_argument("--max-patches", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()

    import openslide
    from PIL import Image
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, mean, std, dim, label = load_model(a.model)
    model = model.to(dev)
    m = torch.tensor(mean, device=dev).view(1, 3, 1, 1)
    s = torch.tensor(std, device=dev).view(1, 3, 1, 1)
    os.makedirs(f"{a.out}/patch_embeddings", exist_ok=True)
    os.makedirs(f"{a.out}/patch_coords", exist_ok=True)

    files = sorted(glob.glob(os.path.join(a.slides, "**", "*.svs"), recursive=True))
    log = {"model": label, "device": dev, "patch_px": a.patch_px, "target_mpp": a.mpp,
           "max_patches": a.max_patches, "slides": []}
    rows = []
    for f in files:
        sid = os.path.basename(f).replace(".svs", "")
        if os.path.exists(f"{a.out}/patch_embeddings/{sid}.npy"):
            continue
        t0 = time.time()
        sl = openslide.OpenSlide(f)
        coords, p0, mpp = tissue_patches(sl, a.patch_px, a.mpp, a.max_patches)
        feats = []
        for i in range(0, len(coords), a.batch):
            imgs = [np.asarray(sl.read_region(c, 0, (p0, p0)).convert("RGB").resize((a.patch_px, a.patch_px), Image.BILINEAR))
                    for c in coords[i:i + a.batch]]
            x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float().to(dev) / 255.0
            with torch.inference_mode(), torch.autocast(device_type=dev, enabled=(dev == "cuda")):
                feats.append(model((x - m) / s).float().cpu().numpy())
        E = np.concatenate(feats) if feats else np.zeros((0, dim), np.float32)
        np.save(f"{a.out}/patch_embeddings/{sid}.npy", E.astype(np.float16))
        np.save(f"{a.out}/patch_coords/{sid}.npy", np.array(coords, dtype=np.int32))
        case = "-".join(sid.split("-")[:3])                 # TCGA-XX-XXXX patient barcode
        rows.append({"slide_id": sid, "case_submitter_id": case, "n_patches": len(coords),
                     **{f"e{j}": v for j, v in enumerate(E.mean(0) if len(E) else np.zeros(dim))}})
        log["slides"].append({"slide_id": sid, "n_patches": len(coords), "slide_mpp": mpp,
                              "seconds": round(time.time() - t0, 1)})
        print(f"{sid}: {len(coords)} patches in {time.time()-t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(f"{a.out}/slide_embeddings.csv", index=False)
        json.dump(log, open(f"{a.out}/run_log.json", "w"), indent=2)
    print("done:", len(rows), "slides")


if __name__ == "__main__":
    main()
