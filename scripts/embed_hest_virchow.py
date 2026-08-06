import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import timm
from PIL import Image
from timm.data import resolve_data_config, create_transform
from tqdm import tqdm


REPO_MODEL = "hf_hub:paige-ai/Virchow2"


def decode_barcode(x):
    """Decode HDF5 barcode values safely."""
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def load_sample_ids(sample_csv: Path, technology: str = "Visium", include_xenium: bool = False):
    df = pd.read_csv(sample_csv)

    if not include_xenium:
        df = df[df["st_technology"].astype(str).str.lower() == technology.lower()]

    sample_ids = df["id"].astype(str).tolist()

    if len(sample_ids) == 0:
        raise ValueError(f"No samples found in {sample_csv} for technology={technology}")

    return sample_ids


def load_virchow(device: str):
    print(f"Loading Virchow model: {REPO_MODEL}")

    model = timm.create_model(
        REPO_MODEL,
        pretrained=True,
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    )

    model.eval()
    model.to(device)

    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    print("Model loaded.")
    print(f"Transform config: {config}")

    return model, transform


def virchow_forward(model, x):
    # Extract features from the Virchow model.
    with torch.no_grad():
        if hasattr(model, "forward_features"):
            out = model.forward_features(x)
        else:
            out = model(x)

    if isinstance(out, dict):
        if "x" in out:
            out = out["x"]
        elif "features" in out:
            out = out["features"]
        else:
            out = next(iter(out.values()))

    if isinstance(out, tuple):
        out = out[0]

    # Case 1: already [B, D]
    if out.ndim == 2:
        return out

    # Case 2: token output [B, tokens, D]
    if out.ndim == 3:
        cls_token = out[:, 0, :]
        patch_tokens = out[:, 1:, :]
        patch_mean = patch_tokens.mean(dim=1)
        return torch.cat([cls_token, patch_mean], dim=1)

    raise RuntimeError(f"Unexpected model output shape: {out.shape}")


def embed_sample(
    sample_id: str,
    patches_dir: Path,
    out_dir: Path,
    model,
    transform,
    device: str,
    batch_size: int,
    max_patches: int | None = None,
    overwrite: bool = False,
):
    patch_path = patches_dir / f"{sample_id}.h5"
    out_npz = out_dir / f"{sample_id}_virchow.npz"
    out_csv = out_dir / f"{sample_id}_barcodes.csv"

    if out_npz.exists() and not overwrite:
        print(f"[SKIP] {sample_id}: output already exists: {out_npz}")
        return

    if not patch_path.exists():
        print(f"[MISSING] {sample_id}: {patch_path}")
        return

    print(f"\nProcessing sample: {sample_id}")
    print(f"Patch file: {patch_path}")

    with h5py.File(patch_path, "r") as f:
        imgs = f["img"]
        coords = f["coords"][:]
        raw_barcodes = f["barcode"][:]

        barcodes = [decode_barcode(b[0]) for b in raw_barcodes]

        n_total = imgs.shape[0]
        n_use = min(n_total, max_patches) if max_patches is not None else n_total

        print(f"Total patches: {n_total}")
        print(f"Using patches: {n_use}")
        print(f"Image shape: {imgs.shape[1:]}")

        all_embeddings = []

        for start in tqdm(range(0, n_use, batch_size), desc=sample_id):
            end = min(start + batch_size, n_use)

            batch_imgs = []
            for i in range(start, end):
                img = Image.fromarray(imgs[i].astype(np.uint8)).convert("RGB")
                batch_imgs.append(transform(img))

            x = torch.stack(batch_imgs, dim=0).to(device)

            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    emb = virchow_forward(model, x)
            else:
                emb = virchow_forward(model, x)

            all_embeddings.append(emb.detach().cpu().float().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)

    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_npz,
        sample_id=sample_id,
        embeddings=embeddings,
        barcodes=np.array(barcodes[:n_use]),
        coords=coords[:n_use],
    )

    pd.DataFrame(
        {
            "sample_id": sample_id,
            "barcode": barcodes[:n_use],
            "coord_x": coords[:n_use, 0],
            "coord_y": coords[:n_use, 1],
        }
    ).to_csv(out_csv, index=False)

    print(f"Saved embeddings: {out_npz}")
    print(f"Saved barcodes:   {out_csv}")
    print(f"Embedding shape:  {embeddings.shape}")


def main():
    parser = argparse.ArgumentParser(description="Extract Virchow embeddings from HEST H&E patches.")
    parser.add_argument("--sample-id", type=str, default=None, help="Run one sample only, e.g. INT1.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-patches", type=int, default=None, help="Use small number for CPU testing.")
    parser.add_argument("--include-xenium", action="store_true", help="Include Xenium samples like TENX105.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent

    dataset_dir = project_dir / "datasets" / "hest_ccrcc"
    sample_csv = project_dir / "datasets" / "hest_metadata" / "ccrcc_samples.csv"

    patches_dir = dataset_dir / "patches"
    out_dir = project_dir / "features" / "virchow_hest_ccrcc"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    if device == "cpu":
        print("WARNING: CUDA is not available. Virchow will run on CPU and may be slow.")

    if args.sample_id is not None:
        sample_ids = [args.sample_id]
    else:
        sample_ids = load_sample_ids(
            sample_csv=sample_csv,
            technology="Visium",
            include_xenium=args.include_xenium,
        )

    print(f"Samples to process: {sample_ids}")

    model, transform = load_virchow(device)

    for sid in sample_ids:
        embed_sample(
            sample_id=sid,
            patches_dir=patches_dir,
            out_dir=out_dir,
            model=model,
            transform=transform,
            device=device,
            batch_size=args.batch_size,
            max_patches=args.max_patches,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()