#!/usr/bin/env python
"""
extract_omiclip_ckpt.py -- both-tower embeddings for every spot, from the pretrained model or
from a fine-tuned checkpoint written by finetune_omiclip_modes.py.

    python scripts/extract_omiclip_ckpt.py --finetuned models/omiclip_ft/dual_lr1e-6/best.pt \
        --output-h5 outputs/omiclip_ft/dual_lr1e-6_embeddings.h5

Rows are written in the order of the pairs file (the same order as
outputs/omiclip/ccrcc_omiclip_embeddings.h5), so the existing metadata files and their
`embedding_row` column still apply. The h5 holds image_embeddings, st_embeddings,
matched_similarity (all L2-normalised / cosine) plus sample_id and barcode, so
make_omiclip_arms.py --rebuild-image can verify the row order.

Only the trainable tensors are stored in the fine-tuned checkpoint; they are loaded into a
freshly built pretrained model, and every key must exist in it (strict check).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from audit_omiclip_params import encode_text

PROJECT_DIR = Path(__file__).resolve().parents[1]


def prepare_patch(array):
    array = np.asarray(array)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating) and array.max() <= 1:
        array = array * 255
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).convert("RGB")


def blur_to_resolution(image, source_um_per_px, target_um_per_px):
    """Make a patch look as if it had been cut from a coarser image: shrink to the target pixel size
    (bicubic, antialiased) and stretch back to the original size with bicubic, which is what
    prepare_response_cohort.py does to the SpaceRanger crops."""
    resample = getattr(Image, "Resampling", Image).BICUBIC
    width, height = image.size
    small = max(8, int(round(width * source_um_per_px / target_um_per_px)))
    return image.resize((small, small), resample).resize((width, height), resample)


class SpotDataset(Dataset):
    def __init__(self, pairs, preprocess, blur=None):
        self.paths = pairs["patch_h5_path"].astype(str).tolist()
        self.index = pairs["patch_index"].astype(int).to_numpy()
        self.sentences = pairs["gene_sentence"].fillna("").astype(str).tolist()
        self.preprocess = preprocess
        self.blur = blur  # (source um/px, target um/px) or None
        self._handles = {}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = str(Path(self.paths[i]).resolve())
        if path not in self._handles:
            self._handles[path] = h5py.File(path, "r")
        array = self._handles[path]["img"][int(self.index[i])]
        image = prepare_patch(array)
        if self.blur:
            image = blur_to_resolution(image, *self.blur)
        return self.preprocess(image), self.sentences[i]


def load_finetuned(model, path):
    ckpt = torch.load(path, map_location="cpu")
    if "trainable_state_dict" not in ckpt:
        raise SystemExit(f"{path} has no 'trainable_state_dict'. Keys: {list(ckpt.keys())}")
    state = ckpt["trainable_state_dict"]
    own = model.state_dict()
    unknown = [k for k in state if k not in own]
    if unknown:
        raise SystemExit(f"{len(unknown)} checkpoint tensors are not in the model, e.g. {unknown[:5]}")
    shape_bad = [k for k in state if own[k].shape != state[k].shape]
    if shape_bad:
        raise SystemExit(f"Shape mismatch for {shape_bad[:5]}")
    model.load_state_dict(state, strict=False)
    print(f"Loaded {len(state)} fine-tuned tensors from {path} "
          f"(mode {ckpt.get('mode')}, lr {ckpt.get('lr')}, epoch {ckpt.get('epoch')}, val loss {ckpt.get('val_loss')})")
    return ckpt


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", type=Path, default=PROJECT_DIR / "datasets" / "omiclip" / "ccrcc_pairs.csv")
    p.add_argument("--base-checkpoint", type=Path, default=PROJECT_DIR / "models" / "omiclip" / "checkpoint.pt")
    p.add_argument("--finetuned", type=Path, default=None, help="checkpoint from finetune_omiclip_modes.py")
    p.add_argument("--output-h5", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--fp32", action="store_true", help="no autocast (the original extraction was fp32)")
    p.add_argument("--max-rows", type=int, default=None, help="small test subset")
    p.add_argument("--blur-um-per-px", type=float, default=None,
                   help="blur the patches to this pixel size before embedding (the kidney cohort images are about "
                        "3.43 um/px); used to train Stage 1 heads on images as blurry as the target cohort")
    p.add_argument("--source-um-per-px", type=float, default=0.5,
                   help="pixel size of the patches as stored (HEST: about 0.5); only used with --blur-um-per-px")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required.")
    device = torch.device("cuda")

    pairs = pd.read_csv(args.pairs)
    for col in ("sample_id", "barcode", "patch_h5_path", "patch_index", "gene_sentence"):
        if col not in pairs.columns:
            raise SystemExit(f"{args.pairs} has no '{col}' column.")
    # same filter as extract_omiclip_embeddings.py, so row i here is row i of the original h5
    pairs = pairs.dropna(subset=["patch_h5_path", "patch_index", "gene_sentence"]).reset_index(drop=True)
    if args.max_rows:
        pairs = pairs.head(args.max_rows).copy()
    print(f"{len(pairs):,} spots on {pairs['sample_id'].nunique()} slides")

    from loki.utils import load_model

    model, preprocess, tokenizer = load_model(str(args.base_checkpoint), str(device))
    if args.finetuned:
        load_finetuned(model, args.finetuned)
    model.eval()

    blur = (args.source_um_per_px, args.blur_um_per_px) if args.blur_um_per_px else None
    if blur:
        side = max(8, int(round(224 * blur[0] / blur[1])))
        print(f"Blurring patches: {blur[0]} -> {blur[1]} um/px (a 224 px patch is shrunk to {side} px and stretched back)")
    loader = DataLoader(SpotDataset(pairs, preprocess, blur), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    n = len(pairs)
    args.output_h5.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with h5py.File(args.output_h5, "w") as out:
        d_img = d_txt = d_sim = None
        cursor = 0
        for b, (images, sentences) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            tokens = tokenizer(list(sentences))
            tokens = tokens.to(device) if torch.is_tensor(tokens) else {k: v.to(device) for k, v in tokens.items()}
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=not args.fp32):
                    img = model.encode_image(images)
                    txt = encode_text(model, tokens)
                img = F.normalize(img.float(), dim=-1)
                txt = F.normalize(txt.float(), dim=-1)
                sim = (img * txt).sum(-1)
            img, txt, sim = img.cpu().numpy(), txt.cpu().numpy(), sim.cpu().numpy()
            if d_img is None:
                dim = img.shape[1]
                chunk = (min(1024, n), dim)
                d_img = out.create_dataset("image_embeddings", (n, dim), "float32", chunks=chunk, compression="lzf")
                d_txt = out.create_dataset("st_embeddings", (n, dim), "float32", chunks=chunk, compression="lzf")
                d_sim = out.create_dataset("matched_similarity", (n,), "float32")
            k = len(img)
            d_img[cursor:cursor + k] = img
            d_txt[cursor:cursor + k] = txt
            d_sim[cursor:cursor + k] = sim
            cursor += k
            if b % 100 == 0 or b == len(loader):
                rate = cursor / (time.time() - start)
                print(f"  {cursor:,}/{n:,} spots | {rate:.0f} spots/s | ETA {(n - cursor) / rate / 60:.1f} min", flush=True)
        if cursor != n:
            raise RuntimeError(f"Expected {n} rows, wrote {cursor}")
        string = h5py.string_dtype()
        out.create_dataset("sample_id", data=pairs["sample_id"].astype(str).to_numpy(dtype=object), dtype=string)
        out.create_dataset("barcode", data=pairs["barcode"].astype(str).to_numpy(dtype=object), dtype=string)
        out.attrs["pairs_file"] = str(args.pairs)
        out.attrs["blur_um_per_px"] = float(args.blur_um_per_px or 0.0)
        out.attrs["base_checkpoint"] = str(args.base_checkpoint)
        out.attrs["finetuned_checkpoint"] = str(args.finetuned) if args.finetuned else ""
    print(f"Wrote {args.output_h5}  ({n:,} x {dim}) in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
