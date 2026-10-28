#!/usr/bin/env python
"""
extract_omiclip_text.py -- text-tower embeddings only, for a pairs file with new sentences.

extract_omiclip_embeddings.py runs both towers on all 73,813 spots. When only the
gene sentence changed (rebuild_sentences_degraded.py), the image side is identical, so
this script encodes just the sentences with the same frozen OmiCLIP checkpoint and the
same settings (Loki load_model, tokenizer, model.encode_text, L2 normalisation) and
writes an h5 with `st_embeddings` in the pairs-file row order.

    python scripts/extract_omiclip_text.py --pairs datasets/omiclip/ccrcc_pairs_sigmasked.csv \
        --output-h5 outputs/omiclip/ccrcc_omiclip_text_sigmasked.h5

Empty sentences are encoded as empty strings (the tokenizer then yields only the
special tokens); they are counted and reported.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--output-h5", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=PROJECT_DIR / "models" / "omiclip" / "checkpoint.pt")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-rows", type=int, default=None, help="small test subset")
    args = p.parse_args()

    import loki.utils

    pairs = pd.read_csv(args.pairs)
    if "gene_sentence" not in pairs.columns:
        raise SystemExit(f"{args.pairs} has no gene_sentence column.")
    sentences = pairs["gene_sentence"].fillna("").astype(str).tolist()
    if args.max_rows:
        sentences = sentences[: args.max_rows]
    n_empty = sum(1 for s in sentences if not s.strip())
    print(f"{len(sentences):,} sentences ({n_empty:,} empty); checkpoint {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _preprocess, tokenizer = loki.utils.load_model(str(args.checkpoint), str(device))
    model.eval()

    args.output_h5.parent.mkdir(parents=True, exist_ok=True)
    out = None
    try:
        for start in tqdm(range(0, len(sentences), args.batch_size), desc="Encoding sentences"):
            batch = sentences[start:start + args.batch_size]
            tokens = tokenizer(batch).to(device)
            with torch.no_grad():
                emb = F.normalize(model.encode_text(tokens), p=2, dim=-1).float().cpu().numpy()
            if out is None:
                out = h5py.File(args.output_h5, "w")
                out.attrs["embedding_dim"] = emb.shape[1]
                out.attrs["number_of_pairs"] = len(sentences)
                out.attrs["pairs_file"] = str(args.pairs)
                out.attrs["checkpoint"] = str(args.checkpoint.resolve())
                dset = out.create_dataset("st_embeddings", shape=(len(sentences), emb.shape[1]), dtype="float32",
                                          chunks=(min(args.batch_size, len(sentences)), emb.shape[1]), compression="lzf")
            dset[start:start + len(batch)] = emb
    finally:
        if out is not None:
            out.close()
    print(f"Wrote {args.output_h5}  ({len(sentences):,} x {emb.shape[1]})")
    print("Next: python scripts/make_omiclip_arms.py --h5 <this file> "
          "--image-h5 outputs/omiclip/ccrcc_omiclip_embeddings.h5 --prefix <name>")


if __name__ == "__main__":
    main()
