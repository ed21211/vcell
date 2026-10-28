#!/usr/bin/env python
"""
rebuild_sentences_degraded.py -- rebuild the gene sentences with part of the ST
information removed, keeping every spot and the row order of the original pairs file.

Why: the four factor scores are computed from the very genes the sentence lists, so
an ST arm partly "reads the answer". Two controls:

  --exclude-signature   remove the 19 signature genes (T-cell, cytotoxicity, TGFb/CAF,
                        proliferation) from every spot BEFORE taking the top 50, so the
                        sentence is the next 50 most abundant genes and the ST arm can no
                        longer see the target genes at all
  --drop-frac F         randomly remove a fraction F of each spot's expressed genes
                        (seeded) before taking the top 50, to mimic sparser / lower-depth ST

Both can be combined. The sentence rule is the original one from
prepare_omiclip_pairs.py (top genes by raw count, positives only), imported from it.

It reads the ORIGINAL pairs file (datasets/omiclip/ccrcc_pairs.csv), so spot order,
sample ids and barcodes are identical; only `gene_sentence` (and `gene_count`) change.
A spot whose new sentence would be empty keeps its row with an empty sentence (counted
and reported) so the row numbers still match the existing embedding files.

    python scripts/rebuild_sentences_degraded.py --exclude-signature --tag sigmasked
    python scripts/rebuild_sentences_degraded.py --drop-frac 0.5 --tag drop50
    python scripts/rebuild_sentences_degraded.py --drop-frac 0.9 --tag drop90

Writes datasets/omiclip/ccrcc_pairs_<tag>.csv. Needs anndata (the omiclip env has it).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from prepare_omiclip_pairs import get_expression_matrix, get_gene_names, make_gene_sentence

PROJECT_DIR = Path(__file__).resolve().parents[1]

SIGNATURE_GENES = {
    "t_cell_infiltration": ["CD3D", "CD3E", "CD2", "CD8A", "CD8B"],
    "cytotoxicity": ["GZMB", "PRF1", "NKG7", "GNLY"],
    "tgfb_caf": ["TGFB1", "COL1A1", "COL1A2", "FAP", "ACTA2"],
    "proliferation": ["MKI67", "TOP2A", "PCNA", "MCM2", "MCM5"],
}
ALL_SIGNATURE = sorted({g for genes in SIGNATURE_GENES.values() for g in genes})


def degrade(matrix, gene_names, exclude, drop_frac, rng):
    """Return a CSR copy of `matrix` with the chosen genes zeroed and a fraction of the
    remaining non-zero entries removed at random."""
    m = sparse.csr_matrix(matrix, dtype=np.float32, copy=True)
    removed_genes = 0
    if exclude:
        mask = np.isin(gene_names, list(exclude))
        removed_genes = int(mask.sum())
        keep = sparse.diags((~mask).astype(np.float32))
        m = (m @ keep).tocsr()
    if drop_frac > 0:
        drop = rng.rand(m.data.shape[0]) < drop_frac
        m.data[drop] = 0.0
    m.eliminate_zeros()
    return m, removed_genes


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", type=Path, default=PROJECT_DIR / "datasets" / "omiclip" / "ccrcc_pairs.csv")
    p.add_argument("--exclude-signature", action="store_true")
    p.add_argument("--drop-frac", type=float, default=0.0)
    p.add_argument("--top-genes", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", required=True, help="output is ccrcc_pairs_<tag>.csv")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    if not args.exclude_signature and args.drop_frac <= 0:
        raise SystemExit("Nothing to do: pass --exclude-signature and/or --drop-frac F (0 < F < 1).")
    if not 0 <= args.drop_frac < 1:
        raise SystemExit("--drop-frac must be in [0, 1).")

    pairs = pd.read_csv(args.pairs)
    for col in ("st_h5ad_path", "st_row_index", "gene_sentence", "sample_id"):
        if col not in pairs.columns:
            raise SystemExit(f"{args.pairs} has no '{col}' column. Columns: {pairs.columns.tolist()}")
    exclude = set(ALL_SIGNATURE) if args.exclude_signature else set()
    print(f"{len(pairs):,} spots, {pairs['sample_id'].nunique()} slides; "
          f"exclude {len(exclude)} signature genes: {args.exclude_signature}; drop fraction {args.drop_frac}")

    new_sentences = np.full(len(pairs), "", dtype=object)
    new_counts = np.zeros(len(pairs), dtype=int)
    sig_hits = 0
    for n, (path, group) in enumerate(pairs.groupby("st_h5ad_path", sort=False)):
        adata = ad.read_h5ad(path)
        names = get_gene_names(adata)
        matrix = get_expression_matrix(adata)
        rng = np.random.RandomState(args.seed + n)
        m, n_removed = degrade(matrix, names, exclude, args.drop_frac, rng)
        if exclude:
            present = sorted(set(names) & set(ALL_SIGNATURE))
            print(f"  {group['sample_id'].iloc[0]}: {len(present)} of {len(ALL_SIGNATURE)} signature genes on this panel, "
                  f"{n_removed} columns removed")
        for idx, row_index in zip(group.index, group["st_row_index"].astype(int)):
            sentence, count = make_gene_sentence(m, row_index, names, args.top_genes)
            new_sentences[idx] = sentence
            new_counts[idx] = count
            if exclude and sentence and (set(sentence.split()) & exclude):
                sig_hits += 1

    out = pairs.copy()
    out["gene_sentence"] = new_sentences
    out["gene_count"] = new_counts
    empty = int((out["gene_sentence"] == "").sum())
    if exclude:
        print(f"Sanity check: spots whose new sentence still contains a signature gene: {sig_hits} (must be 0)")
        if sig_hits:
            raise SystemExit("Signature genes survived the exclusion; not writing the file.")
    print(f"Empty new sentences kept as empty rows: {empty:,} ({empty / len(out):.2%})")
    print(f"Sentence length: min {out['gene_count'].min()}, median {int(out['gene_count'].median())}, "
          f"max {out['gene_count'].max()}")
    same = (out["gene_sentence"] == pairs["gene_sentence"]).mean()
    print(f"Share of spots whose sentence is unchanged from the original: {same:.2%}")

    path = args.out or (args.pairs.parent / f"ccrcc_pairs_{args.tag}.csv")
    out.to_csv(path, index=False)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
