#!/usr/bin/env python
"""
signature_in_sentences.py -- do the four signatures even appear in the text OmiCLIP sees?

OmiCLIP's text tower reads a "gene sentence": the top-50 expressed genes of a
spot, ranked. Abundant genes (collagens, ribosomal, MT genes) make that list;
rarely-detected immune genes usually do not. If CD3D, GZMB or MKI67 almost never
appear in the sentence, the text side cannot teach the image side anything about
T cells, cytotoxicity or proliferation, whatever the fine-tuning does.

Reports, per signature gene and per factor:
  * share of spots whose sentence contains the gene
  * mean position (rank) when present
  * share of spots with at least one signature gene in the sentence
  * optionally (--factor-csv): the same share among the top 10% of spots by
    factor score vs all other spots

    python scripts/signature_in_sentences.py
    python scripts/signature_in_sentences.py --factor-csv features/hest_factor_scores/all_hest_ccrcc_factor_scores.csv

Pure pandas, runs in any env. Writes outputs/audit/signature_in_sentences.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]

SIGNATURES = {
    "t_cell_infiltration": ["CD3D", "CD3E", "CD2", "CD8A", "CD8B"],
    "cytotoxicity": ["GZMB", "PRF1", "NKG7", "GNLY"],
    "tgfb_caf": ["TGFB1", "COL1A1", "COL1A2", "FAP", "ACTA2"],  # CTGF is not in the panel
    "proliferation": ["MKI67", "TOP2A", "PCNA", "MCM2", "MCM5"],
}


def tokenise(sentence):
    return str(sentence).replace(",", " ").upper().split()


def norm_barcode(s):
    s = s.astype(str).str.strip().str.upper()
    return s.str.replace(r"-1$", "", regex=True)


def gene_table(tokens):
    """tokens: list of token lists. Returns DataFrame with presence rate and mean rank per gene."""
    rows = []
    for factor, genes in SIGNATURES.items():
        for gene in genes:
            ranks = [t.index(gene) + 1 for t in tokens if gene in t]
            rows.append(
                dict(
                    factor=factor,
                    gene=gene,
                    share_of_spots=len(ranks) / len(tokens),
                    mean_rank_when_present=float(np.mean(ranks)) if ranks else np.nan,
                )
            )
    return pd.DataFrame(rows)


def factor_table(tokens):
    sets = [set(t) for t in tokens]
    rows = []
    counts = {}
    for factor, genes in SIGNATURES.items():
        n_in = np.array([len(s.intersection(genes)) for s in sets])
        counts[factor] = n_in
        rows.append(
            dict(
                factor=factor,
                n_genes=len(genes),
                share_with_any=float((n_in > 0).mean()),
                mean_genes_present=float(n_in.mean()),
            )
        )
    return pd.DataFrame(rows), counts


def enriched_table(counts, scores, top_q=0.9):
    """share with >=1 signature gene in top decile of the factor vs the rest."""
    rows = []
    for factor, n_in in counts.items():
        s = scores[factor].to_numpy()
        ok = ~np.isnan(s)
        cut = np.quantile(s[ok], top_q)
        top = ok & (s >= cut)
        rest = ok & (s < cut)
        rows.append(
            dict(
                factor=factor,
                share_any_top10pct=float((n_in[top] > 0).mean()),
                share_any_rest=float((n_in[rest] > 0).mean()),
                n_top=int(top.sum()),
            )
        )
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", type=Path, default=PROJECT_DIR / "datasets" / "omiclip" / "ccrcc_pairs.csv")
    p.add_argument("--factor-csv", type=Path, default=None)
    p.add_argument("--out", type=Path, default=PROJECT_DIR / "outputs" / "audit" / "signature_in_sentences.csv")
    args = p.parse_args()

    df = pd.read_csv(args.pairs)
    if "gene_sentence" not in df.columns:
        raise SystemExit(f"{args.pairs} has no gene_sentence column. Columns: {df.columns.tolist()}")
    tokens = [tokenise(s) for s in df["gene_sentence"]]
    lengths = np.array([len(t) for t in tokens])
    print(f"{len(df):,} spots; sentence length min/median/max = {lengths.min()}/{int(np.median(lengths))}/{lengths.max()}")

    genes = gene_table(tokens)
    print("\nPer signature gene (share of spots whose sentence contains it):")
    print(genes.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    facs, counts = factor_table(tokens)
    print("\nPer factor:")
    print(facs.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    out = genes.merge(facs, on="factor")
    if args.factor_csv:
        sc = pd.read_csv(args.factor_csv)
        sc["_s"] = sc["sample_id"].astype(str).str.strip().str.upper()
        sc["_b"] = norm_barcode(sc["barcode"])
        df = df.copy()
        df["_s"] = df["sample_id"].astype(str).str.strip().str.upper()
        df["_b"] = norm_barcode(df["barcode"])
        merged = df[["_s", "_b"]].merge(sc, on=["_s", "_b"], how="left")
        missing = [f for f in SIGNATURES if f not in merged.columns]
        if missing:
            raise SystemExit(f"{args.factor_csv} lacks factor columns {missing}. Columns: {sc.columns.tolist()}")
        enr = enriched_table(counts, merged[list(SIGNATURES)])
        print("\nShare of spots with at least one signature gene in the sentence,")
        print("top 10% of spots by factor score vs all other spots:")
        print(enr.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        out = out.merge(enr, on="factor")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()