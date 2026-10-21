# Elsa VCell: project state and handoff

Last updated: Thu 22 Oct 2026. Read this first in any new session (Claude Code, Claude chat, or a new chat with another assistant). It replaces the development journal as the source of truth; the journal is out of date in places (see section 8).

---

## 1. The question

Thesis research question (22 Oct): does combining H&E and spatial transcriptomics provide more informative prediction of immunotherapy response than either modality alone?

What the HEST ccRCC work is: Stage 1 of that question. It asks how well each encoder recovers ICI-relevant tumour biology (four ST-derived factor scores per spot) from H&E, and gives frozen embeddings to build on. It cannot answer the response question itself: the 24 slides have no treatment or outcome information (`treatment_comment` and `disease_comment` are empty). The response labels would have to come from another cohort. The journal lists GSE235672 (GBM Visium, 6 ICI-treated patients, no paired baseline) and the BCC Xenium set (6 patients); a search for larger public cohorts with H&E + ST + response has not been done yet.

Stage 1 setting: clear-cell renal cell carcinoma (ccRCC), HEST-1K Visium, 24 slides (INT1 to INT24), one patient per slide. Target: four ST-derived factor scores per spot.

### Decisions so far (22 Oct planning discussion; not final until Elsa confirms)

- Use OmiCLIP only as the main model. Freeze it and take one embedding per tower: image tower = H&E-only arm, text tower on the gene sentence = ST-only arm, both concatenated = combined arm. Same small predictor for all three. This is "late fusion" in the methodology document's terms; many papers call concatenating embeddings "intermediate/joint" fusion and averaging predictions "late", so define the terms explicitly in the thesis.
- Cheap add-ons: (a) embedding-level vs prediction-level fusion; (b) degrade the ST input (mask genes or spots) and see whether H&E helps more; (c) Virchow2 as an alternative H&E arm, to check that a weak OmiCLIP image tower is not driving the result; (d) the existing fine-tuning pilots as a short negative-result row.
- Early and intermediate fusion: discussed, not run (few labelled patients would overfit a trained fusion module; OmiCLIP's towers are separate).
- Pretrained OmiCLIP's image embeddings are weak on T-cell and cytotoxicity (test Pearson about 0 with Ridge vs 0.23 for Virchow2), so the H&E arm may add little on immune signals. State this when interpreting any "combined is not better than ST-only" result.
- Evaluation change (built 22 Oct, in `make_splits.py` and `evaluate_embeddings.py --split-csv`): replace the fixed INT1 to 18 / 19 to 21 / 22 to 24 split with (1) 6-fold patient-grouped CV with 2 frozen + 2 FFPE slides held out per fold, and (2) a balanced fixed split of 16 train / 4 val / 4 test (half frozen, half FFPE) for the slow fine-tuning runs. See section 7, item 7 for why. Results on the old split stay as "pilot, original split". The fine-tuning script has not been rewritten yet; when it is, it must read the same split file.
  - The split file in the repo was generated from the sample table as printed on 22 Oct. Re-run `make_splits.py --out outputs/splits_check.csv` on the real table and `diff` it against `datasets/splits/ccrcc_splits_v1.csv`; they should be identical.

Research questions in the methodology document:

| RQ | Question |
|---|---|
| RQ1 | Frozen vs single-tower fine-tune vs dual-tower fine-tune of OmiCLIP (all scored with an H&E-only late-fusion head) |
| RQ2 | Late vs intermediate vs early fusion |
| RQ3 | Virchow2 as the external, image-only benchmark |
| RQ4 | Optional: factor-regression objective added to the contrastive loss |

## 2. Data and split

- 73,813 matched H&E patch to ST spot pairs. The factor table has 74,220 spots; the 407 unmatched spots have no embedding. Virchow2 and OmiCLIP both have 73,813 rows (journal, section 4 and 7), so they are probably scored on the same spots. The new harness checks this with `--restrict-to`.
- Split is by slide: train INT1 to INT18 (45,981 spots), val INT19 to INT21 (14,768), test INT22 to INT24 (13,064).
- Patient overlap checked 22 Oct: `datasets/hest_metadata/ccrcc_samples.csv` has 24 distinct patients, one slide each. So there is no cross-split patient leakage, and a slide is a patient for CV purposes. The harness still verifies this (`--sample-table`).
- The metadata also has `treatment_comment`, `disease_comment`, `dataset_title` and `study_link`. These have not been inspected for treatment or outcome information yet.
- Targets: expression normalised to 10k counts, log1p, then Scanpy `score_genes` on
  - T-cell: CD3D, CD3E, CD2, CD8A, CD8B
  - Cytotoxicity: GZMB, PRF1, NKG7, GNLY
  - TGFb/CAF: TGFB1, COL1A1, COL1A2, FAP, ACTA2 (CTGF is not in the panel)
  - Proliferation: MKI67, TOP2A, PCNA, MCM2, MCM5

## 3. Representations

| Name | Dim | How |
|---|---|---|
| Virchow2 | 2,560 | CLS token + mean of patch tokens. The code uses `out[:, 1:]`, which includes the 4 register tokens; the model card uses `out[:, 5:]`. Small effect, but it should be fixed before the final run. |
| OmiCLIP (Loki, CoCa via open_clip) | 768 | Image encoder. The text input is a sentence of the top-50 expressed genes. |

## 4. Results so far (Pearson, pooled across the spots of the split)

Taken from the metric CSVs you sent on 22 Oct. Ridge always used alpha = 100 (hard-coded in all four Ridge scripts, never tuned). MLPs are 768 or 2560 to 256 to 64 to 4.

Validation (INT19 to INT21):

| Model | T-cell | Cytotox | TGFb/CAF | Prolif | Mean |
|---|---|---|---|---|---|
| Virchow2 + Ridge | 0.130 | 0.109 | 0.629 | 0.039 | 0.226 |
| Virchow2 + MLP | 0.176 | 0.114 | 0.706 | 0.104 | 0.275 |
| OmiCLIP pretrained + Ridge | 0.135 | 0.090 | 0.469 | 0.058 | 0.188 |
| OmiCLIP pretrained + MLP | 0.162 | 0.113 | 0.494 | 0.040 | 0.202 |
| OmiCLIP FT (2e-6) + Ridge | 0.135 | 0.092 | 0.481 | 0.037 | 0.186 |
| OmiCLIP FT (1e-6) + Ridge | 0.131 | 0.100 | 0.432 | 0.058 | 0.180 |

Test (INT22 to INT24):

| Model | T-cell | Cytotox | TGFb/CAF | Prolif | Mean |
|---|---|---|---|---|---|
| Virchow2 + Ridge | 0.229 | 0.227 | 0.517 | 0.161 | 0.284 |
| Virchow2 + MLP | 0.306 | 0.219 | 0.604 | 0.363 | 0.373 |
| OmiCLIP pretrained + Ridge | -0.014 | 0.005 | 0.193 | 0.167 | 0.088 |
| OmiCLIP pretrained + MLP | 0.040 | 0.051 | 0.152 | 0.099 | 0.086 |
| OmiCLIP FT (2e-6) + Ridge | 0.002 | -0.002 | 0.161 | -0.006 | 0.039 |
| OmiCLIP FT (1e-6) + Ridge | -0.046 | -0.012 | 0.164 | -0.012 | 0.023 |

What can and cannot be concluded:

- Virchow2 beats OmiCLIP on test, and the gap is large (0.28 vs 0.09 with Ridge).
- On validation the gap is small (0.23 vs 0.19). Val and test disagree a lot for the same model (OmiCLIP TGFb/CAF 0.47 val vs 0.19 test; Virchow2 MLP proliferation 0.10 val vs 0.36 test). With three slides per split, this is what a very noisy estimate looks like.
- Fine-tuning has not been shown to help. The two pilot FT models are no better than pretrained on val and slightly worse on test. But the pilots cannot settle this (section 5).
- R2 is negative for almost every model on val and test, so predictions are poorly calibrated between slides. Pearson on pooled spots mixes between-slide offsets with within-slide signal; the new harness also reports per-slide Pearson.

## 5. Fine-tuning pilots

Script: `finetune_omiclip_full.py`. Symmetric CLIP loss, batch 32, AdamW (`foreach=False`), weight decay 0.01, 5% warmup then cosine, grad clip 1.0, bf16, gradient checkpointing.

| Run | Epochs | LR | Val loss by epoch | Train loss by epoch |
|---|---|---|---|---|
| `models/omiclip_ccrcc_full` | 5 | 2e-6 | 3.389, 3.423, 3.599, 3.769, 3.815 | 0.950, 0.447, 0.350, 0.294, 0.262 |
| `models/omiclip_ccrcc_full_e2_lr1e6` | 2 | 1e-6 | 3.359, 3.438 | 0.987, 0.543 |

Why these pilots do not settle whether fine-tuning helps:

1. Validation loss rises from epoch 1 in both runs, and `best.pt` is only saved when val loss improves. So both probed checkpoints are the epoch 1 checkpoints. The "5 epoch" run was effectively evaluated after one epoch.
2. There is no epoch 0 baseline. Without the pretrained model's own val loss and retrieval, nobody knows whether epoch 1 helped or hurt.
3. Train and val loss are not comparable. The train loader is shuffled, so negatives come from many slides. The val loader is unshuffled and sorted by sample, so negatives are neighbouring spots of one slide. Val loss near ln(32) = 3.47 is therefore partly a property of the loader.
4. The contrastive loss never sees the four factors, so lower contrastive loss does not have to mean better factor prediction.
5. All parameters have `requires_grad=True`, but `encode_image` and `encode_text` never call the CoCa `text_decoder`. If the decoder gets no gradient, "full FT" was really dual-tower FT already. `audit_omiclip_params.py` measures this.

## 6. Locked protocol (decide once, do not revisit per result)

- Model and hyperparameter selection uses validation only. Test is scored once per final configuration.
- `validation_score` = mean validation Pearson over the four factors. The harness can use pooled or within-slide Pearson (`--select`).
- Score epoch 0 (pretrained) with the same probe pipeline.
- Save a checkpoint every epoch, not only on improved val loss. Choose the epoch by downstream validation Pearson, not by contrastive val loss.
- Shuffle the validation loader.
- A fusion model that needs ST at test time is a different task. It must not be ranked against H&E-only models.
- Final claims come from patient-grouped cross-validation, not from the fixed three-slide test set (section 7).

## 7. Where the plan is weak

1. **Statistical power.** Three val and three test slides, and the gaps between frozen, single and dual are of the order of 0.01 to 0.05 in the pilots. The fixed split cannot resolve differences that small. A bootstrap over three patients has about 10 distinct resamples and means nothing. The fix is patient-grouped K-fold.
2. **CV leaks for fine-tuned encoders.** An encoder fine-tuned on the gene sentences of the training patients has seen those patients' expression. For frozen encoders (Virchow2, pretrained OmiCLIP) direct CV is clean (`--cv-frozen-encoder`). For fine-tuned arms the whole pipeline (adapt, then probe) has to be rerun per fold, which is only affordable with short 1 to 2 epoch runs.
3. **Weak and fragile targets.** Only TGFb/CAF is predicted well. Check how reliable T-cell, cytotoxicity and proliferation are as targets: 4 to 5 genes each, Visium dropout, per-slide offsets.
4. **Hypothesis, not yet checked:** top-50 gene sentences are dominated by abundant genes. COL1A1/COL1A2 are abundant in stroma, which may explain why OmiCLIP does relatively better on TGFb/CAF. CD3D, GZMB and MKI67 are rarely in the top 50, which may explain its weak T-cell, cytotoxicity and proliferation results. Count signature genes in `gene_sentence` to test it.
5. **Frozen-probe caveat for adapted encoders.** The probe is fit on training spots whose embeddings were optimised against their own gene sentences, so train embeddings are more aligned than val and test ones.
6. **Fusion (RQ2) is the riskiest part.** Intermediate and early fusion need ST at inference or a different task definition. Keep it conditional or secondary.
7. **Technology confound in the split (found 22 Oct).** All 24 slides come from one study (PubMed 35231421, tertiary lymphoid structures in RCC). INT1 to INT12 are fresh-frozen (36,601 genes, 1,084 to 2,580 spots each) and INT13 to INT24 are FFPE (17,943 genes, about 3,200 to 5,000 spots each). Train = 12 frozen + 6 FFPE (INT13 to INT18); val = INT19 to INT21 and test = INT22 to INT24 are all FFPE. So every result so far is "mostly frozen training, FFPE evaluation". The two chemistries capture different genes, which changes factor-score scale and the top-50 gene sentences OmiCLIP reads. This does not explain val vs test disagreement (both FFPE). `preservation_method` is empty in the metadata; use `nb_genes` or the `subseries` name. Plan: folds that mix both technologies, FFPE and frozen reported separately, and a train-FFPE-only control.

## 8. Corrections to earlier statements

| Earlier statement | Status |
|---|---|
| "Virchow2 and OmiCLIP may not be scored on the same spots" | Probably wrong. Both have 73,813 rows. `--restrict-to` will confirm. |
| "Bootstrap over patients on the test set" | Retracted. Three patients are too few. Use patient-grouped CV. |
| Methodology Figure 3 legend: "Dual FT" vs "Full FT" | Inconsistent naming. Pick one. |
| "E0/I1/I2 equivalents" in the methodology timeline (weeks 6 to 8) | Typo. The grid is fusion type (rows: Late L, Intermediate I, Early E) by encoder adaptation (columns 0 frozen, 1 single tower, 2 dual tower), so it should read E0/E1/E2. |
| Development journal | Out of date. It says the Virchow2 side of the Virchow2 vs OmiCLIP table "may have been an older existing file" and needs a timestamp check. The Virchow2 Ridge numbers in section 4 equal the ones in the journal's first Virchow2 run, but a regenerated-and-timestamp-checked run has not been recorded. |
| Early-stopping checkpoints of the MLP pilots | Virchow2 MLP val loss is lowest at epoch 1 and OmiCLIP MLP at epoch 2 (stopped at 11 and 12). So the MLP scores above are barely trained models selected on loss. Whether Pearson-based selection changes this is untested. |

## 9. Scripts

Existing pipeline (in `scripts/`, run order roughly top to bottom):

`download_hest.py`, `inspect_hest_data.py`, `check_hest_sample.py`, `compute_hest_factor_scores.py`, `embed_hest_virchow.py`, `check_virchow_embeddings.py`, `join_virchow_factor_scores.py`, `train_hest_factor_baseline.py`, `train_virchow_factor_mlp.py`, `prepare_omiclip_pairs.py`, `check_omiclip.py`, `extract_omiclip_embeddings.py`, `join_omiclip_factor_scores.py`, `train_omiclip_factor_baseline.py`, `train_omiclip_factor_mlp.py`, `finetune_omiclip_full.py`, `extract_finetuned_omiclip_embeddings(_e2).py`, `join_finetuned_omiclip_factor_scores(_e2).py`, `train_finetuned_omiclip_factor_baseline(_e2).py`, `compare_virchow_omiclip.py`, `plot_*_factor_maps.py`.

New (this session):

| Script | Purpose |
|---|---|
| `evaluate_embeddings.py` | One evaluation harness for every embedding file. Alpha and learning-rate sweeps chosen on val, pooled and within-slide Pearson, per-slide table, patient-leakage check, row and barcode alignment checks, optional patient-grouped CV for frozen encoders. Replaces the six Ridge/MLP scripts. |
| `compare_cv.py` | Paired comparison of models from the CV outputs with a patient-level bootstrap; also reports the difference separately for frozen and FFPE slides. |
| `make_splits.py` | Writes `datasets/splits/ccrcc_splits_v1.csv`: a technology-balanced fixed split (16 train / 4 val / 4 test, half frozen and half FFPE each) and 6 CV folds (2 frozen + 2 FFPE held out per fold). Technology is read from `subseries`. Every script should read this file instead of hard-coding INT numbers. |
| `audit_omiclip_params.py` | Which OmiCLIP modules actually receive gradient. Defines the frozen, single and dual modes and asserts each one. |
| `signature_in_sentences.py` | How often each signature gene appears in the top-50 gene sentence OmiCLIP reads (hypothesis in section 7.4). Pandas only. |

## 10. Reproduce the current baselines with the new harness

Dataset files already on disk (all produced by the join scripts):

| Embedding | npz | metadata |
|---|---|---|
| Virchow2 | `features/hest_ccrcc_joined/hest_ccrcc_virchow2_factor_dataset.npz` | `features/hest_ccrcc_joined/hest_ccrcc_virchow2_factor_metadata.csv` |
| OmiCLIP pretrained | `outputs/omiclip_factor_baseline/omiclip_factor_dataset.npz` | `outputs/omiclip_factor_baseline/omiclip_factor_metadata.csv` |
| OmiCLIP FT 2e-6 | `outputs/omiclip_finetuned_factor_baseline/omiclip_finetuned_factor_dataset.npz` | same folder, `omiclip_finetuned_factor_metadata.csv` |
| OmiCLIP FT 1e-6 | `outputs/omiclip_finetuned_e2_lr1e6_factor_baseline/omiclip_finetuned_factor_dataset.npz` | same folder, `omiclip_finetuned_factor_metadata.csv` |

```bash
cd /media/rokny/DATA2/Elsa/vcell
python scripts/make_splits.py                                                      # writes datasets/splits/ccrcc_splits_v1.csv (a copy is already provided)
python scripts/audit_omiclip_params.py | tee outputs/audit_omiclip_params.txt      # omiclip env, GPU, about 10 min
python scripts/signature_in_sentences.py \
  --factor-csv features/hest_factor_scores/all_hest_ccrcc_factor_scores.csv      # any env, under a minute

python scripts/evaluate_embeddings.py --name virchow2 \
  --npz features/hest_ccrcc_joined/hest_ccrcc_virchow2_factor_dataset.npz \
  --meta features/hest_ccrcc_joined/hest_ccrcc_virchow2_factor_metadata.csv \
  --sample-table datasets/hest_metadata/ccrcc_samples.csv \
  --split-csv datasets/splits/ccrcc_splits_v1.csv --no-mlp

python scripts/evaluate_embeddings.py --name omiclip_pretrained \
  --npz outputs/omiclip_factor_baseline/omiclip_factor_dataset.npz \
  --meta outputs/omiclip_factor_baseline/omiclip_factor_metadata.csv \
  --restrict-to features/hest_ccrcc_joined/hest_ccrcc_virchow2_factor_metadata.csv \
  --sample-table datasets/hest_metadata/ccrcc_samples.csv \
  --split-csv datasets/splits/ccrcc_splits_v1.csv --no-mlp
```

Then add `--skip-standard --cv-frozen-encoder 6` to each, and `python scripts/compare_cv.py --reference virchow2 --models virchow2 omiclip_pretrained`.

Note: these scripts were written without access to your real data, so they were tested on synthetic arrays only (Ridge, MLP, CV, leakage checks and the audit on a toy CoCa-like model). The audit's real-model parts (Loki loading, reading patches from `datasets/hest_ccrcc/patches/*.h5`, the tokenizer call) mirror `finetune_omiclip_full.py` but have not run against the real checkpoint. If any of them errors on your machine, send me the traceback.

## 11. To do

- [ ] Run the parameter audit; send the printout back.
- [ ] Run `evaluate_embeddings.py` for Virchow2 and pretrained OmiCLIP; check same-spot overlap (patient leakage already ruled out: 24 patients, one slide each).
- [ ] Print `treatment_comment`, `disease_comment`, `dataset_title`, `study_link` from the sample table and look for treatment or outcome information.
- [ ] Run the signature-gene count in gene sentences (hypothesis in section 7.4).
- [ ] Fix the Virchow2 register-token slice (`out[:, 5:]`) and re-embed, or state the choice in the thesis.
- [ ] Write the frozen/single/dual fine-tuning script (per-epoch checkpoints, shuffled val loader, epoch 0 evaluation, LR as a CLI argument), using the module names the audit prints.
- [ ] Six-run pilot: LR {5e-7, 1e-6, 2e-6} x {single, dual}; select by downstream val Pearson.
- [ ] Patient-grouped CV for the final comparison (frozen arms directly; adapted arms with the whole pipeline per fold).
- [ ] Decide whether RQ2 (fusion) stays as a main result or becomes secondary.
- [ ] Check that the GitHub push of `hest-ccrcc` went through (`git log origin/hest-ccrcc`). Never commit `.hf_home/`, `venv/`, data, or model weights; never paste tokens into chats or files.

## 12. Environment

- Workstation (TOWER3): 2x RTX 6000 Ada, about 96 GB RAM. Project in `/media/rokny/DATA2/Elsa/vcell`, `HF_HOME` set to `.hf_home` inside it.
- Virchow2 runs in the project venv; OmiCLIP/Loki in conda env `omiclip` (Python 3.9). Do not mix them.
- Repo: `ed21211/vcell`, branch `hest-ccrcc`. The workstation SSH key belongs to another account (`jade0530`); pushes use HTTPS with a fine-grained token scoped to this repo, and git identity is set per repo only.