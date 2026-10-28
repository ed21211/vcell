# Elsa VCell: project state and handoff

Last updated: Thu 22 Oct 2026. Read this first in any new session (Claude Code, Claude chat, or a new chat with another assistant). It replaces the development journal as the source of truth; the journal is out of date in places (see section 8).

---

## 1. The question

Thesis research question (22 Oct): does combining H&E and spatial transcriptomics provide more informative prediction of immunotherapy response than either modality alone?

What the HEST ccRCC work is: Stage 1 of that question. It asks how well each encoder recovers ICI-relevant tumour biology (four ST-derived factor scores per spot) from H&E, and gives frozen embeddings to build on. It cannot answer the response question itself: the 24 slides have no treatment or outcome information (`treatment_comment` and `disease_comment` are empty). The response labels would have to come from another cohort. The journal lists GSE235672 (GBM Visium, 6 ICI-treated patients, no paired baseline) and the BCC Xenium set (6 patients); a web search for public cohorts with H&E + ST + response was done 22 Oct (section 1b). It found only very small ones (6 to 13 patients), none large enough to train a fusion model.

Stage 1 setting: clear-cell renal cell carcinoma (ccRCC), HEST-1K Visium, 24 slides (INT1 to INT24), one patient per slide. Target: four ST-derived factor scores per spot.

### Decisions so far (22 Oct planning discussion; not final until Elsa confirms)

- Use OmiCLIP only as the main model. Freeze it and take one embedding per tower: image tower = H&E-only arm, text tower on the gene sentence = ST-only arm, both concatenated = combined arm. Same small predictor for all three. This is "late fusion" in the methodology document's terms; many papers call concatenating embeddings "intermediate/joint" fusion and averaging predictions "late", so define the terms explicitly in the thesis.
- Cheap add-ons: (a) embedding-level vs prediction-level fusion; (b) degrade the ST input (mask genes or spots) and see whether H&E helps more; (c) Virchow2 as an alternative H&E arm, to check that a weak OmiCLIP image tower is not driving the result; (d) the existing fine-tuning pilots as a short negative-result row.
- Response stage (Elsa, 22 Oct): a small public cohort is acceptable; no supervisor check needed. The thesis is Stage 1 on HEST ccRCC plus an exploratory response test on the best small cohort (section 1b: JITC mRCC first, HCC second), reported as exploratory with exact permutation p-values and no claim of a trained fusion model.
- Early and intermediate fusion: discussed, not run (few labelled patients would overfit a trained fusion module; OmiCLIP's towers are separate).
- Pretrained OmiCLIP's image embeddings carry about half of Virchow2's signal on every factor (balanced CV, section 4b, primary within-slide selection: per-slide mean 0.118 vs 0.235; T-cell 0.123 vs 0.228; cytotoxicity 0.074 vs 0.153). The H&E arm is therefore weak overall, not only on immune signals. State this when interpreting any "combined is not better than ST-only" result. Open question: should the multimodal comparison use OmiCLIP's image tower or a stronger H&E encoder? Also (section 4c): the native OmiCLIP text tower is a viable ST representation (it predicts T-cell and cytotoxicity as well as or better than Virchow2), but in Stage 1 that is circular, and adding H&E to it changed nothing measurable.
- Evaluation change (built 22 Oct, in `make_splits.py` and `evaluate_embeddings.py --split-csv`): replace the fixed INT1 to 18 / 19 to 21 / 22 to 24 split with (1) 6-fold patient-grouped CV with 2 frozen + 2 FFPE slides held out per fold, and (2) a balanced fixed split of 16 train / 4 val / 4 test (half frozen, half FFPE) for the slow fine-tuning runs. See section 7, item 7 for why. Results on the old split stay as "pilot, original split". The fine-tuning script has not been rewritten yet; when it is, it must read the same split file.
  - The split file in the repo was generated from the sample table as printed on 22 Oct. VERIFIED 22 Oct on the workstation: `make_splits.py --out outputs/splits_check.csv` on the real table reproduces all 24 rows exactly (the only byte difference is a trailing newline). Same seed gives the same split on both machines.

### 1b. Candidate response cohorts (web search 22 Oct; fields marked "stated" come from the paper or record text, nothing was downloaded)

| Cohort | Patients with ST | Response labels | Platform | Access | Caveats |
|---|---|---|---|---|---|
| JITC 2025;13:e012991, metastatic RCC (10 of 12 clear cell), first-line immunotherapy (8 nivo+ipi) | 18 Visium datasets (FFPE, 30,269 spots in total, about 1,700 per sample) = most likely 12 primary tumours (one per patient) + 6 paired therapy-naive metastases (paper: "primary tumors n=12, metastatic lesions n=6", "18 Visium"; CONFIRMED 29 Oct from the 18 file names in Visium_annotations.zip: primaries W3 W10 W31 W33 W34 W35 W37 W38 W44 W47 W50 W55 (12 patients) and metastases W3 W10 W33 W38 W47 W55 (6); annotation CSVs are 50 to 150 KB each, one per sample, named RCC_<patient>_<PRIM|MET>_classes.csv). Metastasis responders W3 W33 W38, non-responders W10 W47 W55. CosMx on 16 datasets. Per-patient response is in Table 1 / Fig 1 of the paper (not captured). The authors found no CosMx niche predictive in primaries, only in metastases | 12-patient cohort: 1 CR, 5 PR, 1 SD, 5 PD at 3 months, plus RECIST % change | Visium + CosMx FFPE, pathologist-annotated H&E | Stated: raw Visium and CosMx deposited on Zenodo doi 10.5281/zenodo.16833780 (with metadata tables, Seurat objects, scripts). Data availability statement read in full by Elsa 29 Oct and the DOI is exactly 10.5281/zenodo.16833780: "Raw spatial transcriptomic datasets (Visium and COSMx) ... Zenodo ... together with metadata tables detailing patient characteristics and pathology annotations", Seurat objects for CosMx only, scripts. The DOI link and the plain record URL do not open because the record is an UNPUBLISHED DRAFT (checked 29 Oct: the link behind reference 48 in the paper opens a Zenodo "Preview" page with the banner "new record that has not yet been published", and the page says the DOI is registered only on publishing). The preview lists: CC BY 4.0, 35.8 GB total, Visium_Objects.zip 446.6 MB ("all raw Visium data"), Visium_annotations.zip 258 KB (pathology annotations), 4 CosMx runs of 7.3 to 7.9 GB, COSMx_Seurat_Objects.zip 5.5 GB, Supplementary_code1.R. Contact: Charis Kalogirou, kalogirou_c@ukw.de. No per-patient response file is listed (labels are in the paper's Table 1 / Fig 1). The paper says Visium FFPE was H&E stained and imaged (NanoZoomer S210) and processed with SpaceRanger 3.1, so the spatial image folder may be in the "raw" deposit. Whether Visium_Objects.zip contains the H&E image folder (SpaceRanger "spatial/") is UNVERIFIED; next step is to download only Visium_Objects.zip and Visium_annotations.zip (not the 30 GB CosMx files) and list the zip contents. Ask the author to publish the record so the thesis can cite a working DOI | Same disease as Stage 1. Mostly metastatic sites (adrenal, brain, liver, nodes, peritoneum, lung), so a tissue shift from the HEST primary tumours. 3 vs 3 responders in the paired Visium subset |
| HCC perioperative nivolumab (+/- ipilimumab) phase II, bioRxiv 2025.06.11.656869 (preprint) | 13 patients, 22 Visium samples after QC (11 pre, 11 post) | 6 responders, 7 non-responders (major pathological response) | Visium + CODEX; H&E on sequential sections (stated in paper) | Zenodo 10.5281/zenodo.15392699 (v0.1, CC BY 4.0, open): Visium-ST.zip 1.2 GB (AnnData, raw counts, tissue positions), CODEX 864 MB, metadata.csv with response, pre/post, treatment | Different cancer. Preprint. CONFIRMED 28 Oct from the record description: NO H&E or tissue images in the Zenodo files (Visium-ST.zip holds visium_all.h5ad plus tissue_positions CSVs in the CODEX coordinate frame). Usable for an ST-only response test, not for H&E vs ST vs combined. Pre-treatment samples are small biopsies |
| HNSCC ICB, bioRxiv 2025.03.24.644582 | Visium 8 patients (4 R, 4 NR); CosMx 24 patients (11 R, 13 NR) | R vs NR | Visium + CosMx | No data statement in fetched text | H&E not mentioned; access unknown |
| NSCLC anti-PD-1/PD-L1, PMC11297140 | 18 patients (7 R, 11 NR) | RECIST, 180-day threshold | GeoMx DSP (ROIs, not spots), TMA cores | K-BDS database; request for anonymised data | No paired H&E mentioned; GeoMx is not spot-level |
| Stanford cohorts in arXiv 2604.03630 (melanoma, lung, kidney, others) | n/a | ICI response | n/a | Not public; data use agreement | Not usable without an agreement |
| Dryad doi 10.5061/dryad.dv41ns28r (RCC nivolumab phase II, 102 patients) | none (bulk and single-cell) | not confirmed in files | not spatial, no H&E | open | Not spatial; response labels unconfirmed |

Reading: no public cohort found is big enough to train a fusion model. With 6 patients split 3 vs 3 the smallest possible exact permutation p is 0.10 (two-sided, C(6,3)=20); with 13 patients split 6 vs 7 it is about 0.001 (C(13,6)=1716). Any response-stage result would be exploratory: frozen encoders, a fixed Stage 1 recipe, patient-level scores, no training on the response labels beyond a one-parameter or logistic readout. Leading candidate for the same disease: JITC e012991 (confirm the Zenodo files first). The HCC Zenodo set has the larger n but no H&E (fallback for an ST-only test). GSE235672 (GBM Visium, public, GEO RAW tar 470 MB with JPG/PNG/JSON files, 20 samples named Pt1 to Pt18; per-sample response labels are not on the GEO page, so they must come from the paper) is the other candidate for all three pieces; CHECKED 29 Oct and DROPPED: its tar listing shows the same SpaceRanger layout (tissue_hires_image.png, scalefactors, tissue_positions) so the H&E is no sharper than the kidney cohort's, and it is a different cancer. The kidney Zenodo listing was also checked 29 Oct (18 Visium slides = 12 primaries + 6 metastases, plus 3 lymph-node/other objects in the zip; see section 11). GSE291246 (BCC Xenium, 106.7 GB, no labels on the GEO page) is dropped. Not searched: GEO accessions for the Visium/CosMx cohorts above, clinical-trial tissue banks, controlled-access repositories.

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

## 4b. Balanced-split results with the new harness (22 Oct; use these, not section 4)

Setup: `ccrcc_splits_v1.csv`, frozen encoders, Ridge with alpha chosen on validation (pooled-Pearson selection, the harness default) and a grid extended to 1e8. Both models scored on the same 73,813 spots (`--restrict-to`: 0 spots absent; targets identical to 1.2e-7). The numpy "invalid value encountered in matmul" warning at `evaluate_embeddings.py:206` is spurious (inputs and Gram matrix are finite; it reproduces outside the harness).

**6-fold patient-grouped CV, 24 slides, mean per-slide Pearson (out-of-fold; paired patient bootstrap, 10,000 draws; alpha chosen on POOLED Pearson, the harness default; the primary within-slide-selection table follows below):**

| Factor | Virchow2 | OmiCLIP pretrained (image tower) | Diff (Omi - Vir) | 95% CI |
|---|---|---|---|---|
| T-cell | 0.228 | 0.127 | -0.101 | -0.134 to -0.068 |
| Cytotoxicity | 0.153 | 0.085 | -0.068 | -0.106 to -0.032 |
| TGFb/CAF | 0.480 | 0.230 | -0.250 | -0.319 to -0.187 |
| Proliferation | 0.081 | 0.044 | -0.037 | -0.063 to -0.010 |
| Mean of factors | 0.235 | 0.121 | -0.114 | -0.144 to -0.086 |

OmiCLIP beats Virchow2 on 0 of 24 slides for the mean of factors. By technology (mean of factors): FFPE Virchow2 0.314 vs OmiCLIP 0.171 (diff -0.142); frozen 0.157 vs 0.072 (diff -0.086). Pooled out-of-fold r (mean of factors): Virchow2 0.295, OmiCLIP 0.170. Per-slide sd 0.09 to 0.20.

**Fixed split (train 16 / val 4 / test 4; test = INT9, 11, 13, 18):** mean pooled Pearson on test: Virchow2 0.324 (within-slide 0.226), OmiCLIP 0.211 (within-slide 0.134). Four test slides only; use the CV.

**Alpha grid:** Virchow2's best alpha is 1e5 and is now an interior optimum (1e6 gives 0.397 vs 0.408 on val), so the earlier grid edge did not matter. OmiCLIP's pooled validation score peaks at 1e6, but its within-slide score falls from 0.109 (alpha <= 1e4) to 0.076 (1e6): pooled selection favours slide-level offsets. The reported CV metric is per-slide, so the primary protocol selects alpha on within-slide Pearson (`--select within`; rule fixed before seeing the result, both reported).

**PRIMARY: 6-fold CV, alpha chosen on within-slide Pearson (`virchow2_within`, `omiclip_pretrained_within`, same grid to 1e8):**

| Factor | Virchow2 | OmiCLIP pretrained | Diff (Omi - Vir) | 95% CI |
|---|---|---|---|---|
| T-cell | 0.228 | 0.123 | -0.105 | -0.135 to -0.074 |
| Cytotoxicity | 0.153 | 0.074 | -0.079 | -0.112 to -0.046 |
| TGFb/CAF | 0.480 | 0.226 | -0.254 | -0.327 to -0.188 |
| Proliferation | 0.081 | 0.048 | -0.032 | -0.067 to +0.001 (includes zero) |
| Mean of factors | 0.235 | 0.118 | -0.118 | -0.146 to -0.091 |

OmiCLIP better on 0 of 24 slides (mean of factors). By technology (mean of factors): FFPE Virchow2 0.314 vs OmiCLIP 0.174 (diff -0.140, CI -0.180 to -0.102); frozen 0.157 vs 0.062 (diff -0.095, CI -0.137 to -0.065). Pooled out-of-fold r (mean of factors): Virchow2 0.295, OmiCLIP 0.170. Fixed split (within selection): OmiCLIP test within-slide 0.120 vs Virchow2 0.226. The selection rule does not matter: OmiCLIP's mean was 0.121 with pooled selection and 0.118 with within-slide selection (I had predicted a small gain; there was none), and the gap is -0.115, -0.114 and -0.118 under default alphas, wide grid with pooled selection, and wide grid with within-slide selection. Virchow2 is identical under all three. Proliferation is near the floor for both encoders (0.081, 0.048), so that one difference is not resolved.

What these results say:
- Virchow2 is the stronger H&E encoder on every factor, and OmiCLIP gets roughly half of Virchow2's per-slide correlation on every factor (ratios 0.47 to 0.59 across the four factors under either selection rule). The gap is uniform, so the sentence-content finding (section 7, item 4) does not explain it, including TGFb/CAF where the genes are in the sentence. It is a weaker image encoder on this task.
- The old pilot values (section 4; e.g. OmiCLIP T-cell about 0 on test) came from an FFPE-only test set, a fixed alpha, and 3 slides. Do not quote them.
- Frozen slides are harder than FFPE for both encoders (Virchow2 0.157 vs 0.314; OmiCLIP 0.072 vs 0.171). Causes untested (FFPE-heavy pretraining of foundation models; frozen slides have fewer spots, so noisier per-slide r).
- MLP heads on the balanced split (fixed split, within-slide selection, 3 lrs x 3 seeds, early stopping on val within-slide Pearson; `virchow2_mlp`, `omiclip_pretrained_mlp`): mean within-slide Pearson, Ridge vs MLP. Virchow2: val 0.242 vs 0.235, test 0.226 vs 0.235. OmiCLIP: val 0.109 vs 0.136, test 0.120 vs 0.099. Pooled test: Virchow2 0.324 vs 0.319; OmiCLIP 0.185 vs 0.144. The MLP does not beat a tuned Ridge reliably (gains on val, losses on test, all within noise for 4 slides), so Ridge stays the head for the main comparison and the MLP is a robustness row. The pilots' large Virchow2 MLP gain (test 0.284 to 0.373) was most likely the MLP regularising better than an untuned Ridge (alpha 100): on the balanced val set Ridge at alpha 100 scores within-slide 0.185, the MLP 0.235 and Ridge at its tuned alpha 0.242. The Virchow2 vs OmiCLIP gap persists with the MLP (test within-slide 0.235 vs 0.099). MLP learning-rate choice is noise (mean val 0.134, 0.133, 0.136 across the three lrs for OmiCLIP).

## 4c. The three OmiCLIP arms on the native sentences (22 Oct)

Arms: H&E-only = image-tower embedding; ST-only = text-tower embedding of the native top-50 gene sentence (the h5 from `extract_omiclip_embeddings.py`, via `make_omiclip_arms.py`); combined = the two concatenated (1536-d). Same predictor (Ridge, alpha chosen on within-slide Pearson), same 6-fold patient-grouped CV, same 24 slides. Mean per-slide Pearson:

| Factor | H&E-only (OmiCLIP image) | ST-only | Combined | Virchow2 (H&E) |
|---|---|---|---|---|
| T-cell | 0.123 | 0.271 | 0.257 | 0.228 |
| Cytotoxicity | 0.074 | 0.194 | 0.176 | 0.153 |
| TGFb/CAF | 0.226 | 0.540 | 0.564 | 0.480 |
| Proliferation | 0.048 | 0.055 | 0.072 | 0.081 |
| Mean of factors | 0.118 | 0.265 | 0.267 | 0.235 |

Paired differences vs H&E-only (patient bootstrap): ST-only +0.147 (CI +0.111 to +0.188; better on 23 of 24 slides); combined +0.150 (CI +0.123 to +0.179; 24 of 24); Virchow2 +0.118 (CI +0.091 to +0.146). Proliferation: ST-only +0.006 (CI -0.028 to +0.040), combined +0.024 (CI +0.001 to +0.049). By technology the gain is the same in both chemistries (ST-only +0.145 FFPE, +0.149 frozen; combined +0.152, +0.147), whereas Virchow2's gain over the image tower is +0.140 FFPE and +0.095 frozen.

Predictions written before the run: (1) ST-only strong on TGFb/CAF, per-slide r above 0.4: CORRECT (0.540). (2) ST-only not clearly better than the image tower on T-cell, cytotoxicity, proliferation: WRONG for T-cell (0.271 vs 0.123) and cytotoxicity (0.194 vs 0.074), where ST-only is also above Virchow2 in point estimate (the paired difference vs Virchow2 includes zero for both); correct for proliferation. (3) Combined at least as good as image-only on TGFb/CAF: CORRECT (0.564).

What this says:
- The native text tower carries T-cell and cytotoxicity signal even though CD3D, GZMB and the other signature genes almost never appear in the sentences (section 7, item 4). The signal must come through other genes in the sentence (composition). It is not explained by correlation between the factors: pooled across spots they are nearly independent (T-cell vs cytotoxicity 0.31; T-cell vs TGFb/CAF 0.07; cytotoxicity vs TGFb/CAF 0.04; proliferation vs the others 0.04 to 0.08), so the T-cell signal is not riding on the stroma score. Still unchecked: which sentence genes separate high from low T-cell spots. The earlier worry that the OmiCLIP text side is blind to immune biology is therefore not supported, and building enriched sentences is lower priority.
- Paired tests with ST-only as the reference: combined +0.002 on the mean (CI -0.012 to +0.016). By factor, TGFb/CAF +0.024 (CI +0.003 to +0.052) and proliferation +0.017 (CI +0.003 to +0.033) are the only intervals that exclude zero; T-cell -0.014 and cytotoxicity -0.018 include zero. So adding H&E to ST adds nothing measurable on the mean. Virchow2 vs ST-only: mean -0.029 (CI -0.057 to -0.002); FFPE -0.005 (CI -0.033 to +0.030), frozen -0.054 (CI -0.095 to -0.015): on FFPE slides a strong H&E encoder matches the ST text embedding, on frozen slides ST is better. OmiCLIP image tower vs ST-only: -0.147 (CI -0.188 to -0.111), ST better on 23 of 24 slides.
- Circularity (section 7, item 8) still applies in full: the targets are computed from the same spots' expression, so ST-only beating H&E-only is the expected direction and says nothing yet about immunotherapy response. ST-only reaches only 0.265 although the targets are a function of its input, so the sentence plus text tower keeps only part of the expression information.
- Frozen-slide difficulty is a property of the image encoders: the ST gain is equal in the two chemistries.

## 4d. Prediction-level fusion and masked ST (29 Oct, real data)

Run on the 24 slides / 24 patients, 6-fold patient-grouped CV, frozen OmiCLIP, Ridge (alpha grid to 1e8, `--select within`). Mean per-slide Pearson, mean of the four factors:

| Arm | Mean r |
|---|---|
| H&E only (image tower) | 0.118 |
| ST only, native top-50 sentence | 0.265 |
| Both, embeddings joined (`omiclip_both_within`) | 0.267 |
| Both, predictions averaged 50/50 | 0.252 |
| Both, predictions stacked (weight learned on other folds) | 0.269 |
| ST only, 19 signature genes removed before the top 50 (`sigmasked`) | 0.260 |
| Both, with masked ST | 0.259 |

Paired patient-bootstrap differences (model minus reference, 95% CI):
- Averaging minus joined embeddings: -0.016 [-0.027, -0.005]. Frozen slides -0.027 [-0.043, -0.011], FFPE -0.004.
- Stacking minus joined embeddings: +0.002 [-0.007, +0.010]. By factor: cytotoxicity +0.018 [+0.003, +0.036], T-cell +0.015 [+0.002, +0.029], TGFb/CAF -0.021 [-0.044, -0.003], proliferation -0.006 [-0.017, +0.003]. Five factors were tested, so single stars are suggestive only. The stacking weight on ST was about 0.8 (about 0.5 for proliferation).
- Native ST minus masked ST: +0.005 [-0.000, +0.010] (T-cell +0.010 [+0.003, +0.019]; proliferation -0.016 [-0.026, -0.007], i.e. masked is higher).
- Masked both minus masked ST: -0.001 [-0.014, +0.012]; no factor interval excludes zero.
- H&E only minus masked ST: -0.142 [-0.184, -0.105]; H&E better on 1 of 24 slides.

Masking details: 19 of 19 signature genes present on every slide panel; 0 spots still contain a signature gene after masking; 0 empty sentences; median length stays 50 genes (the next-ranked genes take the places); 0.03% of sentences unchanged. Files: `outputs/rebuild_sigmasked.txt`, `outputs/compare_cv_sigmasked.txt`, `outputs/fuse_native.txt`, `outputs/compare_cv_fusion_levels.txt`. A RuntimeWarning "invalid value encountered in matmul" appears in every ridge fit; the results reproduce the 22 Oct numbers exactly (0.118 and 0.265), so it is treated as a harmless floating-point flag from the BLAS library, not a data problem.

Reading: (1) removing the 19 target genes costs about 0.005, so the ST arm is mostly not copying the target genes; it reads cell-type composition from the other genes, which still co-vary with the signatures (for example other T-cell genes). It is still not independent of the target, because the targets are computed from the same spot's expression. (2) H&E adds nothing measurable on top of ST in this stage, whether the ST arm is native (0.267 or 0.269 against 0.265) or masked (0.259 against 0.260). (3) H&E alone reaches about 45% of the ST level. This is Stage 1 only: it asks which input recovers ST-derived programmes, not which predicts response.

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
5. **Audit result (22 Oct, real checkpoint, RTX 6000 Ada).** OmiCLIP has 638.45M parameters: image tower (`visual.*`) 306.72M, text tower (`text.*`) 123.65M, CoCa `text_decoder.*` 208.07M, `logit_scale` 1 scalar. With every parameter `requires_grad=True`, the image tower, text tower and `logit_scale` all received gradient; the entire `text_decoder` (208.07M) received none, so AdamW never moved it. The pilot "full fine-tuning" runs were therefore already dual-tower fine-tuning (430.4M parameters updated). The single-tower condition has not been tested yet, and "full" should not be used in the thesis. The decoder weights in the pilot checkpoints are unchanged from pretrained and play no part in `encode_image` / `encode_text`, so the frozen arms are unaffected. Mode check (one backward pass each): frozen = 0.00M trainable; single = 306.72M (`image_tower`, `scalars`); dual = 430.38M (`image_tower`, `scalars`, `text_tower`); all PASS. Name prefixes for the new fine-tuning script: `visual.` = image tower, `text.` = text tower, `text_decoder.` = decoder (set `requires_grad=False` explicitly; it saves nothing, it just avoids confusion). `logit_scale.exp()` = 87.2. Contrastive loss of the pretrained model on one random 8-patch ccRCC batch from INT1-18 was 2.052 against a chance level of ln 8 = 2.079; this is one batch, but it agrees with the pilot val loss at epoch 1 (about 3.39 against ln 32 = 3.47): the pretrained model barely matches ccRCC patches to their own gene sentences. The "None of the inputs have requires_grad" warning comes from the frozen pass and is harmless.

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
4. **Confirmed 22 Oct (`signature_in_sentences.py`, 73,813 spots, sentences 15 to 50 genes, median 50):** the top-50 gene sentences almost never contain the signature genes of three of the four factors. Share of spots whose sentence contains the gene: T-cell CD3D, CD3E, CD2, CD8A, CD8B all 0.000; cytotoxicity GZMB, PRF1, GNLY 0.000 and NKG7 0.001; proliferation MKI67, TOP2A, PCNA, MCM2, MCM5 all 0.000. TGFb/CAF is the exception: COL1A1 0.450, COL1A2 0.478, ACTA2 0.203 (TGFB1 0.001, FAP 0.000). Share of spots with at least one signature gene: T-cell 0.001, cytotoxicity 0.001, proliferation 0.000, TGFb/CAF 0.577; in the top 10% of spots by factor score the TGFb/CAF share is 0.993 against 0.531 in the rest, while T-cell is 0.006 vs 0.000 and cytotoxicity 0.013 vs 0.000. First reading (partly superseded, see sections 4b and 4c): the sentences contain almost none of these genes, so the image tower, trained to align with that text, might have little reason to encode those signals. But the OmiCLIP vs Virchow2 gap is the same proportion on every factor (section 4b), and the text embedding does predict T-cell and cytotoxicity (section 4c). I predicted from this that an OmiCLIP-text "ST arm" would be strong on TGFb/CAF and near zero on the other three. That prediction was half wrong (section 4c): the ST arm is also strong on T-cell and cytotoxicity, so presence of the signature genes in the top-50 sentence is not what carries the signal. Only proliferation stays at the floor.
5. **Frozen-probe caveat for adapted encoders.** The probe is fit on training spots whose embeddings were optimised against their own gene sentences, so train embeddings are more aligned than val and test ones.
6. **Fusion (RQ2) is the riskiest part.** Intermediate and early fusion need ST at inference or a different task definition. Keep it conditional or secondary.
7. **Technology confound in the split (found 22 Oct).** All 24 slides come from one study (PubMed 35231421, tertiary lymphoid structures in RCC). INT1 to INT12 are fresh-frozen (36,601 genes, 1,084 to 2,580 spots each) and INT13 to INT24 are FFPE (17,943 genes, about 3,200 to 5,000 spots each). Train = 12 frozen + 6 FFPE (INT13 to INT18); val = INT19 to INT21 and test = INT22 to INT24 are all FFPE. So every result so far is "mostly frozen training, FFPE evaluation". The two chemistries capture different genes, which changes factor-score scale and the top-50 gene sentences OmiCLIP reads. This does not explain val vs test disagreement (both FFPE). `preservation_method` is empty in the metadata; use `nb_genes` or the `subseries` name. Plan: folds that mix both technologies, FFPE and frozen reported separately, and a train-FFPE-only control.
8. **Circularity of an ST arm in Stage 1 (raised 22 Oct).** The four factor scores are computed from the same spots' expression. An ST-based arm (text tower on the gene sentence, or the concatenation) predicts a function of its own input; for TGFb/CAF the sentence literally contains COL1A1/COL1A2 in 99% of high-score spots. So in Stage 1 the ST and combined arms are a ceiling or sanity check, and "combined beats H&E-only" there cannot support the thesis claim. The fair multimodal test needs a label that is not derived from the ST input, which is the response cohort. Open decision for Elsa and her supervisor: which ST representation goes into that comparison (OmiCLIP text embedding of the top-50 sentence, which cannot see immune genes, or something that does, such as signature scores or pseudo-bulk expression).

## 8. Corrections to earlier statements

| Earlier statement | Status |
|---|---|
| "Virchow2 and OmiCLIP may not be scored on the same spots" | Probably wrong. Both have 73,813 rows. `--restrict-to` will confirm. |
| "Bootstrap over patients on the test set" | Retracted. Three patients are too few. Use patient-grouped CV. |
| Methodology Figure 3 legend: "Dual FT" vs "Full FT" | Inconsistent naming. Pick one. |
| "E0/I1/I2 equivalents" in the methodology timeline (weeks 6 to 8) | Typo. The grid is fusion type (rows: Late L, Intermediate I, Early E) by encoder adaptation (columns 0 frozen, 1 single tower, 2 dual tower), so it should read E0/E1/E2. |
| Development journal | Out of date. It says the Virchow2 side of the Virchow2 vs OmiCLIP table "may have been an older existing file" and needs a timestamp check. The Virchow2 Ridge numbers in section 4 equal the ones in the journal's first Virchow2 run, but a regenerated-and-timestamp-checked run has not been recorded. |
| "OmiCLIP's image embeddings are weak on T-cell and cytotoxicity (test about 0 with Ridge)" (section 1 decisions) | Superseded 22 Oct. Under balanced CV the per-slide values are T-cell 0.123 and cytotoxicity 0.074 (Virchow2 0.228, 0.153). The pilot near-zero was a split/alpha artefact. |
| "The sentence finding matches the pilots and explains OmiCLIP's weakness" (22 Oct, same day) | Over-stated. The sentences do lack immune and proliferation genes (section 7, item 4), but the OmiCLIP vs Virchow2 gap is the same proportion (about half) on all four factors, including TGFb/CAF. Sentence content matters for an ST arm, not for the image-tower gap. |
| "OmiCLIP's text side is effectively blind to T-cell, cytotoxicity and proliferation, so an ST arm will be near zero on them" (22 Oct, from the signature-gene count) | Wrong for T-cell and cytotoxicity. Measured (section 4c): ST-only 0.271 and 0.194 per-slide r, above the image tower (0.123, 0.074) and, in point estimate, above Virchow2 (0.228, 0.153; that difference is not significant). Correct only for proliferation (0.055). Counting signature genes in the sentence does not tell you what the embedding encodes. |
| "MLP clearly beats Ridge for Virchow2" (pilot: test 0.373 vs 0.284) | Superseded 22 Oct. With a tuned Ridge on the balanced split the two are equal (test within-slide 0.226 vs 0.235). The pilot gap came from the untuned alpha = 100. |
| Early-stopping checkpoints of the MLP pilots | Virchow2 MLP val loss is lowest at epoch 1 and OmiCLIP MLP at epoch 2 (stopped at 11 and 12). So the MLP scores above are barely trained models selected on loss. Whether Pearson-based selection changes this is untested. |

## 9. Scripts

Existing pipeline (in `scripts/`, run order roughly top to bottom):

`download_hest.py`, `inspect_hest_data.py`, `check_hest_sample.py`, `compute_hest_factor_scores.py`, `embed_hest_virchow.py`, `check_virchow_embeddings.py`, `join_virchow_factor_scores.py`, `train_hest_factor_baseline.py`, `train_virchow_factor_mlp.py`, `prepare_omiclip_pairs.py`, `check_omiclip.py`, `extract_omiclip_embeddings.py`, `join_omiclip_factor_scores.py`, `train_omiclip_factor_baseline.py`, `train_omiclip_factor_mlp.py`, `finetune_omiclip_full.py`, `extract_finetuned_omiclip_embeddings(_e2).py`, `join_finetuned_omiclip_factor_scores(_e2).py`, `train_finetuned_omiclip_factor_baseline(_e2).py`, `compare_virchow_omiclip.py`, `plot_*_factor_maps.py`.

New (this session):

| Script | Purpose |
|---|---|
| `evaluate_embeddings.py` | One evaluation harness for every embedding file. Alpha and learning-rate sweeps chosen on val, pooled and within-slide Pearson, per-slide table, patient-leakage check, row and barcode alignment checks, optional patient-grouped CV for frozen encoders (with `--save-predictions` the CV also writes `cv_oof_predictions.csv`, added 22 Oct). Replaces the six Ridge/MLP scripts. |
| `fuse_predictions.py` | Prediction-level fusion of two arms from their out-of-fold predictions: equal-weight average and a stacked weight chosen on the other folds only. Writes `<name>_avg` and `<name>_stack` folders that `compare_cv.py` reads. Has a self-check (w = 0 and w = 1 must reproduce each arm's own per-slide Pearson). Tested on synthetic data only. |
| `rebuild_sentences_degraded.py` | Rebuilds the gene sentences with the 19 signature genes removed (`--exclude-signature`) and/or a random fraction of each spot's expressed genes dropped (`--drop-frac`), from the original pairs file, keeping spot order. Imports the original rule from `prepare_omiclip_pairs.py`. Sanity check: no signature gene may survive. Tested on synthetic AnnData only. |
| `extract_omiclip_text.py` | Text-tower-only embeddings for a pairs file with new sentences (same checkpoint, tokenizer, L2 normalisation as the original extraction). Tested only with a stub model. |
| `compare_cv.py` | Paired comparison of models from the CV outputs with a patient-level bootstrap; also reports the difference separately for frozen and FFPE slides. |
| `make_splits.py` | Writes `datasets/splits/ccrcc_splits_v1.csv`: a technology-balanced fixed split (16 train / 4 val / 4 test, half frozen and half FFPE each) and 6 CV folds (2 frozen + 2 FFPE held out per fold). Technology is read from `subseries`. Every script should read this file instead of hard-coding INT numbers. |
| `audit_omiclip_params.py` | Which OmiCLIP modules actually receive gradient. Defines the frozen, single and dual modes and asserts each one. |
| `signature_in_sentences.py` | How often each signature gene appears in the top-50 gene sentence OmiCLIP reads (result in section 7.4). Pandas only. |
| `make_omiclip_arms.py` | Builds the ST-only (text tower) and combined (image + text, 1536-d) datasets from embeddings already on disk. `extract_omiclip_embeddings.py` saved both towers (`image_embeddings`, `st_embeddings`, L2-normalised) for every spot; the join script kept only the image tower. Same row order as the image dataset, so `omiclip_factor_metadata.csv` is reused. Checks alignment against the h5. Also usable for a second sentence version (`--h5`, `--prefix`). Tested on synthetic data only. |
| `finetune_omiclip_modes.py` | Frozen / single / dual fine-tuning on the balanced fixed split (reads `ccrcc_splits_v1.csv`); LR, epochs, mode, out-dir are arguments; epoch-0 validation of the pretrained model; fixed shuffled validation order; gradient check after the first backward; saves only the trainable tensors (`best.pt` = lowest val contrastive loss over epochs 1..N, `last.pt`). Pieces tested on a toy CoCa-like model (mode groups, gradients, save/load round trip); the real model, Loki and the GPU path are UNTESTED (29 Oct). |
| `extract_omiclip_ckpt.py` | Both-tower embeddings of every spot from the pretrained model or a fine-tuned checkpoint, in pairs-file order, plus `sample_id`/`barcode` for row-order checks. Also used for the JITC cohort (`--pairs datasets/jitc_mrcc/jitc_pairs.csv`). GPU path untested (29 Oct). |
| `run_finetune_pilots.sh` | Six pilot runs (single/dual x LR 5e-7, 1e-6, 2e-6): plumbing check, frozen reference, then per run fine-tune, embed, build image/ST/both datasets (`make_omiclip_arms.py --rebuild-image`), evaluate each arm (Ridge, fixed split, `--select within`). Resumable. Selection is by validation Pearson in the eval logs. |
| `prepare_response_cohort.py` | JITC mRCC SpaceRanger folders to HEST-style objects: 224 px patches cut from the downsized hires image (about 3 um/px, so blurry), gene sentences (Stage 1 rule), pairs CSV, measured factor scores (Stage 1 definition; scanpy if installed, else a close re-implementation). Tested on synthetic folders only; the scanpy branch is untested. |
| `response_test.py` | Exploratory response test: Stage 1 ridge heads (alpha per factor from 6-fold CV) applied to the JITC spots, per-slide mean scores, transfer Pearson (predicted vs measured), responders vs non-responders with AUC and exact permutation p. Primary contrast pre-specified: T-cell score on metastasis slides. Tested on synthetic data with a planted signal. |

Facts about the existing sentence builder (`prepare_omiclip_pairs.py`, read 22 Oct): per spot it takes the top 50 genes by RAW value from `adata.layers['counts']` (else `adata.X`), positive values only, ties broken arbitrarily, gene symbols deduplicated, space-separated; spots with no expressed gene are dropped; sentences can be shorter than 50 (minimum seen 15). No normalisation, no gene-specific reweighting.

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

- [x] Run the parameter audit; send the printout back. DONE 22 Oct (section 5, item 5).
- [x] Verify the split file against the real table. DONE 22 Oct (section 1).
- [x] Run `evaluate_embeddings.py` for Virchow2 and pretrained OmiCLIP; check same-spot overlap. DONE 22 Oct (section 4b): same 73,813 spots, no patient shared between splits.
- [x] Re-run both with `--select within` (primary metric = per-slide Pearson). DONE 22 Oct (section 4b); no change in conclusion.
- [x] DONE 29 Oct (section 4d). Prediction-level fusion (go-ahead 22 Oct): re-run `omiclip_image_oof` and `omiclip_st_oof` with `--save-predictions --skip-standard --cv-frozen-encoder 6 --select within`, then `fuse_predictions.py --a omiclip_image_oof --b omiclip_st_oof --out-name omiclip_predfuse`, then `compare_cv.py --reference omiclip_both_within --models omiclip_both_within omiclip_predfuse_avg omiclip_predfuse_stack`.
- [ ] Degraded ST: `sigmasked` DONE 29 Oct (section 4d); `drop50` and `drop90` not run. (go-ahead 22 Oct; original note:) `rebuild_sentences_degraded.py` for `sigmasked`, `drop50`, `drop90`; `extract_omiclip_text.py` for each; `make_omiclip_arms.py --h5 <text h5> --image-h5 outputs/omiclip/ccrcc_omiclip_embeddings.h5 --prefix omiclip_<tag>`; evaluate ST-only and combined for each; compare against the image-only arm. Question: does H&E add more when ST is degraded?
- [x] DONE 29 Oct as `sigmasked` (section 4d). (Description of the leave-signature-out part, go-ahead given 22 Oct): rebuild each spot's sentence after removing the 19 signature genes (zero them before taking the top 50), embed with the frozen text tower only, re-run ST-only and combined. It removes literal access to the target genes and shows how much of the ST advantage is composition signal rather than copying. Needs a masked-sentence builder (wrapper around `prepare_omiclip_pairs.py`, row order must match the original pairs file) and a text-tower-only extractor; `make_omiclip_arms.py` would need an `--image-h5` option.
- [x] DONE 22 Oct. Pairwise tests with reference = ST-only (`compare_cv.py --reference omiclip_st_within --models omiclip_st_within omiclip_both_within omiclip_pretrained_within virchow2_within`), and the factor-factor correlation check (section 4c).
- [x] DONE 22 Oct (section 4c). Run the NATIVE ST-only and combined arms from the existing text embeddings (`make_omiclip_arms.py`, then `evaluate_embeddings.py --select within --cv-frozen-encoder 6` on `omiclip_st_factor_dataset.npz` and `omiclip_both_factor_dataset.npz`, then `compare_cv.py`). Predictions written down before running (22 Oct): ST-only is strong on TGFb/CAF (per-slide r above 0.4) and not clearly better than the image tower on T-cell, cytotoxicity and proliferation; combined is at least as good as image-only on TGFb/CAF.
- [ ] (Lower priority after section 4c; mainly relevant for proliferation and for any fine-tuning.) The sentence builder is `scripts/prepare_omiclip_pairs.py` (found 22 Oct). Next: write an "enriched" sentence version (rank genes by elevation vs the gene's mean across spots of the same technology) and re-run `signature_in_sentences.py` on it. Only embed it with the frozen text tower if immune genes now appear.
- [x] Run the MLP heads on the fixed split (`virchow2_mlp`, `omiclip_pretrained_mlp`). DONE 22 Oct (section 4b): no reliable gain over tuned Ridge.
- [ ] Print `treatment_comment`, `disease_comment`, `dataset_title`, `study_link` from the sample table and look for treatment or outcome information.
- [ ] Response cohort (section 1b): Elsa opens https://zenodo.org/records/16833780 (JITC mRCC) and https://zenodo.org/records/15392699 (HCC) in a browser and reports file list, licence, and whether H&E images or spatial image folders are included. DECIDED by Elsa 22 Oct: the thesis answers the response question with the small exploratory test (no supervisor check). JITC labels (29 Oct, from Table 1 and Fig 1): Table 1 is aggregate only (12 patients, Visium 12, CosMx 10, CR 1, PR 5, SD 1, PD 5, clinical benefit 7 vs 5). Per-patient labels are known only for the 6 metastasis patients: responders W3 (the CR), W33, W38; non-responders W10, W47, W55 (Fig 1B bars: W10/W47/W55 tumours grew, W3/W33/W38 shrank). By subtraction the other six patients (W31 W34 W35 W37 W44 W50) hold 3 PR, 1 SD, 2 PD, but which is which is NOT public in the text; ask the author for per-patient RECIST category and % volume change at 3 months for all 12. Command to list a record's files from the workstation: `curl -s https://zenodo.org/api/records/<id> | python -m json.tool | grep -E '"(title|key|size|access_right|license)"'`.
- [x] Run the signature-gene count in gene sentences. DONE 22 Oct (section 7, item 4).
- [ ] Decide how the ST arm is defined for the response-prediction stage (section 7, item 8).
- [ ] Fix the Virchow2 register-token slice (`out[:, 5:]`) and re-embed, or state the choice in the thesis.
- [x] WRITTEN 29 Oct, not yet run on the GPU: `finetune_omiclip_modes.py`, `extract_omiclip_ckpt.py`, `run_finetune_pilots.sh` (section 9). First thing the pilot script does is a smoke test per mode; if that fails, read the error before anything else.
- [ ] Six-run pilot (run LAST, it is the longest): `bash scripts/run_finetune_pilots.sh 3 2>&1 | tee outputs/ft_pilots.log`. LR {5e-7, 1e-6, 2e-6} x {single, dual}; select by downstream val Pearson.
- [ ] JITC response test (29 Oct plan; resolution caveat: the only H&E is SpaceRanger's `tissue_hires_image.png`, about 3.2 um/px against about 0.5 um/px for HEST, so the JITC patches are upsampled and blurry and the H&E arm is expected to be weaker there; report this as a limitation). Steps on the workstation: (1) put `Visium_Objects.zip` in e.g. `/media/rokny/DATA2/Elsa/response_cohorts/jitc/` and unzip it (leave out `__MACOSX`); (2) `python scripts/prepare_response_cohort.py --root <that>/Visium_Objects` (CPU); (3) `python scripts/extract_omiclip_ckpt.py --pairs datasets/jitc_mrcc/jitc_pairs.csv --output-h5 outputs/jitc/jitc_omiclip_embeddings.h5` (GPU); (4) `python scripts/response_test.py` (writes `outputs/jitc_response/`). Check first: HEST patch field of view is assumed 112 um; confirm with `pixel_size_um_estimated` in the HEST metadata. Labels known for 6 patients only (R: W3, W33, W38; NR: W10, W47, W55); primary contrast is T-cell score on metastasis slides, with exact min p = 0.10 for 3 vs 3, so everything is descriptive. Not done: Virchow2 embeddings for JITC (control arm); fine-tuned arms through `response_test.py`.
- [ ] Patient-grouped CV for the final comparison (frozen arms directly; adapted arms with the whole pipeline per fold).
- [ ] Decide whether RQ2 (fusion) stays as a main result or becomes secondary.
- [ ] Check that the GitHub push of `hest-ccrcc` went through (`git log origin/hest-ccrcc`). Never commit `.hf_home/`, `venv/`, data, or model weights; never paste tokens into chats or files.

## 12. Environment

- Workstation (TOWER3): 2x RTX 6000 Ada, about 96 GB RAM. Project in `/media/rokny/DATA2/Elsa/vcell`, `HF_HOME` set to `.hf_home` inside it.
- Virchow2 runs in the project venv; OmiCLIP/Loki in conda env `omiclip` (Python 3.9). Do not mix them.
- Repo: `ed21211/vcell`, branch `hest-ccrcc`. The workstation SSH key belongs to another account (`jade0530`); pushes use HTTPS with a fine-grained token scoped to this repo, and git identity is set per repo only.
