# JEPA results report — GLUE, BEIR, Rankify

Generated 2026-09-02 (ICT). Our numbers are parsed directly from the run artifacts on disk; reference numbers are transcribed from the cited papers.

---

## Method

**Every GLUE cell is `ours (JEPA) / published`.** The published column is:

| our column | reference | source |
|---|---|---|
| validation | GLUE **dev**, large models | arXiv 2006.03654 (DeBERTa), Table 1 |
| test | AdvGLUE **test**, large models | arXiv 2111.02840 (AdvGLUE), Table 4 |

Those papers report **large** models only, so `bert-large` is matched to BERT-large, `roberta-large` to RoBERTa-large and `deberta-large` to DeBERTa-large. `deberta-base` is deliberately **not** matched to DeBERTa-large, a different model size.

**For the three base models the comparator is our own `fresh` baseline** (the same architecture trained without JEPA pretraining), since no published large-model number applies to them. So the second element of a cell means different things by column, and the two kinds of comparison should not be read against each other:

| model | second element = |
|---|---|
| bert-base, roberta-base, deberta-base | our fresh (non-JEPA) baseline |
| bert-large | BERT-large, published |
| roberta-large | RoBERTa-large, published |
| deberta-large | DeBERTa-large, published |

**Metrics** (as implemented in `src/jepa_humor/finetune.py`): CoLA = Matthews corr, STS-B = Pearson, all other tasks = accuracy — the same metrics the DeBERTa paper reports. AdvGLUE figures are **accuracy only**. The AdvGLUE paper's own Avg column folds QQP F1 into the mean — verified, since `Avg = mean(SST-2, mean(MNLI-m, MNLI-mm), mean(QQP-acc, QQP-F1), QNLI, RTE)` reproduces its printed value on all six of its models — and those F1 numbers run far below the accuracies (BERT-large 16.56 vs 37.91), so that column is not used anywhere here.

**The four statistics**, computed across seeds within one run:

| statistic | meaning |
|---|---|
| med-val | median over seeds of (best validation score over epochs) |
| med-best-test | median over seeds of (best test score over epochs) |
| med-test@val | median over seeds of (test score at the best-validation epoch) |
| max-test@val | max over seeds of (test score at the best-validation epoch) |

**Run selection — every statistic is maximised independently.** Each model has many runs (LR sweeps, dropout variants, reruns). For each (model, task) cell, *each statistic is taken from whichever run maximises that statistic*, so the validation number and the test numbers may come from **different runs**. Runs with >= 3 seeds are preferred. Section 3 lists the source run behind every cell. Note this is a multiple-comparisons procedure: it answers "what is the best number we obtained", which inflates our side relative to a single pre-registered run, while the published column is a single reported result.

**AdvGLUE test set.** Our `src/jepa_humor/test_ann.json` is byte-identical to the official `adversarialglue.github.io/dataset/test_ann.json`. That file bundles benign GLUE dev examples tagged `method: "glue"`; `load_test_set_from_json` drops every one of them, leaving exactly SST-2 1,420 / QQP 422 / MNLI-m 766 / MNLI-mm 1,098 / QNLI 968 / RTE 304 = **4,978**, matching the paper's counts on all six tasks. The comparison is like-for-like.

> **Excluded: MNLI-transfer runs.** All results are single-task — finetuned from the base/JEPA encoder, never from an MNLI checkpoint. One cell was dropped to enforce this: roberta-large RTE from the `hd00` run (both arms reading exactly 90.61), which `run_hd00_singletask_mrpcstsb.sh` shows was left MNLI-transfer initialised when hd00's MRPC and STS-B were overwritten with single-task runs. Note the DeBERTa paper's RoBERTa-large RTE (86.6) *does* use MNLI-transfer initialisation, so our single-task 87.0 is if anything conservative against it.
>
> For `deberta-large` this exclusion removes far more: its whole `mnli_transfer/` tree and `second_run`'s Stage-D `transfer_*` / `lr2e5_transfer_*` tags (2,307 result files). Those dominate exactly the three tasks MNLI-init is known to help — left in, RTE would read 91.70 against the paper's 88.3 instead of the single-task 87.36. `second_run`'s Stage-B `notransfer_*` tags are own-encoder runs and are kept. `deberta-large` has no WNLI run, so that cell is blank; neither paper reports WNLI for any large model either.

---

## 1. Headline — GLUE validation (ours / published)

| task | bert-base | bert-large | roberta-base | roberta-large | deberta-base | deberta-large |
|---|---|---|---|---|---|---|
| CoLA | 59.2 / 59.9 | 63.2 / 60.6 | 63.7 / 63.6 | 69.0 / 68.0 | 65.8 / 65.3 | 70.4 / 70.5 |
| SST-2 | 92.9 / 92.8 | 93.7 / 93.2 | 95.0 / 95.1 | 96.4 / 96.4 | 95.4 / 95.4 | 96.9 / 96.8 |
| MRPC | 84.1 / 83.6 | 86.3 / 88.0 | 88.7 / 89.2 | 90.7 / 90.9 | 90.0 / 89.7 | 91.2 / 91.9 |
| STS-B | 90.2 / 90.1 | 90.7 / 90.0 | 91.1 / 91.1 | 92.3 / 92.4 | 91.4 / 91.3 | 92.3 / 92.8 |
| QQP | 91.3 / 91.4 | 91.7 / 91.3 | 91.7 / 91.7 | 92.3 / 92.2 | 91.9 / 91.9 | 92.4 / 92.3 |
| MNLI-m | 84.6 / 84.7 | 86.6 / 86.6 | 87.6 / 87.6 | 90.4 / 90.2 | 88.3 / 88.1 | 91.1 / 91.1 |
| MNLI-mm | 84.7 / 84.7 | 86.4 / - | 87.3 / 87.4 | 90.4 / 90.2 | 88.1 / 88.1 | 90.9 / 91.1 |
| QNLI | 91.3 / 91.4 | 92.3 / 92.3 | 92.9 / 92.9 | 94.9 / 93.9 | 93.5 / 93.5 | 95.2 / 95.3 |
| RTE | 71.1 / 71.8 | 75.8 / 70.4 | 79.4 / 77.6 | 87.0 / 86.6 | 78.7 / 78.3 | 87.4 / 88.3 |
| WNLI | 73.1 / 73.1 | 80.8 / - | 83.7 / 85.6 | 91.3 / - | 70.2 / 70.2 | - / - |
| **GLUE avg (8)** | **83.1** / 83.2 | **85.0** / 84.0 | **86.3** / 86.1 | **89.1** / 88.8 | **86.9** / 86.7 | **89.6** / 89.9 |

| model | ours (JEPA) | compared with | comparator | delta |
|---|---:|---|---:|---:|
| bert-base | 83.1 | our fresh baseline | 83.2 | -0.1 |
| bert-large | 85.0 | BERT-large (published) | 84.0 | +1.0 |
| roberta-base | 86.3 | our fresh baseline | 86.1 | +0.2 |
| roberta-large | 89.1 | RoBERTa-large (published) | 88.8 | +0.3 |
| deberta-base | 86.9 | our fresh baseline | 86.7 | +0.2 |
| deberta-large | 89.6 | DeBERTa-large (published) | 89.9 | -0.3 |

GLUE avg = mean of the 8 tasks (CoLA, SST-2, MRPC, STS-B, QQP, MNLI-m, QNLI, RTE); MNLI-mm and WNLI excluded. The DeBERTa paper's Avg uses this same definition — recomputing from its own columns gives 84.05 for BERT-large (printed 84.05) and 88.83 for RoBERTa-large (printed 88.82).

---

## 2. GLUE — four views (ours / published)

Each table independently selects the run maximising that table's statistic, so a model's row in 2.1 and its row in 2.3 may come from different runs. Tasks with no AdvGLUE test set (CoLA, MRPC, STS-B, WNLI) appear only in 2.1. The published column is the same AdvGLUE figure across 2.2-2.4 — only our side changes, so 2.3 is the row that should carry the comparison.

### 2.1 med-val

Median over seeds of the best validation score over epochs, against the DeBERTa paper's GLUE **dev** numbers. One decimal, matching the paper.

| task | bert-base | bert-large | roberta-base | roberta-large | deberta-base | deberta-large |
|---|---|---|---|---|---|---|
| CoLA | 59.2 / 59.9 | 63.2 / 60.6 | 63.7 / 63.6 | 69.0 / 68.0 | 65.8 / 65.3 | 70.4 / 70.5 |
| SST-2 | 92.9 / 92.8 | 93.7 / 93.2 | 95.0 / 95.1 | 96.4 / 96.4 | 95.4 / 95.4 | 96.9 / 96.8 |
| MRPC | 84.1 / 83.6 | 86.3 / 88.0 | 88.7 / 89.2 | 90.7 / 90.9 | 90.0 / 89.7 | 91.2 / 91.9 |
| STS-B | 90.2 / 90.1 | 90.7 / 90.0 | 91.1 / 91.1 | 92.3 / 92.4 | 91.4 / 91.3 | 92.3 / 92.8 |
| QQP | 91.3 / 91.4 | 91.7 / 91.3 | 91.7 / 91.7 | 92.3 / 92.2 | 91.9 / 91.9 | 92.4 / 92.3 |
| MNLI-m | 84.6 / 84.7 | 86.6 / 86.6 | 87.6 / 87.6 | 90.4 / 90.2 | 88.3 / 88.1 | 91.1 / 91.1 |
| MNLI-mm | 84.7 / 84.7 | 86.4 / - | 87.3 / 87.4 | 90.4 / 90.2 | 88.1 / 88.1 | 90.9 / 91.1 |
| QNLI | 91.3 / 91.4 | 92.3 / 92.3 | 92.9 / 92.9 | 94.9 / 93.9 | 93.5 / 93.5 | 95.2 / 95.3 |
| RTE | 71.1 / 71.8 | 75.8 / 70.4 | 79.4 / 77.6 | 87.0 / 86.6 | 78.7 / 78.3 | 87.4 / 88.3 |
| WNLI | 73.1 / 73.1 | 80.8 / - | 83.7 / 85.6 | 91.3 / - | 70.2 / 70.2 | - / - |
| **GLUE avg (8)** | **83.1** / 83.2 | **85.0** / 84.0 | **86.3** / 86.1 | **89.1** / 88.8 | **86.9** / 86.7 | **89.6** / 89.9 |

### 2.2 med-best-test

Median over seeds of the best AdvGLUE test score over epochs — the epoch is chosen using the test set, so this is optimistically biased and the reference column is not a fair target for it.

| task | bert-base | bert-large | roberta-base | roberta-large | deberta-base | deberta-large |
|---|---|---|---|---|---|---|
| SST-2 | 26.83 / 26.34 | 35.99 / 33.03 | 27.96 / 27.68 | 60.63 / 58.52 | 43.24 / 43.10 | 65.70 / 57.89 |
| QQP | 57.11 / 44.31 | 52.13 / 37.91 | 52.37 / 50.24 | 63.27 / 57.11 | 57.11 / 59.24 | 67.06 / 60.43 |
| MNLI-m | 29.50 / 27.15 | 30.42 / 28.72 | 23.76 / 20.63 | 52.22 / 50.78 | 38.38 / 38.51 | 58.62 / 58.36 |
| MNLI-mm | 24.95 / 24.50 | 26.68 / 27.05 | 20.67 / 19.67 | 42.35 / 39.62 | 30.33 / 30.15 | 51.55 / 52.46 |
| QNLI | 36.88 / 36.16 | 41.12 / 39.77 | 36.47 / 28.10 | 58.06 / 52.48 | 45.76 / 44.73 | 65.60 / 57.85 |
| RTE | 41.45 / 41.12 | 42.11 / 40.46 | 40.46 / 40.46 | 68.09 / 45.39 | 58.88 / 56.58 | 70.39 / 78.95 |
| **avg (5, accuracy)** | **38.35** / 35.02 | **40.35** / 35.98 | **36.20** / 33.42 | **60.45** / 52.86 | **48.68** / 48.43 | **65.48** / 62.70 |

### 2.3 med-test@val

Median over seeds of the AdvGLUE test score at the best-validation epoch. **This is the row to compare with the paper** — the epoch is chosen on validation only.

| task | bert-base | bert-large | roberta-base | roberta-large | deberta-base | deberta-large |
|---|---|---|---|---|---|---|
| SST-2 | 24.37 / 24.08 | 33.24 / 33.03 | 22.32 / 21.83 | 58.52 / 58.52 | 41.41 / 40.77 | 64.08 / 57.89 |
| QQP | 44.31 / 36.26 | 38.39 / 37.91 | 32.23 / 27.49 | 56.04 / 57.11 | 41.94 / 42.65 | 63.03 / 60.43 |
| MNLI-m | 27.81 / 26.76 | 30.29 / 28.72 | 18.93 / 19.19 | 52.22 / 50.78 | 36.81 / 36.29 | 58.09 / 58.36 |
| MNLI-mm | 22.95 / 22.86 | 25.32 / 27.05 | 18.40 / 18.67 | 40.80 / 39.62 | 30.15 / 29.33 | 50.64 / 52.46 |
| QNLI | 34.19 / 34.09 | 40.91 / 39.77 | 27.58 / 21.18 | 56.71 / 52.48 | 45.14 / 43.80 | 63.64 / 57.85 |
| RTE | 40.79 / 39.80 | 38.82 / 40.46 | 35.53 / 34.54 | 66.78 / 45.39 | 44.41 / 41.45 | 69.41 / 78.95 |
| **avg (5, accuracy)** | **34.29** / 32.20 | **36.33** / 35.98 | **27.32** / 24.85 | **58.05** / 52.86 | **41.94** / 40.99 | **63.65** / 62.70 |

### 2.4 max-test@val

Max over seeds of the AdvGLUE test score at the best-validation epoch — a best-case seed, so the noisiest figure here.

| task | bert-base | bert-large | roberta-base | roberta-large | deberta-base | deberta-large |
|---|---|---|---|---|---|---|
| SST-2 | 25.00 / 24.37 | 37.04 / 33.03 | 28.66 / 25.35 | 61.06 / 58.52 | 42.82 / 41.76 | 66.55 / 57.89 |
| QQP | 50.47 / 37.68 | 41.23 / 37.91 | 34.83 / 31.99 | 58.77 / 57.11 | 48.34 / 44.08 | 66.11 / 60.43 |
| MNLI-m | 29.11 / 28.20 | 31.72 / 28.72 | 22.58 / 21.15 | 54.18 / 50.78 | 37.99 / 39.56 | 59.14 / 58.36 |
| MNLI-mm | 25.05 / 24.50 | 27.32 / 27.05 | 21.86 / 19.40 | 41.99 / 39.62 | 30.51 / 30.69 | 52.73 / 52.46 |
| QNLI | 35.02 / 34.71 | 41.12 / 39.77 | 29.55 / 23.76 | 57.85 / 52.48 | 47.11 / 45.66 | 66.43 / 57.85 |
| RTE | 43.75 / 41.78 | 42.43 / 40.46 | 46.71 / 39.80 | 70.72 / 45.39 | 46.38 / 48.36 | 73.68 / 78.95 |
| **avg (5, accuracy)** | **36.67** / 33.35 | **38.71** / 35.98 | **32.47** / 28.41 | **60.52** / 52.86 | **44.53** / 43.88 | **66.38** / 62.70 |

---

## 3. Source run behind every cell (ours)

`run [seed count]`. Because each statistic is maximised independently, these differ between tables — expected, not an inconsistency.

### 2.1 med-val — source runs

| task | bert-base | bert-large | roberta-base | roberta-large | deberta-base | deberta-large |
|---|---|---|---|---|---|---|
| CoLA | first_try/lr3e5 [5] | (root):bertlarge [5] | first_try/lr3e5 [5] | hd00:glue [7] | (root) [5] | paper_sweep:lr2e5_bs32_w100_p1c1 [5] |
| SST-2 | sst2_pre1ep_lr1e6 [5] | (root):bertlarge [5] | first_try [5] | hd01:glue9_knn [5] | ft_lr1e5 [5] | bs4864:sst2_lr5e6_bs64_w500_p1c1 [5] |
| MRPC | first_try [5] | (root):bertlarge [5] | first_try [5] | hd01:glue9_knn [5] | (root) [5] | paper_sweep:lr3e5_bs16_w100_p1c1 [5] |
| STS-B | first_try/lr3e5 [5] | stsb_sweep/best_lr2.0e-5_bs16 [5] | first_try/lr3e5 [5] | stsb_sweep/best_lr2.0e-5_bs16 [5] | ft_lr1e5 [5] | biascorr_lr2:stsb_base5e6 [5] |
| QQP | (root) [5] | (root):bertlarge [5] | first_try [5] | hd00:glue_no_knn [5] | (root) [5] | dropsweep:p0c1 [5] |
| MNLI-m | first_try [5] | (root):bertlarge [5] | forth_run:mnli1ep [5] | hd00:glue_no_knn [5] | (root) [5] | bs3264_len128:mnli_bs64_len128 [5] |
| MNLI-mm | first_try [5] | (root):bertlarge [5] | first_try [5] | hd01:glue9_noknn [3] | large_cls15 [5] | bs3264_len128:mnli_bs64_len128 [5] |
| QNLI | first_try [5] | (root):bertlarge [5] | first_try [5] | hd01:glue9_noknn [3] | ft_lr1e5 [5] | paper_sweep:lr8e6_bs32_w500_p0c1 [5] |
| RTE | first_try [5] | second_run [5] | first_try/lr2e5 [5] | hd01:glue9_knn [5] | (root) [5] | paper_sweep:lr9e6_bs16_w50_p0c1 [5] |
| WNLI | first_try [5] | (root):bertlarge [5] | first_try [5] | hd00:glue_no_knn [3] | (root) [5] | - |

### 2.2 med-best-test — source runs

| task | bert-base | bert-large | roberta-base | roberta-large | deberta-base | deberta-large |
|---|---|---|---|---|---|---|
| SST-2 | (root) [5] | sst2_pre1ep_lr1e6 [5] | first_try [5] | hd01:glue9_knn [5] | ft_lr1e5 [5] | bs4864:sst2_lr5e6_bs64_w500_p0c1 [5] |
| QQP | sched5_cap1ep:qqp [5] | (root):bertlarge [5] | (root) [5] | hd01:glue9_knn [5] | large_cls15 [5] | second_run:p0c1 [5] |
| MNLI-m | (root) [5] | third_run [5] | forth_run:mnli2ep [5] | hd01:glue9_knn [5] | (root) [5] | bs3264_len128:mnli_bs64_len128 [5] |
| MNLI-mm | sched5_cap1ep:mnli [5] | (root):bertlarge [5] | forth_run:mnli2ep [5] | hd01:glue9_knn [5] | (root) [5] | second_run:p0c0 [5] |
| QNLI | (root) [5] | (root):bertlarge [5] | forth_run:qnli [5] | hd01:glue9_noknn [3] | ft_lr1e5 [5] | paper_sweep:lr8e6_bs32_w500_p0c15 [5] |
| RTE | first_try [5] | third_run_1ep [5] | first_try/lr2e5 [5] | hd01:glue9_knn [5] | ft_lr1e5 [5] | paper_sweep:lr2e5_bs16_w500_p0c0 [5] |

### 2.3 med-test@val — source runs

| task | bert-base | bert-large | roberta-base | roberta-large | deberta-base | deberta-large |
|---|---|---|---|---|---|---|
| SST-2 | sst2_pre1ep_lr1e6 [5] | sst2_pre1ep_lr1e6 [5] | (root) [5] | hd01:glue9_knn [5] | ft_lr1e5 [5] | bs4864:sst2_lr5e6_bs64_w500_p1c15 [5] |
| QQP | sched5_cap1ep:qqp [5] | (root):bertlarge [5] | (root) [5] | hd00:glue [8] | (root) [5] | second_run:p0c1 [5] |
| MNLI-m | forth_run:mnli [5] | second_run [5] | (root) [5] | hd01:glue9_knn [5] | (root) [5] | bs3264_len128:mnli_bs64_len128 [5] |
| MNLI-mm | first_try [5] | (root):bertlarge [5] | first_try [5] | hd00:glue_no_knn [5] | (root) [5] | bs3264_len128:mnli_bs32_len128 [5] |
| QNLI | (root) [5] | (root):bertlarge [5] | forth_run:qnli [5] | qnli_pre1ep_lr1e6 [5] | (root) [5] | lr8e6:p0c1 [5] |
| RTE | (root) [5] | third_run_1ep [5] | first_try/lr2e5 [5] | hd01:glue9_knn [5] | (root) [5] | biascorr_lr2:rte_base8e6 [5] |

### 2.4 max-test@val — source runs

| task | bert-base | bert-large | roberta-base | roberta-large | deberta-base | deberta-large |
|---|---|---|---|---|---|---|
| SST-2 | first_try [5] | second_run [5] | first_try [5] | hd01:glue9_knn [5] | ft_lr1e5 [5] | bs4864:sst2_lr5e6_bs64_w500_p1c15 [5] |
| QQP | sched5_cap1ep:qqp [5] | (root):bertlarge [5] | first_try [5] | hd00:glue [8] | (root) [5] | second_run:p0c1 [5] |
| MNLI-m | first_try [5] | second_run [5] | (root) [5] | hd01:glue9_knn [5] | (root) [5] | bs3264_len128:mnli_bs64_len128 [5] |
| MNLI-mm | first_try [5] | second_run [5] | first_try [5] | hd00:glue [8] | large_cls15 [5] | bs3264_len128:mnli_bs32_len128 [5] |
| QNLI | (root) [5] | (root):bertlarge [5] | forth_run:qnli [5] | hd00:glue [8] | (root) [5] | second_run:p0c1 [5] |
| RTE | first_try/lr3e5 [5] | second_run [5] | first_try/lr2e5 [5] | hd01:glue9_noknn [5] | (root) [5] | paper_sweep:lr2e5_bs32_w500_p0c1 [5] |

---

## 4. roberta-large — kNN vs no-kNN in the JEPA pretrain

kNN and no-kNN are both ours; the last column repeats the published RoBERTa-large reference (GLUE dev in 4.1, AdvGLUE test in 4.2-4.4) for context. The kNN arm is the better of `hd00:glue` / `hd01:glue9_knn`, the no-kNN arm the better of `hd00:glue_no_knn` / `hd01:glue9_noknn` (hd00/hd01 = head_dropout 0.0/0.1), maximised independently per statistic. Seed counts are uneven (kNN 5-8, no-kNN 3-5) because the no-kNN sweeps were stopped earlier, so differences under ~0.3 are inside the noise.

### 4.1 med-val — kNN / no-kNN

| task | kNN | no-kNN | Δ | published (RoBERTa-large) |
|---|---:|---:|---:|---:|
| CoLA | 69.0 | 68.7 | +0.3 | 68.0 |
| SST-2 | 96.4 | 96.3 | +0.1 | 96.4 |
| MRPC | 90.7 | 90.0 | +0.7 | 90.9 |
| STS-B | 90.8 | 91.0 | -0.2 | 92.4 |
| QQP | 92.2 | 92.3 | -0.1 | 92.2 |
| MNLI-m | 90.4 | 90.4 | -0.0 | 90.2 |
| MNLI-mm | 90.3 | 90.4 | -0.1 | 90.2 |
| QNLI | 94.8 | 94.9 | -0.1 | 93.9 |
| RTE | 87.0 | 86.3 | +0.7 | 86.6 |
| WNLI | 89.9 | 91.3 | -1.4 | - |
| **GLUE avg (8)** | **88.9** | 88.7 | +0.2 | 88.8 |

### 4.2 med-best-test — kNN / no-kNN

| task | kNN | no-kNN | Δ | published (RoBERTa-large) |
|---|---:|---:|---:|---:|
| SST-2 | 60.63 | 60.21 | +0.42 | 58.52 |
| QQP | 63.27 | 63.03 | +0.24 | 57.11 |
| MNLI-m | 52.22 | 50.91 | +1.31 | 50.78 |
| MNLI-mm | 42.35 | 41.99 | +0.36 | 39.62 |
| QNLI | 57.28 | 58.06 | -0.77 | 52.48 |
| RTE | 68.09 | 66.12 | +1.97 | 45.39 |
| **avg (5, accuracy)** | **60.30** | 59.67 | +0.63 | 52.86 |

### 4.3 med-test@val — kNN / no-kNN

| task | kNN | no-kNN | Δ | published (RoBERTa-large) |
|---|---:|---:|---:|---:|
| SST-2 | 58.52 | 57.39 | +1.13 | 58.52 |
| QQP | 56.04 | 55.21 | +0.83 | 57.11 |
| MNLI-m | 52.22 | 50.39 | +1.83 | 50.78 |
| MNLI-mm | 39.89 | 40.80 | -0.91 | 39.62 |
| QNLI | 56.20 | 55.99 | +0.21 | 52.48 |
| RTE | 66.78 | 66.12 | +0.66 | 45.39 |
| **avg (5, accuracy)** | **57.95** | 57.02 | +0.93 | 52.86 |

### 4.4 max-test@val — kNN / no-kNN

| task | kNN | no-kNN | Δ | published (RoBERTa-large) |
|---|---:|---:|---:|---:|
| SST-2 | 61.06 | 60.49 | +0.56 | 58.52 |
| QQP | 58.77 | 58.53 | +0.24 | 57.11 |
| MNLI-m | 54.18 | 52.74 | +1.44 | 50.78 |
| MNLI-mm | 41.99 | 41.89 | +0.09 | 39.62 |
| QNLI | 57.85 | 56.61 | +1.24 | 52.48 |
| RTE | 68.42 | 70.72 | -2.30 | 45.39 |
| **avg (5, accuracy)** | **60.05** | 59.82 | +0.23 | 52.86 |

---

## 5. BEIR — DPR checkpoints vs published DPR

NDCG@10. `ckpt2` = best small-BEIR transfer checkpoint, `ckpt39` = final epoch, `ckpt36` = best in-domain avg-rank checkpoint. The **public DPR** column is the DPR column of BEIR Table 2 (arXiv 2104.08663). No locally-trained *fresh* DPR BEIR evaluation exists, so the paper's numbers are the only available reference.

| dataset | ckpt2 | ckpt39 | ckpt36 | public DPR |
|---|---:|---:|---:|---:|
| MS MARCO* | **0.2072** | 0.2064 | 0.2052 | 0.177 |
| TREC-COVID | **0.4156** | 0.3829 | 0.3819 | 0.332 |
| NFCorpus | **0.2198** | 0.2139 | 0.2126 | 0.189 |
| NQ | 0.4740 | **0.5188** | 0.5170 | 0.474 |
| HotpotQA | **0.4485** | 0.4444 | 0.4385 | 0.391 |
| FiQA-2018 | **0.1531** | 0.1350 | 0.1337 | 0.112 |
| ArguAna | **0.2654** | 0.1856 | 0.1815 | 0.175 |
| Touche-2020 | 0.2109 | **0.2149** | 0.2104 | 0.131 |
| CQADupStack | **0.2036** | 0.1804 | 0.1807 | 0.153 |
| Quora | **0.2770** | 0.2683 | 0.2567 | 0.248 |
| DBPedia | **0.2846** | 0.2790 | 0.2754 | 0.263 |
| SCIDOCS | **0.1104** | 0.0856 | 0.0835 | 0.077 |
| FEVER | 0.5539 | **0.6124** | 0.6067 | 0.562 |
| Climate-FEVER | **0.1902** | 0.1823 | 0.1844 | 0.148 |
| SciFact | **0.4541** | 0.3969 | 0.4042 | 0.318 |
| **MEAN (15)** | **0.2979** | 0.2871 | 0.2848 | 0.2500 |
| **MEAN (14, zero-shot)** | **0.3044** | 0.2929 | 0.2905 | 0.2552 |

\* MS MARCO is **in-domain** for DPR-style training, not zero-shot; BEIR reports it separately, so both means are given.

- ckpt2 beats the published DPR on **14/15**.
- ckpt39 beats the published DPR on **15/15**.
- ckpt36 beats the published DPR on **15/15**.

Only the public 15 are covered: BioASQ, Signal-1M, TREC-NEWS and Robust04 need licensed access, so the paper column is restricted to the same 15 for an apples-to-apples mean.

---

## 6. Rankify Table 3 (R setting) — Exact Match

Zero-shot RAG, retrieval-only context, n_docs=10, greedy decoding, instruction-tuned `-it` Gemma. Here `public DPR` is a **locally run** control through the identical pipeline, so unlike the BEIR table this is a true like-for-like comparison.

| cell | ckpt2 | ckpt39 | ckpt36 | public DPR |
|---|---|---|---|---|
| NQ / 2B | 19.72 | **21.47** | 21.11 | 20.94 |
| TriviaQA / 2B | 37.73 | **39.25** | 38.95 | 38.35 |
| WebQ / 2B | 11.32 | **12.25** | 11.71 | 12.01 |
| NQ / 9B | 31.94 | **32.74** | 32.38 | 32.66 |
| TriviaQA / 9B | — | 60.97 | **61.06** | 59.96 |
| WebQ / 9B | — | 20.47 | **20.52** | 20.37 |
| **MEAN (4 common cells)** | 25.18 | **26.43** | 26.04 | 25.99 |

Common cells = NQ / 2B, TriviaQA / 2B, WebQ / 2B, NQ / 9B. ckpt2's TriviaQA/9B and WebQ/9B were never generated (that run was stopped), so only the common-cell mean is comparable across all four.

---

## 7. Reading the two retrieval benchmarks together

BEIR and Rankify rank the checkpoints in **opposite** order, and that is the substantive finding:

- **ckpt2** wins BEIR (broad zero-shot retrieval) — its gains concentrate in the domains furthest from Wikipedia QA (SciFact, ArguAna, TREC-COVID, Touche).
- **ckpt39** wins Rankify Table 3 (Wikipedia open-domain QA) and beats ckpt2 on exactly the Wikipedia-native BEIR sets (NQ, FEVER).

The extra training epochs between ckpt2 and ckpt39 bought in-domain accuracy at the cost of out-of-domain transfer. Which checkpoint is 'best' depends entirely on which benchmark you privilege, so both should be reported rather than picking one.


---

## 8. Continued-pretraining objective arms — what was compared

Sections 5–7 compare checkpoints of a single DPR run. This section compares **four DPR runs that differ only in how the BERT encoder was pretrained before DPR training**. Everything downstream is byte-identical: the same 4-dataset multiset (NQ, TriviaQA, WebQuestions, CuratedTREC; 122,892 examples), global batch 256 over 8 GPUs, 40 epochs, and the same BEIR / Rankify evaluation code.

| arm | encoder init | pretraining objective | objective head |
|---|---|---|---|
| reference | bert-base-uncased | none (DPR only) | — |
| arm A | bert-base-uncased | JEPA, `knn_weight=0.0`, 5 000 steps, lr 5e-6 | TransformerPredictor, random init |
| arm B-warm | bert-base-uncased | BERT MLM, 5 000 steps, lr 5e-6 | `cls.predictions`, **restored from BERT** |
| arm B-cold | bert-base-uncased | BERT MLM, 5 000 steps, lr 5e-6 | `cls.predictions`, **random init** |

Arm B-cold exists to remove a confound: arm B-warm resumes a task BERT already has ~1M steps of experience on, while JEPA's predictor starts from scratch. B-cold makes both arms warm-encoder + cold-objective-head, so the comparison isolates the objective rather than prior task experience.

---

## 9. Objective arms — BEIR (NDCG@10, 14 zero-shot sets)

MS MARCO is omitted: it is in-domain for DPR-style training and was not re-run for the arms, so these are the 14 zero-shot sets only. CQADupStack is the unweighted mean of its 12 sub-forums. The `public DPR` column is the same published reference used in section 5 (DPR column of BEIR Table 2, arXiv 2104.08663) — it is a **paper number, not a local re-run**, so unlike the Rankify table in section 10 it is not strictly like-for-like; it is repeated here only so the arms can be read against the same baseline as section 5.

### 9.1 ckpt39 (final epoch)

| dataset | reference | arm A | arm B-warm | arm B-cold | public DPR |
|---|---|---|---|---|---|
| TREC-COVID | 0.3829 | 0.3804 | **0.3954** | 0.3824 | 0.332 |
| NFCorpus | 0.2139 | 0.2135 | **0.2185** | 0.2173 | 0.189 |
| NQ | 0.5188 | **0.5196** | 0.5093 | 0.5171 | 0.474 |
| HotpotQA | 0.4444 | 0.4427 | 0.4468 | **0.4477** | 0.391 |
| FiQA-2018 | 0.1350 | 0.1326 | **0.1355** | 0.1321 | 0.112 |
| ArguAna | 0.1856 | 0.1712 | 0.1765 | **0.1943** | 0.175 |
| Touche-2020 | 0.2149 | 0.2127 | **0.2316** | 0.2167 | 0.131 |
| CQADupStack | 0.1804 | 0.1805 | **0.1931** | 0.1854 | 0.153 |
| Quora | **0.2683** | 0.2175 | 0.2364 | 0.1938 | 0.248 |
| DBPedia | 0.2790 | 0.2776 | 0.2759 | **0.2834** | 0.263 |
| SCIDOCS | 0.0856 | 0.0832 | 0.0897 | **0.0904** | 0.077 |
| FEVER | 0.6124 | **0.6126** | 0.6055 | 0.6121 | 0.562 |
| Climate-FEVER | 0.1823 | **0.1850** | 0.1821 | 0.1806 | 0.148 |
| SciFact | 0.3969 | 0.3837 | **0.4208** | 0.4038 | 0.318 |
| **MEAN (14)** | 0.2929 | 0.2866 | **0.2941** | 0.2898 | 0.2552 |

### 9.2 ckpt2 (the BEIR-peak checkpoint)

| dataset | reference | arm A | arm B-warm | arm B-cold | public DPR |
|---|---|---|---|---|---|
| TREC-COVID | 0.4156 | 0.4165 | **0.4203** | 0.4157 | 0.332 |
| NFCorpus | 0.2198 | 0.2206 | **0.2252** | 0.2183 | 0.189 |
| NQ | **0.4740** | 0.4692 | 0.4721 | 0.4625 | 0.474 |
| HotpotQA | 0.4485 | **0.4486** | 0.4395 | 0.4444 | 0.391 |
| FiQA-2018 | **0.1531** | 0.1510 | 0.1488 | 0.1439 | 0.112 |
| ArguAna | 0.2654 | 0.2484 | 0.2507 | **0.2862** | 0.175 |
| Touche-2020 | 0.2109 | 0.2130 | **0.2216** | 0.2145 | 0.131 |
| CQADupStack | 0.2036 | 0.2036 | 0.2035 | **0.2114** | 0.153 |
| Quora | 0.2770 | 0.1947 | **0.2817** | 0.1798 | 0.248 |
| DBPedia | 0.2846 | **0.2849** | 0.2754 | 0.2811 | 0.263 |
| SCIDOCS | 0.1104 | 0.1082 | 0.1077 | **0.1117** | 0.077 |
| FEVER | 0.5539 | 0.5431 | **0.5621** | 0.5120 | 0.562 |
| Climate-FEVER | 0.1902 | **0.1925** | 0.1802 | 0.1831 | 0.148 |
| SciFact | **0.4541** | 0.4341 | 0.4432 | 0.4531 | 0.318 |
| **MEAN (14)** | **0.3044** | 0.2949 | 0.3023 | 0.2941 | 0.2552 |

Against the published DPR baseline (counts over the 14 sets, MEAN row excluded):

| | ckpt39 | ckpt2 |
|---|---|---|
| reference | 14/14 | 12/14 — loses NQ, FEVER |
| arm A | 12/14 — loses ArguAna, Quora | 11/14 — loses NQ, Quora, FEVER |
| arm B-warm | 13/14 — loses Quora | 13/14 — loses NQ |
| arm B-cold | 13/14 — loses Quora | 11/14 — loses NQ, Quora, FEVER |

Note the ckpt2 losses on NQ and FEVER are shared with the reference and reflect that ckpt2 is an early checkpoint on the two most Wikipedia-native sets, not an arm effect. Quora is the only loss specific to the continued-pretraining arms.

Three observations:

- **ckpt2 > ckpt39 for every arm** (reference −0.0115, arm A −0.0083, B-warm −0.0082, B-cold −0.0043). The early-epoch BEIR peak reported in section 5 is not specific to the reference run; it reproduces under all three pretraining objectives, so it is a property of the DPR schedule rather than of any one encoder.
- **No arm beats the reference at ckpt2**, and only arm B-warm does at ckpt39 (+0.0012). Continued pretraining at this budget does not improve zero-shot retrieval.
- **Quora dominates the arm differences.** It is the only set where the spread exceeds 0.01 at ckpt39 (0.1938–0.2683, a 0.075 range, versus ≤0.037 everywhere else). Both JEPA and cold-head MLM lose heavily on it while warm-head MLM does not — at ckpt2 arm B-warm actually leads (0.2817 vs reference 0.2770). Quora is the only duplicate-**question** set in BEIR (short symmetric query-to-query matching), but the fact that the effect is objective-specific *and* checkpoint-specific is not explained by that alone and is left open.

---

## 10. Objective arms — Rankify Table 3 (R setting), Exact Match

Same pipeline as section 6 (basic-rag, retrieval-only context, `n_docs=10`, greedy, `max_new_tokens=64`), all four arms at **ckpt39**, and `public DPR` again run locally through the identical pipeline. Four instruction-tuned generators instead of two.

| cell | reference | arm A | arm B-warm | arm B-cold | public DPR |
|---|---|---|---|---|---|
| NQ / Gemma-2-9b | 32.74 | **32.83** | 32.52 | 32.60 | 32.66 |
| TriviaQA / Gemma-2-9b | 60.97 | **61.17** | 60.66 | 61.04 | 59.96 |
| WebQ / Gemma-2-9b | 20.47 | 20.57 | 20.72 | **21.11** | 20.37 |
| NQ / Llama-3.1-8B | 24.46 | **24.63** | 24.16 | 24.35 | 22.80 |
| TriviaQA / Llama-3.1-8B | 46.38 | 46.31 | 46.50 | **46.55** | 45.03 |
| WebQ / Llama-3.1-8B | **15.60** | 15.35 | 14.86 | 15.45 | 14.37 |
| NQ / Gemma-2-2b | **21.47** | 21.36 | 21.39 | 21.30 | 20.94 |
| TriviaQA / Gemma-2-2b | 39.25 | **39.28** | 38.99 | 38.90 | 38.35 |
| WebQ / Gemma-2-2b | 12.25 | 12.65 | 12.55 | **12.84** | 12.01 |
| NQ / Llama-3.2-3B | 6.87 | 7.40 | 7.53 | **7.65** | 7.51 |
| TriviaQA / Llama-3.2-3B | 26.09 | 26.25 | 25.90 | **26.29** | 24.86 |
| WebQ / Llama-3.2-3B | 6.45 | **7.09** | 6.99 | 6.64 | 6.40 |
| **MEAN (12 cells)** | 26.08 | **26.24** | 26.06 | 26.23 | 25.44 |

Mean over the three datasets, per generator:

| generator | reference | arm A | arm B-warm | arm B-cold | public DPR |
|---|---|---|---|---|---|
| Gemma-2-9b-it | 38.06 | 38.19 | 37.97 | **38.25** | 37.66 |
| Llama-3.1-8B-Instruct | **28.81** | 28.76 | 28.51 | 28.78 | 27.40 |
| Gemma-2-2b-it | 24.32 | **24.43** | 24.31 | 24.35 | 23.76 |
| Llama-3.2-3B-Instruct | 13.14 | **13.58** | 13.47 | 13.53 | 12.92 |

Retrieval Top-10 for the same runs (generator-independent):

| dataset | reference | arm A | arm B-warm | arm B-cold | public DPR |
|---|---|---|---|---|---|
| NQ | 68.06 | **68.23** | 67.89 | 68.03 | 67.98 |
| TriviaQA | 74.04 | 73.89 | **74.21** | 74.00 | 73.69 |
| WebQ | 64.96 | 64.96 | 64.62 | 65.06 | **65.50** |

---

## 11. Reading the arms together

**The four trained retrievers are not distinguishable on ODQA.** Within each generator the spread across reference / arm A / B-warm / B-cold is 0.12–0.44 EM, and the winner changes with the generator (B-cold on Gemma-9b, reference on Llama-8B, arm A on both small models). Top-10 retrieval differs by ≤0.34 across the four, so the readers are working from near-identical evidence. These are single runs without seed variance; no ordering among the four should be claimed.

**All four beat public DPR**, in 11 of 12 cells and in every per-generator mean, by 0.40–1.41 EM. That is the one separation larger than the noise floor, and it is consistent across generators.

**Generator choice dominates retriever choice by roughly two orders of magnitude.** Gemma-2-9b (38.06) versus Llama-3.2-3B (13.14) is a 25-point swing on identical passages; the retriever moves the same metric by under half a point.

**BEIR and Rankify still disagree, as in section 7.** On BEIR-14 arm A is last at both checkpoints (0.2866 / 0.2949) while on ODQA it has the highest 12-cell mean (26.24). Broad zero-shot retrieval quality and downstream Wikipedia QA accuracy do not track each other here, so both should be reported.

**On the confound arm B-cold was built to test:** B-warm and B-cold differ by −0.0043 (BEIR-14 ckpt39), +0.0082 (BEIR-14 ckpt2, favouring warm) and −0.17 EM (Rankify 12-cell mean, favouring cold). The pretrained MLM head confers no consistent advantage, so arm B-warm's results are not attributable to its head having prior task experience.

---

## 12. Objective arms on the DPR corpus itself — BEIR (NDCG@10, 14 zero-shot sets)

Sections 8–11 pretrain on MS MARCO for 5 000 steps. These three arms instead pretrain on the **DPR multiset passages themselves** (the same 6.26M passages DPR is then trained on, fed as the `[CLS] title [SEP] passage [SEP]` pair DPR's context encoder sees), for one full epoch (48 886 steps, lr 5e-6, batch 128, bert-base-uncased). DPR training and BEIR evaluation are byte-identical to sections 8–11. Only the pretraining objective differs:

| arm | pretraining objective |
|---|---|
| jepa | JEPA, `knn_weight=0.1` (default) |
| knn0 | JEPA, `knn_weight=0.0` |
| mlm | BERT MLM (80/10/10, mask 0.15), `cls.predictions` restored from BERT |

MS MARCO was not evaluated for these arms. `public DPR` is the BEIR Table 2 paper number, as in section 9. Steps differ from sections 8–11 (48 886 vs 5 000), so these arms are comparable to each other, not to the MS MARCO arms.

### 12.1 ckpt39 (final epoch)

| dataset | jepa | knn0 | mlm | public DPR |
|---|---|---|---|---|
| TREC-COVID | **0.4092** | 0.3345 | 0.3582 | 0.332 |
| NFCorpus | 0.2046 | 0.1927 | **0.2141** | 0.189 |
| NQ | 0.5009 | 0.4762 | **0.5152** | 0.474 |
| HotpotQA | 0.4411 | **0.4700** | 0.4451 | 0.391 |
| FiQA-2018 | 0.1262 | 0.1037 | **0.1351** | 0.112 |
| ArguAna | 0.1993 | 0.1665 | **0.2060** | 0.175 |
| Touche-2020 | 0.1567 | 0.1391 | **0.1931** | 0.131 |
| CQADupStack | **0.1865** | 0.1852 | 0.1811 | 0.153 |
| Quora | 0.6344 | **0.6602** | 0.3352 | 0.248 |
| DBPedia | **0.2841** | 0.2823 | 0.2814 | 0.263 |
| SCIDOCS | **0.0874** | 0.0852 | 0.0825 | 0.077 |
| FEVER | 0.5923 | 0.5011 | **0.6035** | 0.562 |
| Climate-FEVER | **0.1892** | 0.1560 | 0.1633 | 0.148 |
| SciFact | 0.4058 | **0.4144** | 0.3647 | 0.318 |
| **MEAN (14)** | **0.3155** | 0.2976 | 0.2913 | 0.2552 |

### 12.2 ckpt2

| dataset | jepa | knn0 | mlm | public DPR |
|---|---|---|---|---|
| TREC-COVID | **0.3812** | 0.3148 | 0.3433 | 0.332 |
| NFCorpus | 0.2054 | 0.1789 | **0.2177** | 0.189 |
| NQ | 0.4211 | 0.3955 | **0.4544** | 0.474 |
| HotpotQA | 0.4392 | **0.4683** | 0.4516 | 0.391 |
| FiQA-2018 | 0.1048 | 0.1118 | **0.1446** | 0.112 |
| ArguAna | **0.2775** | 0.1091 | 0.2760 | 0.175 |
| Touche-2020 | **0.1880** | 0.1265 | 0.1871 | 0.131 |
| CQADupStack | 0.1845 | 0.1898 | **0.2075** | 0.153 |
| Quora | 0.5133 | **0.6082** | 0.3699 | 0.248 |
| DBPedia | 0.2834 | **0.2916** | 0.2868 | 0.263 |
| SCIDOCS | 0.0953 | **0.1089** | 0.1083 | 0.077 |
| FEVER | 0.4818 | 0.3834 | **0.5184** | 0.562 |
| Climate-FEVER | **0.2048** | 0.1543 | 0.1694 | 0.148 |
| SciFact | 0.4292 | **0.4353** | 0.4342 | 0.318 |
| **MEAN (14)** | **0.3007** | 0.2769 | 0.2978 | 0.2552 |

Against the published DPR baseline (counts over the 14 sets):

|  | ckpt39 | ckpt2 |
|---|---|---|
| jepa | 14/14 | 11/14 |
| knn0 | 11/14 | 7/14 |
| mlm | 14/14 | 12/14 |

Observations:

- **jepa ckpt39 is the best retriever of any arm in this report** (MEAN-14 0.3155, +2.4 points over mlm ckpt39 and +1.8 over mlm's better checkpoint), and it beats public DPR on 14/14.

- **Quora carries most of that margin.** Both JEPA arms reach 0.634/0.660 on Quora against mlm's 0.335. Without Quora the 13-set means are jepa 0.2910, mlm 0.2879, knn0 0.2698, so the jepa–mlm gap shrinks to 0.3 points. Dataset wins at ckpt39 are split (jepa 5, mlm 6, knn0 3).

- **The kNN term matters here**: jepa beats knn0 by 1.8 (ckpt39) and 2.4 (ckpt2) points, mostly on FEVER, TREC-COVID, ArguAna and Climate-FEVER.

- **Unlike sections 5 and 9, ckpt39 > ckpt2 for both JEPA arms** (mlm still peaks at ckpt2). Single runs, no seed variance.

---

## 13. Task-adaptive pretraining (TAPT) — Don't Stop Pretraining tasks

Gururangan et al. (ACL 2020) setting: continue pretraining roberta-base on each task's own unlabeled training text, then finetune on the task. Our arms replace their MLM objective with JEPA (`knn_weight=0.1`) or JEPA without the kNN term (`knn0`). Metric is the paper's: test F1 at the best-dev epoch (micro-F1 for CHEMPROT and RCT, macro-F1 otherwise), mean over 5 seeds. Paper columns are their Table 5 (RoBERTa = no continued pretraining, TAPT = their MLM TAPT).

**Run selection:** each cell is the best finished 5-seed run for that task and arm, over every TAPT variant (100-epoch TAPT for the four small tasks; 5-epoch and 100-epoch TAPT for the four large ones) and finetune recipe tried — the same maximise-over-runs procedure as the GLUE sections, so it is optimistic relative to the paper's single reported result. TAPT used lr 5e-6 (the paper used 1e-4 to 5e-4). `final/tapt/<task>.json` lists the selected run, per-seed values and every candidate.

| domain | task | paper RoBERTa | paper TAPT | jepa | knn0 |
|---|---|---|---|---|---|
| BioMed | CHEMPROT | 81.9 | **82.6** | 82.34 ±0.40 | 82.44 ±0.17 |
| BioMed | RCT | 87.2 | **87.7** | 87.15 ±0.03 | 87.16 ±0.13 |
| CS | ACL-ARC | 63.0 | 67.4 | **70.13 ±2.06** | 68.16 ±2.68 |
| CS | SCIERC | 77.3 | 79.3 | 79.34 ±1.05 | **80.07 ±0.66** |
| News | HYPERPARTISAN | 86.6 | 90.4 | **95.15 ±2.73** | 94.51 ±1.31 |
| News | AGNEWS | 93.9 | **94.5** | 93.96 ±0.09 | 93.97 ±0.19 |
| Reviews | HELPFULNESS | 65.1 | 68.5 | 69.81 ±1.06 | **70.05 ±1.02** |
| Reviews | IMDB | 95.0 | 95.5 | **95.52 ±0.07** | 95.42 ±0.10 |
|  | **Mean** | 81.25 | 83.24 | **84.17** | 83.97 |

Selected runs:

| task | jepa | knn0 |
|---|---|---|
| CHEMPROT | 100-epoch TAPT (static mask), finetune bs16 lr2e-5 wd0.01 | 100-epoch TAPT (static mask), finetune bs16 lr2e-5 wd0.01 |
| RCT | 100-epoch TAPT (H100, batch 2048, lr 5e-6, fresh remask) epoch100 | 100-epoch TAPT (H100, batch 2048, lr 5e-6, fresh remask) best |
| ACL-ARC | 100-epoch TAPT (fresh remask), finetune bs16 lr2e-5 wd0.01 | 100-epoch TAPT (static mask), finetune bs16 lr2e-5 wd0.01 |
| SCIERC | 100-epoch TAPT (static mask), finetune bs16 lr2e-5 wd0.01 | 100-epoch TAPT (static mask), finetune bs16 lr2e-5 wd0.01 |
| HYPERPARTISAN | 100-epoch TAPT (static mask), finetune bs16 lr2e-5 wd0.01 | 100-epoch TAPT (fresh remask), finetune bs16 lr2e-5 wd0.01 |
| AGNEWS | 100-epoch TAPT (H100, batch 2048, lr 5e-6, fresh remask) best | 100-epoch TAPT (H100, batch 2048, lr 5e-6, fresh remask) epoch20 |
| HELPFULNESS | 5-epoch TAPT (fresh remask), finetune bs32 lr2e-5 wd0.1 | 5-epoch TAPT (fresh remask), finetune bs32 lr1e-5 wd0.1 |
| IMDB | 100-epoch TAPT (H100, batch 2048, lr 5e-6, fresh remask) epoch60 | 100-epoch TAPT (H100, batch 2048, lr 5e-6, fresh remask) epoch60 |

Observations:

- **Both arms beat the paper's TAPT mean** (jepa 84.17, knn0 83.97 vs 83.24), winning on ACL-ARC, HYPERPARTISAN, HELPFULNESS and SCIERC and tying IMDB. They trail it on CHEMPROT, RCT and AGNEWS, where they sit at or just under the paper's no-TAPT RoBERTa.

- **The margin sits on the smallest test sets.** HYPERPARTISAN (65 test articles) and ACL-ARC (139 test examples) carry 2–3 point seed standard deviations and contribute most of the mean gap; over the other six tasks jepa is level with the paper's TAPT.

- **knn0 ≈ jepa**: knn0's pretraining loss barely moves (0.0974 throughout), so knn0 is close to roberta-base plus our finetuning, yet it matches jepa on most tasks. Much of the gain over the paper therefore likely comes from the finetuning recipe, not the objective. We ran no local no-TAPT baseline to confirm this.

- **Longer TAPT does not help**: on AGNEWS, RCT and IMDB, the 100-epoch TAPT checkpoints at epochs 20/40/60/80/100 all finetune within ~0.2 of the 5-epoch TAPT checkpoint.
