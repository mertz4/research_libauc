# LibAUC Melanoma Project — Study Notes

*Notes from a planning/learning session (Nov 2026). Covers report structure, dataset framing, and the evaluation metrics behind the project. Items marked ⚠️ still need verification.*

---

## Part 1 — The Report

### Working title
**Deep AUC Maximization for Imbalanced Melanoma Classification: A Preliminary Study**

- "Deep AUC Maximization" is the term used by my advisor's group (Yuan, Yan, Sonka & Yang, ICCV 2021), so the title places the work in that line of research.
- Drop the subtitle later and it can become the thesis title.
- Alternatives considered:
  - *Evaluating AUC-Oriented Training for Imbalanced Melanoma Classification: A Preliminary Study Plan* (best if there are no results yet)
  - *Directly Optimizing AUC for Rare Malignant Cases: LibAUC on SIIM-ISIC 2020*
  - *Beyond Cross-Entropy: AUC Maximization for Benign vs. Malignant Skin Lesion Classification* (save for the thesis, since "Beyond" implies a result)
- Put "LibAUC" in the abstract rather than the title. The library is the tool; the research idea is AUC maximization.

### Structure (3–5 pages)

1. **Introduction and Research Question (~½ page)**
   Introduces the problem of detecting rare malignant cases in imbalanced medical image data. States the main question: does AUC-optimized training with LibAUC outperform standard binary cross-entropy for melanoma detection?

2. **Background (~¾ page)**
   Why accuracy is misleading under severe class imbalance, the difference between AUROC and AUPRC, and the idea behind deep AUC maximization, with reference to prior work (Yuan et al., ICCV 2021).
   - Hook: at 1.76% prevalence, predicting "benign" every time gets **~98.2% accuracy** and catches **zero** melanomas.

3. **Dataset (~¾ page)**
   The SIIM-ISIC 2020 melanoma dataset: labels, class distribution (~1.8% malignant), and why splitting by patient is needed to avoid data leakage. Include a class-distribution table.

4. **Proposed Methodology (~1¼ pages): the most important section**
   - **Split:** Kaggle's test labels are hidden, so the held-out test set must come out of `train.csv`. Use a patient-grouped, stratified split (e.g., `StratifiedGroupKFold` on `patient_id`).
   - **Two conditions:** BCE baseline vs. LibAUC (`AUCMLoss` + `PESG` + `DualSampler`).
   - **Fairness controls:** same backbone, preprocessing, compute budget, and starting weights. LibAUC's usual recipe is CE pretraining followed by AUCM fine-tuning, so decide up front how the baseline gets matching compute and initialization. This is the most likely place for a reviewer to call the comparison unfair.
   - **Metrics:** AUROC and AUPRC as primary; sensitivity, specificity, precision, and F1 at a threshold **chosen on validation**.
   - **Seeds and success criteria:** e.g., mean ± SD over 3–5 seeds. Define what counts as "better" before running.
   - Include an experiment-conditions table.

5. **Current Status and Feasibility (~¼–½ page)**
   Done-vs-planned table (data access, audit, split, baseline, LibAUC run). Mention the TAMU Grace HPRC setup in one line; the download troubles don't belong in the report.

6. **Limitations and Next Steps (~½–¾ page)**
   Dataset limitations, no clinical claims, a short timeline, and possible thesis extensions (AUPRC-oriented losses like SOAP, focal loss, partial AUC). Keep the extensions to one or two sentences.

7. **References:** Kaggle competition, LibAUC GitHub repo, ICCV 2021 paper. Use BibTeX from the start so references carry over to the thesis.

Tables: class distribution (Section 3), experiment conditions (Section 4), status (Section 5). No figures needed yet. Later: ROC and PR curves of BCE vs. LibAUC side by side.

### LaTeX template
- Pick: the **arXiv preprint style** (Overleaf: "Style and Template for Preprints (arXiv, bio-arXiv)"; GitHub: `kourgeorge/arxiv-style`). Single column, clean, no conference branding.
- Alternatives: NeurIPS style with the `preprint` option, or plain `article` with `geometry`, `microtype`, `booktabs`, `hyperref`, and `natbib`.
- Avoid two-column formats (IEEEtran/ACM); they make a short report look cramped.
- Keep sections in separate files (`\input{sections/intro}`) so the content can move into the official URS thesis template later.

---

## Part 2 — Dataset Framing

### "Tumors" vs. "skin lesions"
- Melanoma **is** a malignant tumor; most benign cases are nevi (moles), which **are** benign tumors. So "tumors" isn't wrong.
- "Skin lesion" is more **accurate**:
  - Not every benign case is a tumor (e.g., lentigos and lichenoid keratoses; many are listed as "unknown" diagnosis ⚠️ check the `diagnosis` column in `train.csv`).
  - It matches the field: ISIC is the International *Skin Imaging* Collaboration, and the literature says "skin lesion classification."
  - It scopes the work honestly; "tumors" sounds like it could mean breast, lung, or brain.
- In the report, say **"melanoma (malignant) vs. benign skin lesions"**, and note once in the intro that this is a case of the broader benign-vs-malignant tumor problem from the original proposal.

### URS topic wording
- I found no official URS rule on title or topic changes. The 2026–27 proposal template says that if the scope changes, it "must be addressed with my Faculty Advisor and the office of Research Compliance & Biosafety prior to implementation." Faculty advisors approve research content; URS staff don't.
- This is a narrowing of wording, not a topic change. Steps: tell Prof. Yang; optionally confirm with ugr@tamu.edu; ask Prof. Yang whether the compliance office needs to know (public de-identified data usually isn't human-subjects research).

### Other benign/malignant datasets (counts approximate ⚠️)
| Dataset | Modality | Approx. size / split | Imbalance |
|---|---|---|---|
| RSNA Screening Mammography (Kaggle 2022–23) | Mammograms | ~54.7k images, ~1.2k cancer | ~2%, severe; best match but huge download |
| CBIS-DDSM | Mammograms | ~3k lesions | Mild (~60/40) |
| BreakHis | Histopathology | ~7.9k images (~2.5k benign / ~5.4k malignant) | Reversed |
| BUSI | Ultrasound | ~780 (437 benign / 210 malignant / 133 normal) | Mild |
| WDBC | Tabular | 569 (357 / 212) | Mild; sanity check only |
| LIDC-IDRI | Lung CT | ~1k patients, malignancy rated 1–5 | Depends on cutoff |
| CMMD | Mammograms | ~1.8k patients | Moderate |

- Most of these are only mildly imbalanced, so the imbalance would have to be created by subsampling. **Keep SIIM-ISIC 2020 as the main dataset**; RSNA or CBIS-DDSM is a possible second dataset for the thesis.

### LibAUC and melanoma
- LibAUC has a melanoma loader: `libauc/datasets/melanoma.py`.
- The README links a tutorial: `examples/08_Optimizing_AUROC_Loss_with_DenseNet121_on_Melanoma.ipynb`.
- The ICCV 2021 paper (Yuan, Yan, Sonka, Yang; arXiv 2012.03173) evaluated on four medical tasks, including "classification of images of skin lesions for identifying melanoma."
- ⚠️ Not confirmed that it's SIIM-ISIC 2020. Check the paper's experiments section, `melanoma.py`, or notebook 08, or ask Prof. Yang.
- LibAUC also has `breastcancer.py` (dataset not checked).
- Why this matters: the project builds on my advisor's work, existing code may save setup time, and there may be published numbers to compare against.

---

## Part 3 — Metrics

### Threshold
The model outputs a **score** (0–1), not a label. The **threshold** turns the score into a decision:

> score ≥ threshold → malignant; score < threshold → benign

| Lesion | Score | Truth |
|---|---|---|
| A | 0.92 | malignant |
| B | 0.75 | benign |
| C | 0.60 | malignant |
| D | 0.40 | benign |
| E | 0.20 | benign |
| F | 0.05 | benign |

- Threshold 0.5 → flags A, B, C: both cancers caught, 1 false alarm.
- Threshold 0.8 → flags A only: no false alarms, misses C.
- Threshold 0.1 → flags everything except F: everything caught, 3 false alarms.

The model is the same each time; only the tradeoff changes.

### Precision, recall, F1
- **Precision:** of the cases flagged malignant, how many really are? (Few false alarms.)
- **Recall / sensitivity / TPR:** of the truly malignant cases, how many were caught? (Few misses.)
- **F1** = 2 × (P × R) / (P + R), the harmonic mean, so it's only high if both are high.
  - Example: P = 0.40, R = 0.70 → F1 ≈ 0.51.
- F1 is computed at **one threshold**, and it **ignores true negatives** (useful when 98% of cases are benign).

### F1 vs. AUPRC (easy to mix up)
- Both use precision and recall.
- **F1 is one point** on the precision–recall curve, the one at the chosen threshold.
- **AUPRC is the area under the whole curve**, across all thresholds.
- F1 scores the final yes/no decisions; AUPRC scores how good the model's scores are overall.

### ROC curve and AUROC
- **Plot:** TPR (y) vs. FPR (x) as the threshold sweeps from high to low.
  - Bottom-left = nothing flagged; top-right = everything flagged; top-left = ideal.
  - The diagonal is random guessing.
- **AUROC** = area under the curve. 1.0 is perfect, 0.5 is random.
- **Pairwise meaning:** the probability that a random malignant case scores higher than a random benign case. Equivalently, form every (malignant, benign) pair and count the fraction ordered correctly.
- A perfect model's ROC curve goes straight up the left edge, then along the top.
- **This is what LibAUC's `AUCMLoss` optimizes.**

### PR curve and AUPRC
- **Plot:** precision (y) vs. recall (x) as the threshold sweeps.
  - Left: high threshold, high precision, low recall. Moving right catches more cancers but adds false alarms, so precision drops.
- **The random baseline is the prevalence (~0.018 here), not 0.5.** So AUPRC values like 0.2–0.4 can be decent on rare-class problems.
- **A perfect model** gives a flat line at precision = 1.0 from recall 0 to 1 (all cancers caught, no false positives), dropping straight down only after recall reaches 1. This requires every malignant case to score above every benign case.
- In practice, compare **how long each model stays high before precision falls**.

### The imbalance trap
A model can have a high AUROC and still poor precision, because a small false-positive *rate* applied to many benign cases is a large *number* of false alarms.

Example: 5,000 test images (~90 malignant), 80% recall, 5% FPR:
- ~72 true positives
- ~245 false positives
- precision ≈ 23%

That's why AUPRC is reported alongside AUROC.

### Choosing a threshold
- Plots don't show threshold values, but the code does:
  ```python
  from sklearn.metrics import roc_curve, precision_recall_curve
  fpr, tpr, roc_thr = roc_curve(y_true, scores)
  precision, recall, pr_thr = precision_recall_curve(y_true, scores)
  ```
- What counts as "best" depends on the goal:
  | Goal | Method | Curve |
  |---|---|---|
  | Balance precision/recall | Maximize F1 | PR |
  | Best separation | Maximize TPR − FPR (Youden's J) | ROC |
  | Don't miss cancers | Lowest threshold with ≥ 90–95% sensitivity | Either |
- For melanoma, setting a sensitivity target is often the most meaningful choice.
- **Rule: choose the threshold on validation, report on test.** Choosing it on the test set inflates the results.

### Metric hierarchy for this study
1. **Primary:** AUROC (what AUCMLoss optimizes) and AUPRC (the harder rare-class test)
2. **Secondary:** sensitivity, specificity, precision, F1 at the validation-chosen threshold
3. **Extension:** partial AUC (pAUC)
4. **Not relevant:** NDCG/DCG

---

## Part 4 — Partial AUC (pAUC)

### What it is
AUROC restricted to one region, usually **FPR ≤ 0.1**, since a hospital can't use a model with a high false-positive rate.

### Why not just use regular AUROC? Reason 1: scoring
FPR from 0.1 to 1.0 is **90% of the x-axis**, so most of the AUROC comes from thresholds that would never be used. Two models can have the same AUROC but behave very differently at low FPR.

Pairwise example: 2 malignant, 6 benign, so 12 pairs.

**Model A**: errors involve the top benign case
| 0.95 | 0.90 | 0.80 | 0.50 | 0.40 | 0.30 | 0.20 | 0.10 |
|---|---|---|---|---|---|---|---|
| **B1** | M1 | M2 | B2 | B3 | B4 | B5 | B6 |

B1 beats both cancers: 2 wrong pairs → AUROC = 10/12 = 0.83. **The first case flagged is a false alarm.**

**Model B**: errors in the middle
| 0.90 | 0.70 | 0.65 | 0.60 | 0.30 | 0.20 | 0.10 | 0.05 |
|---|---|---|---|---|---|---|---|
| M1 | **B2** | **B3** | M2 | B1 | B4 | B5 | B6 |

M2 loses to B2 and B3: 2 wrong pairs → AUROC = 10/12 = 0.83. **At threshold 0.8, M1 is caught with zero false positives.**

Same AUROC, but B is clearly better clinically. pAUC detects this because it only counts pairs with the highest-scoring benign cases.

### Reason 2: training
- **Hard negatives:** benign lesions that look malignant and score high. They cause false alarms at strict thresholds.
- **Easy negatives:** obviously benign, scored low, already ranked below every melanoma.
- **AUCMLoss (full AUC)** weights all pairs equally, so it can't tell A from B. An update can fix many middle pairs while hurting a few top pairs: AUROC goes up, low-FPR performance goes down.
- **A pAUC loss**, each batch: takes the current scores of the benign cases → keeps only the top ones (e.g., top 10%) → trains on pairs with those only. The model focuses on the confusing lesions.

### At SIIM-ISIC scale
- 584 malignant × 32,542 benign ≈ **19 million pairs**.
- Top 10% of benign ≈ **3,250 lesions**, the ones that matter at FPR ≤ 10%.
- The other ~29,000 benign lesions (~90% of pairs) are mostly easy. Full AUROC mostly measures how well the model handles easy cases.

> **One sentence:** full AUROC weights every pair equally, so it can't tell whether the model's mistakes are on hard cases or easy ones; pAUC looks only at the hard benign cases, both for scoring and, with a pAUC loss, for training.

### Practical note
- Raw pAUC over FPR ≤ 0.1 has a maximum of 0.1, so it's often reported **normalized**.
- `roc_auc_score(y, scores, max_fpr=0.1)` returns a standardized version (McClish correction, where 0.5 = random). State which version is reported.

### Why doesn't everyone use pAUC?
The idea of focusing on hard examples **is** common: hard negative mining (object detection, face recognition) and focal loss (down-weights easy examples). pAUC losses aren't the default because:

1. **Less training signal:** most negative pairs are dropped (see below).
2. **Hard negatives can be label noise:** they're the cases most likely to be ambiguous or mislabeled (many SIIM-ISIC benign labels aren't biopsy-confirmed).
3. **"Hard" is meaningless early in training:** scores start out essentially random, so pAUC usually needs a **warm start** (pretrain with CE or full AUC first).
4. **The FPR range must be chosen** (10%? 5%? 1%?) and depends on how the model will be used.
5. **Optimization is harder and newer:** selecting the top-scoring negatives is a sorting step, which isn't smooth. Efficient algorithms are recent (~2021–22), much of it from the LibAUC team.
6. **The benefit isn't guaranteed:** it can lower full AUROC, and gains depend on the dataset.

### "Memorizing noise", explained
- **Case A, wrong label** (actually melanoma, labeled benign): it scores high, gets treated as a hard negative, and is pushed down. The model learns that melanoma-like features mean benign, which hurts **real melanomas that look similar**, including ones in the test set.
- **Case B, correct label but truly ambiguous**: no real visual feature separates it, so the model latches onto anything unique to that image (lighting, ruler marks, hair, camera, patient-specific skin). It fits this image but learns nothing that generalizes.
- In both cases the training loss improves while real performance gets worse: **overfitting**, concentrated on the riskiest cases.

### "Less training signal", explained
All data is still used; every image goes through the model. What shrinks is **how many comparisons count in each update**.

Batch: 2 melanomas, 100 benign
| Loss | Pairs counted | Based on |
|---|---|---|
| Full AUC | 2 × 100 = 200 | All 100 benign |
| pAUC (top 10%) | 2 × 10 = 20 | Only the 10 highest-scoring benign |

- **Noisier updates:** each step depends on a few images (like estimating an average from 10 samples instead of 100).
- **A moving target:** which images are "hard" changes as the model learns.
- **Noise has more influence:** 2 mislabeled cases among 10 hard negatives make up 20% of the update, versus 2% with full AUC. Problems 1 and 2 make each other worse.
- Mitigations: LibAUC samplers (DualSampler oversamples positives) and warm-starting.

### Thesis angle
> *Does focusing training on hard negatives (pAUC optimization) improve melanoma detection at clinically usable false-positive rates, compared with full AUROC optimization and cross-entropy?*

- Preliminary report: BCE vs. AUCMLoss; report AUROC, AUPRC, **and pAUC as a metric**.
- Thesis: add a pAUC loss starting from the AUCM model.
- Analysis idea: **look at the hard negatives.** Which benign lesions get confused with melanoma: nevi, or "unknown" diagnoses? This shows whether pAUC training learns real features or fits noisy labels.

---

## Part 5 — Gain, DCG, NDCG (and why they're out of scope)

These are **ranking metrics for recommendation and search** (e.g., LibAUC's `movielens.py`), not binary classification.

- **Gain:** how relevant one item is (binary 0/1, or graded, e.g., star ratings).
- **DCG** = Σ gainᵢ / log₂(i + 1): relevance added down the list, with lower positions counting less (rank 1 = full credit, rank 2 ≈ 0.63, rank 10 ≈ 0.29).
- **NDCG** = DCG / best possible DCG, on a 0–1 scale. **NDCG@k** looks only at the top k.
- LibAUC has NDCG losses, the same idea as AUCMLoss: optimize the metric you're evaluated on.

| Metric | Emphasizes |
|---|---|
| AUROC | Overall ordering; every pair equal |
| AUPRC | Ordering, weighted toward the top of the list |
| NDCG@k | The very top, with a steep discount by position |

**Not relevant to this project.** At most, one line in the Background:
> "LibAUC supports a family of ranking-based objectives (AUROC, AUPRC, partial AUC, and NDCG for recommendation); this study focuses on AUROC and AUPRC, which are the standard metrics for imbalanced binary classification."

---

## ⚠️ To-Do / Verify
- [ ] Confirm dataset counts on Grace: `train.csv` row count, `target` value counts, number of unique `patient_id`s, `diagnosis` breakdown.
- [ ] Confirm the full image download and extraction finished.
- [ ] Check which melanoma dataset the ICCV 2021 paper and `melanoma.py` / notebook 08 use.
- [ ] Check the installed LibAUC version's API for `AUCMLoss`, `PESG`, and `DualSampler`.
- [ ] Tell Prof. Yang about the wording change; optionally email ugr@tamu.edu.
- [ ] Update the "Current Status" section of the report before sending the outline.
