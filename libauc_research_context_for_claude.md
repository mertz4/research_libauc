# Research Context and Preliminary Report Brief for Claude

> **Purpose of this document:** This is a handoff of my research background, dataset, intended experiments, and writing goals. Use it as context when helping me plan or write a **3–5 page preliminary research report**. I am **not trying to write my full thesis yet**. The report should be a practical, credible starting point that can later be expanded into a more rigorous undergraduate thesis.
>
> **Important:** Distinguish what I have *already done* from what I *intend to do*. Do not invent training runs, numerical results, downloaded files, citations, statistical improvements, or conclusions. If experiment outputs are missing, label the work as a **proposed approach / preliminary feasibility study** and describe the planned analysis rather than fabricating findings.

## 1. Project overview

**Working thesis title from my research form:**

> Applying LibAUC with imbalanced data on benign and malignant tumors.

**Working project objective from that form, summarized:**

I want to apply **LibAUC** to a medical binary-classification problem involving **benign and malignant tumors**. The underlying issue is that tumor/lesion datasets can have very uneven distributions between the two classes. I want to explore whether **directly optimizing AUC-oriented losses** produces better classification/ranking behavior than a standard approach, especially for identifying the rare malignant cases. The main evaluation metric is **AUROC**, with **AUPRC / average precision** also of interest. The longer-term goal is to assess whether LibAUC is a useful way to approach class imbalance in a consequential medical prediction task.

**Academic/research setting:**
- This is my undergraduate research / prospective senior thesis work at **Texas A&M University**.
- My advisor is **Professor Tianbao Yang**.
- It is being developed in the **Fall 2026–Spring 2027** period.
- I am interested in ultimately developing a substantive thesis, but the **immediate deliverable is a short report** that explains the problem, describes the dataset, establishes a valid study design, and shows why this approach is worth investigating.

### The central research question

**How does training with a directly AUC-optimized objective using LibAUC compare with conventional binary cross-entropy training for detecting malignant melanoma in a severely imbalanced image dataset?**

Secondary questions, if the work progresses far enough:
- Does the approach help rank malignant cases above benign ones, as measured by AUROC?
- Does it improve rare-class performance as measured by AUPRC and malignant recall/sensitivity?
- Are any improvements consistent across random seeds or class-imbalance settings?
- What tradeoffs appear between AUROC, AUPRC, specificity, precision, thresholds, and calibration?

These are **questions to test**, not conclusions I have already reached.

## 2. Why this is interesting

Ordinary binary classification is often trained with **binary cross-entropy (BCE)**, which assesses predictions sample by sample. With a severe class imbalance, a model can predict the majority class frequently and obtain superficially high **accuracy** while missing important minority-class cases. This is particularly concerning for a malignant-versus-benign task, where false negatives matter.

**AUROC** measures how well a model ranks positive cases above negative ones across discrimination thresholds. Rather than optimizing accuracy at only one threshold, **LibAUC** provides AUC-oriented loss functions and optimization methods intended to optimize ranking-related objectives more directly. **AUPRC** is also important because the positive class is very rare; an AUROC improvement should not automatically be interpreted as improved precision or real-world clinical usefulness.

The core idea is not that LibAUC is guaranteed to be better, but that **direct AUC optimization is a plausible method to evaluate under severe imbalance**. Any report should make that distinction clear.

## 3. The actual dataset

**Selected dataset:** **SIIM-ISIC Melanoma Classification** (2020) on Kaggle.

**Competition identifier:** `siim-isic-melanoma-classification`

**Dataset modality:** Dermoscopic images of skin lesions, with metadata. The task can be framed as predicting **melanoma (malignant) versus non-melanoma / benign** from the images. Avoid implying this dataset covers every type of benign and malignant tumor: it is specifically a **skin-lesion / melanoma** task.

**Previously discussed training-set statistics (confirm against the actual downloaded `train.csv` before reporting them as measured results):**

| Item | Previously identified value |
| --- | ---: |
| Training images / training records | Approximately **33,126** |
| Negative / non-malignant (`target = 0`) | Approximately **32,542** |
| Positive / malignant (`target = 1`) | Approximately **584** |
| Positive-class prevalence | Approximately **1.76%** |
| Original Kaggle competition evaluation focus | **ROC-AUC** |

**Potentially relevant metadata:** the training CSV contains target labels and fields such as `patient_id` (verify actual columns). A **patient-level split**, when possible, is important so related images from one patient do not leak across train and validation/test sets.

**Data-access status / caveat:** I have been working on obtaining the **entire Kaggle competition dataset**, not just the CSV or a small subset. Do **not** assume the complete image download and extraction are finished; check the actual directory and logs before saying the data is ready. I have experienced issues with Kaggle credentials/versions, downloads on the HPC cluster, network/proxy connectivity, and memory limits.

## 4. Computing environment and download history

I am using **Texas A&M High Performance Research Computing (HPRC)**, specifically the **Grace** cluster, for data storage and eventual model training. I have used **Slurm** batch jobs instead of relying on an interactive session for long-running downloads.

**Most recently discussed working location:**

```text
/scratch/group/optmai/kademertins/melanoma
```

There may also be relevant files under the shared `optmai` group allocation; do not assume the directory above is fully populated. Before using it, verify with filesystem commands and inspect the data layout. Avoid writing large files into my small home quota if scratch space is available.

**Known setup issues encountered:**
- Kaggle API credential/configuration problems earlier in setup.
- An attempted download was killed by Slurm for exceeding available memory.
- Other attempts encountered Kaggle API version or cluster network/proxy failures.
- At one point a download job was running; later there were no jobs shown. That does **not** establish that the download succeeded.

If helping with execution, first check dataset presence, `train.csv`, image archives/directories, file counts, Slurm logs, and available scratch space. Do not claim I have run models simply because the data source and cluster were identified.

## 5. Tools, methods, and comparisons under consideration

**Research library:** **LibAUC** (Python/deep-learning library for AUC-oriented optimization).

**Main comparison / minimum viable experiment:**

1. **Baseline:** Train a binary skin-lesion image classifier using **binary cross-entropy (BCE)**.
2. **LibAUC experiment:** Train the **same architecture** with an AUC-oriented objective, initially using **`AUCMLoss` + `PESG`** with an appropriate **`DualSampler`** configuration.
3. Keep data splits, image sizes, preprocessing/augmentations, model backbone, initialization approach, compute budget, and evaluation procedure as comparable as practicable.
4. Evaluate both methods on a **held-out, preferably patient-separated** test set using **AUROC** and **AUPRC**, plus malignant recall/sensitivity, specificity, precision, F1, and confusion matrices at explicitly defined thresholds when appropriate.
5. Explain whether results support or do not support the original hypothesis. Report uncertainty/limitations instead of overselling a single run.

**Other methods we have discussed as possible extensions, *not mandatory for the initial report*:**
- **`CompositionalAUCLoss` / `PDSCA`**, to investigate combining classification and ranking objectives.
- **`AveragePrecisionLoss` / `SOAP`** as a possible AUPRC-oriented objective (implementation may require per-sample indices).
- **Focal loss** as another imbalance-sensitive baseline.
- **Partial AUC / pAUC**, particularly behavior at low false-positive rates.
- LibAUC sampler designs including **DualSampler**, and the broader ideas of **TriSampler** and **DistributedSampler** if they become relevant.
- Multiple training seeds, deliberately varied imbalance ratios, calibration, and analysis across decision thresholds.

The exact LibAUC API/version and compatibility between loss, sampler, and optimizer **must be checked against the installed library**; these are candidate methods, not a claim that a working pipeline already exists.

### Experimental validity principles

- Avoid data leakage, especially across patients.
- Do not compare methods using different architectures or favorable splits without explaining the difference.
- Record class counts for every split and note that minority examples are limited.
- Accuracy alone is insufficient evidence on this dataset.
- Report **both** AUROC and AUPRC if possible, and identify which quantity a loss directly optimizes versus which metrics are used for evaluation.
- Consider variability from seeds, class proportions, class weighting/sampling, and any threshold selection.
- Do not make clinical-effectiveness or real-world deployment claims from this dataset alone.

## 6. What I want RIGHT NOW: a 3–5 page preliminary report

I do **not** need a full thesis, a dense theoretical paper, or a highly technical derivation right now. I want a **clear, reasonably polished 3–5 page report** that can **precede** the thesis and potentially be **revised or expanded into thesis chapters later**.

The immediate goal is to establish:
1. **What the research problem is:** extreme imbalance in malignant-vs-benign skin-lesion data.
2. **Why it is worth studying:** majority-class accuracy can hide poor rare-class detection, and AUC-style optimization offers a different training objective.
3. **What dataset I selected:** SIIM-ISIC 2020, including size, labels, prevalence, and why it is appropriate.
4. **What I plan to test:** a fair BCE baseline against a LibAUC/AUC-optimization approach.
5. **How I will decide whether it works:** held-out AUROC, AUPRC, recall/sensitivity, and other relevant metrics.
6. **Why it is a plausible thesis direction:** a concrete, reproducible, testable comparison with potential to support more extensive experiments later.

### Desired tone and depth

- Write at the level of a **thoughtful undergraduate research progress / preliminary study report**.
- Be understandable to an engineering/computer science reader who might not know LibAUC in detail.
- Keep technical vocabulary accurate but **explain it in plain language**.
- Favor tight prose, meaningful headings, and perhaps **one small table or figure** (class distribution or experiment comparison) rather than pages of equations.
- It should feel **substantive and evidence-aware**, not padded, but should **not read like a completed thesis or journal paper**.
- Use approximately **3–5 pages of body text**, depending on formatting, and a short references section where appropriate.
- Keep the distinction clear between **observed data properties**, **planned methods**, and **actual experimental findings**.

### Suggested report structure

1. **Title and short overview / introduction** (about ½ page)  
   State the clinical-data classification problem and core research question.
2. **Background and motivation** (about ¾–1 page)  
   Explain class imbalance, why accuracy can mislead, AUROC versus AUPRC, and why LibAUC is relevant without advanced mathematics.
3. **Dataset description and initial feasibility** (about ¾–1 page)  
   Describe SIIM-ISIC, target labels, approximate counts/imbalance, patient-split concern, and the feasibility of using Grace. If a verified dataset audit is available, include it; otherwise label figures as public dataset descriptions or previously identified counts pending verification.
4. **Proposed methodology / comparison** (about 1–1¼ pages)  
   Explain BCE baseline versus a LibAUC training approach, intended controls, split strategy, evaluation metrics, and practical implementation plan.
5. **Expected insights, limitations, and next steps** (about ½–1 page)  
   Describe what evidence would validate the research direction, limitations of the data and the preliminary stage, and how this could develop into a larger thesis.
6. **References**  
   Use accurate, verifiable citations to the dataset/competition and LibAUC documentation or literature if available. Do not fabricate references or page numbers.

**If actual model results are not yet available:** Prefer a title such as *“Evaluating AUC-Oriented Training for Imbalanced Melanoma Classification: A Preliminary Study Plan.”* Do not insert an invented Results section. A short **dataset overview, proposed experiments, and research feasibility** is a completely acceptable first report.

**If real experimental results become available later:** Replace part of the methodology/next-steps discussion with a compact results table, interpretation, and limitations. The same report can evolve organically toward a thesis.

## 7. How I want you (Claude) to help

Please use this brief as the project's background and keep the current stage in mind. A helpful workflow is:

1. **First** inspect whatever actual dataset files, scripts, logs, or experimental outputs I provide. Confirm which steps are done and which remain plans.
2. **Then** produce a clear **3–5 page draft report** following the outline above, based only on verifiable dataset properties and work completed so far. If needed, use explicit placeholders like `[Insert verified train-set counts]` or `[Results pending]` instead of made-up claims.
3. Identify a **small, feasible first experiment** that will provide evidence for the report: dataset audit → patient-level split → BCE baseline → AUCMLoss/PESG comparison → AUROC/AUPRC evaluation.
4. When reviewing code or data, prioritize reproducibility and fair comparison over adding many complex loss functions at once.
5. Make the report **reusable**: I should be able to expand the background, experiments, and analysis into a thesis later, but I do not want to spend time on a complete thesis structure now.
6. Preserve my research intent rather than changing the topic to general medical AI. **The specific focus is LibAUC, severe class imbalance, and benign-versus-malignant melanoma classification using SIIM-ISIC.**

## 8. Useful references to locate or verify

- **Kaggle dataset / competition:** `https://www.kaggle.com/competitions/siim-isic-melanoma-classification`
- **LibAUC project:** `https://github.com/Optimization-AI/LibAUC`

These are starting points; validate any specific API claims, published results, or reported dataset statistics before citing them as evidence in the final report.

---

### One-sentence summary

I am doing undergraduate research with Professor Tianbao Yang on whether **LibAUC's AUC-oriented training can better handle the severe malignant-versus-benign class imbalance in the SIIM-ISIC 2020 melanoma dataset than standard cross-entropy**, and I currently need a **concise 3–5 page preliminary report** describing the problem, dataset, rationale, and fair experimental plan—not a full thesis or fabricated experimental results.
