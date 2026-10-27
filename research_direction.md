# Research Directions: Evidence-Aligned Supervision for Medical MLLMs

Last updated: 2026-10-24 (after advisor meeting). Sources: `med_mllm_gap_notes.pdf` (Oct 18–19),
`med_mllm_overlap_notes.pdf` (Oct 19–20), Radiopaedia audit, meeting notes.
The previous version of this file (evidence–supervision mismatch note) is kept in §6 and backed up at
`/data/aofei/output/ContextMLLM/research_direction.orig.md`.

## 0. One-paragraph thesis

Med-MLLMs are trained to maximize p(T | I), but the target text T is written by an author who also had
context C: history, labs, prior studies, pathology, and the rest of the article. Text that the image cannot
explain still has to be predicted, so the model explains it with the language prior. This is the
**attribution gap** (the "missing context" term I(Y; C | I) in the loss decomposition). Its other faces are
hallucination, fabricated comparisons to priors, and weak perception of unreported findings. **Main direction
(agreed in the meeting): close the gap at the source by giving training the context the author had, and
decide how much context to add by measuring what the model can and cannot get from the image.**

## 1. Main direction: context-expanded training on PubMed (selected)

| Component | Meeting idea | Concrete form |
|---|---|---|
| 1a. Source the context | "pre-extract the context in PubMed" | Map each PubMedVision image to its PMC article; pull the real caption, the paragraphs that cite the figure, the abstract, and case details. **Feasibility confirmed, see §1.1.** |
| 1b. Uncertainty-gated expansion | "measure uncertainty during training; if uncertain, retrieve more context from the source article" | Per-claim gate: add context only where the target is unsupported by the image *and* supported by the context. Raw uncertainty alone is the wrong gate (§2 of the exploration doc). |
| 1c. Knowledge context | "include radiology / diagnosis guidelines" | Retrieve guidelines only for *inferential* claims (findings → diagnosis/management). Guidelines verify inference, not perception (overlap notes §4.2). |
| 1d. Not simply pretraining | "this process is not simply pretraining" | A model-in-the-loop data loop: the current model decides what context each sample needs, so the dose falls as perception improves (self-paced). This naturally extends to a learned "request context" action, a form of **self-awareness**. |

Full exploration: [Plans/context_expansion/README.md](Plans/context_expansion/README.md).
Post-training plan, literature review and the first isolation experiment (2026-10-24):
[Plans/context_expansion/post_training_plan.md](Plans/context_expansion/post_training_plan.md). Headline: raw uncertainty
does not separate needs-context from needs-perception (AUROC 0.48–0.61). Content typing is easy but
textual (probes unchanged without the image), so image-specific self-awareness must be taught. Redacted
context removes 37–53% of the visual leak. **Working axis (2026-10-25): visual context** (sentences that restate
findings, a shortcut) **vs. non-visual information context** (history, labs, pathology, outcome, priors: what a
request should return). The overnight P1b–P6 run tests this split (plan §4).

### 1.1 Feasibility result: PubMedVision → PMC articles (done 2026-10-24)

PubMedVision records carry no PMCID (images are `pmc_N_k.jpg`), but the source articles can be recovered:

- **Offline exact caption match against PMC-OA** (1.6M captions with PMCIDs) maps **317,314 / 636,092
  (49.9%)** captions uniquely to **136,748 articles**; 1,358 are ambiguous.
- **Europe PMC phrase search** on a random 200: 172 unique hits (86%). On 93 cases where both methods
  return a unique ID, **they agree in all 93**. Together they cover **178/200 (89%)**.
- **Full text (NCBI BioC API)** for 165 of the 178 mapped articles (the other 13 were rate-limited):
  - 115 are true figure captions; **80 (70%) have ≥1 body paragraph citing that figure**
    (median ≈700 characters of new context per image).
  - **~26% of all PubMedVision "Original_Caption" fields are not captions** but body paragraphs mentioning
    a figure (PMC-Inline source); 40 of 50 such cases were found verbatim in the article. For these, the
    missing context is the real caption plus the surrounding paragraphs.
- Scale-up path: map the remaining ~50% with phrase search (~320k queries, roughly half a day at the rate
  limit). Then get full text for ~250k articles from the PMC OA bulk XML packages, not the per-article API.

Scripts: `experiments/context_expansion/`. Outputs: `/data/aofei/output/ContextMLLM/`.

## 2. Second testbed: longitudinal context from MIMIC-CXR

- Radiologists write MIMIC-CXR reports *with the prior studies open*. "Unchanged from prior" or "interval
  improvement" are context-derived claims, and models fabricate them when priors are absent.
- Sample previous studies for the same patient (one or more priors, plus prior reports and indication) as C.
  The gate from 1b decides how many priors a sample needs.
- Cleaner than PubMed: C is well defined (time-ordered studies), and claims can be tagged with existing tools
  (RadGraph, CheXbert, sentence-level comparison detection).
- Bridges to forecasting (§4.1): the same data with the latest study held out becomes a prediction task.

## 3. Supporting / measurement directions (from the gap and overlap notes)

These are the tools that make the main direction measurable. Several are prerequisites.

| # | Direction | Role for the main direction | Status |
|---|---|---|---|
| 3a | **Groundability tagging**: classify target claims as image-derivable / context-derived / reader-specific | Defines which claims need context and which must stay image-only | Planned (gap notes §6.1) |
| 3b | **Pointwise V-information (PVI)**: log-lik gain from adding the image or the context | The gate signal for 1b; per-source "text groundedness" estimates | Pilot on 12 facts: paraphrased context gains ≈0.4–0.46 nats (`experiments/information_gap`) |
| 3c | **Oracle ladder / layer-wise probing** (image → own description → oracle findings → +context) | Locates the perception vs. missing-context share of error (Δperc vs. Δctx); tells us how much context expansion *can* buy | Planned (overlap notes §5.1) |
| 3d | **Counterfactual evaluation** (lesion edit, prior swap, indication swap) | Primary metric that context training did not make the model stop looking | Planned |
| 3e | **Step-level PVI audit of reasoning datasets** (ChestX-Reasoner, Chiron-o1) | Shows current reasoning data turns context into fake perception, which motivates provenance | Planned |
| 3f | **Provenance-typed reasoning** (seen / from context / inferred, each verified differently) | Post-training counterpart: the model states where each claim comes from | Planned (overlap notes §6) |
| 3g | **Counterfactual preference optimization** (clinically meaningful edits, not random corruption) | Fixes the "use" gap (shortcut redundancy) that context expansion does not address | Planned |
| 3h | **Typed uncertainty + conformal risk control** | Deployment guarantees; gap-aware scores reduce abstention at equal risk | Theory sketched (gap notes §4) |

## 4. Future directions (lower priority, noted in meeting)

1. **Forecasting**: predict the next study or outcome from longitudinal images and context (MIMIC-CXR priors,
   PubMed case series with follow-up). Tests whether the model learned disease dynamics rather than templates.
2. **Post-training on Radiopaedia**: use cases with images, history, and final diagnosis to supervise
   evidence-aligned reasoning: what the image supports, what extra information settles the diagnosis, and how
   the conclusion changes when it arrives. Audit work exists in `Plans/radiopaedia_audit/`.

## 5. How the pieces fit (loss decomposition view)

E[−log pθ(Y | Z)] = H(Y | I, C) [intrinsic] + I(Y; C | I) [missing context] + I(Y; I | Z) [lost in encoding]
+ KL [present but not extracted].

- **Main direction (1, 2)** attacks *missing context* by moving C into the input during training.
- **Gate (1b, 3b)** must avoid moving image-derivable content into C, or it converts the *not-extracted* term
  into a text shortcut and perception gets worse.
- **3c** measures the *lost-in-encoding* floor, which no amount of context can fix.
- **Intrinsic** uncertainty is what the model should state honestly (3h).

## 6. Background: the evidence–supervision mismatch (previous version of this file)

The supervision can require information that the model's actual input does not contain. Three cases:

| Situation | What happens | What would help |
|---|---|---|
| Evidence is missing | The target depends on another image, sequence, history, or pathology result | Supply the evidence or adjust the target |
| Evidence is present but not perceived | The finding is visible but missed (resolution, frame selection, visual understanding) | Improve perception and image–finding supervision |
| Evidence is inherently nonspecific | Even an expert could not determine the answer from the input | Supervise plausible alternatives and appropriate uncertainty |

Two qualifications still hold. (1) Medical knowledge and language priors are necessary. The failure is a prior
that overrides contradictory evidence or substitutes for missing evidence without acknowledging uncertainty.
(2) An eventual diagnosis is a valid target even when the image is ambiguous. The defect is supervision that
presents it as *uniquely established by the image*, or a rationale claiming an unobserved feature was seen.

Research question carried forward: **does aligning each target with the evidence actually supplied reduce
unsupported claims and improve diagnostic discrimination?** The main direction answers this for the
"evidence is missing" case by supplying the evidence.
