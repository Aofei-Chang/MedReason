# Uncertainty-Gated Context Seeking as Post-Training

Plan + preliminary results, 2026-10-24. Parent docs: [README.md](README.md) (context expansion design),
[research_direction.md](../../research_direction.md) §1.
Code: `experiments/context_expansion/`. Data and outputs: `/data/aofei/output/ContextMLLM/`.

## 0. Idea in one paragraph

A Med-MLLM trained on PubMed pairs has learned to state things the image cannot show (age, history,
pathology, time course), because the training text was written with context the model never saw. We
post-train the model to act on its uncertainty:
- answer from the image when the image suffices;
- request context (the source article, prior studies, guidelines) when the missing piece is non-visual;
- look again (zoom) when the missing piece is a visual finding it failed to see;
- revise its earlier answer once the new evidence arrives.

The two uncertainty sources, **needs context** and **needs perception**, call for different actions. Giving
context for the second is harmful: it lets the model copy findings instead of seeing them. So the method only
works if the two can be told apart. The preliminary experiment (§3) tests exactly that.

## 1. Related work

### 1.1 When to retrieve: uncertainty-triggered retrieval (text LLMs, inference time)

| Work | Trigger signal | Relation to us |
|---|---|---|
| FLARE (Jiang et al., EMNLP 2023); DRAGIN (Su et al., ACL 2024) | Low-confidence upcoming tokens | Raw token uncertainty as the gate. Our pilot tests whether that signal separates the two causes. |
| Self-RAG (Asai et al., ICLR 2024) | Learned reflection tokens ("retrieve?", "supported?") | Learned gate via SFT on critic labels. Closest in spirit to a learned `<request>` action. |
| [SeaKR](https://arxiv.org/abs/2406.19215) (ACL 2025) | Uncertainty from internal states; reranks snippets by uncertainty reduction | Internal-state gate. Our hidden-state probe is the multimodal analogue. |
| [Decide Then Retrieve](https://arxiv.org/pdf/2601.03908) (2026) | Normalized NLL of the generated answer, training-free | The raw-NLL baseline in our pilot. |
| [Interpretable Uncertainty for Adaptive Retrieval and Reasoning](https://arxiv.org/abs/2607.07380) (Jul 2026) | Hidden-state estimates of *knowledge insufficiency* vs *ambiguity/conflict*; retrieve for the first, reason more for the second | **Closest typed-uncertainty precedent** (text only). Same move as ours: different uncertainty types get different actions. |

### 1.2 Learning to search with RL (the "actionable" part)

| Work | What it shows |
|---|---|
| [Search-R1](https://arxiv.org/abs/2503.09516) (Jin et al.) | Outcome-reward RL with interleaved `<search>`/`<information>` turns. Retrieved tokens are masked from the loss. This is the protocol we adopt. |
| [MMSearch-R1](https://arxiv.org/abs/2506.20670) (ACL 2026) | First end-to-end RL for **multimodal** on-demand search. Outcome reward plus a **search penalty**; a search-balanced dataset (needs-search vs not) is essential; >30% fewer searches. **Closest multimodal precedent.** Its gate is learned purely from outcome and cost, with no notion of *why* the model is uncertain. |
| UR², GraphRAG-R1, RFM-RAG (see search results) | Reward shaping against under- and over-retrieval; learned sufficiency feedback |
| [Information self-locking in RL for active reasoning](https://arxiv.org/pdf/2603.12109) (2026) | RL information-seeking agents can stop asking. A risk we must monitor. |

### 1.3 Perception vs. knowledge: attributing VLM failures

| Work | What it shows |
|---|---|
| [Failing to See or Failing to Know?](https://arxiv.org/abs/2607.04683) (Jul 2026) | Tree-structured attribution of VLM errors to pre-recognition (perception) vs. post-recognition (knowledge) failures. **Visual-token states predict recognition failures; prompt hidden states predict correctness; lack of factual access gives only a weak signal.** Routes to image repair vs. retrieval. General-domain entity VQA, inference only. |
| [Delineating Knowledge Boundaries for Honest LVLMs](https://arxiv.org/abs/2604.26419) (Apr 2026) | Visual-IDK data from multi-sample consistency; SFT + DPO/ORPO to refuse; probes to check that refusal is not a memorized pattern; reports OOD generalization to medical domains |
| "Look Again or Look It Up" ([hypothesis page](https://lacuna.tiptreesystems.com/hypothesis/look-again-or-look-it-up-attributing-uncertainty-to-choose-between-perception-281a2c6267d792b2), not a paper) | Proposes splitting uncertainty into perception and knowledge parts by perturbing the image reading vs. the retrieved documents. Unpublished, but indicates the idea is in the air. |

### 1.4 Medical: retrieval, information seeking, uncertainty

| Work | What it shows |
|---|---|
| RULE (Xia et al., EMNLP 2024); MMed-RAG (Xia et al., ICLR 2025); [HeteroRAG](https://arxiv.org/pdf/2508.12778) | Med-VLM RAG at inference: calibrated retrieval amount, domain-aware retrieval, preference tuning against over-reliance on retrieved text. **Always retrieve.** No uncertainty gate, no training-target repair. |
| [Uncertainty-Driven Expert Control](https://openaccess.thecvf.com/content/ICCV2025/papers/Liang_Uncertainty-Driven_Expert_Control_Enhancing_the_Reliability_of_Medical_Vision-Language_Models_ICCV_2025_paper.pdf) (ICCV 2025) | Med-VLM baseline: RAG only on the top-5% highest-entropy samples. Retrieved knowledge conflicted with internal knowledge. A raw-entropy gate in medicine. |
| [MediQ](https://arxiv.org/abs/2406.00922) (NeurIPS 2024) and RL follow-ups (DoctorAgent-RL, Doctor-R1) | Clinical LLMs that ask for missing information. Naive asking hurts; abstention helps. Text only. |
| [PCQC](https://arxiv.org/abs/2609.27987) (Sep 2026) | **Privileged counterfactual question credit**: privileged patient facts answer questions the policy did *not* ask, and a frozen scorer turns this into per-question credit on top of GRPO. +4.4 pt over GRPO with 33% fewer turns. **The credit-assignment trick we reuse:** the source article plays the role of the privileged patient record. |

### 1.5 Training with privileged context

| Work | What it shows |
|---|---|
| Learning using privileged information (Vapnik); generalized distillation (Lopez-Paz et al.) | The formal setting: extra information at training time only |
| Context distillation (Snell et al. 2022) | Teacher sees context, student does not. **PubMedVision is effectively this**: GPT-4V wrote answers with the caption, and the student learns them without it. Correct for visual content, wrong for non-visual content. |
| [DOPD](https://arxiv.org/abs/2606.30626) (Jun 2026) | Names the **"privilege illusion"**: part of a privileged teacher's advantage is information asymmetry the student can never learn. Uniform distillation on it causes entropy collapse. Routes token-level supervision by advantage gap. A general-ML statement of our problem; our typing is the medical, claim-level version. |

### 1.6 What is not covered (our gap)

No work we found combines all three of the following:
- (i) typing a multimodal model's uncertainty into *needs context* vs. *needs perception* using counterfactual image/context ablations;
- (ii) using that typing as **privileged credit** in RL, so the model learns *which* action (request / look again / answer) to take;
- (iii) applying it to repair a known training-data defect (PubMed supervision exceeding the image).

MMSearch-R1 learns *whether* to search from outcome only. "Failing to See or Failing to Know" attributes but does not train. DOPD and PCQC provide the credit machinery in text settings. This review was done in one session with web search. It is not exhaustive, so a citation-chasing pass from MMSearch-R1, PCQC, and 2607.04683 is still needed.

## 2. Why post-training first (and how pretraining fits)

- **The base model already perceives roughly.** The gate relies on the image-ablation gain dI = l(y|I) − l(y|∅).
  For a model that cannot yet see, dI ≈ 0 everywhere, so "needs context" and "needs perception" look the same
  to any likelihood-based gate. The isolation signal therefore *requires* a reasonably aligned model. That
  argues for post-training first. For pretraining, the gate should come from a frozen post-trained reference
  model, not from the model being trained.
- **The defect is a learned bias, so correcting it is a policy change.** The model confidently emits
  non-visual claims (§3, Q4). Post-training can reward "request, then state" over "state from the prior", and
  reward revising after context arrives. That is an RL problem with actions, not just a better loss.
- **Cost.** RL on a 3–7B model with a few thousand items is feasible on 3–4 A6000s. A pretraining ablation is
  not, at a meaningful scale, before the post-training signal is validated.
- **Pretraining remains open.** The same typing (offline, from a reference model) gives the pretraining
  variants M2/M3 in [README.md](README.md) §3: complementary context for context-derived claims, and masking
  of unsupported claims. Paper story: post-training corrects the bias; gated pretraining avoids installing it;
  compare both on the same evaluation.

## 3. Preliminary experiment P1: can "needs context" be isolated? (DONE)

**Setup.**
- **Data:** 600 single-image CT/MRI/US PubMedVision records, one per article. Each is linked to its PMC
  article: true caption, figure-citing paragraphs (via JATS `<xref>`), abstract.
- **Models:** Qwen2.5-VL-7B-Instruct (general) and Lingshu-7B (medical, a PubMedVision-family training mix).
- **Two targets:**
  - (A) the GPT-4V instruction answer, whose author context is the original caption;
  - (B) the original caption, whose author context is the article.
- **Scoring:** teacher-forced per-sentence log-likelihood under image/context ablations:

  | Gain | Definition |
  |---|---|
  | dC | l(I,C) − l(I) |
  | dI | l(I) − l(∅) |
  | dIC | l(I,C) − l(C) |

- **Semantic label:** an independent text-only judge (Qwen3-8B) tags each sentence as V (visual finding),
  P (non-visual patient/clinical), K (knowledge), or M (figure meta).

**Questions.**
- **Q1:** is context gain concentrated where it should be?
- **Q2 (key):** among sentences that are *uncertain* (NLL above the median) and *context-resolvable*
  (dC ≥ 0.25 nats/token), can signals available **without the context** tell P (needs context) from
  V (needs perception)? Candidate signals:
  - raw NLL and entropy;
  - image-ablation gain dI;
  - zero-shot self-report "verifiable from the image alone?";
  - linear probes on hidden states at the decision point (before the sentence) and in-sentence;
  - a text-only TF-IDF baseline.
- **Q3:** does redacting figure-citing sentences from the context reduce the visual leak while keeping the
  non-visual gain?
- **Q4:** does the image raise the likelihood of non-visual claims (a learned spurious shortcut)?

**Results.** 600 records, 3,503 sentences per model. AUROC values carry 95% record-clustered bootstrap CIs. Full
numbers are in `/data/aofei/output/ContextMLLM/pilot_needs_context/analysis.json`.

Q2 subset sizes: A, Qwen 199 P / 188 V; A, Lingshu 134 / 163; B, Qwen 103 / 186; B, Lingshu 147 / 202.

**Q2 — separating needs-context (P) from needs-perception (V)** among uncertain, context-resolvable
sentences. The signals marked † are available before the sentence is produced, so they can drive an
inference-time decision.

| Signal | A Qwen | A Lingshu | B Qwen | B Lingshu |
|---|---|---|---|---|
| Raw NLL † | 0.53 [.48,.59] | 0.52 [.46,.59] | 0.60 [.53,.67] | 0.58 [.53,.64] |
| Mean token entropy † | 0.60 | 0.48 | 0.61 | 0.57 |
| Image-ablation gain (−dI) | 0.59 [.53,.64] | 0.55 | 0.48 | 0.51 |
| Zero-shot self-report "not verifiable from image" | 0.71 [.66,.77] | 0.79 [.73,.84] | 0.81 [.76,.86] | 0.79 [.74,.84] |
| Hidden-state probe, decision point (best layer) † | 0.73 (L21) | 0.76 (L21) | 0.68 (L7) | 0.66 (L21) |
| … same probe, **no image in the input** | 0.70 | 0.74 | 0.66 | 0.69 |
| Hidden-state probe, in-sentence (best layer) | 0.94 | 0.94 | 0.94 | 0.95 |
| … same probe, **no image in the input** | 0.94 | 0.94 | 0.94 | 0.94 |
| Text-only TF-IDF on the sentence | 0.66 | 0.71 | 0.77 | 0.74 |

**Q1 — where context helps** (target A, Qwen; mean nats/token). Lingshu shows the same ordering.

| Class | n | dC | dI | Share context-resolvable |
|---|---|---|---|---|
| V | 588 | 0.41 | 0.16 | 53% |
| P | 961 | 0.21 | 0.04 | 28% |
| K | 575 | 0.04 | −0.02 | 6% |

**Q3 — redacting figure-citing sentences from the article context** (target B; dC full → redacted):

| Class | Qwen | Lingshu |
|---|---|---|
| V | 0.39 → 0.18 (−53%) | 0.51 → 0.32 (−37%) |
| P | 0.29 → 0.23 (−20%) | 0.43 → 0.35 (−17%) |

Image gain given context, for V sentences, rises slightly after redaction (Qwen 0.113 → 0.130).

**Q4 — image gain on non-visual (P) claims** (share of P sentences with dI > 0.1 nats/token):

| Target | Qwen | Lingshu |
|---|---|---|
| A | 24% | 36% |
| B | 52% | 46% |

**Interpretation.**

1. **"Needs context" is a real but not self-evident category.** Context gain is *larger* on visual sentences
   than on non-visual ones, because captions and citing paragraphs restate the findings. Gating on "context
   resolves the uncertainty" would therefore fire mostly on perception cases: about half the
   context-resolvable uncertain sentences are visual. This is the leak, now measured.
2. **Raw uncertainty cannot isolate it** (AUROC 0.48–0.61). The meeting proposal "if uncertain, retrieve" needs
   a typed gate. A raw-entropy gate (e.g. the ICCV 2025 top-5% entropy RAG) would retrieve for perception
   failures as often as for missing information.
3. **The image-ablation gain is not a usable separator either** (0.48–0.59). For a model that cannot extract a
   finding, visible-but-missed and not-visible look the same (§2). Counterfactual likelihoods alone cannot
   type the uncertainty.
4. **Content type is easy to decode, but it is textual, not image-aware.** In-sentence probes reach 0.94, but
   the **same probe with no image gets the same score**. The probe reads what kind of statement it is; it does
   not read whether *this image* supports it. Decision-point probes (0.66–0.76) are modest and also unchanged
   without the image. So we found **no evidence of latent image-specific self-awareness** in these models.
   The capacity to know "I can't see this" versus "this isn't visible" must be **taught**, which is what
   post-training would add.
5. **Practical consequences for the method.**
   - At **training time** the target is known, so claim typing is cheap and reliable: a text judge or
     content probe works. That is exactly what privileged necessity credit needs.
   - At **inference time** the best available signal is zero-shot self-report on a drafted claim (0.71–0.81).
     That suggests a draft → check → request protocol, and a strong SFT/RL starting point.
6. **Complementary context works as designed.** Redaction removes 37–53% of the visual leak while keeping
   80–83% of the non-visual gain.
7. **Bias (Q4) is suggestive but not established.** On instruction answers, Lingshu (trained on a
   PubMedVision-family mix) raises the likelihood of non-visual claims from the image more often than Qwen (36%
   vs 24%), consistent with a learned image → non-visual shortcut. On captions the order reverses. Lingshu's
   much lower NLL on target A (≈1.0 vs 1.7) also indicates exposure to these answers, so memorization
   confounds this. P3 (free generation, checked against the article) is the proper test.

**Caveats.**
- All 600 records are true-caption records (PMC-OA-mapped), so inline-paragraph records are not yet included.
- Semantic labels come from one LLM judge with no human audit yet.
- τ = 0.25 nats/token is a single threshold, with no sensitivity sweep yet.
- Probes are linear, 5-fold grouped by article; n ≈ 300–390 per cell.
- Self-report was not run without the image, so its image-dependence is untested (add this control).

## 4. Further preliminary experiments

### 4.0 Central axis: visual context vs. non-visual information context (agreed 2026-10-25)

P1 showed that "context" is two different things:

- **Visual context:** caption or article sentences that restate what the image shows (findings, arrows,
  locations). It resolves *perception* uncertainty by handing over the answer: a text shortcut.
- **Non-visual information context:** history, labs, pathology, treatment, outcome, priors. It resolves
  *missing-information* uncertainty, which is the only kind the image cannot resolve.

Every experiment below now types the *context sentences* the same way as the target sentences (V/M = visual,
P/K = non-visual) and reports the two parts separately. The method consequence:

- Non-visual context is what a `<request>` should return.
- Visual context is privileged supervision: a verifier for `<look>` and a perception reward, never an input
  the policy can ask for.

### 4.1 Validation run (started 2026-10-25 17:00, ≤8 h budget, GPUs 1–3)

**Done (3.96 h). Results: [validation_results_2026-10-25.md](validation_results_2026-10-25.md).** In short:
- Clean crossover: visual context helps V sentences, non-visual context helps P sentences.
- Semantic context typing removes 68–85% of the visual leak; regex redaction removes 20–25%.
- Self-report is mostly textual.
- Zoom gives no gain over a random crop.
- Zero-shot requesting is decided by the question wording.
- On MIMIC-CXR, the prior report helps comparison sentences (Qwen), and Lingshu predicts "has a prior" from the image.


Scope: exploring and validating the ideas, with **no RL training for now** (P5 is dropped from this run).
Driver: `experiments/context_expansion/run_overnight.py`, a dependency queue with 2 slots per GPU. Outputs and
logs are in `/data/aofei/output/ContextMLLM/overnight/`; the results land in `RESULTS.md` and `results.json`.

**Inputs.**
- `d2k`: 2,000 records, the 600 pilot records plus 1,400 new articles.
- `box`: 600 MedSAM3-box records from disjoint articles.
- `mimic`: 900 MIMIC-CXR studies, 600 with a prior report and 300 first studies.
- QA items generated from `d2k`.

**Stages.**

| Stage | Script | Model(s) | Est. |
|---|---|---|---|
| Sentence typing (targets + context sentences, 48.7k) | `label_sentences.py` | Qwen3-8B | 40 min |
| P1b + P2a typed-context scoring | `score_typed_context.py` (3 shards × 2 models) | Qwen2.5-VL-7B, Lingshu-7B | 1–1.5 h |
| P2b zoom vs. context | `score_zoom.py` | both | 30 min |
| P3 free generation → judge with article | `gen_free.py`, `judge_free.py` | both → Qwen3-8B | 30 + 60 min |
| QA set (image / context items) | `qa_build.py` | Qwen3-8B | 20 min |
| P4 zero-shot request policy | `p4_zero_shot.py` | both | 20 min |
| P6 MIMIC-CXR scoring + generation | `mimic_p6.py` | both | 40 min |
| Analysis | `analyze_overnight.py` | CPU | 5 min |

**Predictions** (what would support the visual vs. non-visual split):

- **P1b:** if self-report AUROC is unchanged without the image, the self-report is also textual, and image-aware
  "can I see this?" must be trained.
- **P2a:** a crossover. Visual context helps V sentences more than P sentences; non-visual context helps P more
  than V. Image reliance given context (dI|ctx) for V sentences stays high under non-visual context and drops
  under visual context (the shortcut). A gate that uses only non-visual context should fire mostly on P,
  whereas the full-context gate fires about half on V (P1).
  - Controls: mismatched context (another article), and the P1 regex redaction.
- **P2b:** zoom minus a random crop of the same size helps V (most where the sentence names a boxed entity)
  and not P. Non-visual context helps P and not V. If so, two actions are justified: `<look>` for
  perception and `<request>` for information.
- **P3:** the reference answers and each model's free text, scored by unsupported patient claims per text,
  per model.
- **P4:** zero-shot request precision and recall. Answer quality with the full vs. non-visual context given
  (does the full context leak the answer to image questions?).
- **P6:** on MIMIC-CXR the prior report is the non-visual context. It should raise the likelihood of
  comparison sentences, not of the others. Models with no prior given will still write comparisons, a
  fabricated-comparison rate, including on first studies.

**Scope cuts versus the original table.**
- P5, the GRPO pilot, is deferred. The current phase validates the ideas, not the method.
- P6 uses the prior *report* as context; the prior *image* needs the full MIMIC image set.

### 4.2 Original design table

| ID | Question | Design | Decision it informs |
|---|---|---|---|
| P1b | Is zero-shot self-report image-dependent? | Rerun self-report with the image removed; if the AUROC holds, it is also textual | Whether self-report can seed the gate or only the protocol |
| P2 | Is the split **actionable** (a crossover)? | For uncertain V- and P-type items, apply (a) redacted article context and (b) a perception aid: zoom/crop to the MedSAM3 box (boxes exist for 10k PubMedVision entities). Predict an interaction: context helps P, not V; zoom helps V, not P. | Whether a `<look>` action is worth adding next to `<request>` |
| P3 | How large is the **bias** in free generation? | Generate descriptions without context. A judge extracts non-visual claims and checks them against the article: fabricated / correct-by-prior / unverifiable. Lingshu vs. Qwen. | Size of the effect post-training must fix; main metric |
| P4 | What is the **zero-shot request policy**? | Give the model the `<request>` protocol with no training, on a balanced set (needs-context / image-answerable). Measure request precision/recall and accuracy. | RL headroom; whether SFT warm-start is needed |
| P5 | Does **typed credit** beat outcome-only RL? | GRPO pilot, Qwen2.5-VL-3B, ~5k items. (a) outcome + cost (MMSearch-R1 style) vs. (b) + privileged necessity credit (PCQC style) from the typing. | Core method claim |
| P6 | Does it **transfer** to real clinical context? | MIMIC-CXR: priors as the requestable context; fabricated "compared with prior" rate. | Generality beyond PubMed |

## 5. Method design: context-seeking post-training

> **2026-10-26:** a privilege-aware revision, which adapts DOPD's privilege-matched advantage gap to typed context,
> is in [privilege_aware_design.md](privilege_aware_design.md). If its validations (V1–V3, V5) hold, it replaces the
> judge-based necessity credit and unsupported-claim penalty below.

**Episode.**
- Input: image I and question q (or "describe").
- The policy writes `<think>` and then one of:
  - `<answer>` — answer now;
  - `<request kind="article|prior|guideline">query</request>` — the environment returns `<context>` from
    that sample's pre-extracted pool (complementary, i.e. visual restatements redacted);
  - `<look box="...">` — the environment returns a zoomed crop (phase 2, if P2 supports it).
- Budget: ≤ 2 actions. Retrieved tokens are masked from the loss (Search-R1).

**Data (from the retrieved PubMed context).** Balanced, because MMSearch-R1 found balance essential:
- **Context-required items:** questions whose answer is a non-visual fact about *this* patient (age, history,
  pathology, follow-up, treatment response). An LLM generates them from the article. Keep only items that are
  answerable from the article, judged non-visual, and low-likelihood from the image across ≥2 models.
- **Image-required items:** visual findings from the caption (laterality, location, sign, modality), with the
  context redacted so a request cannot leak the answer.
- **Paired items:** the same image with one question of each type. These are the counterfactual check that
  the policy reads the question, not the image style.

**Reward.** R = correctness − λ·(#actions) + necessity credit − unsupported-claim penalty.
- *Correctness:* exact/short-answer match, or an LLM judge against the reference.
- *Necessity credit (privileged, PCQC-style):* the offline typing says whether the item needed context.
  - Reward requests on needs-context items.
  - Penalize requests on image-answerable items. Without this term the policy can learn to request
    everything, since context often contains the visual answer.
- *Unsupported-claim penalty (the bias correction):* non-visual claims in the answer that are not supported by
  the context the policy actually received are penalized. The verifier is a judge with the article
  (privileged). This directly targets the PubMedVision-installed behaviour.
- *Self-correction variant:* turn 1 answers without context; the environment reveals context; turn 2 revises.
  - Reward correct revisions.
  - Score turn-1 stated confidence with a proper scoring rule. On needs-context items, calibrated
    hedging in turn 1 is the desired behaviour.

**Training.** Optional SFT warm-start on protocol-formatted traces (from P4 failures plus templated traces),
then GRPO. Use Qwen2.5-VL-3B for pilots, then Lingshu-7B / Qwen2.5-VL-7B. This needs TRL or verl installed;
neither is in the current envs.

**Evaluation (must catch shortcut and surface compliance).**
- Request precision/recall vs. the typing; accuracy per item type.
- Unsupported non-visual claim rate in free descriptions with no context available (P3 metric).
- Image reliance: the image-ablation gap and paired counterfactual items must *not* drop.
- Standard VQA (VQA-RAD, SLAKE, PMC-VQA) unchanged; MIMIC-CXR comparison fabrication (P6).
- Ablations:
  - outcome-only reward;
  - raw-entropy gate (SFT on "request iff NLL high");
  - always-context;
  - never-context.

**Risks.**
- **Over-requesting:** context contains visual answers. Mitigated by redaction plus necessity credit.
- **Self-locking:** the policy stops requesting. Monitor the request rate per item type.
- **Typing noise:** use two scorers plus a judge, and audit a 200-item human-checked subset.
- **Judge leakage** of the reference answer.
- **Memorization:** Lingshu may have seen these PubMedVision items. That is part of the bias, but report
  results on held-out articles too.

## 6. Timeline (proposal)

| Week | Work |
|---|---|
| 1 | Finish mapping and full text; run P2–P4 on the 600-record pilot set, expanded to ~3k including inline-type records |
| 2 | Build the balanced QA set (~8k); human-check 200 items; SFT warm-start; zero-shot baselines |
| 3–4 | P5 GRPO pilot (3B), with and without necessity credit; P6 MIMIC-CXR setup |
| 5+ | Scale to 7B; pretraining-side ablation (M2 with the reference gate) on a small alignment run |
