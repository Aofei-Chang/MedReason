# Context Expansion with Uncertainty / Self-Awareness

Exploration note, 2026-10-24. Parent: [research_direction.md](../../research_direction.md) §1.
Feasibility data: `/data/aofei/output/ContextMLLM/` (scripts in `experiments/context_expansion/`).
Follow-up: [post_training_plan.md](post_training_plan.md) — post-training (RL) version, literature, pilot P1 results,
and the overnight P1b–P6 run (§4), organized around visual vs. non-visual information context.
Next: [privilege_aware_design.md](privilege_aware_design.md) — DOPD-style privilege-aware distillation design and validations V0–V6.
Results of V0–V6 (awareness of needing context): [awareness_results_2026-10-26.md](awareness_results_2026-10-26.md).

## 1. Problem setup

A PubMed training pair is (I, T), where T is a caption or a GPT-4V reformulation of it. The author of T had
the article: history, labs, earlier timepoints, pathology, follow-up. Write the article-derived pool as

C = {real caption, figure-citing paragraphs, abstract, case presentation, other figures' captions} ∪ {guidelines}.

Standard training fits p(T | I). We want the model to learn what the image supports, and to get everything
else from C, but only when C is actually needed.

Per claim y in T, three log-likelihoods under the current (or a reference) model define the typing:

| Quantity | Formula | Reads as |
|---|---|---|
| Context gain | Δ_C(y) = log p(y \| I, C) − log p(y \| I) | Context explains y |
| Image gain given context | Δ_I(y) = log p(y \| I, C) − log p(y \| ∅, C) | Image still matters once C is present |
| Residual | −log p(y \| I, C) | Neither explains y (style, GPT-4V embellishment, intrinsic) |

Claim types:

- **image-derivable**: Δ_I high, Δ_C low.
- **context-derived**: Δ_C high, Δ_I low.
- **redundant**: both sources suffice.
- **unsupported**: the residual stays high.

## 2. Key design insight: raw uncertainty is the wrong gate

The meeting proposal is: if the model is uncertain, retrieve more context. High loss on a claim has two
different causes:

1. **The information is not in the image** (attribution gap). Context is the right fix.
2. **The information is in the image, but the model cannot extract it yet** (perception: the not-extracted or
   lost-in-encoding term). Context is the *wrong* fix. If the caption says "arrow marks a 2 cm mass in the
   left lobe", adding it lets the model copy the finding instead of learning to see it. Its perception
   gradient disappears, and the text shortcut we are trying to remove gets *stronger*.

Raw entropy treats both cases the same. Worse, an "uncertain → add context" loop has a failure mode. The
model learns to lean on context, its image-only uncertainty stays high, so more context is added, and
perception degrades further. So the gate must be **typed**: retrieve when Δ_C is high and Δ_I is low.

Two consequences shape the method:

- **Complementary context, not full context.** Give C *with image-derivable statements redacted*. Keep
  history, other timepoints, labs, pathology, and outcomes. Remove sentences that restate visible findings.
  This adds C-only information without leaking the perceptual answer.
- **The gate must be robust to the model's own drift.** Compute types with a frozen reference model, or
  re-estimate periodically with image-ablation controls. Keep a share of no-context passes where
  context-derived tokens are masked from the loss.

This is the claim the project should test first (E2 below), because it decides whether "uncertainty" can be
the model's raw confidence or must be counterfactual.

## 3. Method ladder

| ID | Training input | Loss | Purpose |
|---|---|---|---|
| M0 | (I, q) → T | all tokens | PubMedVision baseline |
| M1 | (I, q, C_full) → T | all tokens | Naive "just add the article". Expected to raise benchmark text scores and lower image reliance |
| M2 | (I, q, C_comp) → T, offline typing | mask unsupported claims | **"Pre-extract the context"**: an LLM with the article splits T into claims, types them, and builds complementary context. No model in the loop |
| M3 | Two-pass: (I, q) first; for gated claims, re-forward with retrieved C_comp | typed gate from Δ_C, Δ_I | **Uncertainty-gated expansion** (meeting idea). Ablate gate = raw entropy vs. typed |
| M4 | The model may emit `<ctx? query>`; retrieval fills from the article pool / guidelines | SFT on gate labels, then RL: reward necessary requests, penalize requests for image-derivable claims | **Self-aware context seeking**: the training-time gate becomes a learned inference-time skill (ask for history / prior / guideline) |

Guidelines (meeting idea) enter only for claims typed *inferential* (diagnosis, staging, management). They
supply the F → D step and should never be retrieved for perceptual claims.

M3 is "not simply pretraining" in a precise sense. Each sample's context depends on the current model, so it
works like active / self-paced learning. Context dose should fall as perception improves, and logging the
gated fraction over training is itself a result.

## 4. Experiments (cheapest first)

**E1. Context-gain audit (no training).** On ~2k mapped PubMedVision samples:

- Compute Δ_C per sentence under 2–3 Med-MLLMs. HuatuoGPT-Vision is trained on PubMedVision, so treat it as
  the memorization control. Also use Lingshu or MedGemma.
- Context ablations: real caption / citing paragraphs / abstract.
- Output: the share of supervision tokens that are context-derived, by modality and source type
  (caption vs. inline). This scales up the 12-fact pilot in `experiments/information_gap`.

**E2. Does raw uncertainty predict context need? (decides M3's gate).** Label claims by type with the
counterfactual definitions (validated on a ~200-claim human-checked subset). Then compute the AUROC of raw
token entropy / loss for "context-derived vs. image-derivable".

- Hypothesis: entropy is a weak predictor, because perception-hard claims are also high-entropy.
- If confirmed, this is a standalone, publishable diagnostic that justifies typed gating.
- If refuted, the simple gate is fine and M3 gets cheaper.

**E3. Small-scale training.** Qwen2.5-VL-3B or a LLaVA-style model, ~100k mapped PubMedVision samples,
M0–M3, same compute. Evaluation must catch the shortcut, not just accuracy:

- Image reliance: image-ablation gap and counterfactual pairs (does the answer flip with the image?).
- Fabrication of non-visible facts with no context given (ages, "compared with prior", pathology).
- Use of context when it is given, and abstention or requests when it is missing.
- Standard benchmarks: VQA-RAD, SLAKE, PMC-VQA, OmniMedVQA.

**E4. Longitudinal (MIMIC-CXR).** Repeat M0/M3 with priors as C. Primary metric: fabricated-comparison rate,
plus comparison accuracy when priors are supplied.

**E5. Self-aware requests (M4).** Precision and recall of `<ctx?>` against the typed labels, and accuracy
gain per request.

## 5. Risks

- **Shortcut amplification** (M1, and M3 with a raw gate): the central risk. Section 2 is the mitigation, and
  E3 measures it.
- **Typing noise**: Δ is relative to the scoring model. Use 2+ scorers and a human-checked subset.
- **GPT-4V embellishment**: some PubMedVision answer content is in neither image nor article. It is the
  unsupported class, and masking it may matter as much as adding context.
- **Leakage / memorization**: a PubMedVision-trained scorer may have memorized T. Use non-PubMedVision
  scorers for gating.
- **Coverage**: ~11% of sampled captions remain unmapped, and ~30% of true captions have no citing paragraph.
  For those, the context is the abstract or nothing.

## 6. Closest related work and differentiation (from memory; verify before writing)

- **Inference-time active retrieval** (FLARE, Jiang et al. 2023; DRAGIN, Su et al. 2024; Self-RAG, Asai et
  al. 2024; self-knowledge-guided retrieval, SKR): these trigger retrieval on low-confidence tokens or learned
  reflection tokens at *inference*. Ours triggers at *training* time to repair the supervision, and
  distinguishes perceptual from missing-information uncertainty.
- **Retrieval-augmented / context-enriched pretraining** (REALM, RETRO; In-Context Pretraining, Shi et al.
  2024): these add context uniformly, with no gate or modality typing.
- **Medical multimodal RAG** (RULE, MMed-RAG, Xia et al. 2024–25): retrieves reports at inference to cut
  hallucination. It is not about training-target alignment.
- **Caption context** (Concadia; KnowAda; Gekhman et al. 2024 on unknown facts): these provide the
  p(T | I, C) motivation and show that training on unsupported targets increases hallucination.

A dedicated literature check on "uncertainty-triggered context during multimodal training" is still needed
before claiming novelty.

## 7. Feasibility status and next steps

Done (2026-10-24, see the parent file §1.1):

- Caption → PMCID mapping: 50% offline via PMC-OA, 89% of a sample with Europe PMC search, and 93/93
  agreement between the two.
- BioC full-text retrieval works.
- Figure-citing paragraphs exist for 70% of true captions.
- 26% of "captions" are actually inline paragraphs.

Next:

1. Map the remaining ~320k captions via Europe PMC search (background job, rate-limited).
2. Bulk full text from PMC OA XML for ~250k articles. Parse into caption, citing paragraphs, abstract, and
   case presentation per image.
3. Build the claim splitter + typing prompt (M2) and validate it on 200 human-checked claims.
4. Run E1 and E2. Decide the gate before any training.
