# Awareness of needing context + privilege decomposition: results (run 2026-10-26)

Results of the V0–V6 validations proposed in [privilege_aware_design.md](privilege_aware_design.md) §4. No training.

The central question (from the user): **does a Med-MLLM know when it needs more context?** — and, for the DOPD-style
design, how much of a privileged teacher's advantage is information the student cannot get.

- Raw outputs: `/data/aofei/output/ContextMLLM/aware/` (`RESULTS_aware.md`, `results_aware.json` from `analyze_aware.py`;
  `logs/`; `audit_labels.csv` = 300-row human-audit sheet, not yet filled).
- Scripts (`experiments/context_expansion/`):
  - driver: `run_aware.py`. Runtime 4.1 h on GPUs 1–3 after a restart: `--resume` adopted the running jobs and the
    hard set was cut to 2,400 items.
  - per-step scripts: `label_aware.py` (V0 time labels, typing of generated sentences, MIMIC comparison detector),
    `build_aware_items.py`, `aware.py` (central experiment), `token_decomp.py` (V1/V2/V5/box), `zoom_qa.py`,
    `mimic_p6.py --set mimic_test`.
  - shared helpers: `aware_common.py`, which caches vision-tower features (no flash-attn on this machine).
- Models: Qwen2.5-VL-7B (general), Lingshu-7B, HuatuoGPT-Vision-7B-Qwen2.5VL (trained on PubMedVision itself).
  All three share one tokenizer, so per-token comparisons across models are exact.

## 0. Setup

**Items.** 3,200 short-answer items, each with a typed context split:
- *qa*: 800 of the P4 QA items. The wording gives the type away ("what is the patient's age?").
- *hard*: 2,400 items from **fixed templates asked of every record**, so the wording is constant and only the case
  decides:
  - diagnosis ("What is the diagnosis for the patient in this image?");
  - nature ("benign or malignant?");
  - finding ("main abnormal finding?").
- Qwen3-8B answered each template from the numbered caption + article sentences and cited evidence. Each item is
  typed by that evidence: only visual sentences → *image* item; any patient sentence → *context* item.

**Ground truth, measured on each model itself:**
- *Correctness-based.* Using greedy answers, token-F1 ≥ 0.5:
  - **S** — right without context;
  - **CR** — context-resolvable: wrong without context, right with the non-visual context;
  - **PL** — perception-limited: wrong with both, right with the visual context;
  - **U** — none of the above.
- *Likelihood-based.* **VOI** = per-token log-likelihood gain of the reference answer from the non-visual context.
  This is independent of the strict F1 threshold.

**Awareness signals.** All measured with no context, each with the correct image (I), a same-modality mismatched
image (M) and no image (0):
- verbal "image alone or needs more patient information?" (log-odds);
- P(IK): "will you answer correctly?";
- the answer-or-request protocol's first-tag log-odds;
- answer NLL, first-token entropy, disagreement among 6 samples;
- linear probes on last-prompt-token hidden states (layers 14/21/28; PCA-256, grouped 5-fold CV);
- baseline: a TF-IDF classifier on the question text alone.

## 1. Headline conclusions

1. **On standard QA, apparent awareness of needing context is question reading.**
   - Task: model-defined need (VOI ≥ 1 vs < 0.25 nats/token).
   - The verbal need signal reaches 0.89 / 0.90 / 0.91 (Qwen / Lingshu / Huatuo), and the request tag 0.89 / 0.89 / 0.90.
   - But the question text alone reaches **0.92 / 0.92 / 0.94**.
   - The signals barely change with a mismatched image (0.87 / 0.86 / 0.88) or no image (0.84 / 0.89 / 0.90).
2. **With the wording held fixed, expressed awareness is at chance.** Template-stratified AUROC, counting only pairs
   within the same template:

   | Signal (correct image) | Qwen | Lingshu | Huatuo |
   |---|---|---|---|
   | Verbal need, CR vs S | 0.45 [0.41, 0.50] | 0.54 [0.50, 0.57] | 0.49 [0.45, 0.53] |
   | Request tag, CR vs S | 0.49 | **0.61 [0.57, 0.65]** | 0.54 |
   | Verbal need, VOI-defined need | 0.53 | 0.57 | 0.60 |
   | Verbal need, item type (context vs image evidence) | 0.49 | 0.54 | 0.52 |
   | Any signal, typed CR vs PL (needs context vs needs perception) | 0.47–0.53 | 0.45–0.56 | 0.50–0.55 |

   - The request decision follows the template, not the case.
     - Qwen and Huatuo request on ~99% of "nature" items in every class, and on 0% of "finding" items.
   - Lingshu is the partial exception.
     - On "nature", its request rate rises S 29% → CR 48% → PL 61% → U 76%.
     - Its stratified request AUROC drops from 0.61 to 0.50 with a mismatched image.
3. **Generic uncertainty weakly flags "I will fail without context", but cannot say why.**
   - Stratified first-token entropy for CR vs S: 0.66 / 0.64 / 0.63. Sample disagreement: 0.60 / 0.62 / 0.57.
   - The same signals give 0.45–0.52 for CR vs PL: they cannot tell missing information from missing perception.
   - At token level (V1), the student's entropy cannot separate tokens that a request would fix (G_acq > 1, G_vis < 0.2)
     from tokens that only the visual context fixes: AUROC 0.44–0.52.
   - This confirms P1 with model-defined ground truth.
4. **Hidden states carry a latent, image-specific need signal that the model does not express.**
   - Within-template probes on layer 21, CR vs S. Each cell is correct image / mismatched image / no image:

     | Template | Qwen | Lingshu | Huatuo |
     |---|---|---|---|
     | nature | 0.69 / 0.51 / 0.61 | 0.75 / 0.48 / 0.47 | 0.73 / 0.49 / 0.61 |
     | finding | 0.52 | 0.64 / 0.44 / 0.47 | 0.75 / 0.37 / 0.35 |

   - Item type on "nature" (correct image / mismatched image): 0.61 / 0.51, 0.62 / 0.48, 0.67 / 0.49.
   - The "diagnosis" template has too few S items to say anything: n_S ≈ 26–61.
   - **Caveat:** such a probe may read image content that predicts difficulty (lesion appearance) rather than
     self-knowledge. It needs an image-encoder-only probe as a control (§4).
   - If the signal survives that control, the gap between latent and expressed need is the clearest motivation for
     post-training: teach the request policy to use what the model already encodes.
5. **Sentence-level "can I see this?" does have an instance-specific image component (V6a; corrects P1b's reading).**
   - p(yes, verifiable from the image) for V sentences, correct vs mismatched image:
     - Qwen: 0.21 vs 0.07 (A); 0.18 vs 0.04 (B).
     - Lingshu: 0.57 vs 0.25 (A); 0.47 vs 0.16 (B).
   - For P sentences the same comparison gives 0.09 vs 0.04 and 0.33 vs 0.16.
   - AUROC of (image − mismatched image), V vs P: 0.66–0.70.
   - So the image effect is not a generic "an image is present" effect. It is weaker than the text effect, but real.
   - Item-level "do I need patient information?" shows no such effect (point 2).
6. **P(IK) is badly calibrated in both directions.**

   | | Qwen | Lingshu | Huatuo |
   |---|---|---|---|
   | Mean p | 0.05 | 0.36 | 0.22 |
   | Accuracy | 0.21 | 0.22 | 0.23 |
   | ECE | 0.22 | 0.30 | 0.31 |

   Huatuo's mean matches its accuracy but it does not discriminate.

### DOPD privilege decomposition

7. **Privilege-blind distillation pushes mostly non-visual information (V1).**
   - Reference instruction answers (A), for each of the three models: of the total gap G_blind = ℓ(I, V+N) − ℓ(I),
     - 37–39% is G_acq (information a request would return);
     - 22–23% is G_post (information never available to the deployed model);
     - only **38–40% is visual privilege**.
   - 53% of the positive gap sits on P sentences.
   - On captions (B), G_post is 31–43%, and on P tokens it is 50–67%.
   - The "post" labels are noisy (point 14), so G_post vs G_acq is approximate. Their sum (non-visual, about 60%) is not
     affected.
   - **Observation matching (the teacher sees only what the student saw, plus V) removes most of the push SFT gets.**
8. **Image-counterfactual attainability fails the go/no-go rule.**
   - Among sentences the visual privilege helps (G_vis > 0.25), V vs P AUROC:
     - A_teach: 0.48–0.55;
     - ρ: 0.53–0.61;
     - student image specificity: 0.51–0.61;
     - all below 0.65, and below raw G_vis itself (0.62–0.67).
   - Median A_teach is about 0.01–0.05 nats/token: the privileged teacher barely needs the right image.
9. **An external medical teacher (V4: Lingshu or Huatuo teaching Qwen, same inputs, same tokens) does not fix it.**
   - The capability gap is uniform over V and P sentences.
     - On captions: 0.41–0.44 (Lingshu) and 0.18–0.19 (Huatuo).
     - On PubMedVision answers: 0.40–0.54, where both teachers are contaminated.
   - V vs P AUROC is 0.41–0.55. It measures domain fluency and memorization, not perception.
   - **Decision:** route by the *typed context* (labels), not by an attainability metric.
10. **Hint-like privilege (boxes, entity names) behaves as DOPD predicts (box set, 600 records).**
    - On V sentences that mention a boxed entity, gain in nats/token:

      | Privilege | Qwen | Lingshu | Huatuo |
      |---|---|---|---|
      | Caption V part | 0.55 | 0.36 | 0.40 |
      | Box coordinates + labels | 0.19 | 0.12 | 0.14 |
      | Entity names only | 0.24 | 0.16 | 0.18 |

    - The hints are about twice as image-dependent: A_teach 0.07–0.10 vs 0.04–0.05.
    - They leak less onto P sentences: 0.03–0.07 vs 0.10–0.17.
    - On V sentences that do not mention the entity, they slightly hurt: −0.03 to −0.05.
    - Hints are the safer privileged input for the teacher.
11. **On the model's own rollouts, the illusion is small (V2; prediction "blind ≫ matched" not confirmed).**
    - A privileged self-teacher gives the model's own free generations *lower* probability in every sentence class
      (−0.10 to −0.25 nats/token).
    - Share of positive pull on P sentences: blind teacher 24–28%, matched teacher 23–26%.
    - Only 1–2.5% of P tokens are pulled up by > 0.5 nats.
    - Generated patient claims are mostly wrong, so the informed teacher mainly *suppresses* them.
    - **The privilege illusion is mainly a property of SFT on reference targets (point 7), less of on-policy
      distillation.** The "contradiction/suppression" regime carries most of the on-policy signal.
12. **MIMIC-CXR held-out (V5).**
    - **The decomposition recovers the expected structure.** The prior report raises comparison sentences more than
      other sentences:
      - Qwen +0.24 vs +0.10;
      - Huatuo +0.25 vs +0.10;
      - Lingshu +0.06 vs +0.03.
    - **Comparison language with no prior in the input** (LLM detector, replacing the P6 regex):

      | Model | Share of reports |
      |---|---|
      | Qwen | 0.3–0.8% |
      | Huatuo | 2–3% |
      | Lingshu, train | 51% |
      | Lingshu, held-out test | **64%** |

      - Lingshu's fabrication persists on held-out data.
      - The P6 figure of "Qwen 33%" was regex noise.
    - **Image use differs sharply.** Image specificity on CXR is near zero for Qwen and Huatuo (0.01–0.02
      nats/token) vs 0.06–0.12 for Lingshu.
    - **Lingshu probably memorized the training reports.** NLL train vs test: Lingshu 1.29 vs 1.79; Qwen 2.38 vs 2.67.
    - **The held-out "first study" group is not a valid control.** 69% of its reference reports compare, because
      earlier exams are missing from the release. Use the has-prior group.
13. **Zoom does not help short answers either (V6b).**
    - Crop minus random crop, F1: +0.006 / +0.001 / −0.004 (CIs ±0.015). Base F1 is 0.09–0.13.
    - The `<look>` action is dropped.
14. **V0 time split.**
    - Qwen3-8B cannot label "general", which it never chose. Methods sentences and titles fall into "post" (69%).
    - The "pre" class looks precise on spot checks.
    - Even that conservative pre-only context carries 64–68% of the non-visual VOI on context items:
      - Qwen 1.47 vs 2.16;
      - Lingshu 1.32 vs 1.86;
      - Huatuo 1.24 vs 1.85.
    - 43% of hard context items cite only "post" evidence (noisy).

## 2. What this means for the central idea

- **The claim "the model is aware it needs more context" is not supported in expressed form.** Every expressed
  signal works only when the question wording reveals the need. It drops to chance when the wording is fixed and
  the case decides. Generic uncertainty knows "I may be wrong" (weakly), not "I am missing information".
- **There is a latent, image-dependent signal**, in two places:
  - within-template probes, 0.69–0.75 with the correct image vs about 0.5 with a mismatched one;
  - the sentence-level self-report image effect.
- The defensible research claim is therefore: *Med-MLLMs do not express context-need awareness. Some of it is
  latent, and post-training with privileged credit can align the request policy with it.* This holds only if the
  image-encoder control does not explain the probe.
- **Consequences for the method design** ([privilege_aware_design.md](privilege_aware_design.md) §6):
  - Keep **observation matching**: the teacher sees only what the student received, plus visual privilege.
  - Replace attainability routing with **label routing**: distil V-typed tokens, credit P-typed ones to the request.
  - Use **hint privileges** (boxes, entity names) in preference to caption sentences.
  - Drop `<look>`.
  - The SFT baseline (privilege-blind vs privilege-filtered) matters more than on-policy routing, because that is
    where the illusion lives.
  - Fixed-template items, with VOI as the label, are the right training and evaluation set for the request decision.
    The P4-style QA is not.

## 3. Caveats

- All V/P/K/M and pre/post labels come from Qwen3-8B. The time labels are poor outside "pre". The human audit sheet
  is ready but not filled.
- Correctness uses token-F1 ≥ 0.5. Very few items are right without context (S = 4–11% on qa), and on "nature" S
  includes lucky benign/malignant guesses. The VOI-based tasks agree with the correctness-based ones.
- The hard-item evidence typing depends on the LLM's evidence citation. "Diagnosis" is almost always typed context
  (736/800).
- Huatuo and Lingshu are contaminated on PubMedVision answers (target A). Lingshu is likely contaminated on MIMIC
  train.
- Probes use PCA-256 and grouped CV with C = 0.01. Within-template n is 280–800, with wide CIs on "diagnosis".

## 4. Next validation steps (no training)

1. **Image-only probe control:** probe the vision tower's pooled features, or BiomedCLIP's, for the same
   within-template labels. If they match the LM-state probe, the "latent awareness" is image content, not
   self-knowledge.
2. **Fill the audit sheet:** `aware/audit_labels.csv`, 200 V/P/K/M + 100 pre/post labels.
3. **Redo V0** with a better time prompt or a stronger labeler; add a "not about this patient" class.
4. **Score hard-item answers with an LLM equivalence judge** instead of token-F1, to firm up S/CR/PL.
5. **Latent-to-expressed test, still without RL:** a few-shot or instruction-only prompt that asks the model to
   reason about the specific image before deciding to request. Check whether stratified request AUROC moves toward
   the probe's 0.7. If prompting cannot surface it, that is direct motivation for the training stage.
