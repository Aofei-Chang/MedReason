# Privilege-aware post-training: next steps after the 2026-10-25 validation run

Status: design and validation plan (2026-10-26). There is no training code yet, per the current phase rule
"validate before method". This builds on [validation_results_2026-10-25.md](validation_results_2026-10-25.md)
and replaces the reward design in [post_training_plan.md](post_training_plan.md) §5 if the validations in §4
below hold.

## 1. What DOPD contributes

DOPD = *Dual On-policy Distillation*, Yu et al., arXiv [2606.30626](https://arxiv.org/abs/2606.30626), Jun 2026.

These notes come from the abstract and the HF paper page. The PDF could not be parsed here, so check the exact
equations before citing them.

- **Privileged information (PI) p.** Extra input that exists at training time only:
  - LLM experiments: step-wise hints, with no final answer.
  - VLM experiments: query-related bounding boxes with labels.
- **Privilege illusion.** If only the teacher gets p, the teacher–student gap mixes two things: real capability
  differences, and advantages that come *only from the extra input*, which the student can never acquire.
  - Uniform distillation on that gap gives early gains, then late degradation and entropy collapse.
  - Giving p to only one side shows the same pattern.
  - Giving p to *both* sides removes most of the apparent advantage.
- **The metric (privilege-matched advantage gap).** Per sampled token, take the gap between the log-probabilities
  of the privileged teacher π_T(·|x,p) and the *privileged student* π_θ(·|x,p): the same weights as the deployed
  student, but also given p. Both sides hold the same privilege, so the gap measures capability, not information.
  - Normalized per batch; the top 5% are dropped as outliers.
- **Routing.** Four regimes, defined by the gap and the teacher/student token probabilities:
  - High gap, teacher ≥ student → full-vocabulary JS divergence to the teacher. These are the
    "capability-bearing" tokens.
  - High gap, student higher → light top-K reverse KL to a stop-gradient self-anchor.
  - Low gap, both probabilities high → light reverse KL to the teacher.
  - Low gap, both probabilities low → weak anchor to self.
  - Settings: β_w = 0.3, β_l = 0.6, K = 128.
- **Result relevant to us.** Giving the *final answer* as PI caused shortcut learning and could do worse than no
  PI at all. PI that the student can partly reproduce (hints, boxes) works best.

Related self-distillation line, where teacher = the same model given PI, and its known failure mode:
- OPSD ([2601.18734](https://arxiv.org/abs/2601.18734)) and π-Distill ([2602.04942](https://arxiv.org/abs/2602.04942)).
- RLSD ([2604.03128](https://arxiv.org/abs/2604.03128)) and DemoPSD ([2607.02502](https://arxiv.org/abs/2607.02502)):
  both report *information leakage* from the privileged teacher.
- [2607.05184](https://arxiv.org/abs/2607.05184): privileged-context OPSD hurts thinking models.

## 2. Our problem *is* a privilege-illusion problem

PubMedVision-style SFT is privilege-blind distillation:
- The "teacher" is GPT-4V with the caption, or the article author with the full patient record.
- The student sees only the image.

Our results already show both halves of DOPD's diagnosis:

| DOPD notion | Our evidence |
|---|---|
| Teacher advantage that is pure information | Non-visual context raises P sentences by +0.47 vs V by +0.09 (P2a). That gain is unreachable from pixels. |
| Shortcut from answer-like PI | Visual context hands over image answers: P4 image F1 goes from 0.08 to 0.73. Caption gains on V sentences barely depend on the image: the image-reliance interaction is only +0.03 to +0.05. |
| Illusion learned by a privilege-blind student | **P6 Lingshu.** It was trained on reports written *with* a prior, but sees no prior. It writes comparison language for 54% of patients who have a prior (21% for first studies) without seeing one, and ignores a real prior when given it (+0.02). That is the expected outcome of distilling an unacquirable advantage. Memorization is still a confound. |

### Where we differ from DOPD, and why it matters

DOPD has *one* kind of privilege, and it is unacquirable. We have three, and P2a lets us separate them:

| Information | Example | Can the deployed model get it? | Treatment |
|---|---|---|---|
| **N_pre**: non-visual, known at reading time | history, labs, indication, prior report | **Yes, by `<request>`** | Matched privilege: the teacher gets it only if the student requested it. Its value is credited to the *request action*, not to the answer tokens. |
| **V**: visual context | caption or article sentences restating findings | As text, no. Its *content* sometimes, by perception. | Distil only on tokens where the content is image-attainable; elsewhere it is an illusion → self-anchor. |
| **N_post**: non-visual, not knowable at reading time | biopsy result, outcome, treatment response | Never | Never given to the teacher or returned by a request. The pure DOPD case. |

The N_pre / N_post split is new. The P labels do not separate them yet; see V0 in §4.

So DOPD's "make both sides privileged" becomes **observation matching**: the teacher sees exactly what the
student observed in its rollout, plus V. The only open question is which V-driven tokens the student can
attain.

## 3. Proposed metric: per-token privilege decomposition

Notation:
- ℓ(o) = log π(y_t | o, y_<t) for a token y_t of a student rollout (on-policy) or of a reference.
- I = the image; Ĩ = a mismatched image from another record of the same modality.
- R = the observations the student actually received: N_pre if it requested, otherwise ∅.

**Total gap.** This is what privilege-blind OPSD/SFT distils:

    G = ℓ(I, V, N_pre, N_post) − ℓ(I, R)

It telescopes into three parts:

    G_post = ℓ(I, V, N_pre, N_post) − ℓ(I, V, N_pre)
        unacquirable → never distilled (the illusion mass)

    G_acq  = ℓ(I, N_pre) − ℓ(I, R)
        zero if the student requested; otherwise the value of information of a request
        → credit goes to the request decision

    G_vis  = ℓ(I, V, N_pre) − ℓ(I, N_pre)
        visual-privilege gain, with non-visual information matched (DOPD's "dual")

(G_vis is shown here with N_pre on both sides. In training, R replaces N_pre and the remainder goes into G_acq.)

**Attainability of G_vis.** Is the teacher's advantage on this token something the *image* can supply? This plays
the role of DOPD's capability gap. Two candidates, compared in V1:

    A_teach = ℓ(I, V, R) − ℓ(Ĩ, V, R)
        Does the privileged teacher still need the right image? If its prediction is the same with the
        wrong image, the token comes from text alone → illusion.

    ρ = [ℓ(Ĩ, V, R) − ℓ(Ĩ, R)] − [ℓ(I, V, R) − ℓ(I, R)]
        Image–caption redundancy. The caption adds less when the right image is present → the right image
        already carries part of the content.

Using Ĩ instead of no image controls for the generic "an image is present" effect suspected in P1b. A third,
external option is a stronger teacher given the same matched inputs. That is DOPD exactly, with a capability gap
between teacher and student. See V4.

**Routing.** A DOPD adaptation, offered for discussion; nothing is implemented yet:

| Regime | Condition | Signal |
|---|---|---|
| Perceptual distillation | G_vis high, A high | JS to the matched teacher (strong) |
| Illusion | G_vis high, A low | No teacher pull; weak stop-gradient self-anchor |
| Contradiction | G_vis strongly negative: the caption contradicts a token the student wrote | Pull toward the teacher (suppression only). This is the hallucination correction; whether it needs A gating is open. |
| Neutral | \|G_vis\| small | Light top-K reverse KL to the teacher (stability) |
| Request decision | — | Advantage = VOI = mean of G_acq over the answer tokens, minus cost c. Penalizes not requesting when VOI is high and requesting when VOI ≈ 0. |
| Unrequested patient claims | P token, no request | Matched teacher has no N → no pull toward lucky guesses. An optional hedge term is open. |

**What this replaces in §5 of the plan.**
- The *necessity credit* and the *unsupported-claim penalty* (both judge-based) become likelihood-based
  privileged credit: VOI and illusion routing.
- No claim judge is needed during training. That matters because the P3 judge failed.
- Typing is needed only for the *context* split (V / N_pre / N_post), done once offline, not for every
  generated claim.

## 4. Next validation steps (no training)

Ordered by what most decides go / no-go. Rough GPU time is for GPUs 1–3, two 7B models.

| # | Experiment | Data | Decides | ≈ GPU-h |
|---|---|---|---|---|
| **V0** | Relabel the P context sentences into N_pre vs N_post with Qwen3-8B, then human-check 200, merged with the existing 200-label audit. | d2k article and caption context | Whether the three-way split is labelable; the N_post share | 0.5 + human |
| **V1** | **Token-level decomposition** on reference targets A/B. Per token, compute G, G_post, G_acq, G_vis, A_teach and ρ. This needs 7 conditions: (I,∅), (I,N_pre), (I,V,N_pre), (I,V,N_pre,N_post), (Ĩ,∅), (Ĩ,V,∅), (Ĩ,V,N_pre). | d2k (2k records); box set for grounded-entity tokens | (a) Is attainability bimodal at token level, even though the sentence-level mean was only +0.03 to +0.05? (b) AUROC of A_teach and ρ for V vs P tokens and for MedSAM-grounded entity tokens. **Go if > 0.7.** (c) How much of G is G_post + G_acq, i.e. the illusion mass under privilege-blind distillation. | 3–4 |
| **V2** | **Illusion audit on on-policy rollouts.** Score the existing `gen_free_*` generations (600 images × 2 prompts × 2 models) with the teacher under three conditions: privilege-blind (full context), matched (V + R), and matched + routing. Type the generated sentences with the existing labeler, not the P3 judge. | `gen_free_*.jsonl` (already generated) | Share of positive teacher pull that lands on unsupported P sentences, by condition. Prediction: blind ≫ matched. This is the direct "privilege illusion" measurement for our setting. | 2 |
| **V3** | **VOI as the request oracle.** Compute G_acq on P4 items, plus **hard items**: one question template whose answer is visual for some images and needs context for others (results §4.3). Compare against entropy, self-report, and the hidden-state probe. | qa.jsonl + new hard items | Can a likelihood-based VOI decide request vs answer when the question wording cannot? It must beat self-report on the hard items. | 1.5 + item build |
| **V4** | **External teacher (true DOPD).** Use HuatuoGPT-Vision-7B (already cached) or Lingshu-7B as the teacher for a Qwen student, with matched inputs. Measure the capability gap on V tokens, compared with A_teach. A 32B teacher needs about 65 GB of the 106 GB free on /data, so it is optional. | d2k subset (500) | Whether a separate teacher gives a cleaner attainability signal than the image counterfactual. | 2 |
| **V5** | **MIMIC as the clean testbed.** Use the same decomposition with N_pre = prior report and indication. Comparison tokens should carry G_acq; findings tokens should carry G_vis. Use a held-out split, or a non-MIMIC CXR set, to address Lingshu memorization (results §4.6). | mimic.jsonl + held-out | Whether the decomposition recovers the known structure. This is the sanity check for the metric itself. | 1.5 |
| V6 | Carried over: the self-report mismatched-image control, and zoom retested with short-answer accuracy. | as in results §4 | Whether `<look>` stays in scope, and how much the generic image effect inflates A. | 1.5 |

V1, V2, V3 and V5 together fit one overnight run (~8–9 GPU-h over 3 GPUs). V1, V2 and V5 reuse the
existing `score_typed_context.py` teacher-forcing code: add per-token log-prob output and the Ĩ conditions.

### Decision rules

> **Outcome (run 2026-10-26, [awareness_results_2026-10-26.md](awareness_results_2026-10-26.md)):**
> - V1: attainability AUROC is 0.48–0.61 → fall back to label routing. The matched-observation teacher holds:
>   about 60% of the privilege-blind gap is non-visual.
> - V2: on-policy illusion mass is small (P share of pull 24–28% blind vs 23–26% matched) → the illusion lives in
>   SFT on reference targets.
> - V3: expressed need is at chance once the wording is fixed. Only within-template probes reach about 0.7 → the
>   request decision needs privileged (VOI) credit.
> - V4: the external-teacher gap is not V-specific.
> - V5: passes for Qwen and Huatuo.
> - V6: zoom is null; the self-report image effect is instance-specific.

- **A_teach / ρ AUROC < 0.65 (V1):** image-counterfactual attainability is too weak at 7B.
  - Fall back to the external teacher (V4) or to label-based routing (V-typed tokens distilled, P-typed not).
  - Either way, the matched-observation teacher still holds.
- **Illusion mass small (V1c / V2 < 10% of positive pull):** DOPD-style routing adds little over plain
  observation matching. Keep the simpler method.
- **VOI fails on hard items (V3):** the request decision needs privileged labels (the PCQC-style necessity
  credit), not VOI.
- **V5 fails to put comparison tokens in G_acq:** the decomposition is not measuring what we think. Stop and debug
  before using it anywhere.

## 5. Open design questions, for the advisor

1. **Self-teacher or external teacher?**
   - Self-teacher (OPSD-style) has no capability gap. Every advantage is privilege, so attainability must come
     from the image counterfactual.
   - External teacher is closer to DOPD but needs a stronger medical VLM.
2. **Should V-driven suppression ("Contradiction") be ungated?** Suppressing a finding the model cannot see is
   conservative, but it could still teach dataset priors.
3. **Is N_post knowledge or illusion?** "This lesion was later biopsy-proven X" is unacquirable for *this*
   patient, but the population-level association (appearance → diagnosis) is partly visual knowledge. DOPD has
   no analogue, and V0 will show how often this occurs.
4. **Is distillation the right vehicle at all?** VOI + routing could equally be advantages in GRPO. On-policy
   distillation gives dense token credit at low cost, and RL gives outcome grounding. Defer until V1–V3 are in.

## 6. End-to-end post-training design (proposal, 2026-10-26)

This is design text only. Nothing gets implemented until V1–V3 and V5 are in.

### 6.1 Target behaviour, and which result each part answers

| The trained model should... | Why it is needed (from the validation results) |
|---|---|
| State visual findings only when the image supports them | The visual context is a text shortcut; image reliance barely moves (+0.03 to +0.05) |
| Request non-visual information when the answer depends on it, *from the image and question jointly* | Zero-shot requesting reads the question wording (P4); self-report is textual (P1b) |
| Not fabricate patient facts or comparisons without the information | Lingshu: comparisons for 54% of patients with a prior vs 21% of first studies, with no prior given |
| *Use* returned context | Lingshu ignores a given prior (+0.02) |
| Keep the output format | Lingshu ignores tags (P4) |

`<look>`/zoom is dropped for v1 because P2b was null. It comes back only if V6's accuracy test is positive.

### 6.2 Episode and environment

- **Input:** image I and a task q, which is one of: a short VQA question, "describe", or "write the report"
  (MIMIC).
- **Actions:**
  - `<answer>…</answer>`
  - at most one `<request kind="history|prior|labs">query</request>`. The environment returns this sample's
    **N_pre pool only**. Visual-context sentences and N_post are never returned, so a request cannot leak the
    image answer.
- Returned context tokens are masked from the loss.

### 6.3 Data mix (balanced, as MMSearch-R1 found essential)

| Pool | Source | Role |
|---|---|---|
| Image items | Caption V sentences → short QA. The answer must not be in N_pre (the existing leak filter). | Answer without requesting |
| Context items | N_pre facts → short QA | Request, then answer |
| **Paired hard items** | Same template on the same kind of image ("most likely diagnosis?", "is this new?"). Labelled by VOI (V3), not by wording. | The only items that test the image-and-question decision |
| Descriptions | PubMedVision images with "describe" | Dense token training against fabrication |
| MIMIC reports | Indication given; prior requestable; first studies have none | Comparison fabrication vs prior use |

Held out: whole articles, plus the MIMIC test split, or a non-MIMIC CXR set because of the memorization confound.

### 6.4 Learning signal: RL decides, distillation shapes tokens

A self-teacher only differs from the student in V. So distillation alone cannot train two things:
- the request decision;
- the use of returned context: teacher and student both see R, so the signal is zero there.

Hence a **hybrid**, in the spirit of RLSD (RLVR + self-distillation):

    L = L_GRPO(r)  +  λ · L_PM-OPD  +  β · KL(π_θ ‖ π_ref)

**Outcome reward r (sequence level):**
- r_correct: exact match or token F1 for short QA. For MIMIC, a label F1 on findings and comparisons (a
  CheXbert-style labeler or an LLM extractor). Descriptions get no outcome term; L_PM-OPD covers them.
- r_req = −c per request, + VOI shaping: a bonus if the item's VOI > τ and the policy requested; a penalty if it
  requested when VOI ≈ 0. VOI is precomputed offline from G_acq on the reference answer.
- r_fab = −α per generated sentence that is typed P (by the existing Qwen3-8B sentence labeler) *and* not
  supported by R. The support check uses only R as the premise, which is small, so it is far easier than the
  failed P3 article judge. On MIMIC: −α per comparison sentence when no prior was received.

**Privilege-matched on-policy distillation L_PM-OPD (token level):**
- **Teacher:** a frozen copy of the initial policy (or EMA), conditioned on (I, q, R, PI). The teacher sees the
  student's own observations (observation matching, DOPD's "dual"), plus visual PI only. It never sees N_post,
  or N_pre the student did not request.
- **Routing** follows §3: strong JS on attainable V tokens, suppression on contradicted tokens, a self-anchor on
  illusion tokens, and light KL elsewhere.
- **The choice of PI is an experiment.** DOPD found answer-like PI causes shortcut learning, and hints or boxes
  work best. Our caption V sentences *are* answer-like. Candidates, weakest last:
  - (a) V sentences;
  - (b) the MedSAM3 boxes + entity labels we already have (the box set) — DOPD's best VLM PI;
  - (c) entity names only.
  - V1 should add (b) and (c) as teacher conditions, to see which keeps attainable gain while cutting
    the leak.

### 6.5 Stages

1. **S0, format warm-start (SFT, small).** About 2k templated traces, with **privilege-filtered targets**: P
   sentences are removed from the answers unless the trace includes a request that returned them. This teaches
   the tags (needed for Lingshu) without reinstalling the illusion.
2. **S1, pilot.** Qwen2.5-VL-3B: hybrid objective on the QA + paired + description mix.
3. **S2, scale.** Qwen2.5-VL-7B and Lingshu-7B, adding MIMIC. Lingshu tests repair of an already-installed
   illusion.

Infrastructure is not in place: TRL or verl must be installed, and /data has 106 GB free.

### 6.6 Baselines and ablations (each isolates one claim)

| Run | What it tests |
|---|---|
| B0: base model | Reference point |
| B1: SFT on PubMedVision targets (privilege-blind) | Should *increase* fabrication. The privilege illusion via SFT. |
| B2: privilege-filtered SFT (= S0 at full scale) | Strong cheap baseline: is any on-policy method needed? |
| B3: OPSD with a full-context teacher (privilege-blind) | DOPD's prediction: early gain, then entropy collapse and fabrication |
| B4: observation-matched OPSD, no routing | Value of matching alone |
| B5: GRPO, outcome only (MMSearch-R1-like) | Value of dense privileged credit |
| **Full:** matched + routing + VOI shaping + r_fab | |
| Ablations | PI type (a/b/c); attainability metric (A_teach / ρ / labels); no VOI shaping; no r_fab |

### 6.7 Evaluation

- **Decision:** request precision and recall on *paired hard* items (the main metric), plus per-type accuracy.
- **Fabrication:**
  - unsupported P sentences in free descriptions with no context;
  - MIMIC comparisons in first studies (LLM detector, against the reference floor);
  - prior use when given.
- **Perception kept:**
  - mismatched-image likelihood gap must not shrink;
  - VQA-RAD, SLAKE and PMC-VQA must not drop.
- **Training health:** entropy curve and fabrication curve over steps. This is where B3's illusion should show.
- All reported on held-out articles; MIMIC also on the held-out split.

**Predicted ordering**, which is the falsifiable claim of the paper:
- fabrication: B1 > B3 > B0 > B5 > B2 ≈ B4 > Full;
- VQA accuracy: Full ≥ B0;
- request precision on hard items: Full ≫ B2 and B0 (they cannot use the image for the decision).
