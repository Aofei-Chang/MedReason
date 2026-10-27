# Validation results: visual vs. non-visual context (run 2026-10-25)

Summary of the validation run P1b–P4 and P6, for later sessions. There was no training: RL (P5) is out of
scope for this phase. Parent plan: [post_training_plan.md](post_training_plan.md) §4. Earlier pilot (P1): same file, §3.

- **Raw outputs:** `/data/aofei/output/ContextMLLM/overnight/`
  - `RESULTS.md` and `results.json` from `analyze_overnight.py`
  - `p4_rescored.json`
  - per-stage logs in `logs/`
- **Scripts:** `experiments/context_expansion/`
  - `run_overnight.py`, the driver; 3.96 h on GPUs 1–3, all stages finished
  - `build_overnight_sets.py`, `label_sentences.py`, `score_typed_context.py`, `score_zoom.py`, `gen_free.py`,
    `judge_free.py`, `qa_build.py`, `protocol.py`, `p4_zero_shot.py`, `mimic_p6.py`

## 0. Setup in one paragraph

**Data and models.**
- **Models:** Qwen2.5-VL-7B-Instruct (general) and Lingshu-7B (medical, trained on a PubMedVision-family mix and
  probably on MIMIC-CXR).
- **d2k:** 2,000 single-image CT/MRI/US PubMedVision records linked to their PMC article: the 600 P1 records plus
  1,400 new ones.
- **box:** 600 records with MedSAM3 entity boxes, from disjoint articles.
- **mimic:** 900 MIMIC-CXR studies, 600 with a prior report and 300 first studies.

**Typing.** Qwen3-8B (text only) typed 48.7k sentences as V (visual finding), P (non-visual patient/clinical),
K (knowledge) or M (figure meta). This covers both the *target* sentences and the *context* sentences.
- Context split: V/M sentences form the **visual context**; P/K sentences form the **non-visual information context**.
- Article excerpt mix: 18.1k P, 4.0k K, 3.6k V, 1.0k M.
- Caption mix: 2.2k V, 1.9k P, 0.9k M, 0.1k K.

**Metric.** Gains are teacher-forced log-likelihood differences in nats/token, with 95% CIs from a
record-clustered bootstrap.
- dC_x = l(I, C_x) − l(I), the context gain.
- dI|x = l(I, C_x) − l(C_x), image reliance once context is present.

**Targets.**
- A = the GPT-4V instruction answer; its context is the caption.
- B = the original caption; its context is the article (title + abstract + figure-citing paragraphs).
- P2a tables use only records where both context parts are non-empty.

## 1. Headline conclusions

1. **The visual / non-visual split is real and large: a clean crossover.**
   - Visual context mainly helps visual sentences; non-visual context mainly helps patient sentences.
   - On captions (target B, Qwen):
     - Visual sentences gain +0.43 from visual context and only +0.09 from non-visual context.
     - Patient sentences gain +0.47 from non-visual context and only +0.03 from visual context.
   - Interaction: +0.77 [0.69, 0.86] for Qwen, +0.68 for Lingshu.
   - Instruction answers (A) show the same direction, smaller: +0.23 Qwen, +0.15 Lingshu.
2. **Typing the context sentences is far better than the P1 regex redaction.**
   - Keeping only the non-visual part (target B, Qwen):
     - removes **85%** of the context gain on visual sentences (0.61 → 0.09);
     - keeps **78%** of the gain on patient sentences (0.59 → 0.47).
   - Lingshu: −68% and 82% kept.
   - Removing sentences that cite the figure (regex) removes only ~20–25% of the visual leak.
   - Semantic typing of context is the way to build "complementary context".
3. **A gate that retrieves only non-visual context fires less on perception cases, but still not cleanly.**
   - Measured among uncertain sentences the context resolves (NLL ≥ median, dC ≥ 0.25). Share of these that
     are visual:

     | Context used by the gate | Target A | Target B |
     |---|---|---|
     | Full context | 38% | 53–56% |
     | Visual context only | 50–54% | 68–75% |
     | Non-visual context only | 24–26% | 41–42% |

   - The remaining visual share is partly label noise, and partly visual findings implied by the clinical text
     (for example, the diagnosis is stated in the history).
4. **Effect on image reliance (the shortcut) is real but small.**
   - For visual sentences, the image gain once context is present is slightly higher with non-visual context than
     with visual context. Interaction: +0.03 to +0.05, CI excludes 0.
   - The big effect of visual context is on the likelihood itself, not on image reliance.
5. **Gains are information-specific, not a priming effect.** Context from a mismatched article *lowers*
   likelihood (−0.03 to −0.33), so the dC values understate the real information gain.
6. **Self-report "verifiable from the image?" is mostly textual (P1b).**
   - It separates P from V just as well without the image:
     - Qwen: 0.73 → 0.78 on A; 0.78 → 0.78 on B.
     - Lingshu: 0.78 → 0.82 on A; 0.78 → 0.78 on B.
   - The image does raise p(yes), more for visual sentences than for patient sentences:
     - Qwen: +0.19 vs +0.08 on A; +0.16 vs +0.05 on B.
     - Lingshu: +0.35 vs +0.24 on A; +0.28 vs +0.15 on B.
   - The image-induced shift alone has AUROC 0.68–0.75, so there is *some* image sensitivity. It may be a generic
     "an image is present" effect rather than "this image shows it". This is untested: it needs a
     mismatched-image control.
   - Conclusion unchanged from P1: image-specific "I can't see this" awareness is not there and would have to be
     taught.
7. **Zoom (P2b) gives no measurable perception gain.**
   - A crop to the MedSAM3 box, added as a second image, changes likelihood exactly as much as a random crop of
     the same size. Zoom − random is ≈ 0.000 ± 0.005 for V, P, and V sentences that name the boxed entity, in both
     models.
   - Non-visual context in the same set does help P more than V: Qwen +0.075 [0.02, 0.13]; Lingshu +0.034
     [−0.00, 0.07].
   - Either the models ignore the second image, or teacher-forced likelihood of long GPT-4V answers is insensitive
     to it. **A `<look>` action is not supported by this test.** It needs a short-answer accuracy test before it
     is dropped.
8. **Zero-shot requesting (P4) is easy because the question text gives the type away. The full context leaks the
   image answers.**
   - Qwen requests on 85% of context questions and 4% of image questions (precision 0.96). Lingshu barely follows
     the format: 94% of its image-question replies and 36% of its context-question replies are untagged.
   - Image-question F1 (rescored so untagged replies count as answers):

     | Model | No context | Non-visual context given | Full context given |
     |---|---|---|---|
     | Qwen | 0.08 | 0.19 | **0.73** |
     | Lingshu | 0.19 | 0.22 | **0.70** |

     The full context hands over the image answer, and the non-visual context does not.
   - Context-question F1 is 0.05–0.06 with no context and about 0.45–0.47 with either context, so the items do
     need context.
   - The current QA cannot test *self-aware* requesting, because the question wording alone decides it. It needs
     items whose wording does not reveal whether they need context.
9. **MIMIC-CXR (P6) shows the same split, and a likely image → "has prior" shortcut in Lingshu.**
   - **Qwen:** the prior report raises comparison sentences about twice as much as other sentences (+0.24 vs +0.11).
     Without a prior, Qwen writes comparison language in 33% of reports whether or not a prior exists (30% for
     first studies). That is fabrication at about the regex noise floor.
   - **Lingshu:**
     - It ignores the prior report (dC ≈ +0.02).
     - The image itself raises the likelihood of comparison sentences: +0.25 with a prior, +0.36 for first
       studies, vs +0.17 for other sentences.
     - In generation without a prior, it writes comparisons for 54% of patients who have a prior vs 21% of
       first-study patients.
   - So Lingshu predicts *that a prior exists* from the image, a learned non-visual inference: devices or
     inpatient appearance → follow-up exam. The other candidate is memorization: Lingshu's NLL is about 1.0–1.5 vs
     about 2.1–2.8 for Qwen, so it has likely seen these reports.
   - For reference, the regex flags comparison language in 85% of reference reports with a prior and in 32% of
     first studies, which is regex noise (words like "new", "increased").
10. **P3 (unsupported patient claims in free text) is not interpretable yet.**
    - The Qwen3-8B judge labels many visual findings and general statements as "patient claims". In a spot
      check of 12, most were not patient facts.
    - The raw counts are directionally plausible but unusable:

      | Source | Unsupported patient claims per text | Contradicted share of patient claims |
      |---|---|---|
      | Qwen | 4.0–4.9 | 8–14% |
      | Lingshu | 2.0–2.8 | 14–22% |
      | PubMedVision reference answers | 1.85 | 3% |

    - It needs a redesign before any conclusion (§4).

## 2. Key tables

### 2.1 P2a typed context, target B (caption | article context), dC in nats/token

| Model | Class (n) | Full context | Visual context | Non-visual context | Regex-redacted | Mismatched article |
|---|---|---|---|---|---|---|
| Qwen | V (1832) | +0.61 | +0.43 | +0.09 | +0.46 | −0.33 |
| Qwen | P (1452) | +0.59 | +0.03 | +0.47 | +0.53 | −0.29 |
| Lingshu | V (1832) | +0.67 | +0.55 | +0.22 | +0.54 | −0.15 |
| Lingshu | P (1452) | +0.66 | +0.19 | +0.54 | +0.60 | −0.14 |

### 2.2 P2a target A (instruction answer | caption context)

| Model | Class (n) | Full context | Visual context | Non-visual context | Crossover V−P |
|---|---|---|---|---|---|
| Qwen | V (595) | +0.41 | +0.30 | +0.14 | +0.23 [0.18, 0.28] |
| Qwen | P (1209) | +0.28 | +0.11 | +0.18 | |
| Lingshu | V (595) | +0.24 | +0.18 | +0.07 | +0.15 [0.11, 0.19] |
| Lingshu | P (1209) | +0.16 | +0.06 | +0.10 | |

On target A, K sentences (673) gain little from any context: ≤ 0.08.

### 2.3 P4 zero-shot (QA test split, 150 image + 150 context items; rescored F1)

| Model | Request rate on image items | Request rate on context items | F1 image: none / non-visual / full | F1 context: none / non-visual / full |
|---|---|---|---|---|
| Qwen | 0.04 | 0.85 | 0.08 / 0.19 / 0.73 | 0.05 / 0.45 / 0.45 |
| Lingshu | 0.01 | 0.52 | 0.19 / 0.22 / 0.70 | 0.06 / 0.45 / 0.47 |

QA set: 1,313 image and 1,898 context items were kept. Another 427 image items were dropped because their
answer also appeared in the non-visual context.

### 2.4 P6 MIMIC-CXR

| Model | Sentences | dI (image gain) | dC from prior report | NLL |
|---|---|---|---|---|
| Qwen | comparison, has prior | +0.04 | +0.24 | 2.75 |
| Qwen | other, has prior | +0.04 | +0.11 | 2.12 |
| Lingshu | comparison, has prior | +0.25 | +0.02 | 1.51 |
| Lingshu | comparison, first study | +0.36 | – | 1.26 |
| Lingshu | other | +0.17–0.18 | +0.01 | 0.98–1.09 |

Share of generated reports with comparison language when no prior is in the input:

| Model | Has prior | First study | Prior given |
|---|---|---|---|
| Qwen | 33% | 30% | 80% |
| Lingshu | 54% | 21% | 75% |
| Reference reports | 85% | 32% | – |

## 3. Caveats

- All V/P/K/M labels come from one LLM (Qwen3-8B) with no human audit. Mixed sentences, such as a visual
  finding plus a history clause, are forced into one class.
- P2a uses records where both context parts are non-empty, which shrinks target A, and a single τ = 0.25.
- Lingshu has likely seen PubMedVision answers and MIMIC-CXR reports (low NLL), which confounds its bias results.
- P3 judge is unreliable (see above). The P6 comparison detector is a regex with a ~30% false-positive floor.
- P4 F1 is token overlap, a strict measure for short answers. Lingshu ignores the tag format.
- P2b's "zoom" is a second image in the prompt, scored by teacher-forced likelihood; a null result here may be
  method-specific.

## 4. Suggested next validation steps (no training)

1. **Human audit:** check 200 sentence labels and 100 P3 claims, to put a number on typing noise.
2. **Self-report instance-specificity:** run it with a *mismatched image*. If p(yes) for V sentences rises just as
   much with the wrong image, the image effect is generic.
3. **Hard P4 items:** use one question template whose answer is visual for some images and needs context for
   others (for example "What is the diagnosis?", typed by whether the caption's visual sentences settle it).
   That tests self-aware requesting rather than reading the question.
4. **Redo P3** with claim extraction first, then per-claim verification with a stronger judge or a human check.
   Keep the reference-vs-model comparison.
5. **Retest zoom with short-answer accuracy** on image items: crop vs random crop, in generation mode.
6. **P6 on held-out data:** use the MIMIC test split or a non-MIMIC CXR set to separate memorization from the
   learned image → "has prior" inference. Replace the regex with an LLM comparison detector.
7. **Scale to inline-type records:** the Europe PMC search is complete (317k captions) and has not been mapped
   into records yet.
