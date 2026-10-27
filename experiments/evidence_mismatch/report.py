"""Render an inspectable local research report from completed measurements."""
import csv
import html
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'
E=html.escape

def main():
    summary=json.loads((OUT/'summary.json').read_text())
    audit=json.loads((OUT/'audit.json').read_text())
    age=json.loads((OUT/'age_provenance.json').read_text())
    facts=json.loads((OUT/'fact_summary.json').read_text())
    manifest=[json.loads(l) for l in (OUT/'manifest.jsonl').read_text().splitlines()]
    records=list(csv.DictReader((OUT/'per_record.csv').open()))
    def stat(model,category,metric,subset='all'):
        return next(s for s in summary['summaries'] if s['model']==model and s['category']==category and s['subset']==subset)['metrics'][metric]
    def fmt(s):
        return f'{s["mean"]:.3f} [{s["ci95"][0]:.3f}, {s["ci95"][1]:.3f}]'
    tables=''
    for subset in ['all','medsight_60k','full_only']:
        tables+=f'<h3>{subset}</h3><table><tr><th>Model / group</th><th>n</th><th>Image identity gain</th><th>True − donor prefix, no image</th><th>Caption specificity</th><th>Image rank top-1</th></tr>'
        for model in ['huatuo','medsight']:
            for cat in ['visual_description','diagnostic_inference','external_context']:
                tables+=f'<tr><td>{model} / {cat}</td><td>{stat(model,cat,"image_identity_gain",subset)["n"]}</td>'
                for metric in ['image_identity_gain','prefix_specificity_no_image','caption_specificity','image_top1']:
                    tables+=f'<td>{fmt(stat(model,cat,metric,subset))}</td>'
                tables+='</tr>'
        tables+='</table>'
    audit_table='<table><tr><th>Corpus</th><th>Records</th><th>Answer marker, category absent from question</th><th>Screening rate</th></tr>'
    for corpus,c in audit['counts'].items():
        n=c['records']; k=c['answer_marker_category_absent_question']
        audit_table+=f'<tr><td>{corpus}</td><td>{n:,}</td><td>{k:,}</td><td>{k/n:.2%}</td></tr>'
    audit_table+='</table>'
    age_table='<table><tr><th>Corpus</th><th>Answer contains numeric age</th><th>Answer age absent question</th><th>Same age in source caption, absent question</th></tr>'
    for corpus,c in age['counts'].items():
        age_table+=f'<tr><td>{corpus}</td><td>{c["answer_numeric_age"]:,}</td><td>{c["answer_age_absent_question"]:,}</td><td>{c["source_shared_age_absent_question"]:,}</td></tr>'
    age_table+='</table>'
    fact_table='<table><tr><th>Model / score scope</th><th>n</th><th>Image identity gain</th><th>True − donor prefix, no image</th><th>Caption specificity</th></tr>'
    for s in facts['summaries']:
        m=s['metrics']
        fact_table+=f'<tr><td>{s["model"]} / {s["scope"]}</td><td>{m["image_identity_gain"]["n"]}</td>'
        for k in ['image_identity_gain','prefix_specificity_no_image','caption_specificity']:
            fact_table+=f'<td>{fmt(m[k])}</td>'
        fact_table+='</tr>'
    fact_table+='</table>'
    conclusions=[]
    for model in ['huatuo','medsight']:
        g=stat(model,'external_context','image_identity_gain')
        p=stat(model,'external_context','prefix_gain_no_image')
        c=stat(model,'external_context','caption_specificity')
        conclusions.append(f'<p><b>{model}, external-context targets:</b> image identity gain {fmt(g)}, '
                           f'preceding-answer benefit without any image {fmt(p)}, and true-versus-donor-caption benefit {fmt(c)} nats/token.</p>')
    contrasts='<table><tr><th>Model</th><th>Metric</th><th>External − visual mean [95% CI]</th></tr>'
    for c in summary['category_contrasts']:
        contrasts+=f'<tr><td>{c["model"]}</td><td>{c["metric"]}</td><td>{fmt(c)}</td></tr>'
    contrasts+='</table>'
    decisions=[]
    for model in ['huatuo','medsight']:
        cc={c['metric']:c for c in summary['category_contrasts'] if c['model']==model}
        decisions.append(dict(model=model,
            external_less_image_sensitive=cc['image_identity_gain']['ci95'][1]<0,
            external_more_caption_sensitive=cc['caption_specificity']['ci95'][0]>0,
            prefix_specificity_distinguishes_groups=not(cc['prefix_specificity_no_image']['ci95'][0]<=0<=cc['prefix_specificity_no_image']['ci95'][1]),
            source_over_image_contrast=cc['source_over_image']))
    conclusion=dict(
        outcome='Both models show lower image identity gain and higher caption specificity for externally screened targets than visual-description controls; prefix specificity does not separate these groups in this pilot.',
        supported='Source-dependent supervision is a concrete and measurable audit target. Explicit biopsy-event and patient-history examples provide stronger structural evidence than low image sensitivity alone.',
        unsupported='No verified population mismatch rate, no evidence that this training data caused the model behavior, no general model ranking, and no validated mismatch detector.',
        recommended_measurement='A claim-level evidence-provenance profile: image identity gain, source-context specificity, and their contrast, accompanied by an independent evidence-sufficiency label. Use prefix dependence as a control or mechanism probe.',
        decisions=decisions)
    (OUT/'conclusions.json').write_text(json.dumps(conclusion,indent=2))
    case_html=''
    # All external-context examples are exposed, avoiding cherry-picked-only reporting.
    for r in manifest:
        if r['category']!='external_context': continue
        case_html+=f'<details><summary>{E(r["id"])} · {r["subset"]} · {E(r["modality"])}</summary><p><b>Question:</b> {E(r["question"])}</p><p><b>Target:</b> {E(r["target"])}</p><p><b>Preceding answer:</b> {E(r["prefix"])}</p><p><b>Original caption:</b> {E(r["caption"])}</p><p><b>Screening match:</b> {E(str(r["screening_matches"]))}</p><p>Image: {E(r["image_path"])}</p>'
        for row in records:
            if row['id']==r['id']:
                case_html+=f'<p>{row["model"]}: image gain {float(row["image_identity_gain"]):.3f}; prefix gain without image {float(row["prefix_gain_no_image"]):.3f}; caption specificity {float(row["caption_specificity"]):.3f}.</p>'
        case_html+='</details>'
    report=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Evidence–supervision mismatch: frozen-model pilot</title>
<style>body{{max-width:1150px;margin:40px auto;padding:0 25px;font:17px/1.6 system-ui;color:#233044;background:#fafbfd}}h1,h2,h3{{line-height:1.2;color:#143e59}}h2{{margin-top:2em}}table{{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}}th,td{{padding:9px;border:1px solid #d2dce3;text-align:left}}th{{background:#e9f0f5}}.note{{padding:18px;background:#edf3f7;border-left:4px solid #327ba3}}code,pre{{background:#eef1f5}}pre{{white-space:pre-wrap;padding:18px;font-size:13px}}details{{border-bottom:1px solid #cdd7df;padding:12px}}summary{{cursor:pointer}}img{{max-width:100%}}a{{color:#176da0}}</style>
<h1>Evidence–supervision mismatch: a training-free preliminary study</h1>
<p>Prepared October 18, 2026. Local source audit plus frozen HuatuoGPT-Vision-7B and MedSIGHT-Comp experiments. No weights were updated.</p>
<div class="note"><b>Research question:</b> Does the training answer demand case-specific information that the supplied question and image do not establish? The defensible preliminary endpoint is a queue of potentially mismatched supervision, with measured dependence on images, source context, and preceding answer text. Model insensitivity alone does not establish missing evidence.</div>
<h2>Conclusion from the completed pilot</h2>
<p><b>{E(conclusion['outcome'])}</b> The external-minus-visual source-over-image contrast is {fmt(decisions[0]['source_over_image_contrast'])} nats/token for Huatuo and {fmt(decisions[1]['source_over_image_contrast'])} for MedSIGHT. These are within-model contrasts; absolute likelihood scales should not be used to rank the two models.</p>
<p>{E(conclusion['supported'])} The most promising proposed measurement is therefore an <b>evidence-provenance profile</b> attached to each supervised claim. Prefix sensitivity is useful as a control but is not supported as a standalone mismatch discriminator. {E(conclusion['unsupported'])}</p>
<h2>What was executed</h2>
<p>The complete 646,759-record instruction corpus and the supplied 60,000-record subset were screened. The model pilot contains 96 distinct single-image records: 16 in each combination of two membership strata (60k subset / remainder) and three target-sentence groups (visual description, diagnostic inference, external context). Both models score the same records. Selection is seeded and independent of model scores, balanced between the recorded CT and MRI modalities. This is an enriched pilot, not a random estimate of mismatch prevalence.</p>
<p>Each target receives 14 conditions per model: correct image, two different images with the same recorded modality and body part, a gray image, or no image, each with and without preceding answer sentences; plus a donor answer-prefix control with the correct image and with no image, and true source caption and donor source caption with the correct image and no preceding answer. Donor prefixes come from a different selected record in the same screening category and recorded modality, chosen by nearest word count; they are not exactly token-length matched. Total: <b>{summary['validation']['n_scores']:,} scored target-condition pairs</b>. A target is always the same token sequence within a model. Only its tokens contribute to its score. Source captions are introduced as additional user context.</p>
<p>Huatuo uses the local 7B LLaVA/Qwen2 checkpoint. MedSIGHT uses <code>Caf0914/MedSIGHT-Comp</code>, the comprehension checkpoint in MedProbe, with its <code>huatuo</code> template. This does not test the region-token/segmentation variant. Checkpoint locations, model configs, library versions, MedProbe commit, corpus hashes, and selected IDs are saved alongside results. The two local copies of the 60k corpus have matching IDs and conversations; their image path prefixes differ.</p>
<h2>Measurements and why they help</h2>
<p>Let ℓ(I,Q,A&lt;j; Cj) be the average log probability of target sentence Cj, optionally conditioned on preceding supervised answer sentences A&lt;j. These are nats per target token, not calibrated probabilities that a medical claim is true.</p>
<table><tr><th>Measurement</th><th>Definition</th><th>Interpretation / limitation</th></tr>
<tr><td>Image identity gain, V</td><td>ℓ(correct image,Q;C) − mean ℓ(matched donor image,Q;C)</td><td>Does the particular image support the target more than other images of the same recorded anatomy and modality? Zero can reflect missing evidence, poor perception, generic language, or question cues.</td></tr>
<tr><td>Answer-prefix dependence, P</td><td>ℓ(no image,Q,A&lt;j;C) − ℓ(no image,Q;C)</td><td>How much can preceding supervision explain the target without pixels? This is normal discourse predictability unless independently linked to unsupported premises.</td></tr>
<tr><td>Prefix specificity</td><td>ℓ(no image,Q,true preceding answer;C) − ℓ(no image,Q,donor preceding answer;C)</td><td>Controls approximately for discourse and prefix length. Still sensitive to pronoun resolution and ordinary answer coherence, so it is not a hallucination measure.</td></tr>
<tr><td>Prefix–image interaction, J</td><td>V with preceding answer − V without preceding answer</td><td>A negative value indicates preceding answer sentences reduce the measured contribution of image identity. It is a candidate mechanism, not proof that training caused hallucinations.</td></tr>
<tr><td>Source-context specificity, R</td><td>ℓ(correct image,Q,true caption;C) − ℓ(correct image,Q,donor caption;C)</td><td>Tests whether privileged source context helps. Captions may directly reveal the target, so this is an information-release control, not independent clinical verification.</td></tr>
<tr><td>Source-over-image contrast</td><td>R − V</td><td>Ranks targets that respond more strongly to source text than to image identity. Keep the components separate; no universal threshold has been validated.</td></tr>
<tr><td>Image retrieval check</td><td>Rank true image among itself and two donors using target likelihood</td><td>Chance is 1/3 only under exchangeability. Donors can share pathology and metadata can be wrong; this is a coarse specificity check.</td></tr></table>
<h2>Results</h2>
{audit_table}
<p>The corpus counts are <b>regex screening rates</b>, not verified mismatch rates. A marker category in an answer but absent from the question can still describe information visible in the image, an explanation, or an appropriate inference. The filters are non-exhaustive and do not check pixels or paraphrased question context. Categories overlap. Exact pattern strings and category counts are in <a href="audit.json">audit.json</a>.</p>
<p>Recorded modality labels are noisy: inspection found radiographs and mixed histology/radiology montages under CT/MRI metadata. Donors are matched on supplied metadata, not on clinically verified image content. A “single image” means one file and may contain multiple panels. All 96 target images have distinct decoded-pixel hashes, and each has two pixel-distinct donors.</p>
<h3>Exact-age provenance audit</h3>{age_table}
<p>Here a numeric age occurs in both the original caption and supervised answer, but not in the question. This is direct evidence of a source-only textual information path. It is not automatically a mismatch: scanner overlays, patient identity, approximate-age questions, and hypothetical examples need checking. Numeric values are compared exactly; spelling and paraphrase matching are not attempted.</p>
<img src="pilot_results.png" alt="Image, prefix and caption dependence with bootstrap intervals">
{''.join(conclusions)}
<p>All table intervals use 5,000 record-bootstrap draws. Category contrasts use within-membership-stratum resampling. They are exploratory pointwise intervals without multiplicity correction. Records, rather than tokens or perturbation runs, are the sampling units. No patient/article identifiers are supplied here, so independence across source articles is not guaranteed.</p>
{tables}
<h3>Does the screened external-context group differ from the visual-description group?</h3>
{contrasts}
<p>A category contrast whose interval spans zero does not support a directional group difference in this pilot. Category-level effects must be interpreted alongside sentence position, length, lexical overlap, question specificity, and source-caption informativeness. The membership comparison is descriptive: full-only does not mean unseen by MedSIGHT, because its alignment corpus and other pretraining exposures can overlap.</p>
<h3>Sensitivity: 12 explicit patient-fact candidates</h3>
<p>Before inspecting model contrasts, the assistant reviewed all 32 external-screen target texts, captions, and image contact sheets. Twelve cases contain explicit ages, clinical-history facts, treatment histories, renal-function status, or biopsy/histopathology events that the supplied question does not state and the displayed radiology figure does not visibly document. This is a structural review, not a clinical adjudication or exhaustive OCR examination. The selected fact spans are saved in <code>fact_spans.json</code>. At least four original screening flags are clear false positives or already supplied evidence; these remain in the primary results for transparency.</p>
{fact_table}
<p>The fact-span analysis uses the same recorded forward passes and scores only tokens overlapping the inspected factual phrase. It still conditions on earlier tokens within that target sentence. Thus it reduces dilution by generic prose but does not remove all autoregressive answer leakage. The selected set is too small to validate a detector. Some clinical histories may be probabilistically inferred; our concern is presentation as known patient-specific facts.</p>
<p>A concrete source-provenance example is <b>Instruction-Tuning_141171</b>: the question asks for the likely diagnosis from a CT image, while the target additionally asserts that a CT-guided biopsy was performed and histology confirmed non-small-cell adenocarcinoma. The caption supplies that event, but the input is a single chest CT image. Distinguish predicting a diagnosis from asserting knowledge of an unprovided test result. A second example, <b>Instruction-Tuning_422538</b>, supplies a breast MRI and an answer stating that biopsy results confirm the diagnosis; the biopsy information is in the caption.</p>
<h2>Novelty assessment and stronger follow-up designs</h2>
<p>Simple image removal, corruption, region masking, and claim verification are established approaches. <a href="https://github.com/hq0709/MedVIGIL">MedVIGIL</a> already tests broken medical visual evidence; <a href="https://papers.miccai.org/miccai-2026/0442-Paper0281.html">CoEV</a> verifies individual claims against visual evidence; <a href="https://arxiv.org/abs/2606.07647">TLVS</a> studies token-level visual sensitivity. The candidate contribution here is the <b>joint source-provenance and supervision-prefix audit of training targets</b>, with an explicit distinction between missing evidence and failed perception. This is a positioning hypothesis, not an exhaustive novelty claim.</p>
<p>The <a href="https://arxiv.org/abs/2406.19280">HuatuoGPT-Vision paper</a> describes using GPT-4V with source information to construct PubMedVision. That makes the source-to-target information path a relevant audit target; it does not imply the dataset is generally defective.</p>
<table><tr><th>Candidate experiment</th><th>Training-free design</th><th>Status / what it establishes</th></tr>
<tr><td>Evidence demand versus supply ledger</td><td>Assign each atomic target claim a required evidence type: image, clinical history, previous study, pathology, molecular test, or general medical knowledge. Compare to the actual supplied evidence inventory.</td><td>Automated lexical screen executed. Expert-adjudicated evidence contracts remain necessary for precision and recall.</td></tr>
<tr><td>Source–image–prefix decomposition</td><td>Cross image identity with preceding-answer availability, then release true or irrelevant source context.</td><td>Executed. Measures competing information sources without changing weights.</td></tr>
<tr><td>Missing-evidence crossover</td><td>For an adjudicated radiology target requiring pathology or history, provide the missing evidence, then replace it with a matched conflicting result. Test whether the diagnostic preference changes appropriately.</td><td>Caption-release proxy executed. A clean test needs non-answer-copying evidence and plausible, reviewed alternative diagnoses.</td></tr>
<tr><td>Minimum sufficient evidence frontier</td><td>Reveal image panels, sequences, history, and pathology one at a time. Measure the first stage where the target is both identifiable to experts and preferred by the model.</td><td>Proposed. Separates a supervision target stated too early from a model that fails despite adequate evidence.</td></tr>
<tr><td>Specificity debt</td><td>For the same case, compare verified morphology, diagnosis family, exact subtype, and molecular identity. Score evidence gains at each level using length-matched contrastive alternatives.</td><td>Proposed. More informative than raw answer likelihood; needs expert-vetted alternatives to avoid false specificity labels.</td></tr>
<tr><td>Premise inheritance</td><td>Replace one evidence-critical premise in the answer prefix with a plausible contradictory premise while holding image and subsequent target fixed; include equally long neutral-prefix controls.</td><td>Proposed. Tests whether a rationale inherits textual premises rather than checking the image. Prefix deletion alone is insufficient.</td></tr>
<tr><td>Perception-versus-availability triangulation</td><td>For low-V targets compare full-resolution crops, expert localized findings, and missing nonvisual evidence. Use independent experts or an independent model rather than self-judging.</td><td>Proposed. Image rescue favors a perception explanation; pathology/history rescue with no available image discriminator favors evidence mismatch.</td></tr>
<tr><td>Training-free target repair audit</td><td>Remove unsupported patient facts, change certainty to a justified differential, or add the missing context. Blindly evaluate target correctness, evidence sufficiency, and specificity with experts.</td><td>Proposed. Can validate a data-cleaning intervention without retraining; downstream training benefit cannot be claimed from it.</td></tr>
</table>
<h2>Interpretation limits and next decision</h2>
<p>This experiment cannot establish that the training objective caused a behavior, that a particular diagnosis is wrong, or that clinicians could not infer it. It measures frozen-model dependencies on deliberately selected source targets. Low image dependence has several explanations; high source-context dependence can be ordinary answer leakage; high preceding-answer dependence can be ordinary coherence. Agreement between two models is not independent clinical ground truth. Neither model generates free responses in the main pilot, so answer accuracy, hallucination rate, and abstention behavior are not measured.</p>
<p>The next decisive study is a blinded audit of about 100–200 cases with two clinical reviewers. Label each claim as visible, supplied in the question, inferentially supported but nonspecific, dependent on missing evidence, or unverifiable. Review complete panels and image text. Predefine a high-confidence subset with clearly absent required evidence, and a matched visible-evidence control subset. Then test whether the measured source/prefix-versus-image contrasts discriminate those labels. Only after this should an aggregate “mismatch score” or a broad dataset prevalence be reported.</p>
<h2>Inspect all 32 externally flagged pilot targets</h2>{case_html}
<h2>Reproduce and inspect</h2>
<pre>cd /home/avc6555/research/MedRL
bash experiments/evidence_mismatch/run_all.sh

# Recompute statistics/report without model inference:
/data/aofei/conda/env/med_edit/bin/python experiments/evidence_mismatch/analyze.py
/data/aofei/conda/env/med_edit/bin/python experiments/evidence_mismatch/report.py</pre>
<p>The runner defaults to GPUs 1 and 2; change HUATUO_GPU and MEDSIGHT_GPU to currently idle devices. It resumes completed condition keys only if the manifest hash and checkpoint match. No paid APIs are used. Native wrapper templates and preprocessing are retained. Target masking is asserted after multimodal image-token expansion; the optimized target-only score is checked against native model cross-entropy.</p>
<p>Artifacts: <a href="manifest.jsonl">manifest</a>, <a href="audit.json">full corpus audit</a>, <a href="summary.json">statistics and intervals</a>, <a href="per_record.csv">per-record contrasts</a>, <a href="huatuo.jsonl">Huatuo token scores</a>, <a href="medsight.jsonl">MedSIGHT token scores</a>, <a href="pilot_results.pdf">PDF figure</a>. Source data and MedProbe code are read only. Development-v1 files are an abandoned screening draft and partial scorer check, excluded from all final analyses. The screening rules were tightened after text inspection and before model contrasts were examined.</p>
</html>'''
    (OUT/'report.html').write_text(report)
    print(OUT/'report.html')

if __name__=='__main__':main()
