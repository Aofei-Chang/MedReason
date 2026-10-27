"""Items for the awareness run. Needs labels.jsonl (overnight) and labels_time.jsonl (label_aware.py --what time).

aware/items.jsonl   short-answer items with the context split V / pre / post / N and a same-modality mismatched
                    image. Two sources:
    qa    the P4 QA set (image vs context items); its question wording mostly gives the type away;
    hard  fixed templates asked of every record, so the wording is constant and only the case decides whether
          the image suffices. Qwen3-8B answers each template from the numbered caption + article sentences
          and cites evidence; the item is typed by its evidence: only V sentences -> image, any P -> context.
aware/zoom_items.jsonl  V6b: a short question about a MedSAM3-boxed entity (box set), for crop vs random crop.
aware/mimic_test.jsonl  V5: MIMIC-CXR official test split (held out), same fields as overnight/mimic.jsonl.
"""
import csv, json, random, re
from collections import defaultdict
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from aware_common import AW, contexts, fmt_context, load_time_labels, mismatched, split_sentences
from build_overnight_sets import MIMIC, MIMIC_META, report_sections
from build_pilot import sentences
from label_semantic import PATH
from qa_build import contains
from protocol import normalize
from score_typed_context import OUT, load_labels

TEMPLATES = {'diagnosis': 'What is the diagnosis for the patient in this image?',
             'nature': 'Is the main lesion in this image benign or malignant?',
             'finding': 'What is the main abnormal finding in this image?'}
HARD_PROMPT = """Below are numbered sentences from a published medical case about one image (modality: {modality}).

{numbered}

Answer each question about the patient in this image using only these sentences.
1. "diagnosis": What is the diagnosis for the patient in this image?
2. "nature": Is the main lesion in this image benign or malignant? (answer exactly "benign" or "malignant")
3. "finding": What is the main abnormal finding in this image?
For each, give a 1-5 word answer and the numbers of the fewest sentences that establish it. Use null if the sentences do not answer it.
Return only JSON: {{"diagnosis": {{"answer": "...", "evidence": [1]}}, "nature": {{"answer": "...", "evidence": [2]}}, "finding": {{"answer": "...", "evidence": [3]}}}}"""
ZOOM_PROMPT = """Sentences describing a medical image (modality: {modality}):
{vis}

Structures marked in the image: {ents}.
Write one short-answer question about a marked structure that is answered by looking at the image (for example its location, side, shape, appearance, or signal/density), with a 1-5 word answer copied from the sentences. Do not mention the answer in the question.
Return only JSON: {{"question": "...", "answer": "..."}}, or null if not possible."""
JPG = Path('/data/yuan/physionet.org/files/mimic-cxr-jpg/2.0.0/files')

def all_parts(r, lab, tlab):
    items = split_sentences(r, lab, tlab, 'caption_sentences') + split_sentences(r, lab, tlab, 'article_sentences')
    return items, {k: fmt_context(items, keep) for k, keep in
                   [('V', {'V'}), ('pre', {'pre'}), ('post', {'post'}), ('N', {'pre', 'post'})]}

class LLM:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(PATH, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.bfloat16).cuda().eval()

    @torch.no_grad()
    def json_gen(self, prompts, max_new=256, bs=24):
        out = []
        for b in range(0, len(prompts), bs):
            flat = [self.tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False) for p in prompts[b:b + bs]]
            enc = self.tok(flat, return_tensors='pt', padding=True).to('cuda')
            seq = self.model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None)
            for s in self.tok.batch_decode(seq[:, enc['input_ids'].shape[1]:], skip_special_tokens=True):
                m = re.search(r'\{.*\}', s, re.S)
                try: out.append(json.loads(m.group(0)))
                except Exception: out.append(None)
            print('gen', b, len(prompts), flush=True)
        return out

def build_items(llm):
    recs = [json.loads(l) for l in open(OUT / 'd2k.jsonl')]
    lab, tlab = load_labels('d2k'), load_time_labels()
    mis = mismatched(recs); idx = {r['id']: i for i, r in enumerate(recs)}
    parts = {r['id']: all_parts(r, lab, tlab) for r in recs}
    def base(r):
        _, c = parts[r['id']]
        return {'rid': r['id'], 'image': r['image'], 'image_mis': recs[mis[idx[r['id']]]]['image'],
                'modality': r['modality'], **{f'ctx_{k}': v for k, v in c.items()}}
    items = []
    for line in open(OUT / 'qa.jsonl'):
        q = json.loads(line); r = recs[idx[q['rid']]]
        items.append({'qid': q['qid'], 'source': 'qa', 'template': None, 'type': q['type'], 'question': q['question'],
                      'answer': q['answer'], 'split': q['split'], 'evidence_parts': None, **base(r)})
    # compute budget: 400 per type, the P4 test items first, then random train items
    rng = random.Random(1); rng.shuffle(items)
    items = sorted(items, key=lambda it: it['split'] != 'test')
    items = [it for t in ('image', 'context') for it in [x for x in items if x['type'] == t][:400]]
    prompts = []
    for r in recs:
        its, _ = parts[r['id']]
        numbered = '\n'.join(f'{i + 1}. {s}' for i, (_, s, _) in enumerate(its[:40]))
        prompts.append(HARD_PROMPT.format(modality=r['modality'], numbered=numbered))
    stats = defaultdict(int)
    with open(AW / 'hard_raw.jsonl', 'w') as raw:
        for r, js in zip(recs, llm.json_gen(prompts)):
            raw.write(json.dumps({'id': r['id'], 'out': js}) + '\n')
            if not isinstance(js, dict): stats['parse_fail'] += 1; continue
            its, c = parts[r['id']]
            for t, q in TEMPLATES.items():
                x = js.get(t)
                if not isinstance(x, dict) or not x.get('answer') or not isinstance(x.get('evidence'), list): continue
                ans = str(x['answer']).strip()
                if len(normalize(ans)) > 6 or len(normalize(ans)) == 0: continue
                if t == 'nature' and ans.lower() not in ('benign', 'malignant'): continue
                ev = [its[e - 1][2] for e in x['evidence'] if isinstance(e, int) and 1 <= e <= len(its[:40])]
                if not ev: continue
                typ = 'image' if all(p == 'V' for p in ev) else 'context'
                stats[f'{t}_{typ}'] += 1
                items.append({'qid': f"{r['id']}_hard_{t}", 'source': 'hard', 'template': t, 'type': typ, 'question': q,
                              'answer': ans, 'split': 'test', 'evidence_parts': ev,
                              'answer_in_V': contains(c['V'] or '', ans), 'answer_in_N': contains(c['N'] or '', ans), **base(r)})
    with open(AW / 'items.jsonl', 'w') as f:
        for it in items: f.write(json.dumps(it) + '\n')
    stats['qa'] = sum(it['source'] == 'qa' for it in items); stats['hard'] = sum(it['source'] == 'hard' for it in items)
    print(dict(stats)); return dict(stats)

def build_zoom(llm):
    recs = [json.loads(l) for l in open(OUT / 'box.jsonl')]
    lab = load_labels('box')
    prompts, keep = [], []
    for r in recs:
        vis = [s for s, c in zip(r['caption_sentences'], lab[(r['id'], 'caption_sentences')]) if c == 0]
        if not vis: continue
        ents = sorted({b['entity'] for b in r['boxes']})
        prompts.append(ZOOM_PROMPT.format(modality=r['modality'], vis='\n'.join(f'- {s}' for s in vis), ents=', '.join(ents)))
        keep.append((r, ' '.join(vis)))
    n = 0
    with open(AW / 'zoom_items.jsonl', 'w') as f:
        for (r, vis), js in zip(keep, llm.json_gen(prompts, max_new=96)):
            if not isinstance(js, dict) or not js.get('question') or not js.get('answer'): continue
            ans = str(js['answer'])
            if not contains(vis, ans) or len(normalize(ans)) > 6 or contains(js['question'], ans): continue
            f.write(json.dumps({'qid': f"{r['id']}_zoom", 'rid': r['id'], 'image': r['image'], 'boxes': r['boxes'],
                                'question': js['question'], 'answer': ans}) + '\n'); n += 1
    print({'zoom_items': n})

def build_mimic_test(n_prior=600, n_first=300):
    studies = defaultdict(dict)
    for r in csv.DictReader(open(MIMIC_META)):
        studies[r['subject_id']][r['study_id']] = r['StudyDate'] + r['StudyTime'].split('.')[0].zfill(6)
    def report(subj, study):
        p = MIMIC / 'reports_files' / f'p{subj[:2]}' / f'p{subj}' / f's{study}.txt'
        return report_sections(p.read_text()) if p.exists() else None
    out = {'prior': [], 'first': []}
    paths = json.load(open(MIMIC / 'sampled_paths_test.json'))
    random.seed(3); random.shuffle(paths)
    for p in paths:
        _, subj, st, _ = p.split('/'); s, t = subj[1:], st[1:]
        img = JPG / p
        d = studies[s].get(t); cur = report(s, t)
        if d is None or cur is None or not img.exists(): continue
        target = ' '.join(x for x in (cur.get('FINDINGS', ''), cur.get('IMPRESSION', '')) if x)
        if len(target.split()) < 8: continue
        earlier = sorted((v, k) for k, v in studies[s].items() if v < d)
        rec = {'id': f'{s}_{t}', 'image': str(img), 'target': target, 'target_sentences': sentences(target),
               'indication': cur.get('INDICATION', '') or cur.get('HISTORY', ''), 'has_prior': bool(earlier)}
        if not rec['target_sentences']: continue
        if earlier:
            pr = report(s, earlier[-1][1])
            if pr is None: continue
            rec['prior_report'] = ' '.join(x for x in (pr.get('FINDINGS', ''), pr.get('IMPRESSION', '')) if x)[:1500]
            if not rec['prior_report']: continue
            key = 'prior'
        else:
            key = 'first'
        if len(out[key]) < (n_prior if key == 'prior' else n_first): out[key].append(rec)
    rows = out['prior'] + out['first']
    with open(AW / 'mimic_test.jsonl', 'w') as f:
        for r in rows: f.write(json.dumps(r) + '\n')
    print({'mimic_test': len(rows), 'with_prior': len(out['prior'])})

if __name__ == '__main__':
    import sys
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('mimic', 'all'): build_mimic_test()
    if what in ('items', 'all'):
        llm = LLM()
        build_zoom(llm)
        s = build_items(llm)
        (AW / 'items_stats.json').write_text(json.dumps(s, indent=1))
