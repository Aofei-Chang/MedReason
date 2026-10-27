"""Text-only labels for the awareness run (Qwen3-8B, no thinking, next-token probabilities; no generation).

--what time  V0: every P-typed context sentence of d2k (caption + article) -> known before/at reading (pre),
             established after the image (post), or general. Writes aware/labels_time.jsonl and an audit sheet
             (aware/audit_labels.csv: 200 V/P/K/M labels + 100 time labels, human column blank).
--what gen   V/P/K/M type of every sentence of the free generations (gen_free_*.jsonl, three models).
--what cmp   MIMIC: does a report sentence compare with a previous exam? References (train + test sets) and
             generated reports. Replaces the regex of P6.
"""
import argparse, csv, json, random, re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from aware_common import AW
from build_pilot import sentences
from label_semantic import PATH, PROMPT
from score_typed_context import OUT, load_labels

TIME_PROMPT = """A sentence from a medical article about a patient's image (modality: {modality}) is given. It states non-visual information. When is this information known, relative to acquiring and reading this image?
A: known before or when the image is read: patient age or sex, presenting symptoms, history, the reason for the exam, earlier treatment, or earlier test results.
B: established only after this image: biopsy, pathology or histology results, surgical findings, the final confirmed diagnosis, treatment decided or given afterwards, follow-up, or outcome.
C: neither: general medical knowledge, study methods, or not about this patient.

Sentence: "{s}"
Answer with a single letter."""

CMP_PROMPT = """A sentence from a chest X-ray report: "{s}"
Does this sentence compare the current exam with a previous exam, or describe a change over time (for example new, unchanged, stable, improved, worsened, interval change, compared with prior)? Answer Yes or No."""

def report_sentences(t):
    return [s for s in re.split(r'(?<=[.!?])\s+', t.strip()) if len(s.split()) >= 2]

class Labeler:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(PATH, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.bfloat16).cuda().eval()

    @torch.no_grad()
    def probs(self, prompts, choices, bs=64):
        ids = [self.tok.encode(c, add_special_tokens=False)[0] for c in choices]
        flat = [self.tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False) for p in prompts]
        order = sorted(range(len(flat)), key=lambda i: len(flat[i])); out = [None] * len(flat)
        for b in range(0, len(order), bs):
            idx = order[b:b + bs]
            enc = self.tok([flat[i] for i in idx], return_tensors='pt', padding=True).to('cuda')
            logits = self.model(**enc, logits_to_keep=1).logits[:, -1].float()
            for i, p in zip(idx, torch.softmax(logits[:, ids], -1).tolist()): out[i] = [round(x, 4) for x in p]
            if b % 6400 == 0: print(b, len(order), flush=True)
        return out

def do_time(L):
    recs = [json.loads(l) for l in open(OUT / 'd2k.jsonl')]
    lab = load_labels('d2k')
    keys, prompts, texts = [], [], []
    for r in recs:
        for fld in ('caption_sentences', 'article_sentences'):
            sents = r[fld] if fld == 'caption_sentences' else [s for _, s in r[fld]]
            for i, (s, c) in enumerate(zip(sents, lab[(r['id'], fld)])):
                if c == 1:
                    keys.append((r['id'], fld, i)); texts.append(s)
                    prompts.append(TIME_PROMPT.format(modality=r['modality'], s=s))
    P = L.probs(prompts, 'ABC')
    rows = {}
    for (rid, fld, i), p in zip(keys, P):
        x = rows.setdefault((rid, fld), {'id': rid, 'field': fld, 'idx': [], 'probs': []})
        x['idx'].append(i); x['probs'].append(p)
    with open(AW / 'labels_time.jsonl', 'w') as f:
        for x in rows.values(): f.write(json.dumps(x) + '\n')
    # audit sheet
    rng = random.Random(0)
    sem = []
    for r in recs:
        for fld in ('answer_sentences', 'caption_sentences', 'article_sentences'):
            sents = r[fld] if fld != 'article_sentences' else [s for _, s in r[fld]]
            for s, c in zip(sents, lab[(r['id'], fld)]): sem.append(('VPKM', r['id'], fld, s, 'VPKM'[c]))
    tim = [('time', rid, fld, t, ['pre', 'post', 'general'][max(range(3), key=p.__getitem__)])
           for (rid, fld, _), t, p in zip(keys, texts, P)]
    with open(AW / 'audit_labels.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['task', 'id', 'field', 'sentence', 'model_label', 'human_label'])
        for row in rng.sample(sem, 200) + rng.sample(tim, 100): w.writerow(list(row) + [''])
    print({'time_labelled': len(keys)})

def do_gen(L):
    recs = {json.loads(l)['id']: json.loads(l) for l in open(OUT / 'd2k.jsonl')}
    keys, prompts = [], []
    for m in ('qwen25vl', 'lingshu', 'huatuo'):
        p = OUT / f'gen_free_{m}.jsonl'
        if not p.exists(): p = AW / f'gen_free_{m}.jsonl'
        for line in open(p):
            g = json.loads(line)
            for i, s in enumerate(sentences(g['text'])):
                keys.append((m, g['id'], g['prompt'], i, s))
                prompts.append(PROMPT.format(modality=recs[g['id']]['modality'], s=s))
    P = L.probs(prompts, 'ABCD')
    with open(AW / 'labels_gen.jsonl', 'w') as f:
        for (m, rid, pr, i, s), p in zip(keys, P):
            f.write(json.dumps({'model': m, 'id': rid, 'prompt': pr, 'i': i, 'sentence': s, 'probs': p}) + '\n')

def do_cmp(L):
    keys, prompts = [], []
    for name, path in [('mimic', OUT / 'mimic.jsonl'), ('mimic_test', AW / 'mimic_test.jsonl')]:
        for line in open(path):
            r = json.loads(line)
            for i, s in enumerate(r['target_sentences']):
                keys.append({'src': 'ref', 'set': name, 'id': r['id'], 'has_prior': r['has_prior'], 'i': i})
                prompts.append(CMP_PROMPT.format(s=s))
    gens = [(f'gen_mimic_{m}', 'mimic', OUT) for m in ('qwen25vl', 'lingshu')] + \
           [(f'gen_mimic_test_{m}', 'mimic_test', AW) for m in ('qwen25vl', 'lingshu', 'huatuo')]
    for stem, name, d in gens:
        if not (d / f'{stem}.jsonl').exists(): continue
        for line in open(d / f'{stem}.jsonl'):
            g = json.loads(line)
            for i, s in enumerate(report_sentences(g['text'])):
                keys.append({'src': stem, 'set': name, 'id': g['id'], 'has_prior': g['has_prior'], 'cond': g['cond'], 'i': i})
                prompts.append(CMP_PROMPT.format(s=s))
    P = L.probs(prompts, ['Yes', 'No'])
    with open(AW / 'labels_cmp.jsonl', 'w') as f:
        for k, p in zip(keys, P): f.write(json.dumps({**k, 'p_yes': p[0]}) + '\n')

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--what', choices=['time', 'gen', 'cmp'], required=True)
    a = ap.parse_args()
    {'time': do_time, 'gen': do_gen, 'cmp': do_cmp}[a.what](Labeler())
