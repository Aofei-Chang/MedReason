"""Balanced short-answer QA for P4/P5, generated from d2k with Qwen3-8B (no thinking).

Per article, one item of each type:
  image   answer is a visual finding stated in the caption/article (V sentences): answerable from the image.
          Kept only if the answer does NOT appear in the non-visual context, so a request under env=nonvis
          cannot leak it (under env=full it can, through the caption).
  context answer is patient information (P sentences: age, history, pathology, treatment, outcome).
          Kept only if the answer appears in the non-visual context.
Output overnight/qa.jsonl with ctx_full and ctx_nonvis, and a test split of 150 + 150 items.
"""
import json, random, re
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from label_semantic import PATH
from protocol import normalize
from score_conditions import article_context
from score_typed_context import OUT, article_parts, caption_parts, load_labels

PROMPT = """Below are sentences from a published medical case about one image (modality: {modality}).

VISUAL FINDINGS (visible in the image):
{vis}

PATIENT INFORMATION (not visible in the image):
{non}

Write two short-answer questions about the patient shown in this image.
1. "image": a question answerable by looking at the image, using a fact from VISUAL FINDINGS (for example location, side, which structure is abnormal, appearance, imaging sequence or plane). It must not need any history.
2. "context": a question about this patient using a fact from PATIENT INFORMATION that cannot be determined from the image (for example age, sex, presenting symptom, pathology or laboratory result, treatment, outcome, duration). Phrase it about "the patient in this image".
Each answer must be 1-5 words copied from its source sentence. Copy the source sentence as "evidence".
Return only JSON: {{"image": {{"question": "...", "answer": "...", "evidence": "..."}}, "context": {{"question": "...", "answer": "...", "evidence": "..."}}}}
Use null for an item if no suitable fact exists."""

def contains(text, ans):
    t, a = ' '.join(normalize(text)), ' '.join(normalize(ans))
    return bool(a) and re.search(rf'(^| ){re.escape(a)}( |$)', t) is not None

def pools(r, lab):
    cap = lab[(r['id'], 'caption_sentences')]; art = lab[(r['id'], 'article_sentences')]
    vis = [s for s, c in zip(r['caption_sentences'], cap) if c == 0] + [s for (_, s), c in zip(r['article_sentences'], art) if c == 0]
    non = [s for s, c in zip(r['caption_sentences'], cap) if c == 1] + [s for (_, s), c in zip(r['article_sentences'], art) if c == 1]
    return vis[:12], non[:12]

def main():
    recs = [json.loads(l) for l in open(OUT / 'd2k.jsonl')]
    lab = load_labels('d2k')
    tok = AutoTokenizer.from_pretrained(PATH, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.bfloat16).cuda().eval()
    jobs = []
    for r in recs:
        vis, non = pools(r, lab)
        if not vis and not non: continue
        p = PROMPT.format(modality=r['modality'], vis='\n'.join(f'- {s}' for s in vis) or '(none)',
                          non='\n'.join(f'- {s}' for s in non) or '(none)')
        jobs.append((r, tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False,
                                                add_generation_prompt=True, enable_thinking=False)))
    items, stats = [], {'records': len(jobs), 'parse_fail': 0, 'image_kept': 0, 'context_kept': 0, 'image_leak_dropped': 0}
    raw = open(OUT / 'qa_raw.jsonl', 'w')
    for b in range(0, len(jobs), 24):
        batch = jobs[b:b + 24]
        enc = tok([t for _, t in batch], return_tensors='pt', padding=True).to('cuda')
        with torch.no_grad():
            seq = model.generate(**enc, max_new_tokens=320, do_sample=False, temperature=None, top_p=None, top_k=None)
        for (r, _), s in zip(batch, tok.batch_decode(seq[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)):
            raw.write(json.dumps({'id': r['id'], 'out': s}) + '\n')
            m = re.search(r'\{.*\}', s, re.S)
            try: js = json.loads(m.group(0))
            except Exception: stats['parse_fail'] += 1; continue
            cv, cn = caption_parts(r, lab); av, an = article_parts(r, lab)
            full = f"Figure caption: {r['original_caption']}\n{article_context(r, False)}"
            nonvis = '\n'.join(x for x in (cn, an) if x) or '(no further information)'
            for t in ('image', 'context'):
                x = js.get(t)
                if not isinstance(x, dict) or not x.get('question') or not x.get('answer'): continue
                ans = str(x['answer'])
                if t == 'image':
                    if not contains(full, ans) or len(normalize(ans)) > 6: continue
                    if contains(nonvis, ans): stats['image_leak_dropped'] += 1; continue
                elif not contains(nonvis, ans) or len(normalize(ans)) > 6: continue
                stats[f'{t}_kept'] += 1
                items.append({'qid': f"{r['id']}_{t}", 'rid': r['id'], 'image': r['image'], 'type': t,
                              'question': x['question'], 'answer': ans, 'evidence': x.get('evidence', ''),
                              'answer_in_full': contains(full, ans), 'ctx_full': full, 'ctx_nonvis': nonvis})
        print(b, len(jobs), stats, flush=True)
    rng = random.Random(0); rids = sorted({it['rid'] for it in items}); rng.shuffle(rids)
    by = {}
    for it in items: by.setdefault(it['rid'], []).append(it)
    test, n = set(), {'image': 0, 'context': 0}
    for rid in rids:
        if min(n.values()) >= 150: break
        if all(n[it['type']] < 150 for it in by[rid]):
            test.add(rid)
            for it in by[rid]: n[it['type']] += 1
    with open(OUT / 'qa.jsonl', 'w') as f:
        for it in items:
            it['split'] = 'test' if it['rid'] in test else 'train'
            f.write(json.dumps(it) + '\n')
    stats['test'] = n; stats['train'] = len(items) - sum(n.values())
    (OUT / 'qa_stats.json').write_text(json.dumps(stats, indent=1)); print(stats)

if __name__ == '__main__':
    main()
