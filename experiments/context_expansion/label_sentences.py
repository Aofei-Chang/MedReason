"""Batched semantic labels (Qwen3-8B, no thinking) for every sentence the overnight experiments use.

Same classes as label_semantic.py: A/V visual finding, B/P non-visual patient/clinical information,
C/K general knowledge, D/M figure meta. Context sentences (caption, article) are labelled the same way, which
splits each context into a visual part (V, M) and a non-visual information part (P, K).
Output: overnight/labels.jsonl, one row per (set, record id, field) with a list of 4-way probabilities.
"""
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from label_semantic import PATH, PROMPT

OUT = Path('/data/aofei/output/ContextMLLM/overnight')
ARTICLE_PROMPT = PROMPT.replace('A sentence from a description of a medical image (modality: {modality}) is given.',
                                'A sentence from the article that contains a medical image (modality: {modality}) is given.'
                                ).replace('observe in this image itself', 'observe in that image itself')

def items():
    for name, fields in [('d2k', ['answer_sentences', 'caption_sentences', 'article_sentences']),
                         ('box', ['answer_sentences', 'caption_sentences']),
                         ('mimic', ['target_sentences'])]:
        for line in open(OUT / f'{name}.jsonl'):
            r = json.loads(line)
            mod = r.get('modality', 'chest X-ray')
            for fld in fields:
                sents = [s for _, s in r[fld]] if fld == 'article_sentences' else r[fld]
                prompt = ARTICLE_PROMPT if fld == 'article_sentences' else PROMPT
                yield (name, r['id'], fld), [prompt.format(modality=mod, s=s) for s in sents]

def main():
    tok = AutoTokenizer.from_pretrained(PATH, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.bfloat16).cuda().eval()
    letters = [tok.encode(x, add_special_tokens=False)[0] for x in 'ABCD']
    keys, flat = [], []
    groups = list(items())
    for gi, (_, prompts) in enumerate(groups):
        for j, p in enumerate(prompts):
            keys.append((gi, j))
            flat.append(tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False,
                                                add_generation_prompt=True, enable_thinking=False))
    order = sorted(range(len(flat)), key=lambda i: len(flat[i]))
    probs = [None] * len(flat)
    for b in range(0, len(order), 64):
        idx = order[b:b + 64]
        enc = tok([flat[i] for i in idx], return_tensors='pt', padding=True).to('cuda')
        with torch.no_grad():
            logits = model(**enc, logits_to_keep=1).logits[:, -1].float()
        for i, p in zip(idx, torch.softmax(logits[:, letters], -1).tolist()): probs[i] = p
        if b % 6400 == 0: print(b, len(order), flush=True)
    res = [[None] * len(p) for _, p in groups]
    for (gi, j), p in zip(keys, probs): res[gi][j] = p
    with open(OUT / 'labels.jsonl', 'w') as f:
        for (key, _), p in zip(groups, res):
            f.write(json.dumps({'set': key[0], 'id': key[1], 'field': key[2], 'probs': p}) + '\n')

if __name__ == '__main__':
    main()
