"""P1b + P2a on d2k: teacher-forced scoring with the context split into its visual and non-visual parts.

Context sentences are typed by labels.jsonl: visual context = V/M sentences (findings, arrows, panels),
non-visual information context = P/K sentences (history, labs, pathology, outcome, knowledge).
Target A (instruction answer), context pool = original caption:
    I, 0, IC, C (full caption), IV, V (visual part), IN, N (non-visual part), IX (image + another record's caption)
Target B (original caption), context pool = article excerpt:
    same, plus IR, R (regex redaction from P1)
P1b: self-report p(Yes) "verifiable from the image alone?" with the image and without it.
"""
import argparse, json
from pathlib import Path
import torch
from PIL import Image
from score_conditions import CAPTION_Q, Scorer, article_context, spans_of

OUT = Path('/data/aofei/output/ContextMLLM/overnight')
VISUAL = (0, 3)  # V, M

def load_labels(name):
    lab = {}
    for line in open(OUT / 'labels.jsonl'):
        x = json.loads(line)
        if x['set'] == name: lab[(x['id'], x['field'])] = [max(range(4), key=p.__getitem__) for p in x['probs']]
    return lab

def caption_parts(r, lab):
    cls = lab[(r['id'], 'caption_sentences')]
    vis = [s for s, c in zip(r['caption_sentences'], cls) if c in VISUAL]
    non = [s for s, c in zip(r['caption_sentences'], cls) if c not in VISUAL]
    fmt = lambda xs: f"Figure caption from the source article (excerpt): {' '.join(xs)}" if xs else None
    return fmt(vis), fmt(non)

def article_parts(r, lab):
    cls = lab[(r['id'], 'article_sentences')]
    def fmt(keep):
        sel = [(src, s) for (src, s), c in zip(r['article_sentences'], cls) if keep(c)]
        if not sel: return None
        out = []
        for src, head in [('title', 'Article title'), ('abstract', 'Abstract'), ('citing', 'Article text')]:
            xs = [s for k, s in sel if k == src]
            if xs: out.append(f"{head}: {' '.join(xs)}")
        return '\n'.join(out)
    return fmt(lambda c: c in VISUAL), fmt(lambda c: c not in VISUAL)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True)
    ap.add_argument('--start', type=int, default=0); ap.add_argument('--end', type=int, default=None)
    args = ap.parse_args()
    torch.set_num_threads(4)
    recs = [json.loads(l) for l in open(OUT / 'd2k.jsonl')]
    lab = load_labels('d2k')
    out_path = OUT / f'scores_typed_{args.model}_s{args.start}.jsonl'
    done = {json.loads(l)['id'] for p in OUT.glob(f'scores_typed_{args.model}_s*.jsonl') for l in open(p)}
    sc = Scorer(args.model)
    with open(out_path, 'a') as f:
        for i in range(args.start, args.end or len(recs)):
            r = recs[i]
            if r['id'] in done: continue
            other = recs[(i + 997) % len(recs)]  # mismatched context from an unrelated article
            img = Image.open(r['image']).convert('RGB')
            ans, cap, q = r['answer'], r['original_caption'], r['question']
            sa, sb = spans_of(ans, r['answer_sentences']), spans_of(cap, r['caption_sentences'])
            ctx = {'A': {'C': f"Figure caption from the source article: {cap}",
                         'X': f"Figure caption from the source article: {other['original_caption']}"},
                   'B': {'C': article_context(r, False), 'R': article_context(r, True), 'X': article_context(other, False)}}
            ctx['A']['V'], ctx['A']['N'] = caption_parts(r, lab)
            ctx['B']['V'], ctx['B']['N'] = article_parts(r, lab)
            row = {'id': r['id'], 'ctx_chars': {t: {k: len(v) if v else 0 for k, v in d.items()} for t, d in ctx.items()}}
            for t, q_t, tgt, spans in [('A', q, ans, sa), ('B', CAPTION_Q, cap, sb)]:
                res = {}
                res['I'], _ = sc.score(q_t, img, tgt, spans)
                res['0'], _ = sc.score(q_t, None, tgt, spans)
                for k, c in ctx[t].items():
                    if c is None: res[f'I{k}'] = res[k] = None; continue
                    res[f'I{k}'], _ = sc.score(f'{c}\n\n{q_t}', img, tgt, spans)
                    if k != 'X': res[k], _ = sc.score(f'{c}\n\n{q_t}', None, tgt, spans)
                sents = r['answer_sentences'] if t == 'A' else r['caption_sentences']
                res['self_yes'] = [sc.self_report(s, img) for s in sents]
                res['self_yes_noimage'] = [sc.self_report(s, None) for s in sents]
                row[t] = res
            f.write(json.dumps(row) + '\n'); f.flush()
            if i % 50 == 0: print(i, flush=True)

if __name__ == '__main__':
    main()
