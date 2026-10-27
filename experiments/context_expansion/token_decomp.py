"""Per-token privilege decomposition (V1, V2, V5 and the box/entity teacher conditions; no training).

Every condition scores the same target tokens, so per-token differences decompose the teacher-student gap
(privilege_aware_design.md section 3). I = correct image, M = same-modality mismatched image.
  --set d2k      target A (instruction answer | caption context) and B (caption | article context), first 1200
                 records. Conditions I, IV, Ipre, IN, IVN, M, MV, MN, MVN (+ token entropy under I), and the
                 P1b self-report with the mismatched image (V6a).
  --set box      target A on the MedSAM3 box set, teacher privileges: V caption part, box coordinates + labels,
                 entity names only. Conditions I, IV, IBOX, IENT, M, MV, MBOX, MENT.
  --set genfree  the model's own free generations (on-policy rollouts, V2) scored by itself with privileged
                 inputs: I (student), IV (teacher matched to a no-request rollout), IVpre (matched, after a
                 request), IVN (privilege-blind teacher), M, MV.
  --set mimic_test  held-out MIMIC-CXR (V5): I, IH (indication), IHP (indication + prior), MH, MHP.
"""
import argparse, json
import numpy as np
import torch
from aware_common import AW, TF, contexts, load_image, load_time_labels, mismatched
from build_aware_items import all_parts
from build_pilot import sentences
from gen_free import DESCRIBE
from mimic_p6 import REPORT_Q
from score_conditions import CAPTION_Q, SELF_Q, spans_of
from score_typed_context import OUT, load_labels

def tok_sentence_index(tok, target, sents):
    offs = tok(target, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
    spans = spans_of(target, sents)
    return [next((j for j, (a, b) in enumerate(spans) if a <= s < b), -1) for s, _ in offs]

def run(tf, prompt, ctxs, conds, img, mis, target, sents):
    """conds: names like 'I', 'IV', 'MVN'; first letter = image, rest = context key in ctxs ('' = none)."""
    out = {'sent': tok_sentence_index(tf.tok, target, sents), 'lp': {}}
    for c in conds:
        im, key = (img if c[0] == 'I' else mis), c[1:]
        ctx = ctxs.get(key) if key else None
        if key and not ctx: out['lp'][c] = None; continue
        text = f'{ctx}\n\n{prompt}' if ctx else prompt
        if c == 'I':
            lp, ent = tf.score(text, im, target, entropy=True)
            out['ent'] = np.round(ent, 3).tolist()
        else:
            lp = tf.score(text, im, target)
        out['lp'][c] = np.round(lp, 3).tolist()
    return out

def box_text(r):
    bs = sorted(r['boxes'], key=lambda b: -b['score'])[:6]
    return ('Regions of interest marked in the image (x, y, width, height as fractions of the image size): ' +
            '; '.join(f"{b['entity']} ({', '.join(f'{v:.2f}' for v in b['xywh'])})" for b in bs) + '.')

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True)
    ap.add_argument('--set', required=True, choices=['d2k', 'box', 'genfree', 'mimic_test'])
    ap.add_argument('--start', type=int, default=0); ap.add_argument('--end', type=int, default=None)
    args = ap.parse_args()
    torch.set_num_threads(4)
    out = AW / f'decomp_{args.set}_{args.model}_s{args.start}.jsonl'
    done = {json.loads(l)['key'] for l in open(out)} if out.exists() else set()
    tf = TF(args.model)
    f = open(out, 'a')
    if args.set in ('d2k', 'genfree'):
        recs = [json.loads(l) for l in open(OUT / 'd2k.jsonl')]
        lab, tlab = load_labels('d2k'), load_time_labels()
        mis = mismatched(recs); idx = {r['id']: i for i, r in enumerate(recs)}
    if args.set == 'd2k':
        for i in range(args.start, args.end or 1200):
            r = recs[i]
            if r['id'] in done: continue
            img, m = load_image(r['image']), load_image(recs[mis[i]]['image'])
            row = {'key': r['id']}
            conds = ['I', 'IV', 'Ipre', 'IN', 'IVN', 'M', 'MV', 'MN', 'MVN']
            row['A'] = run(tf, r['question'], contexts(r, lab, tlab, 'caption_sentences'), conds, img, m, r['answer'], r['answer_sentences'])
            row['B'] = run(tf, CAPTION_Q, contexts(r, lab, tlab, 'article_sentences'), conds, img, m, r['original_caption'], r['caption_sentences'])
            for t, sents in (('A', r['answer_sentences']), ('B', r['caption_sentences'])):
                if not sents: row[t]['self_yes_mis'] = []; continue
                p = tf.next_token_probs([SELF_Q.format(s=s) for s in sents], [m] * len(sents), ['Yes', 'No'])
                row[t]['self_yes_mis'] = [round(float(x[0]), 4) for x in p]
            f.write(json.dumps(row) + '\n'); f.flush()
            if i % 50 == 0: print(i, flush=True)
    elif args.set == 'box':
        recs = [json.loads(l) for l in open(OUT / 'box.jsonl')]
        lab = load_labels('box'); mis = mismatched(recs)
        for i in range(args.start, args.end or len(recs)):
            r = recs[i]
            if r['id'] in done: continue
            img, m = load_image(r['image']), load_image(recs[mis[i]]['image'])
            vis = [s for s, c in zip(r['caption_sentences'], lab[(r['id'], 'caption_sentences')]) if c in (0, 3)]
            ctxs = {'V': f"Figure caption from the source article (excerpt): {' '.join(vis)}" if vis else None,
                    'BOX': box_text(r),
                    'ENT': 'Structures of interest in this image: ' + ', '.join(sorted({b['entity'] for b in r['boxes']})) + '.'}
            row = {'key': r['id'], 'mentions_entity': [any(b['entity'].lower() in s.lower() for b in r['boxes']) for s in r['answer_sentences']]}
            row['A'] = run(tf, r['question'], ctxs, ['I', 'IV', 'IBOX', 'IENT', 'M', 'MV', 'MBOX', 'MENT'], img, m, r['answer'], r['answer_sentences'])
            f.write(json.dumps(row) + '\n'); f.flush()
            if i % 50 == 0: print(i, flush=True)
    elif args.set == 'genfree':
        gens = [json.loads(l) for l in open(OUT / f'gen_free_{args.model}.jsonl')][args.start:args.end]
        for g in gens:
            key = f"{g['id']}|{g['prompt']}"
            if key in done: continue
            r = recs[idx[g['id']]]
            _, c = all_parts(r, lab, tlab)
            ctxs = {'V': c['V'], 'Vpre': '\n'.join(x for x in (c['V'], c['pre']) if x) or None,
                    'VN': '\n'.join(x for x in (c['V'], c['N']) if x) or None}
            sents = sentences(g['text'])
            if not sents: continue
            img, m = load_image(r['image']), load_image(recs[mis[idx[g['id']]]]['image'])
            prompt = r['question'] if g['prompt'] == 'q' else DESCRIBE
            row = {'key': key, 'sentences': sents}
            row['G'] = run(tf, prompt, ctxs, ['I', 'IV', 'IVpre', 'IVN', 'M', 'MV'], img, m, g['text'], sents)
            f.write(json.dumps(row) + '\n'); f.flush()
    else:
        recs = [json.loads(l) for l in open(AW / 'mimic_test.jsonl')]
        n = len(recs)
        for i in range(args.start, args.end or n):
            r = recs[i]
            if r['id'] in done: continue
            img, m = load_image(r['image']), load_image(recs[(i + 451) % n]['image'])
            ctxs = {'H': f"Indication: {r['indication']}" if r['indication'] else None}
            ctxs['HP'] = '\n'.join(x for x in (ctxs['H'], f"Prior report for this patient: {r['prior_report']}" if r['has_prior'] else None) if x) or None
            if not r['has_prior']: ctxs['HP'] = None
            row = {'key': r['id'], 'has_prior': r['has_prior']}
            row['R'] = run(tf, REPORT_Q, ctxs, ['I', 'IH', 'IHP', 'MH', 'MHP'], img, m, r['target'], r['target_sentences'])
            f.write(json.dumps(row) + '\n'); f.flush()
            if i % 100 == 0: print(i, flush=True)

if __name__ == '__main__':
    main()
