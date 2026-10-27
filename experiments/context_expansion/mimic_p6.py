"""P6: the same question on MIMIC-CXR, where the non-visual context is well defined: the prior report.

Scoring (teacher-forced, findings + impression of the current report):
  I, 0, IP (image + prior report), P (prior report only), IH (image + indication), H (indication only).
  Comparison sentences ("unchanged", "compared with prior", "new", "interval") need the prior; the others
  should be carried by the image.
Generation (greedy): the report with no context, and (patients with a prior) with the prior report given.
Comparison language without a prior in the input is a fabricated comparison; for first-study patients no
prior exists at all.
"""
import argparse, json, re
from PIL import Image
from protocol import generate, load_vl
from score_conditions import Scorer, spans_of
from score_typed_context import OUT

REPORT_Q = 'Write the FINDINGS and IMPRESSION sections of the radiology report for this chest X-ray.'
COMPARISON = re.compile(r'\b(compar\w*|prior|previous(ly)?|interval|unchanged|stable|again|new(ly)?|persist\w*|'
                        r'redemonstrat\w*|improv\w*|wors\w*|increas\w*|decreas\w*|resolv\w*|no longer|since|remain\w*|recurr\w*)\b', re.I)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True)
    ap.add_argument('--stage', choices=['score', 'gen'], required=True); ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--set', default='mimic', choices=['mimic', 'mimic_test'])  # mimic_test: held-out split (aware/)
    args = ap.parse_args()
    d = OUT if args.set == 'mimic' else OUT.parent / 'aware'
    recs = [json.loads(l) for l in open(d / f'{args.set}.jsonl')][:args.limit]
    if args.stage == 'score':
        sc = Scorer(args.model)
        out = OUT / f'scores_mimic_{args.model}.jsonl'
        done = {json.loads(l)['id'] for l in open(out)} if out.exists() else set()
        with open(out, 'a') as f:
            for i, r in enumerate(recs):
                if r['id'] in done: continue
                img = Image.open(r['image']).convert('RGB'); t = r['target']
                spans = spans_of(t, r['target_sentences'])
                row = {'id': r['id'], 'has_prior': r['has_prior'],
                       'comparison': [bool(COMPARISON.search(s)) for s in r['target_sentences']]}
                row['I'], _ = sc.score(REPORT_Q, img, t, spans)
                row['0'], _ = sc.score(REPORT_Q, None, t, spans)
                ctxs = {'H': f"Indication: {r['indication']}" if r['indication'] else None,
                        'P': f"Prior report for this patient: {r['prior_report']}" if r['has_prior'] else None}
                for k, c in ctxs.items():
                    row[f'I{k}'] = sc.score(f'{c}\n\n{REPORT_Q}', img, t, spans)[0] if c else None
                    row[k] = sc.score(f'{c}\n\n{REPORT_Q}', None, t, spans)[0] if c else None
                f.write(json.dumps(row) + '\n'); f.flush()
                if i % 100 == 0: print(i, flush=True)
    else:
        model, proc = load_vl(args.model, max_pixels=1024 * 28 * 28)
        with open(d / f"gen_{args.set}_{args.model}.jsonl", 'w') as f:
            for cond in ('none', 'prior'):
                rs = [r for r in recs if cond == 'none' or r['has_prior']]
                texts = [REPORT_Q if cond == 'none' else f"Prior report for this patient: {r['prior_report']}\n\n{REPORT_Q}" for r in rs]
                for b in range(0, len(rs), 8):
                    gens = generate(model, proc, texts[b:b + 8], [Image.open(r['image']).convert('RGB') for r in rs[b:b + 8]], max_new_tokens=200)
                    for r, g in zip(rs[b:b + 8], gens):
                        f.write(json.dumps({'id': r['id'], 'has_prior': r['has_prior'], 'cond': cond, 'text': g,
                                            'n_comparison_sentences': sum(bool(COMPARISON.search(s)) for s in re.split(r'(?<=[.!?])\s+', g))}) + '\n')
                    f.flush(); print(cond, b, flush=True)

if __name__ == '__main__':
    main()
