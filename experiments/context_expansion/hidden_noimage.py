"""Control for the hidden-state probes: same targets, same positions, but no image and no context.
If probes on these states match the image-conditioned probes, the probe signal is textual content, not
image-dependent self-knowledge."""
import argparse, json
import numpy as np
from score_conditions import BASE, LAYERS, CAPTION_Q, Scorer, spans_of

ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True)
ap.add_argument('--start', type=int, default=0); ap.add_argument('--end', type=int, default=None)
args = ap.parse_args()
out = BASE / f'hidden0_{args.model}'; out.mkdir(exist_ok=True)
sc = Scorer(args.model)
for r in [json.loads(l) for l in open(BASE / 'pilot.jsonl')][args.start:args.end]:
    if (out / f"{r['id']}.npz").exists(): continue
    hs = {}
    for key, text, sents, q in [('A', r['answer'], r['answer_sentences'], r['question']),
                                ('B', r['original_caption'], r['caption_sentences'], CAPTION_Q)]:
        _, h = sc.score(q, None, text, spans_of(text, sents), hidden=True)
        hs[key] = np.stack([x if x is not None else np.zeros((len(LAYERS), 7168), np.float16) for x in h]) if h else np.zeros((0,))
    np.savez_compressed(out / f"{r['id']}.npz", **hs)
