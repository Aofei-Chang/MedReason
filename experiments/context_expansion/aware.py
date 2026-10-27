"""Central experiment: does the model know when it needs more context? (no training)

For every short-answer item (aware/items.jsonl), with no context in the input, collect the model's awareness
signals, each with the correct image (I), a same-modality mismatched image (M) and no image (0):
  need    verbal: "image alone (A) or needs additional patient information (B)?" -> log p(B)/p(A)
  ik      verbal: "will you answer this correctly from this image?" -> log p(Yes)/p(No)  (P(IK)-style)
  req     the answer-or-request protocol's first tag: log p(<request>)/p(<answer>), teacher-forced
  hidden  last prompt-token states (layers 14/21/28) of the plain answer prompt, for probes
  greedy answer mean NLL, first-token entropy, and agreement of 6 samples (T = 1)
and the ground truth, measured on the model itself:
  correctness (token F1) of greedy answers with: no context (I), non-visual context (IN), the visual context
  (IV), and the mismatched image (M);
  VOI = per-token log-likelihood gain of the reference answer from each context (teacher-forced).
"""
import argparse, json
import numpy as np
import torch
from aware_common import AW, TF, load_image
from protocol import PROTOCOL, f1

ANS = 'Answer the question about this medical image with a short phrase of 1-5 words only.\n\nQuestion: {q}'
ANS_CTX = 'Text from the source article:\n{ctx}\n\nAnswer the question about this medical image with a short phrase of 1-5 words only.\n\nQuestion: {q}'
NEED = ('Question about this medical image: "{q}"\n\nCan this question be answered from the image alone, or does answering it '
        'require additional information about the patient that is not visible in the image (such as history, symptoms, '
        'laboratory or pathology results, treatment, outcome, or prior studies)?\n'
        'A: the image alone is enough\nB: additional patient information is needed\nAnswer with a single letter.')
IK = 'Question about this medical image: "{q}"\n\nWill you be able to answer this question correctly from this image? Answer Yes or No.'
LAYERS = [14, 21, 28]
CTX = {'IN': 'ctx_N', 'Ipre': 'ctx_pre', 'IV': 'ctx_V'}
GEN_CTX = ('IN', 'IV')  # Ipre is teacher-forced only

def run_batch(tf, its, ims):
    I = [ims[it['image']] for it in its]; M = [ims[it['image_mis']] for it in its]; Z = [None] * len(its)
    rows = [{'qid': it['qid']} for it in its]
    views = {'I': I, 'M': M, '0': Z}
    # ---- awareness signals (no context)
    for v, im in views.items():
        p = tf.next_token_probs([NEED.format(q=it['question']) for it in its], im, ['B', 'A'], log_odds=True)
        for r, x in zip(rows, p): r[f'need_{v}'] = float(x)
        p = tf.next_token_probs([IK.format(q=it['question']) for it in its], im, ['Yes', 'No'], log_odds=True)
        for r, x in zip(rows, p): r[f'ik_{v}'] = float(x)
        texts = [PROTOCOL.format(q=it['question']) for it in its]
        la, _ = tf.score_batch(texts, im, ['<answer>'] * len(its))
        lr, _ = tf.score_batch(texts, im, ['<request>'] * len(its))
        for r, a, b in zip(rows, la, lr): r[f'req_{v}'] = b[0] - a[0]  # log-odds request vs answer
    # ---- teacher-forced reference answer (VOI) + hidden states
    hid = {}
    plain = [ANS.format(q=it['question']) for it in its]
    tg = [it['answer'] for it in its]
    for v, im in views.items():
        res, h = tf.score_batch(plain, im, tg, hidden_layers=LAYERS)
        hid[v] = h
        for r, (s, n) in zip(rows, res): r[f'lp_{v}'] = s; r['n_tok'] = n
    for c, key in CTX.items():
        sel = [i for i, it in enumerate(its) if it[key]]
        if sel:
            res, _ = tf.score_batch([ANS_CTX.format(ctx=its[i][key], q=its[i]['question']) for i in sel],
                                    [I[i] for i in sel], [tg[i] for i in sel])
            for i, (s, _) in zip(sel, res): rows[i][f'lp_{c}'] = s
    # ---- generation
    for i, g in enumerate(tf.generate(plain, I)):
        t, nll, e0 = g[0]; rows[i].update(ans_I=t, f1_I=f1(t, tg[i]), nll_I=nll, ent0_I=e0)
    for i, g in enumerate(tf.generate(plain, M)):
        rows[i].update(ans_M=g[0][0], f1_M=f1(g[0][0], tg[i]))
    for c in GEN_CTX:
        key = CTX[c]
        sel = [i for i, it in enumerate(its) if it[key]]
        if not sel: continue
        gens = tf.generate([ANS_CTX.format(ctx=its[i][key], q=its[i]['question']) for i in sel], [I[i] for i in sel])
        for i, g in zip(sel, gens): rows[i].update({f'ans_{c}': g[0][0], f'f1_{c}': f1(g[0][0], tg[i])})
    for b in range(0, len(its), 8):
        for i, g in enumerate(tf.generate(plain[b:b + 8], I[b:b + 8], n=6, sample=True), start=b):
            s = [x[0] for x in g]
            ref = rows[i]['ans_I']
            rows[i]['agree'] = float(np.mean([f1(x, ref) >= 0.5 for x in s]))
            rows[i]['samples_f1'] = float(np.mean([f1(x, tg[i]) for x in s]))
            rows[i]['samples'] = s
    return rows, np.stack([hid[v] for v in ('I', 'M', '0')], 1)  # [B, 3 views, layers, d]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True)
    ap.add_argument('--shard', type=int, default=0); ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--bs', type=int, default=16); ap.add_argument('--items', default='items.jsonl')
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    torch.set_num_threads(4)
    items = [json.loads(l) for l in open(AW / args.items)]
    items = [it for i, it in enumerate(items) if i % args.nshards == args.shard]
    args.start = args.shard
    out = AW / f'aware{args.tag}_{args.model}_s{args.start}.jsonl'
    done = {json.loads(l)['qid'] for l in open(out)} if out.exists() else set()
    items = [it for it in items if it['qid'] not in done]
    tf = TF(args.model, max_pixels=256 * 28 * 28)
    H, Q, chunk = [], [], len(list(AW.glob(f'aware{args.tag}_hidden_{args.model}_s{args.start}_c*.npz')))
    with open(out, 'a') as f:
        for b in range(0, len(items), args.bs):
            its = items[b:b + args.bs]
            ims = {p: load_image(p) for it in its for p in (it['image'], it['image_mis'])}
            rows, h = run_batch(tf, its, ims)
            for r in rows: f.write(json.dumps(r) + '\n')
            f.flush(); H.append(h); Q += [r['qid'] for r in rows]
            if len(Q) >= 400 or b + args.bs >= len(items):
                np.savez_compressed(AW / f'aware{args.tag}_hidden_{args.model}_s{args.start}_c{chunk}.npz', h=np.concatenate(H), qid=np.array(Q))
                H, Q, chunk = [], [], chunk + 1
            print(b, len(items), flush=True)

if __name__ == '__main__':
    main()
