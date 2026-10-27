"""Analyses for the needs-context pilot.

Per sentence (nats/token): dC = l(I,C)-l(I), dI = l(I)-l(0), dIC = l(I,C)-l(C), NLL = -l(I).
Semantic class from Qwen3-8B: V visual, P non-visual patient/clinical, K knowledge, M meta.

Q1 Typing: how gains distribute by semantic class (is "needs context" a real, separable category?).
Q2 Isolation: among uncertain & context-resolvable sentences, can no-context signals separate
   P (needs context) from V (needs perception)? AUROC of NLL, entropy, dI, self-report, hidden probes,
   with a text-only baseline. Record-clustered bootstrap CIs.
Q3 Complementary context (target B): does removing figure-citing sentences cut the leak on V while keeping P gains?
Q4 Bias: does the image raise the likelihood of non-visual claims (spurious image gain), per model?
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

BASE = Path('/data/aofei/output/ContextMLLM/pilot_needs_context')
CLASSES = 'VPKM'
TAU = 0.25  # nats/token: "context resolves it"
rng = np.random.default_rng(0)

def load(model):
    pilot = {json.loads(l)['id']: json.loads(l) for l in open(BASE / 'pilot.jsonl')}
    sem = {json.loads(l)['id']: json.loads(l) for l in open(BASE / 'semantic_labels.jsonl')}
    rows = []
    for line in (l for p in sorted(BASE.glob(f'scores_{model}*.jsonl')) for l in open(p)):
        s = json.loads(line)
        if s['id'] not in sem: continue
        hs = np.load(BASE / f'hidden_{model}' / f"{s['id']}.npz")
        p0 = BASE / f'hidden0_{model}' / f"{s['id']}.npz"
        hs0 = np.load(p0) if p0.exists() else None
        for tgt, skey in [('A', 'answer_sentences'), ('B', 'caption_sentences')]:
            sc, sents = s[tgt], pilot[s['id']][skey]
            H = hs[tgt]
            for j, text in enumerate(sents):
                if sc['I'][j] is None: continue
                n = sc['I'][j]['n']; l = {c: sc[c][j]['sum'] / n for c in sc if c != 'self_yes'}
                p = np.array(sem[s['id']][tgt][j])
                r = dict(id=s['id'], tgt=tgt, text=text, n=n, NLL=-l['I'], ent=sc['I'][j]['mean_entropy'],
                         maxent=sc['I'][j]['max_entropy'], dC=l['IC'] - l['I'], dI=l['I'] - l['0'],
                         dIC=l['IC'] - l['C'], self_yes=sc['self_yes'][j],
                         sem=CLASSES[p.argmax()] if p.max() >= 0.5 else '?', pP=p[1], h=H[j],
                         h0=hs0[tgt][j] if hs0 is not None else None)
                if tgt == 'B':
                    r.update(dR=l['IR'] - l['I'], dIR=l['IR'] - l['R'])
                rows.append(r)
    return rows

def boot_auc(y, score, groups, n=1000):
    y, score, groups = map(np.asarray, (y, score, groups))
    point = roc_auc_score(y, score)
    ug = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in ug}
    vals = []
    for _ in range(n):
        take = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        if len(set(y[take])) == 2: vals.append(roc_auc_score(y[take], score[take]))
    return round(point, 3), [round(float(np.percentile(vals, q)), 3) for q in (2.5, 97.5)]

def cv_scores(X, y, groups):
    out = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):
        clf = LogisticRegression(C=0.1, max_iter=3000).fit(X[tr], y[tr])
        out[te] = clf.predict_proba(X[te])[:, 1]
    return out

def summarize(model):
    rows = load(model); res = {'model': model, 'n_sentences': len(rows)}
    for tgt in 'AB':
        R = [r for r in rows if r['tgt'] == tgt]
        # Q1
        by = defaultdict(list)
        for r in R: by[r['sem']].append(r)
        res[f'{tgt}_by_class'] = {c: {'n': len(v), 'dC': round(np.mean([r['dC'] for r in v]), 3),
                                      'dI': round(np.mean([r['dI'] for r in v]), 3),
                                      'dIC': round(np.mean([r['dIC'] for r in v]), 3),
                                      'NLL': round(np.mean([r['NLL'] for r in v]), 3),
                                      'frac_context_resolvable': round(np.mean([r['dC'] >= TAU for r in v]), 3)}
                                  for c, v in sorted(by.items()) if len(v) >= 10}
        # Q2: uncertain & context-resolvable, P vs V
        med = np.median([r['NLL'] for r in R])
        S = [r for r in R if r['NLL'] >= med and r['dC'] >= TAU and r['sem'] in 'PV']
        y = np.array([r['sem'] == 'P' for r in S]).astype(int); g = [r['id'] for r in S]
        q2 = {'n': len(S), 'n_P': int(y.sum()), 'n_V': int(len(y) - y.sum())}
        if len(S) >= 40 and 10 <= y.sum() <= len(y) - 10:
            q2['NLL'] = boot_auc(y, [r['NLL'] for r in S], g)
            q2['mean_entropy'] = boot_auc(y, [r['ent'] for r in S], g)
            q2['neg_dI (image-ablation gain)'] = boot_auc(y, [-r['dI'] for r in S], g)
            q2['neg_self_report_yes'] = boot_auc(y, [-r['self_yes'] for r in S], g)
            H = np.stack([r['h'] for r in S]).astype(np.float32)  # [n, layers, 2*d]
            d = H.shape[2] // 2
            for li, layer in enumerate([7, 14, 21, 28]):
                for part, sl in [('decision', slice(0, d)), ('in_sentence', slice(d, 2 * d))]:
                    X = H[:, li, sl]; X = (X - X.mean(0)) / (X.std(0) + 1e-6)
                    q2[f'probe_L{layer}_{part}'] = boot_auc(y, cv_scores(X, y, g), g)
            if all(r['h0'] is not None for r in S):
                H0 = np.stack([r['h0'] for r in S]).astype(np.float32)
                for li, layer in enumerate([7, 14, 21, 28]):
                    for part, sl in [('decision', slice(0, d)), ('in_sentence', slice(d, 2 * d))]:
                        X = H0[:, li, sl]; X = (X - X.mean(0)) / (X.std(0) + 1e-6)
                        q2[f'noimage_probe_L{layer}_{part}'] = boot_auc(y, cv_scores(X, y, g), g)
            Xt = TfidfVectorizer(min_df=2, ngram_range=(1, 2)).fit_transform([r['text'] for r in S])
            q2['text_only_tfidf'] = boot_auc(y, cv_scores(Xt, y, g), g)
        res[f'{tgt}_isolation'] = q2
        # Q4: spurious image gain on non-visual claims
        P = [r for r in R if r['sem'] == 'P']; V = [r for r in R if r['sem'] == 'V']
        if P and V:
            res[f'{tgt}_image_gain'] = {'P_mean_dI': round(np.mean([r['dI'] for r in P]), 3),
                                       'P_frac_dI_gt_0.1': round(np.mean([r['dI'] > 0.1 for r in P]), 3),
                                       'V_mean_dI': round(np.mean([r['dI'] for r in V]), 3)}
    # Q3 (target B only)
    B = [r for r in rows if r['tgt'] == 'B']
    res['B_redaction'] = {c: {'n': len(v), 'dC_full': round(np.mean([r['dC'] for r in v]), 3),
                              'dC_redacted': round(np.mean([r['dR'] for r in v]), 3),
                              'dI_given_full': round(np.mean([r['dIC'] for r in v]), 3),
                              'dI_given_redacted': round(np.mean([r['dIR'] for r in v]), 3)}
                          for c in 'VP' for v in [[r for r in B if r['sem'] == c]] if len(v) >= 10}
    return res

if __name__ == '__main__':
    out = [summarize(m) for m in sys.argv[1:] or ['qwen25vl', 'lingshu']]
    (BASE / 'analysis.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
