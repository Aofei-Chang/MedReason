"""Analysis of the awareness / privilege-decomposition run -> aware/results_aware.json and aware/RESULTS_aware.md.

A  awareness (central): can no-context signals tell when the model needs more context?
   ground truth per item, measured on the model: S = right without context; CR = wrong without, right with the
   non-visual context (context-resolvable); PL = wrong with both, right with the visual context (perception-
   limited); and VOI = per-token log-lik gain of the reference answer from the non-visual context.
B  V1 token decomposition on d2k (+ V6a self-report with a mismatched image)
C  privilege type on the box set (caption V part vs box coordinates vs entity names)
D  V2 on-policy rollouts: where a privileged self-teacher pulls the model's own generations
E  V5 MIMIC-CXR held-out test split
F  V6b zoom short-answer accuracy;  V0 time labels
"""
import json, re, traceback
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
import analyze_overnight
from analyze_overnight import boot_mean, labels

def boot_auc(y, s, groups, n=300):
    return analyze_overnight.boot_auc(y, s, groups, n)

AW = Path('/data/aofei/output/ContextMLLM/aware'); OUT = AW.parent / 'overnight'
MODELS = ('qwen25vl', 'lingshu', 'huatuo')
CORRECT = 0.5

def jl(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []

def cv_probe(X, y, g, C=0.05):
    out = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        if len(set(y[tr])) < 2: out[te] = y[tr].mean(); continue
        clf = LogisticRegression(C=C, max_iter=2000).fit(X[tr], y[tr])
        out[te] = clf.predict_proba(X[te])[:, 1]
    return out

def zs(X):
    X = X.astype(np.float32); return (X - X.mean(0)) / (X.std(0) + 1e-5)

# ------------------------------------------------------------------ A awareness
SIGNALS = {  # name -> (field, sign): larger = "needs more context" / less confident
    'verbal_need': ('need', 1), 'neg_P(IK)': ('ik', -1), 'request_tag': ('req', 1)}

def stratified_auc(y, s, strata, groups, n=300):
    """Mann-Whitney AUROC counting only positive/negative pairs within the same stratum; record-clustered CI."""
    from scipy.stats import rankdata
    def est(ix):
        num = den = 0.0
        for t in np.unique(strata[ix]):
            m = ix[strata[ix] == t]; yt, st = y[m], s[m]
            npos, nneg = yt.sum(), len(yt) - yt.sum()
            if npos == 0 or nneg == 0: continue
            rk = rankdata(st); num += rk[yt == 1].sum() - npos * (npos + 1) / 2; den += npos * nneg
        return num / den if den else np.nan
    ug = np.unique(groups); idx = {x: np.where(groups == x)[0] for x in ug}
    rng_ = np.random.default_rng(0)
    bs = [est(np.concatenate([idx[x] for x in rng_.choice(ug, len(ug))])) for _ in range(n)]
    return [round(float(est(np.arange(len(y)))), 3), round(float(np.nanpercentile(bs, 2.5)), 3), round(float(np.nanpercentile(bs, 97.5)), 3)]

def aware_rows(model):
    items = {it['qid']: it for it in jl(AW / 'items.jsonl')}
    rows = []
    for p in sorted(AW.glob(f'aware_{model}_s*.jsonl')):
        for r in jl(p):
            it = items.get(r['qid'])
            if it is None: continue
            n = r['n_tok']
            voi = lambda c: (r[f'lp_{c}'] - r['lp_I']) / n if r.get(f'lp_{c}') is not None else None
            cI = r['f1_I'] >= CORRECT
            cN = r.get('f1_IN') is not None and r['f1_IN'] >= CORRECT
            cV = r.get('f1_IV') is not None and r['f1_IV'] >= CORRECT
            cls = 'S' if cI else 'CR' if cN else 'PL' if cV else 'U'
            rows.append({**r, 'rid': it['rid'], 'source': it['source'], 'template': it['template'], 'type': it['type'],
                         'question': it['question'], 'cls': cls, 'voi_N': voi('IN'), 'voi_pre': voi('Ipre'), 'voi_V': voi('IV'),
                         'img_gain': (r['lp_I'] - r['lp_M']) / n, 'neg_agree': -r['agree'],
                         'evidence_post': bool(it.get('evidence_parts')) and 'post' in it['evidence_parts'] and 'pre' not in it['evidence_parts']})
    return rows

def load_hidden(model, k=256):
    """Hidden states [n, view, layer, d], standardized and PCA-reduced to k dims per (view, layer) for fast probes."""
    H, Q = [], []
    for p in sorted(AW.glob(f'aware_hidden_{model}_s*_c*.npz')):
        d = np.load(p); H.append(d['h']); Q += d['qid'].tolist()
    if not H: return None, {}
    H = np.concatenate(H).astype(np.float32); out = np.zeros(H.shape[:3] + (k,), np.float32)
    for v in range(H.shape[1]):
        for l in range(H.shape[2]):
            X = zs(H[:, v, l]); out[:, v, l] = PCA(k, random_state=0).fit_transform(X)
    return out, {q: i for i, q in enumerate(Q)}

def signal_table(R, y, H=None, hidx=None, text=True):
    """AUROC (record-clustered CI) of every awareness signal for binary target y (1 = needs context / will fail)."""
    g = np.array([r['rid'] for r in R]); out = {}
    if len(set(y)) < 2 or min(y.sum(), len(y) - y.sum()) < 10: return {'n': len(R), 'n_pos': int(y.sum())}
    out['n'] = len(R); out['n_pos'] = int(y.sum())
    for name, (f, s) in SIGNALS.items():
        for v in ('I', 'M', '0'):
            out[f'{name}[{v}]'] = boot_auc(y, [s * r[f'{f}_{v}'] for r in R], g)
        out[f'{name}[I-M]'] = boot_auc(y, [s * (r[f'{f}_I'] - r[f'{f}_M']) for r in R], g)
    out['answer_NLL'] = boot_auc(y, [r['nll_I'] for r in R], g)
    out['first_token_entropy'] = boot_auc(y, [r['ent0_I'] for r in R], g)
    out['sample_disagreement'] = boot_auc(y, [r['neg_agree'] for r in R], g)
    if text:
        Xt = TfidfVectorizer(min_df=2, ngram_range=(1, 2)).fit_transform([r['question'] for r in R])
        out['question_text_tfidf'] = boot_auc(y, cv_probe(Xt, y, g, C=1.0), g)
    if H is not None and all(r['qid'] in hidx for r in R):
        ix = [hidx[r['qid']] for r in R]
        for li, layer in enumerate((14, 21, 28)):
            for vi, v in enumerate(('I', 'M', '0')):
                if layer != 21 and v != 'I': continue
                out[f'probe_L{layer}[{v}]'] = boot_auc(y, cv_probe(zs(H[ix, vi, li]), y, g, C=0.01), g)
    return out

def section_a(model):
    R = aware_rows(model)
    if not R: return None
    H, hidx = load_hidden(model)
    res = {'n_items': len(R)}
    res['class_mix'] = {src: {c: round(np.mean([r['cls'] == c for r in R if r['source'] == src]), 3) for c in ('S', 'CR', 'PL', 'U')}
                        | {'n': sum(r['source'] == src for r in R)} for src in ('qa', 'hard')}
    res['f1_by_type'] = {f'{src}/{t}': {c: round(np.mean([r[f'f1_{c}'] for r in R if r['source'] == src and r['type'] == t and r.get(f'f1_{c}') is not None]), 3)
                                        for c in ('I', 'M', 'IN', 'IV')}
                         for src in ('qa', 'hard') for t in ('image', 'context') if any(r['source'] == src and r['type'] == t for r in R)}
    tasks = {}
    for src in ('qa', 'hard', 'all'):
        S = [r for r in R if src == 'all' or r['source'] == src]
        # T1 needs context: CR vs S
        T = [r for r in S if r['cls'] in ('CR', 'S')]
        tasks[f'{src}: needs-context (CR vs S)'] = signal_table(T, np.array([r['cls'] == 'CR' for r in T]).astype(int), H, hidx)
        # T2 typed: CR vs PL (needs context vs needs perception), among failures
        T = [r for r in S if r['cls'] in ('CR', 'PL')]
        tasks[f'{src}: typed (CR vs PL)'] = signal_table(T, np.array([r['cls'] == 'CR' for r in T]).astype(int), H, hidx)
        # T3 generic failure awareness
        tasks[f'{src}: will fail (not S vs S)'] = signal_table(S, np.array([r['cls'] != 'S' for r in S]).astype(int), H, hidx)
        # T4 data-defined type
        tasks[f'{src}: item type (context vs image)'] = signal_table(S, np.array([r['type'] == 'context' for r in S]).astype(int), H, hidx)
        # T5/T6 likelihood-defined need (does not depend on the strict F1 threshold), nats/token of the reference
        T = [r for r in S if r['voi_N'] is not None and (r['voi_N'] >= 1 or r['voi_N'] < 0.25)]
        tasks[f'{src}: VOI needs-context (voi_N>=1 vs <0.25)'] = signal_table(T, np.array([r['voi_N'] >= 1 for r in T]).astype(int), H, hidx)
        T = [r for r in S if r['voi_N'] is not None and r['voi_V'] is not None and
             ((r['voi_N'] >= 1 and r['voi_V'] < 0.25) or (r['voi_V'] >= 1 and r['voi_N'] < 0.25))]
        tasks[f'{src}: VOI typed (only N helps vs only V helps)'] = signal_table(T, np.array([r['voi_N'] >= 1 for r in T]).astype(int), H, hidx)
    # template-stratified AUROC over the hard items: only pairs within the same template count
    H_ = [r for r in R if r['source'] == 'hard']
    strat = {}
    for task, sel, pos in (('needs-context (CR vs S)', lambda r: r['cls'] in ('CR', 'S'), lambda r: r['cls'] == 'CR'),
                           ('typed (CR vs PL)', lambda r: r['cls'] in ('CR', 'PL'), lambda r: r['cls'] == 'CR'),
                           ('item type (context vs image)', lambda r: True, lambda r: r['type'] == 'context'),
                           ('VOI needs-context', lambda r: r['voi_N'] is not None and (r['voi_N'] >= 1 or r['voi_N'] < 0.25), lambda r: r['voi_N'] >= 1)):
        T = [r for r in H_ if sel(r)]
        y = np.array([pos(r) for r in T]).astype(int); g = np.array([r['rid'] for r in T]); tpl = np.array([r['template'] for r in T])
        out = {}
        for name, f in [(f'{n}[{v}]', (lambda r, f=f, s=s_, v=v: s * r[f'{f}_{v}'])) for n, (f, s_) in SIGNALS.items() for v in ('I', 'M')] + \
                       [('answer_NLL', lambda r: r['nll_I']), ('first_token_entropy', lambda r: r['ent0_I']), ('sample_disagreement', lambda r: r['neg_agree'])]:
            sc = np.array([f(r) for r in T])
            out[name] = stratified_auc(y, sc, tpl, g)
        strat[task] = out
    res['hard_template_stratified'] = strat
    res['request_rate_by_template_class'] = {t: {c: round(np.mean([r['req_I'] > 0 for r in R if r['template'] == t and r['cls'] == c]), 3)
                                                 for c in ('S', 'CR', 'PL', 'U') if any(r['template'] == t and r['cls'] == c for r in R)}
                                             for t in ('diagnosis', 'nature', 'finding', None)}
    for t in ('diagnosis', 'nature', 'finding'):  # wording held fixed: the question text carries no signal
        S = [r for r in R if r['template'] == t]
        T = [r for r in S if r['cls'] in ('CR', 'S')]
        tasks[f'hard/{t}: needs-context (CR vs S)'] = signal_table(T, np.array([r['cls'] == 'CR' for r in T]).astype(int), H, hidx, text=False)
        T = [r for r in S if r['cls'] in ('CR', 'PL')]
        tasks[f'hard/{t}: typed (CR vs PL)'] = signal_table(T, np.array([r['cls'] == 'CR' for r in T]).astype(int), H, hidx, text=False)
        tasks[f'hard/{t}: item type (context vs image)'] = signal_table(S, np.array([r['type'] == 'context' for r in S]).astype(int), H, hidx, text=False)
        T = [r for r in S if r['voi_N'] is not None and (r['voi_N'] >= 1 or r['voi_N'] < 0.25)]
        tasks[f'hard/{t}: VOI needs-context (voi_N>=1 vs <0.25)'] = signal_table(T, np.array([r['voi_N'] >= 1 for r in T]).astype(int), H, hidx, text=False)
    res['tasks'] = tasks
    # continuous: rank correlation with VOI (model-defined need)
    S = [r for r in R if r['voi_N'] is not None]
    res['spearman_with_VOI_N'] = {n: round(float(spearmanr([s * r[f'{f}_I'] for r in S], [r['voi_N'] for r in S])[0]), 3)
                                  for n, (f, s) in SIGNALS.items()}
    res['spearman_with_VOI_N'].update({k: round(float(spearmanr([r[f] for r in S], [r['voi_N'] for r in S])[0]), 3)
                                       for k, f in (('answer_NLL', 'nll_I'), ('sample_disagreement', 'neg_agree'))})
    # knowing vs acting: the protocol's request decision (log-odds > 0) per class
    res['request_rate_by_class'] = {c: round(np.mean([r['req_I'] > 0 for r in R if r['cls'] == c]), 3) for c in ('S', 'CR', 'PL', 'U')}
    res['verbal_need_rate_by_class'] = {c: round(np.mean([r['need_I'] > 0 for r in R if r['cls'] == c]), 3) for c in ('S', 'CR', 'PL', 'U')}
    # P(IK) calibration
    p = 1 / (1 + np.exp(-np.array([r['ik_I'] for r in R]))); c = np.array([r['cls'] == 'S' for r in R])
    bins = np.clip((p * 10).astype(int), 0, 9)
    res['P(IK)_calibration'] = {'mean_p': round(float(p.mean()), 3), 'accuracy': round(float(c.mean()), 3),
                                'ECE': round(float(sum(abs(p[bins == b].mean() - c[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any())), 3)}
    # V0: does the pre-reading part carry the context gain?
    C = [r for r in R if r['type'] == 'context' and r['voi_N'] is not None and r['voi_pre'] is not None]
    res['V0_context_items'] = {'voi_N': boot_mean([r['voi_N'] for r in C], [r['rid'] for r in C]),
                               'voi_pre': boot_mean([r['voi_pre'] for r in C], [r['rid'] for r in C]),
                               'share_evidence_post_only': round(np.mean([r['evidence_post'] for r in R if r['source'] == 'hard' and r['type'] == 'context']), 3)}
    return res

# ------------------------------------------------------------------ B token decomposition (V1, V6a)
def tok_arrays(x):
    lp = {c: (np.array(v) if v is not None else None) for c, v in x['lp'].items()}
    return lp, np.array(x['sent']), np.array(x.get('ent', []))

def section_b(model):
    rows = [r for p in sorted(AW.glob(f'decomp_d2k_{model}_s*.jsonl')) for r in jl(p)]
    if not rows: return None
    lab = labels('d2k')
    res = {'n_records': len(rows)}
    typed = {}
    for typed_scores in (sorted(OUT.glob(f'scores_typed_{model}_s*.jsonl'))):
        for s in jl(typed_scores): typed[s['id']] = s
    for tgt, fld in (('A', 'answer_sentences'), ('B', 'caption_sentences')):
        T = defaultdict(list)  # per token
        sents = []  # per sentence
        for r in rows:
            x = r[tgt]; lp, si, ent = tok_arrays(x)
            sem = lab[(r['key'], fld)]
            have = lambda *cs: all(lp.get(c) is not None for c in cs)
            if not have('I', 'M'): continue
            comp = {}
            if have('Ipre'): comp['G_acq'] = lp['Ipre'] - lp['I']
            if have('IN', 'Ipre'): comp['G_post'] = lp['IN'] - lp['Ipre']
            if have('IV'): comp['G_vis|0'] = lp['IV'] - lp['I']
            if have('IVN', 'IN'): comp['G_vis|N'] = lp['IVN'] - lp['IN']
            if have('IVN'): comp['G_blind'] = lp['IVN'] - lp['I']
            if have('IV', 'MV'): comp['A_teach|0'] = lp['IV'] - lp['MV']
            if have('IVN', 'MVN'): comp['A_teach|N'] = lp['IVN'] - lp['MVN']
            if have('IV', 'MV'): comp['rho|0'] = (lp['MV'] - lp['M']) - (lp['IV'] - lp['I'])
            if have('IVN', 'MVN', 'IN', 'MN'): comp['rho|N'] = (lp['MVN'] - lp['MN']) - (lp['IVN'] - lp['IN'])
            comp['S_img'] = lp['I'] - lp['M']; comp['entropy'] = ent
            for k, v in comp.items(): T[k].append(v)
            T['sem'].append(np.array([sem[j] if 0 <= j < len(sem) else '-' for j in si]))
            T['rid'].append(np.array([r['key']] * len(si)))
            full = all(k in comp for k in ('G_acq', 'G_post', 'G_vis|N'))
            T['full'].append(np.full(len(si), full))
            for j in range(len(sem)):
                m = si == j
                if not m.any(): continue
                sents.append({'rid': r['key'], 'sem': sem[j], **{k: float(v[m].mean()) for k, v in comp.items()},
                              'self_mis': x['self_yes_mis'][j] if j < len(x.get('self_yes_mis', [])) else None,
                              'self_img': typed[r['key']][tgt]['self_yes'][j] if r['key'] in typed else None,
                              'self_noimg': typed[r['key']][tgt]['self_yes_noimage'][j] if r['key'] in typed else None})
        # token-level arrays restricted to records with every condition (so the components sum to G_blind)
        sem_all = np.concatenate(T['sem']); rid_all = np.concatenate(T['rid']); full_all = np.concatenate(T['full'])
        def cat(k):
            out = []
            for v, f in zip(T[k] if k in T else [], T['full']): out.append(v)
            return out
        R = {'by_class_sentence_mean': {}}
        for c in 'VPKM':
            S = [s for s in sents if s['sem'] == c]
            if len(S) < 20: continue
            R['by_class_sentence_mean'][c] = {'n': len(S), **{k: round(float(np.nanmean([s[k] for s in S if k in s])), 3)
                                                             for k in ('G_acq', 'G_post', 'G_vis|0', 'G_vis|N', 'G_blind', 'A_teach|0', 'A_teach|N', 'rho|0', 'rho|N', 'S_img', 'entropy')}}
        # illusion mass: where does the privilege-blind gap go (records with all parts)
        idx = [i for i, f in enumerate(T['full']) if f.all()]
        if idx:
            # components exist for these records; rebuild aligned arrays
            comps = {k: [] for k in ('G_acq', 'G_post', 'G_vis|N', 'G_blind')}
            sem_f = []
            for r_i, r in enumerate([r for r in rows if all(r[tgt]['lp'].get(c) is not None for c in ('I', 'M', 'Ipre', 'IN', 'IVN'))]):
                lp, si, _ = tok_arrays(r[tgt])
                sem = lab[(r['key'], fld)]
                comps['G_acq'].append(lp['Ipre'] - lp['I']); comps['G_post'].append(lp['IN'] - lp['Ipre'])
                comps['G_vis|N'].append(lp['IVN'] - lp['IN']); comps['G_blind'].append(lp['IVN'] - lp['I'])
                sem_f.append(np.array([sem[j] if 0 <= j < len(sem) else '-' for j in si]))
            comps = {k: np.concatenate(v) for k, v in comps.items()}; sem_f = np.concatenate(sem_f)
            pos = comps['G_blind'] > 0
            R['blind_gap_mass'] = {'n_tokens': int(len(sem_f)),
                                   'share_of_positive_blind_gap_on_class': {c: round(float(comps['G_blind'][pos & (sem_f == c)].sum() / comps['G_blind'][pos].sum()), 3) for c in 'VPKM'},
                                   'decomposition_share (sum G_x / sum G_blind)': {k: round(float(comps[k].sum() / comps['G_blind'].sum()), 3) for k in ('G_acq', 'G_post', 'G_vis|N')},
                                   'decomposition_on_P_tokens': {k: round(float(comps[k][sem_f == 'P'].sum() / comps['G_blind'][sem_f == 'P'].sum()), 3) for k in ('G_acq', 'G_post', 'G_vis|N')},
                                   'decomposition_on_V_tokens': {k: round(float(comps[k][sem_f == 'V'].sum() / comps['G_blind'][sem_f == 'V'].sum()), 3) for k in ('G_acq', 'G_post', 'G_vis|N')}}
        # attainability: among sentences the visual privilege helps, does A_teach / rho / S_img pick out V over P?
        att = {}
        for gate in ('G_vis|0',):
            S = [s for s in sents if s['sem'] in 'VP' and s.get(gate) is not None and s[gate] > 0.25]
            if len(S) < 40: continue
            y = np.array([s['sem'] == 'V' for s in S]).astype(int); g = [s['rid'] for s in S]
            att[f'V vs P among sentences with {gate} > 0.25'] = {'n': len(S), 'n_V': int(y.sum()),
                **{k: boot_auc(y, [s[k] for s in S], g) for k in ('A_teach|0', 'rho|0', 'S_img') if all(k in s for s in S)},
                'G_vis|0 itself': boot_auc(y, [s['G_vis|0'] for s in S], g)}
            for c in 'VP':
                v = np.array([s['A_teach|0'] for s in S if s['sem'] == c])
                att[f'A_teach|0 quantiles (10/50/90) on {c}'] = [round(float(np.percentile(v, q)), 3) for q in (10, 50, 90)]
        R['attainability'] = att
        # token-level awareness: does the student's own entropy (no context) flag tokens the request would fix
        # (G_acq high) differently from tokens only the visual privilege fixes (G_vis high, G_acq low)?
        tk = []
        for r in rows:
            lp, si, ent = tok_arrays(r[tgt])
            if lp.get('Ipre') is None or lp.get('IV') is None or not len(ent): continue
            ga, gv = lp['Ipre'] - lp['I'], lp['IV'] - lp['I']
            for k in np.where((ga > 1) & (gv < 0.2))[0]: tk.append((1, ent[k], -lp['I'][k], r['key']))
            for k in np.where((gv > 1) & (ga < 0.2))[0]: tk.append((0, ent[k], -lp['I'][k], r['key']))
        if len(tk) > 100:
            y = np.array([t[0] for t in tk]); g = [t[3] for t in tk]
            R['token_awareness (needs-request vs needs-visual tokens)'] = {
                'n_request_tokens': int(y.sum()), 'n_visual_tokens': int(len(y) - y.sum()),
                'entropy': boot_auc(y, [t[1] for t in tk], g), 'surprisal': boot_auc(y, [t[2] for t in tk], g)}
        # V6a self-report: is the image effect instance-specific? (correct vs mismatched image)
        S = [s for s in sents if s['sem'] in 'VP' and s['self_mis'] is not None and s['self_img'] is not None]
        if S:
            y = np.array([s['sem'] == 'V' for s in S]).astype(int); g = [s['rid'] for s in S]
            R['V6a_self_report'] = {c: {'p_yes_image': round(np.mean([s['self_img'] for s in S if s['sem'] == c]), 3),
                                        'p_yes_mismatched': round(np.mean([s['self_mis'] for s in S if s['sem'] == c]), 3),
                                        'p_yes_noimage': round(np.mean([s['self_noimg'] for s in S if s['sem'] == c]), 3)} for c in 'VP'}
            R['V6a_self_report']['AUROC V vs P'] = {k: boot_auc(y, v, g) for k, v in (
                ('image', [s['self_img'] for s in S]), ('mismatched', [s['self_mis'] for s in S]),
                ('image - mismatched', [s['self_img'] - s['self_mis'] for s in S]),
                ('image - noimage', [s['self_img'] - s['self_noimg'] for s in S]))}
        res[tgt] = R
    return res

# ------------------------------------------------------------------ V4 external teacher (true DOPD gap)
def section_v4(teacher, student='qwen25vl'):
    """Same tokens (shared tokenizer), same matched inputs: teacher minus student log-prob is a capability gap."""
    load = lambda m: {r['key']: r for p in sorted(AW.glob(f'decomp_d2k_{m}_s*.jsonl')) for r in jl(p)}
    T, S_ = load(teacher), load(student)
    if not T or not S_: return None
    lab = labels('d2k'); out = {}
    for tgt, fld in (('A', 'answer_sentences'), ('B', 'caption_sentences')):
        sents = []
        for k in T.keys() & S_.keys():
            lt, ls = T[k][tgt]['lp'], S_[k][tgt]['lp']; si = np.array(T[k][tgt]['sent']); sem = lab[(k, fld)]
            for j in range(len(sem)):
                m = si == j
                if not m.any() or sem[j] not in 'VP': continue
                d = {'rid': k, 'sem': sem[j], 'gap|I': float((np.array(lt['I']) - np.array(ls['I']))[m].mean())}
                if lt.get('IVN') is not None: d['gap|VN (matched privilege)'] = float((np.array(lt['IVN']) - np.array(ls['IVN']))[m].mean())
                if lt.get('IV') is not None:
                    d['gap|V'] = float((np.array(lt['IV']) - np.array(ls['IV']))[m].mean())
                    d['G_vis|0 (student)'] = float((np.array(ls['IV']) - np.array(ls['I']))[m].mean())
                sents.append(d)
        R = {c: {k: boot_mean([s[k] for s in sents if s['sem'] == c and k in s], [s['rid'] for s in sents if s['sem'] == c and k in s])
                 for k in ('gap|I', 'gap|V', 'gap|VN (matched privilege)')} for c in 'VP'}
        G = [s for s in sents if 'gap|V' in s and s['G_vis|0 (student)'] > 0.25]
        if len(G) > 40:
            y = np.array([s['sem'] == 'V' for s in G]).astype(int); g = [s['rid'] for s in G]
            R['V vs P among sentences with student G_vis|0 > 0.25'] = {k: boot_auc(y, [s[k] for s in G], g) for k in ('gap|V', 'gap|I', 'G_vis|0 (student)')}
        out[tgt] = R
    return out

# ------------------------------------------------------------------ C box privilege types
def section_c(model):
    rows = [r for p in sorted(AW.glob(f'decomp_box_{model}_s*.jsonl')) for r in jl(p)]
    if not rows: return None
    lab = labels('box'); recs = {r['id']: r for r in jl(OUT / 'box.jsonl')}
    S = []
    for r in rows:
        lp, si, _ = tok_arrays(r['A']); sem = lab[(r['key'], 'answer_sentences')]
        for j in range(len(sem)):
            m = si == j
            if not m.any(): continue
            d = {'rid': r['key'], 'grp': ('V-entity' if r['mentions_entity'][j] else 'V-other') if sem[j] == 'V' else sem[j]}
            for k in ('V', 'BOX', 'ENT'):
                if lp.get('I' + k) is None: continue
                d[f'gain_{k}'] = float((lp['I' + k] - lp['I'])[m].mean())
                d[f'A_teach_{k}'] = float((lp['I' + k] - lp['M' + k])[m].mean())
            S.append(d)
    out = {}
    for grp in ('V-entity', 'V-other', 'P', 'K'):
        G = [s for s in S if s['grp'] == grp]
        if len(G) < 20: continue
        out[grp] = {'n': len(G), **{k: boot_mean([s[k] for s in G if k in s], [s['rid'] for s in G if k in s])
                                    for k in ('gain_V', 'gain_BOX', 'gain_ENT', 'A_teach_V', 'A_teach_BOX', 'A_teach_ENT')}}
    return out

# ------------------------------------------------------------------ D on-policy rollouts (V2)
def section_d(model):
    rows = [r for p in sorted(AW.glob(f'decomp_genfree_{model}_s*.jsonl')) for r in jl(p)]
    if not rows: return None
    lg = {(x['id'], x['prompt'], x['i']): 'VPKM'[int(np.argmax(x['probs']))] for x in jl(AW / 'labels_gen.jsonl') if x['model'] == model}
    out = {'n_texts': len(rows)}
    toks = defaultdict(list)
    for r in rows:
        rid, pr = r['key'].split('|')
        lp, si, _ = tok_arrays(r['G'])
        cls = np.array([lg.get((rid, pr, j), '-') if j >= 0 else '-' for j in si])
        toks['cls'].append(cls); toks['rid'].append(np.array([rid] * len(si))); toks['prompt'].append(np.array([pr] * len(si)))
        for name, c in (('blind (V+pre+post)', 'IVN'), ('matched, no request (V)', 'IV'), ('matched, after request (V+pre)', 'IVpre')):
            toks[name].append(lp[c] - lp['I'] if lp.get(c) is not None else np.full(len(si), np.nan))
        toks['A_teach'].append(lp['IV'] - lp['MV'] if lp.get('IV') is not None and lp.get('MV') is not None else np.full(len(si), np.nan))
    T = {k: np.concatenate(v) for k, v in toks.items()}
    out['sentence_type_share'] = {pr: {c: round(float(np.mean(T['cls'][(T['prompt'] == pr) & (T['cls'] != '-')] == c)), 3) for c in 'VPKM'} for pr in ('q', 'd')}
    for name in ('blind (V+pre+post)', 'matched, no request (V)', 'matched, after request (V+pre)'):
        g = T[name]; ok = ~np.isnan(g); pos = ok & (g > 0)
        out[name] = {'mean_gap_by_class': {c: round(float(np.nanmean(g[ok & (T['cls'] == c)])), 3) for c in 'VPKM'},
                     'share_of_positive_pull_on_class': {c: round(float(g[pos & (T['cls'] == c)].sum() / g[pos].sum()), 3) for c in 'VPKM'},
                     'frac_P_tokens_pulled_up_>0.5': round(float(np.mean(g[ok & (T['cls'] == 'P')] > 0.5)), 3)}
    # routing: keep the matched (no request) teacher's positive pull only where it depends on the right image
    g = T['matched, no request (V)']; a = T['A_teach']; ok = ~np.isnan(g) & ~np.isnan(a)
    for th in (0.0, 0.5):
        keep = ok & (g > 0) & (a > th)
        out[f'routed (A_teach > {th})'] = {'share_of_kept_pull_on_class': {c: round(float(g[keep & (T['cls'] == c)].sum() / max(g[keep].sum(), 1e-9)), 3) for c in 'VPKM'},
                                           'kept_fraction_of_pull': round(float(g[keep].sum() / g[ok & (g > 0)].sum()), 3)}
    return out

# ------------------------------------------------------------------ E MIMIC held-out (V5)
def section_e(model):
    rows = [r for p in sorted(AW.glob(f'decomp_mimic_test_{model}_s*.jsonl')) for r in jl(p)]
    cmp = defaultdict(dict)
    for x in jl(AW / 'labels_cmp.jsonl'):
        cmp[(x['src'], x['set'], x['id'], x.get('cond'))][x['i']] = x['p_yes']
    out = {}
    if rows:
        S = []
        for r in rows:
            lp, si, ent = tok_arrays(r['R']); base = 'IH' if lp.get('IH') is not None else 'I'
            flags = cmp.get(('ref', 'mimic_test', r['key'], None), {})
            for j in sorted(set(si.tolist()) - {-1}):
                m = si == j
                d = {'rid': r['key'], 'has_prior': r['has_prior'], 'cmp': flags.get(j, 0) > 0.5, 'nll': float(-lp['I'][m].mean()),
                     'entropy': float(ent[m].mean())}
                if lp.get('IHP') is not None: d['G_acq(prior)'] = float((lp['IHP'] - lp[base])[m].mean())
                if lp.get('IH') is not None and lp.get('MH') is not None: d['S_img|H'] = float((lp['IH'] - lp['MH'])[m].mean())
                if lp.get('IHP') is not None and lp.get('MHP') is not None: d['S_img|HP'] = float((lp['IHP'] - lp['MHP'])[m].mean())
                S.append(d)
        for name, sel in (('comparison, has prior', lambda s: s['cmp'] and s['has_prior']), ('other, has prior', lambda s: not s['cmp'] and s['has_prior']),
                          ('comparison, first study', lambda s: s['cmp'] and not s['has_prior']), ('other, first study', lambda s: not s['cmp'] and not s['has_prior'])):
            G = [s for s in S if sel(s)]
            if G: out[name] = {'n': len(G), **{k: boot_mean([s[k] for s in G if k in s], [s['rid'] for s in G if k in s])
                                              for k in ('G_acq(prior)', 'S_img|H', 'S_img|HP', 'nll', 'entropy') if any(k in s for s in G)}}
        G = [s for s in S if s['has_prior'] and 'G_acq(prior)' in s]
        y = np.array([s['cmp'] for s in G]).astype(int); g = [s['rid'] for s in G]
        out['awareness: comparison vs other sentences (has prior), no-context signal'] = {
            'entropy': boot_auc(y, [s['entropy'] for s in G], g), 'nll': boot_auc(y, [s['nll'] for s in G], g),
            'G_acq(prior) (oracle)': boot_auc(y, [s['G_acq(prior)'] for s in G], g)}
    # memorization check: NLL on train (overnight P6) vs held-out test
    tr = jl(OUT / f'scores_mimic_{model}.jsonl')
    if tr and rows:
        nt = [sum(x['sum'] for x in r['I'] if x) / max(1, sum(x['n'] for x in r['I'] if x)) for r in tr]
        te = [float(np.mean(r['R']['lp']['I'])) for r in rows]
        out['NLL per token, train vs held-out test'] = [round(-float(np.mean(nt)), 3), round(-float(np.mean(te)), 3)]
    # fabricated comparisons in generation (LLM detector)
    for stem, st in ((f'gen_mimic_{model}', 'mimic'), (f'gen_mimic_test_{model}', 'mimic_test')):
        G = defaultdict(list)
        for (src, s, rid, cond), v in cmp.items():
            if src != stem: continue
            G[(cond, rid)] = v
        gens = jl((OUT if st == 'mimic' else AW) / f'{stem}.jsonl')
        if not gens: continue
        tab = {}
        for cond in ('none', 'prior'):
            for hp in (True, False):
                xs = [any(p > 0.5 for p in G.get((cond, g['id']), {}).values()) for g in gens if g['cond'] == cond and g['has_prior'] == hp]
                if xs: tab[f"{cond} | {'has prior' if hp else 'first study'}"] = [round(float(np.mean(xs)), 3), len(xs)]
        out[f'share of reports with a comparison sentence ({st})'] = tab
    ref = {}
    for st in ('mimic', 'mimic_test'):
        recs = jl((OUT if st == 'mimic' else AW) / f'{st}.jsonl')
        for hp in (True, False):
            xs = [any(p > 0.5 for p in cmp.get(('ref', st, r['id'], None), {}).values()) for r in recs if r['has_prior'] == hp]
            if xs: ref[f"{st} | {'has prior' if hp else 'first study'}"] = round(float(np.mean(xs)), 3)
    out['reference reports with a comparison sentence'] = ref
    return out

# ------------------------------------------------------------------ F zoom, V0
def section_f(model):
    R = jl(AW / f'zoom_qa_{model}.jsonl')
    if not R: return None
    g = [r['qid'] for r in R]
    return {'n': len(R), **{f'F1_{v}': boot_mean([r[f'f1_{v}'] for r in R], g) for v in ('I', 'Z', 'ZR')},
            'F1 Z - ZR': boot_mean([r['f1_Z'] - r['f1_ZR'] for r in R], g),
            'lp/token Z - ZR': boot_mean([r['lp_Z'] - r['lp_ZR'] for r in R], g),
            'lp/token Z - I': boot_mean([r['lp_Z'] - r['lp_I'] for r in R], g)}

def section_v0():
    c = defaultdict(int)
    for x in jl(AW / 'labels_time.jsonl'):
        for p in x['probs']: c[('pre', 'post', 'general')[int(np.argmax(p))]] += 1
    n = sum(c.values())
    return {'P_context_sentences': n, **{k: round(v / n, 3) for k, v in c.items()}} if n else None

# ------------------------------------------------------------------ report
def fmt(x):
    if isinstance(x, list) and len(x) >= 3 and all(isinstance(v, (int, float)) for v in x[:3]):
        return f'{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]' if len(x) == 4 else f'{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]'
    return json.dumps(x)

def md(res):
    L = ['# Awareness / privilege-decomposition run (V0-V6)', '',
         'Generated by analyze_aware.py. AUROC entries: point [95% record-clustered CI]; boot_mean entries: mean [CI] (n).', '']
    for sec, title in (('A', 'A. Awareness of needing context'), ('B', 'B. Token decomposition (V1) + self-report control (V6a)'),
                       ('C', 'C. Teacher privilege type (box set)'), ('D', 'D. On-policy rollouts (V2)'),
                       ('E', 'E. MIMIC-CXR held-out (V5)'), ('F', 'F. Zoom short-answer (V6b)')):
        L += [f'## {title}', '']
        for m in MODELS:
            x = res.get(sec, {}).get(m)
            if not x: continue
            L += [f'### {m}', '']
            def walk(d, depth=0):
                for k, v in d.items():
                    if isinstance(v, dict) and not all(not isinstance(u, dict) for u in v.values()) or (isinstance(v, dict) and len(v) > 6):
                        L.append(f"{'  ' * depth}- **{k}**"); walk(v, depth + 1)
                    elif isinstance(v, dict):
                        L.append(f"{'  ' * depth}- {k}: " + '; '.join(f'{a} {fmt(b)}' for a, b in v.items()))
                    else:
                        L.append(f"{'  ' * depth}- {k}: {fmt(v)}")
            walk(x); L.append('')
    L += ['## V4 external teacher (matched inputs, same tokens)', '']
    for k, v in (res.get('V4') or {}).items():
        L.append(f'### {k}')
        for t, x in (v or {}).items(): L.append(f'- {t}: {json.dumps(x)}')
        L.append('')
    L += ['## V0 time labels', '', f"- {fmt(res.get('V0'))}", '']
    return '\n'.join(L)

def main():
    res = {}
    for sec, fn in (('A', section_a), ('B', section_b), ('C', section_c), ('D', section_d), ('E', section_e), ('F', section_f)):
        res[sec] = {}
        for m in MODELS:
            try: res[sec][m] = fn(m)
            except Exception: res[sec][m] = {'error': traceback.format_exc()[-1500:]}
    res['V4'] = {}
    for t in ('lingshu', 'huatuo'):
        try: res['V4'][f'{t} teacher -> qwen25vl student'] = section_v4(t)
        except Exception: res['V4'][t] = {'error': traceback.format_exc()[-1500:]}
    res['V0'] = section_v0()
    (AW / 'results_aware.json').write_text(json.dumps(res, indent=1))
    (AW / 'RESULTS_aware.md').write_text(md(res))
    print(md(res)[:3000])

if __name__ == '__main__':
    main()
