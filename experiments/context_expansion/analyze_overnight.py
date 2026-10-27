"""Analysis for the validation runs (P1b-P4, P6). Writes overnight/results.json and overnight/RESULTS.md.

Gains are nats/token, per target sentence. The question running through every section: does the
*visual* part of the context act differently from the *non-visual information* part?
  dC_full = l(I,C) - l(I)    dC_vis = l(I,C_vis) - l(I)    dC_non = l(I,C_non) - l(I)    dC_mis = l(I,C_other) - l(I)
  image reliance given context: dI|ctx = l(I,ctx) - l(ctx)
Each section runs independently; a failed section records its error and the rest still runs.
"""
import json, traceback
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

OUT = Path('/data/aofei/output/ContextMLLM/overnight')
CLASSES = 'VPKM'; TAU = 0.25
rng = np.random.default_rng(0)

def labels(name):
    lab = {}
    for line in open(OUT / 'labels.jsonl'):
        x = json.loads(line)
        if x['set'] == name: lab[(x['id'], x['field'])] = [CLASSES[int(np.argmax(p))] if max(p) >= 0.5 else '?' for p in x['probs']]
    return lab

def per_tok(sc, c, j):
    v = sc.get(c)
    if v is None or v[j] is None: return None
    return v[j]['sum'] / v[j]['n']

def boot_mean(vals, groups, n=1000):
    vals, groups = np.asarray(vals, float), np.asarray(groups)
    if len(vals) == 0: return None
    ug = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in ug}
    bs = [vals[np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])].mean() for _ in range(n)]
    return [round(float(vals.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3), len(vals)]

def boot_auc(y, s, groups, n=1000):
    y, s, groups = np.asarray(y), np.asarray(s, float), np.asarray(groups)
    if len(set(y)) < 2: return None
    ug = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in ug}; vals = []
    for _ in range(n):
        t = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        if len(set(y[t])) == 2: vals.append(roc_auc_score(y[t], s[t]))
    return [round(float(roc_auc_score(y, s)), 3), round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]

def interaction(rows, a, b, ca='V', cb='P'):
    """(a - b) on class ca minus (a - b) on class cb, record-clustered bootstrap."""
    R = [r for r in rows if r.get(a) is not None and r.get(b) is not None and r['sem'] in (ca, cb)]
    if not R: return None
    g = np.array([r['id'] for r in R]); d = np.array([r[a] - r[b] for r in R]); is_a = np.array([r['sem'] == ca for r in R])
    ug = np.unique(g); idx = {x: np.where(g == x)[0] for x in ug}
    def est(t): return d[t][is_a[t]].mean() - d[t][~is_a[t]].mean()
    bs = [est(np.concatenate([idx[x] for x in rng.choice(ug, len(ug))])) for _ in range(1000)]
    return [round(float(est(np.arange(len(d)))), 3), round(float(np.nanpercentile(bs, 2.5)), 3), round(float(np.nanpercentile(bs, 97.5)), 3)]

# ------------------------------------------------------------------ P1b + P2a
def typed_rows(model):
    recs = {json.loads(l)['id']: json.loads(l) for l in open(OUT / 'd2k.jsonl')}
    lab = labels('d2k'); rows = []
    for p in sorted(OUT.glob(f'scores_typed_{model}_s*.jsonl')):
        for line in open(p):
            s = json.loads(line); r = recs[s['id']]
            for tgt, fld in [('A', 'answer_sentences'), ('B', 'caption_sentences')]:
                sc = s[tgt]; sem = lab[(s['id'], fld)]
                for j in range(len(r[fld])):
                    l = {c: per_tok(sc, c, j) for c in ('I', '0', 'IC', 'C', 'IV', 'V', 'IN', 'N', 'IX', 'IR', 'R')}
                    if l['I'] is None: continue
                    d = lambda x, y: l[x] - l[y] if l[x] is not None and l[y] is not None else None
                    rows.append({'id': s['id'], 'tgt': tgt, 'sem': sem[j], 'NLL': -l['I'], 'dI': d('I', '0'),
                                 'dC_full': d('IC', 'I'), 'dC_vis': d('IV', 'I'), 'dC_non': d('IN', 'I'), 'dC_mis': d('IX', 'I'),
                                 'dC_regex': d('IR', 'I'), 'dI|full': d('IC', 'C'), 'dI|vis': d('IV', 'V'), 'dI|non': d('IN', 'N'),
                                 'dI|regex': d('IR', 'R'), 'self_img': sc['self_yes'][j], 'self_noimg': sc['self_yes_noimage'][j],
                                 'both_pools': l['IV'] is not None and l['IN'] is not None})
    return rows

def p2a(model):
    rows = typed_rows(model); res = {'n_sentences': len(rows), 'n_records': len({r['id'] for r in rows})}
    keys = ['dC_full', 'dC_vis', 'dC_non', 'dC_mis', 'dC_regex', 'dI', 'dI|full', 'dI|vis', 'dI|non', 'dI|regex']
    for tgt in 'AB':
        R = [r for r in rows if r['tgt'] == tgt and r['both_pools']]
        res[tgt] = {'by_class': {c: {k: boot_mean([r[k] for r in R if r['sem'] == c and r[k] is not None],
                                                   [r['id'] for r in R if r['sem'] == c and r[k] is not None]) for k in keys}
                                 for c in 'VPK'}}
        res[tgt]['crossover_(vis-non)_V_minus_P'] = interaction(R, 'dC_vis', 'dC_non')
        res[tgt]['image_reliance_(dI|non - dI|vis)_V_minus_P'] = interaction(R, 'dI|non', 'dI|vis')
        med = np.median([r['NLL'] for r in R]) if R else 0
        gate = {}
        for k in ('dC_full', 'dC_vis', 'dC_non', 'dC_regex'):
            fired = [r for r in R if r['NLL'] >= med and r[k] is not None and r[k] >= TAU and r['sem'] in 'PV']
            gate[k] = {'n_fired': len(fired), 'share_V': round(float(np.mean([r['sem'] == 'V' for r in fired])), 3) if fired else None}
        res[tgt]['uncertainty_gate_fires_on'] = gate
        # P1b: is the self-report image-dependent? Same subset as P1 Q2 (uncertain, resolvable by full context)
        S = [r for r in R if r['NLL'] >= med and r['dC_full'] is not None and r['dC_full'] >= TAU and r['sem'] in 'PV']
        y = [int(r['sem'] == 'P') for r in S]; g = [r['id'] for r in S]
        allPV = [r for r in R if r['sem'] in 'PV']
        res[tgt]['P1b_self_report'] = {
            'q2_subset_n': len(S),
            'auroc_with_image': boot_auc(y, [-r['self_img'] for r in S], g),
            'auroc_no_image': boot_auc(y, [-r['self_noimg'] for r in S], g),
            'auroc_image_minus_noimage': boot_auc(y, [-(r['self_img'] - r['self_noimg']) for r in S], g),
            'all_PV_auroc_with_image': boot_auc([int(r['sem'] == 'P') for r in allPV], [-r['self_img'] for r in allPV], [r['id'] for r in allPV]),
            'all_PV_auroc_no_image': boot_auc([int(r['sem'] == 'P') for r in allPV], [-r['self_noimg'] for r in allPV], [r['id'] for r in allPV]),
            'corr_img_noimg': round(float(np.corrcoef([r['self_img'] for r in allPV], [r['self_noimg'] for r in allPV])[0, 1]), 3) if allPV else None,
            'mean_shift_V': boot_mean([r['self_img'] - r['self_noimg'] for r in allPV if r['sem'] == 'V'], [r['id'] for r in allPV if r['sem'] == 'V']),
            'mean_shift_P': boot_mean([r['self_img'] - r['self_noimg'] for r in allPV if r['sem'] == 'P'], [r['id'] for r in allPV if r['sem'] == 'P'])}
    return res

# ------------------------------------------------------------------ P2b zoom
def p2b(model):
    recs = {json.loads(l)['id']: json.loads(l) for l in open(OUT / 'box.jsonl')}
    lab = labels('box'); rows = []
    for p in sorted(OUT.glob(f'scores_zoom_{model}_s*.jsonl')):
        for line in open(p):
            s = json.loads(line); sem = lab[(s['id'], 'answer_sentences')]
            for j in range(len(recs[s['id']]['answer_sentences'])):
                l = {c: per_tok(s, c, j) for c in ('I', '0', 'IC', 'IV', 'IN', 'Z', 'ZR')}
                if l['I'] is None: continue
                d = lambda x: l[x] - l['I'] if l[x] is not None else None
                rows.append({'id': s['id'], 'sem': sem[j], 'mentions': s['mentions_entity'][j], 'dZ': d('Z'), 'dZR': d('ZR'),
                             'dZ_specific': (l['Z'] - l['ZR']) if l['Z'] is not None and l['ZR'] is not None else None,
                             'dC_full': d('IC'), 'dC_vis': d('IV'), 'dC_non': d('IN'), 'dI': l['I'] - l['0']})
    keys = ['dZ', 'dZR', 'dZ_specific', 'dC_full', 'dC_vis', 'dC_non', 'dI']
    res = {'n_sentences': len(rows), 'by_class': {}}
    for name, sel in [('V', lambda r: r['sem'] == 'V'), ('V_mentions_boxed_entity', lambda r: r['sem'] == 'V' and r['mentions']),
                      ('V_other', lambda r: r['sem'] == 'V' and not r['mentions']), ('P', lambda r: r['sem'] == 'P'), ('K', lambda r: r['sem'] == 'K')]:
        R = [r for r in rows if sel(r)]
        res['by_class'][name] = {k: boot_mean([r[k] for r in R if r[k] is not None], [r['id'] for r in R if r[k] is not None]) for k in keys}
    zr = [dict(r, zs=r['dZ_specific'], z0=0.0) for r in rows if r['dZ_specific'] is not None]
    res['zoom_specific_V_minus_P'] = interaction(zr, 'zs', 'z0')
    nonv = [dict(r, z0=0.0) for r in rows if r['dC_non'] is not None]
    res['nonvisual_context_P_minus_V'] = interaction(nonv, 'dC_non', 'z0', ca='P', cb='V')
    return res

# ------------------------------------------------------------------ P3
def p3():
    out = defaultdict(lambda: {'texts': 0, 'parse_fail': 0, 'patient': defaultdict(int), 'diagnosis': defaultdict(int),
                               'texts_with_unsupported_patient_claim': 0, 'sentences': 0})
    for p in sorted(OUT.glob('judge_free_s*.jsonl')):
        for line in open(p):
            x = json.loads(line); o = out[f"{x['source']}:{x['prompt']}"]; o['texts'] += 1
            js = x['judge']
            if not isinstance(js, dict) or not isinstance(js.get('claims'), list): o['parse_fail'] += 1; continue
            o['sentences'] += int(js.get('n_sentences') or 0); bad = False
            for c in js['claims']:
                if not isinstance(c, dict): continue
                kind = c.get('kind') if c.get('kind') in ('patient', 'diagnosis') else 'patient'
                v = str(c.get('verdict', '')).upper(); o[kind][v] += 1
                bad |= kind == 'patient' and v in ('UNVERIFIED', 'CONTRADICTED')
            o['texts_with_unsupported_patient_claim'] += bad
    res = {}
    for k, o in out.items():
        ok = o['texts'] - o['parse_fail']
        res[k] = {'texts': o['texts'], 'parse_fail': o['parse_fail'], 'patient_claims_per_text': round(sum(o['patient'].values()) / max(ok, 1), 2),
                  'patient_verdicts': dict(o['patient']), 'diagnosis_claims_per_text': round(sum(o['diagnosis'].values()) / max(ok, 1), 2),
                  'diagnosis_verdicts': dict(o['diagnosis']),
                  'unsupported_patient_claims_per_text': round((o['patient']['UNVERIFIED'] + o['patient']['CONTRADICTED']) / max(ok, 1), 2),
                  'share_texts_with_unsupported_patient_claim': round(o['texts_with_unsupported_patient_claim'] / max(ok, 1), 3)}
    return res

# ------------------------------------------------------------------ P4
def p4():
    return {m: json.loads((OUT / f'p4_{m}.json').read_text()) for m in ('qwen25vl', 'lingshu') if (OUT / f'p4_{m}.json').exists()}

# ------------------------------------------------------------------ P6
def p6(model):
    lab = labels('mimic'); res = {}
    rows = []
    p = OUT / f'scores_mimic_{model}.jsonl'
    if p.exists():
        for line in open(p):
            s = json.loads(line)
            for j, comp in enumerate(s['comparison']):
                l = {c: per_tok(s, c, j) for c in ('I', '0', 'IP', 'P', 'IH', 'H')}
                if l['I'] is None: continue
                d = lambda x, y: l[x] - l[y] if l[x] is not None and l[y] is not None else None
                rows.append({'id': s['id'], 'has_prior': s['has_prior'], 'comparison': comp, 'sem': lab[(s['id'], 'target_sentences')][j],
                             'dI': d('I', '0'), 'dC_prior': d('IP', 'I'), 'dC_indication': d('IH', 'I'), 'dI|prior': d('IP', 'P'), 'NLL': -l['I']})
        for name, sel in [('comparison_sentences', lambda r: r['comparison']), ('other_sentences', lambda r: not r['comparison'])]:
            for hp in (True, False):
                R = [r for r in rows if sel(r) and r['has_prior'] == hp]
                res[f"{name}|{'has_prior' if hp else 'first_study'}"] = {
                    k: boot_mean([r[k] for r in R if r[k] is not None], [r['id'] for r in R if r[k] is not None])
                    for k in ('NLL', 'dI', 'dC_prior', 'dI|prior', 'dC_indication')}
    g = OUT / f'gen_mimic_{model}.jsonl'
    if g.exists():
        G = [json.loads(l) for l in open(g)]
        for cond, hp in [('none', True), ('none', False), ('prior', True)]:
            X = [x for x in G if x['cond'] == cond and x['has_prior'] == hp]
            if X: res[f"gen|{cond}|{'has_prior' if hp else 'first_study'}"] = {
                'n': len(X), 'share_with_comparison': round(float(np.mean([x['n_comparison_sentences'] > 0 for x in X])), 3),
                'comparison_sentences_per_report': round(float(np.mean([x['n_comparison_sentences'] for x in X])), 2)}
    recs = [json.loads(l) for l in open(OUT / 'mimic.jsonl')]
    import re
    from mimic_p6 import COMPARISON
    for hp in (True, False):
        X = [r for r in recs if r['has_prior'] == hp]
        res[f"reference|{'has_prior' if hp else 'first_study'}"] = {
            'n': len(X), 'share_with_comparison': round(float(np.mean([any(COMPARISON.search(s) for s in r['target_sentences']) for r in X])), 3)}
    return res

# ------------------------------------------------------------------ report
def fmt(x):
    if x is None: return '–'
    if isinstance(x, list) and len(x) >= 3: return f'{x[0]:+.3f} [{x[1]:+.2f},{x[2]:+.2f}]'
    return str(x)

def md(res):
    L = ['# Validation results (P1b–P4, P6)', '', 'Generated by `analyze_overnight.py`. Gains in nats/token, 95% record-clustered bootstrap CIs.', '']
    for m in ('qwen25vl', 'lingshu'):
        r = res.get(f'P2a_{m}')
        if not isinstance(r, dict) or 'A' not in r: continue
        L += [f'## P2a typed context — {m} ({r["n_records"]} records, {r["n_sentences"]} sentences)', '']
        for tgt, name in [('A', 'instruction answer, context = caption'), ('B', 'caption, context = article')]:
            L += [f'**Target {tgt}** ({name})', '', '| class | dC full | dC visual ctx | dC non-visual ctx | dC mismatched | dI\\|vis | dI\\|non |', '|---|---|---|---|---|---|---|']
            for c in 'VPK':
                b = r[tgt]['by_class'][c]
                L.append(f"| {c} (n={b['dC_full'][3] if b['dC_full'] else 0}) | " + ' | '.join(fmt(b[k]) for k in ('dC_full', 'dC_vis', 'dC_non', 'dC_mis', 'dI|vis', 'dI|non')) + ' |')
            L += ['', f"- crossover (dC_vis − dC_non), V minus P: {fmt(r[tgt]['crossover_(vis-non)_V_minus_P'])}",
                  f"- image reliance (dI|non − dI|vis), V minus P: {fmt(r[tgt]['image_reliance_(dI|non - dI|vis)_V_minus_P'])}",
                  f"- uncertain & context-resolvable gate, share visual: " + ', '.join(f"{k}: {v['share_V']} (n={v['n_fired']})" for k, v in r[tgt]['uncertainty_gate_fires_on'].items()),
                  f"- P1b self-report AUROC (P vs V, Q2 subset n={r[tgt]['P1b_self_report']['q2_subset_n']}): with image {fmt(r[tgt]['P1b_self_report']['auroc_with_image'])}, "
                  f"without image {fmt(r[tgt]['P1b_self_report']['auroc_no_image'])}, image-minus-no-image {fmt(r[tgt]['P1b_self_report']['auroc_image_minus_noimage'])}; "
                  f"corr {r[tgt]['P1b_self_report']['corr_img_noimg']}", '']
    for m in ('qwen25vl', 'lingshu'):
        r = res.get(f'P2b_{m}')
        if not isinstance(r, dict) or 'by_class' not in r: continue
        L += [f'## P2b zoom vs. context — {m}', '', '| class | zoom | random crop | zoom − random | dC visual | dC non-visual |', '|---|---|---|---|---|---|']
        for c, b in r['by_class'].items():
            L.append(f"| {c} (n={b['dZ'][3] if b['dZ'] else 0}) | " + ' | '.join(fmt(b[k]) for k in ('dZ', 'dZR', 'dZ_specific', 'dC_vis', 'dC_non')) + ' |')
        L += ['', f"- zoom-specific gain, V minus P: {fmt(r['zoom_specific_V_minus_P'])}; non-visual context gain, P minus V: {fmt(r['nonvisual_context_P_minus_V'])}", '']
    if isinstance(res.get('P3'), dict) and res['P3'] and 'error' not in res['P3']:
        L += ['## P3 non-visual claims in free generation (judge: Qwen3-8B with the article)', '',
              '| source:prompt | texts | patient claims / text | unsupported patient claims / text | texts with ≥1 unsupported | diagnosis claims / text | verdicts (patient) |', '|---|---|---|---|---|---|---|']
        for k, v in sorted(res['P3'].items()):
            L.append(f"| {k} | {v['texts']} | {v['patient_claims_per_text']} | {v['unsupported_patient_claims_per_text']} | {v['share_texts_with_unsupported_patient_claim']} | {v['diagnosis_claims_per_text']} | {v['patient_verdicts']} |")
        L.append('')
    if isinstance(res.get('P4'), dict) and res['P4'] and 'error' not in res['P4']:
        L += ['## P4 zero-shot answer-or-request (QA test split)', '', '| model | condition | req image | req context | precision | F1 image | F1 context |', '|---|---|---|---|---|---|---|']
        for m, r in res['P4'].items():
            for c in ('protocol_full', 'protocol_nonvis', 'direct', 'given_full', 'given_nonvis'):
                x = r.get(c, {})
                if 'image' in x: L.append(f"| {m} | {c} | {x['image']['request_rate']} | {x['context']['request_rate']} | {x.get('request_precision')} | {x['image']['f1']} | {x['context']['f1']} |")
        L.append('')
    for m in ('qwen25vl', 'lingshu'):
        r = res.get(f'P6_{m}')
        if not isinstance(r, dict) or 'error' in r: continue
        L += [f'## P6 MIMIC-CXR — {m}', '', '| subset | NLL | dI | dC prior report | dI\\|prior | dC indication |', '|---|---|---|---|---|---|']
        for k, v in r.items():
            if k.startswith(('comparison', 'other')): L.append(f"| {k} | " + ' | '.join(fmt(v[x]) for x in ('NLL', 'dI', 'dC_prior', 'dI|prior', 'dC_indication')) + ' |')
        L += [''] + [f"- {k}: {v}" for k, v in r.items() if k.startswith(('gen', 'reference'))] + ['']
    errs = {k: v['error'] for k, v in res.items() if isinstance(v, dict) and 'error' in v}
    if errs: L += ['## Section errors', ''] + [f'- {k}: `{v.splitlines()[-1]}`' for k, v in errs.items()]
    return '\n'.join(L)

def main():
    res = {}
    tasks = [(f'P2a_{m}', lambda m=m: p2a(m)) for m in ('qwen25vl', 'lingshu')] + \
            [(f'P2b_{m}', lambda m=m: p2b(m)) for m in ('qwen25vl', 'lingshu')] + \
            [('P3', p3), ('P4', p4)] + [(f'P6_{m}', lambda m=m: p6(m)) for m in ('qwen25vl', 'lingshu')]
    for name, fn in tasks:
        try: res[name] = fn()
        except Exception: res[name] = {'error': traceback.format_exc()}
        print(name, 'ok' if 'error' not in res[name] else 'error', flush=True)
    (OUT / 'results.json').write_text(json.dumps(res, indent=1, default=str))
    (OUT / 'RESULTS.md').write_text(md(res))

if __name__ == '__main__':
    main()
