"""Deterministic, model-independent source audit and pilot selection.

Heuristic strata are screening labels, never adjudicated clinical truth.
"""
import argparse
import collections
import hashlib
import json
import random
import re
from pathlib import Path

ROOT = Path('/data/aofei/Medical_datasets/PubMedVision')
OUT = Path(__file__).parent / 'results'
PATTERNS = {
    'age': r'\b\d{1,3}[ -]year[ -]old\b|\baged?\s+\d{1,3}\b',
    'history': r'\b(?:history of|presented with|was admitted|had undergone|previously diagnosed)\b',
    'external_test': r'\b(?:biopsy|histopatholog\w*|patholog\w* examination|genetic testing|laboratory (?:tests?|results?))\b.{0,65}\b(?:confirm\w*|reveal\w*|show\w*|diagnos\w*)\b',
    'outcome': r'\b(?:follow[ -]up|months? later|years? later|subsequently|eventually)\b.{0,65}\b(?:show\w*|reveal\w*|confirm\w*|resolv\w*|recover\w*|died|improv\w*)\b',
    'reference': r'\b(?:reference (?:information|text)|provided (?:context|history)|clinical history|patient.s history)\b',
}
RX = {k: re.compile(v, re.I) for k, v in PATTERNS.items()}
DIAG = re.compile(r'\b(?:consistent with|suggestive of|indicative of|diagnos\w*|characteristic of|compatible with)\b', re.I)
VIS = re.compile(r'\b(?:shows?|visible|seen|demonstrates?|appears?|depicts?|displays?)\b', re.I)
GENERAL = re.compile(r'\b(?:if|should|would|could|may|might|typically|usually|often|recommend\w*|further|necessary|important|can help)\b', re.I)
ADVICE = re.compile(r'\b(?:crucial|imperative|inquire|importance|need for|requiring|require[sd]?|to (?:confirm|establish|reach|ensure)|help in|essential)\b', re.I)
MODALITIES = {'Computed Tomography', 'Magnetic Resonance Imaging'}

def flags(t):
    return {k: m.group() for k, rx in RX.items() if (m := rx.search(t))}

def sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])|\n+', text) if s.strip()]

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()

def image_path(relative):
    for p in [ROOT/'instruct_images'/relative, ROOT/'images_all'/Path(relative).name]:
        if p.exists():
            return p
    return None

def add_prefix_controls(path):
    rows=[json.loads(l) for l in path.read_text().splitlines()]
    for r in rows:
        pool=[d for d in rows if d['id']!=r['id'] and d['category']==r['category'] and d['modality']==r['modality']]
        donor=min(pool,key=lambda d:(abs(len(d['prefix'].split())-len(r['prefix'].split())),d['id']))
        r['shuffled_prefix']=donor['prefix']
        r['shuffled_prefix_id']=donor['id']
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-cell', type=int, default=16)
    ap.add_argument('--seed', type=int, default=20261018)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    rng = random.Random(args.seed)
    fullpath = ROOT / 'PubMedVision_InstructionTuning_VQA.json'
    subpath = ROOT / '60k_PubMedVision_InstructionTuning_VQA.json'
    cappath = ROOT / 'PubMedVision_Original_Caption.json'
    small = json.load(open(subpath))
    captions = {r['id'].rsplit('_', 1)[-1]: r['Original_Caption'] for r in json.load(open(cappath))}
    sub = {x['id']: x for x in small}
    full = json.load(open(fullpath))
    counts = {k: collections.Counter() for k in ['full', 'medsight_60k']}
    donor_pools = collections.defaultdict(list)
    pools = collections.defaultdict(list)
    audit_examples = []
    exact_subset = 0
    full_ids = set()
    for r in full:
        rid = r['id']
        full_ids.add(rid)
        if len(r['image']) == 1:
            donor_pools[(r['modality'],r['body_part'])].append(r)
        is_sub = rid in sub
        if is_sub and r == sub[rid]:
            exact_subset += 1
        cs = r['conversations']
        q = cs[0]['value'].replace('<image>', '').strip()
        a = cs[1]['value']
        af, qf = flags(a), flags(q)
        missing = {k: v for k, v in af.items() if k not in qf}
        for group in ['full'] + (['medsight_60k'] if is_sub else []):
            c = counts[group]
            c['records'] += 1
            c['single_image'] += len(r['image']) == 1
            c['any_answer_marker'] += bool(af)
            c['answer_marker_category_absent_question'] += bool(missing)
            for k in missing:
                c['question_absent_' + k] += 1
        if missing and len(audit_examples) < 30:
            audit_examples.append(dict(id=rid, question=q, answer=a, flags=missing))
        # A constrained radiology pilot: single image, later answer sentence,
        # no explicit same-category external-context cue in the question.
        if len(r['image']) != 1 or r['modality'] not in MODALITIES:
            continue
        ss = sentences(a)
        found = set()
        for j, s in enumerate(ss):
            if j == 0 or not 8 <= len(s.split()) <= 70:
                continue
            prefix = ' '.join(ss[:j])
            if len(prefix.split()) > 220:
                continue
            sf = flags(s)
            substantive = set(sf) - {'reference'}
            if substantive and any(k not in qf for k in substantive) and not GENERAL.search(s) and not ADVICE.search(s):
                kind = 'external_context'
            elif not sf and DIAG.search(s):
                kind = 'diagnostic_inference'
            elif not sf and VIS.search(s) and not DIAG.search(s) and not GENERAL.search(s):
                kind = 'visual_description'
            else:
                continue
            if kind in found:
                continue
            found.add(kind)
            pools[('medsight_60k' if is_sub else 'full_only', kind)].append(dict(
                id=rid, image=r['image'][0], question=q, answer=a,
                target=s, prefix=prefix, sentence_index=j, category=kind,
                modality=r['modality'], body_part=r['body_part'],
                subset='medsight_60k' if is_sub else 'full_only', screening_matches=sf))
    selected, used = [], set()
    for (subset, kind), pool in sorted(pools.items()):
        rng.shuffle(pool)
        # Round-robin modalities to reduce category/modality confounding.
        bins = {m: [r for r in pool if r['modality'] == m] for m in sorted(MODALITIES)}
        n = 0
        while n < args.per_cell and any(bins.values()):
            for m, rows in bins.items():
                while rows and rows[-1]['id'] in used:
                    rows.pop()
                if rows and n < args.per_cell:
                    r = rows.pop()
                    if not captions.get(r['id'].rsplit('_', 1)[-1]):
                        continue
                    path = image_path(r['image'])
                    if path is None:
                        continue
                    r['image_path'] = str(path)
                    selected.append(r)
                    used.add(r['id'])
                    n += 1
        if n < args.per_cell:
            raise RuntimeError(f'Only {n} eligible examples for {subset}/{kind}')
    for r in selected:
        r['caption'] = captions.get(r['id'].rsplit('_', 1)[-1], '')
        donors = [d for d in donor_pools[r['modality'],r['body_part']] if d['id'] != r['id']]
        rng.shuffle(donors)
        valid_donors = []
        for d in donors:
            if image_path(d['image'][0]) is not None and captions.get(d['id'].rsplit('_',1)[-1]):
                valid_donors.append(d)
            if len(valid_donors) == 2:
                break
        donors = valid_donors
        assert len(donors) == 2
        r['donors'] = [dict(id=d['id'], image_path=str(image_path(d['image'][0])), modality=d['modality'], body_part=d['body_part']) for d in donors]
        r['shuffled_caption'] = captions.get(donors[0]['id'].rsplit('_', 1)[-1], '')
        # Exact phrase overlap is descriptive provenance, not entailment.
        r['caption_marker_matches'] = flags(r['caption'])
    audit = dict(seed=args.seed, per_cell=args.per_cell, counts=counts,
                 selection_version=2,
                 pilot_excludes='Reference-only markers and advisory language; missing source captions; unavailable exact-basename images',
                 subset_exact_record_matches=exact_subset,
                 subset_ids_missing_from_full=sorted(set(sub)-full_ids),
                 screening_patterns=PATTERNS,
                 candidate_pool_sizes={f'{k[0]}/{k[1]}': len(v) for k,v in pools.items()},
                 pilot_n=len(selected), examples=audit_examples,
                 files={str(p): dict(sha256=sha(p), bytes=p.stat().st_size) for p in [fullpath, subpath, cappath]})
    (OUT/'audit.json').write_text(json.dumps(audit, indent=2))
    with open(OUT/'manifest.jsonl', 'w') as f:
        for r in selected:
            f.write(json.dumps(r)+'\n')
    add_prefix_controls(OUT/'manifest.jsonl')
    print(json.dumps({k:v for k,v in audit.items() if k not in ['examples','screening_patterns','files']},indent=2))

if __name__ == '__main__':
    main()
