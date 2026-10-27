"""Verify caption->PMCID via BioC full text and extract figure-citing paragraphs (the candidate context)."""
import json, re, sys, time, urllib.request
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path('/data/aofei/Medical_datasets/PubMedVision')
BASE = Path('/data/aofei/output/ContextMLLM')
CACHE = BASE / 'bioc_cache'; CACHE.mkdir(parents=True, exist_ok=True)

def bioc(pmcid):
    p = CACHE / f'{pmcid}.json'
    if not p.exists():
        url = f'https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/unicode'
        with urllib.request.urlopen(url, timeout=60) as r: p.write_bytes(r.read())
        time.sleep(1.2)
    d = json.loads(p.read_text())
    d = d[0] if isinstance(d, list) else d
    return d['documents'][0]['passages']

def norm(s): return re.sub(r'\s+', ' ', s.lower()).strip()

def figure_context(passages, caption):
    figs = [p for p in passages if p['infons'].get('type') in ('fig_caption', 'fig_title_caption')]
    best = max(figs, key=lambda p: SequenceMatcher(None, norm(p['text'])[:400], norm(caption)[:400]).ratio(), default=None)
    if best is None: return None
    sim = SequenceMatcher(None, norm(best['text'])[:400], norm(caption)[:400]).ratio()
    m = re.search(r'(\d+)', best['infons'].get('id', '') or '')
    num = m.group(1) if m else None
    refs = []
    if num:
        pat = re.compile(rf'\b(fig(ure)?s?\.?)\s*{num}(?![0-9])', re.I)
        refs = [p['text'] for p in passages if p['infons'].get('type') == 'paragraph' and pat.search(p['text'])]
    sections = {p['infons'].get('section_type') for p in passages}
    return {'fig_id': best['infons'].get('id'), 'caption_sim': round(sim, 3), 'n_citing_paragraphs': len(refs),
            'citing_chars': sum(map(len, refs)), 'citing_paragraphs': refs, 'sections': sorted(s for s in sections if s),
            'abstract': next((p['text'] for p in passages if p['infons'].get('section_type') == 'ABSTRACT'), '')}

if __name__ == '__main__':
    pilot = [json.loads(l) for l in open(BASE / 'pilot_epmc/pilot.jsonl')]
    oa = {}
    for l in open(BASE / 'mapping/pmc_oa_match.jsonl'):
        r = json.loads(l); oa[r['id']] = r['pmcids']
    caps = {r['id']: r['Original_Caption'] for r in json.load(open(ROOT / 'PubMedVision_Original_Caption.json'))}
    out = open(BASE / 'pilot_epmc/verified.jsonl', 'w')
    for r in pilot:
        rec = {'id': r['id'], 'epmc': r['pmcids'], 'pmc_oa': oa.get(r['id'], [])}
        cand = rec['pmc_oa'][:1] if len(rec['pmc_oa']) == 1 else (r['pmcids'][:1] if len(r['pmcids']) == 1 else [])
        if cand:
            try: rec.update(pmcid=cand[0], **(figure_context(bioc(cand[0]), caps[r['id']]) or {}))
            except Exception as e: rec.update(pmcid=cand[0], error=str(e)[:200])
        out.write(json.dumps(rec) + '\n')
