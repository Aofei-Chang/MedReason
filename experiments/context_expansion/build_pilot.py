"""Build the needs-context pilot set: single-image radiology PubMedVision records with linked article context.

Target A: instruction answer (GPT-4V output). Its author context is the original caption.
Target B: original caption (alignment target). Its author context is the rest of the article.
"""
import json, random, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_jats import parse_article, link_record

ROOT = Path('/data/aofei/Medical_datasets/PubMedVision')
BASE = Path('/data/aofei/output/ContextMLLM')
OUT = BASE / 'pilot_needs_context'; OUT.mkdir(parents=True, exist_ok=True)
N = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 600
MODALITIES = {'Computed Tomography', 'Magnetic Resonance Imaging', 'Ultrasound'}
ABBREV = re.compile(r'\b(Fig|Figs|e\.g|i\.e|vs|et al|approx|No|Dr)\.$', re.I)

def sentences(text):
    out, buf = [], ''
    for piece in re.split(r'(?<=[.!?])\s+', text.strip()):
        buf = f'{buf} {piece}'.strip() if buf else piece
        if not ABBREV.search(buf):
            out.append(buf); buf = ''
    if buf: out.append(buf)
    return [s for s in out if len(s.split()) >= 4]

def main():
    mapping = {}
    for line in open(BASE / 'mapping/pmc_oa_match.jsonl'):
        r = json.loads(line)
        if len(r['pmcids']) == 1: mapping[r['id'].rsplit('_', 1)[1]] = r['pmcids'][0]
    caps = {r['id'].rsplit('_', 1)[1]: r for r in json.load(open(ROOT / 'PubMedVision_Original_Caption.json'))}
    pool = []
    for r in json.load(open(ROOT / 'PubMedVision_InstructionTuning_VQA.json')):
        k = r['id'].rsplit('_', 1)[1]
        c = caps.get(k)
        if len(r['image']) != 1 or r['modality'] not in MODALITIES or not c or c['image'] != r['image']: continue
        pmcid = mapping.get(k)
        if pmcid and (BASE / 'fulltext/xml' / pmcid[-2:] / f'{pmcid}.xml').exists():
            pool.append((pmcid, r, c))
    random.seed(20261024); random.shuffle(pool)
    seen, kept, n_fail = set(), [], 0
    for pmcid, r, c in pool:
        if pmcid in seen: continue
        try: link = link_record(parse_article(BASE / 'fulltext/xml' / pmcid[-2:] / f'{pmcid}.xml'), c['Original_Caption'])
        except Exception: link = None
        if link is None: n_fail += 1; continue
        seen.add(pmcid)
        img = ROOT / 'images_all' / Path(r['image'][0]).name
        if not img.exists(): continue
        conv = r['conversations']
        kept.append({'id': r['id'], 'pmcid': pmcid, 'image': str(img), 'modality': r['modality'], 'body_part': r['body_part'],
                     'question': conv[0]['value'].replace('<image>', '').strip(), 'answer': conv[1]['value'],
                     'answer_sentences': sentences(conv[1]['value']), 'original_caption': c['Original_Caption'].strip(),
                     'caption_sentences': sentences(c['Original_Caption']), **link})
        if len(kept) >= N: break
    with open(OUT / 'pilot.jsonl', 'w') as f:
        for k in kept: f.write(json.dumps(k) + '\n')
    types = [k['record_type'] for k in kept]
    print({'eligible_pool': len(pool), 'kept': len(kept), 'link_failures': n_fail,
           'caption_type': types.count('caption'), 'inline_type': types.count('inline'),
           'with_citing_paragraphs': sum(bool(k['citing_paragraphs']) for k in kept),
           'answer_sentences': sum(len(k['answer_sentences']) for k in kept),
           'caption_sentences': sum(len(k['caption_sentences']) for k in kept)})

if __name__ == '__main__':
    main()
