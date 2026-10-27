"""Build the input sets for the overnight preliminary experiments (P1b-P6).

d2k.jsonl   2,000 single-image CT/MRI/US records linked to their article: the 600-record pilot first, then
            1,400 new articles. Adds the article context split into sentences, so each context sentence
            can be typed visual / non-visual.
box.jsonl   600 records from the MedSAM3 box set (disjoint images), same fields plus entity boxes (P2 zoom arm).
mimic.jsonl MIMIC-CXR images with their report split into sentences, prior report (if any) and indication (P6).
"""
import csv, json, random, re, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_pilot import MODALITIES, ROOT, sentences
from parse_jats import parse_article, link_record

BASE = Path('/data/aofei/output/ContextMLLM')
OUT = BASE / 'overnight'; OUT.mkdir(exist_ok=True)
BOXES = Path('/data/aofei/Medical_datasets/SpaceAlign/combined_vqa_bboxes_medsam3_xywh.json')
MIMIC = Path('/data/aofei/hallucination/MIMIC_CXR')
MIMIC_META = Path('/data/yuan/mimic-cxr-2.0.0-metadata.csv')

def article_sentences(link):
    """Article context as (source, sentence), in the order and truncation used for scoring."""
    out = [('title', link['title'])] if link['title'] else []
    out += [('abstract', s) for s in sentences(link['abstract'][:800])]
    paras, n = [], 0
    for p in link['citing_paragraphs']:
        for s in sentences(p['text']):
            if n + len(s) > 1500: break
            paras.append(('citing', s)); n += len(s) + 1
    return out + paras

def load_mapping():
    m = {}
    for line in open(BASE / 'mapping/pmc_oa_match.jsonl'):
        r = json.loads(line)
        if len(r['pmcids']) == 1: m[r['id'].rsplit('_', 1)[1]] = r['pmcids'][0]
    return m

def make_record(r, c, pmcid, seen):
    """Mirrors build_pilot.py: an article counts as used once its figure links, even if the image is missing."""
    xml = BASE / 'fulltext/xml' / pmcid[-2:] / f'{pmcid}.xml'
    if not xml.exists(): return None
    try: link = link_record(parse_article(xml), c['Original_Caption'])
    except Exception: return None
    if link is None: return None
    seen.add(pmcid)
    img = ROOT / 'images_all' / Path(r['image'][0]).name
    if not img.exists(): return None
    conv = r['conversations']
    return {'id': r['id'], 'pmcid': pmcid, 'image': str(img), 'modality': r['modality'], 'body_part': r['body_part'],
            'question': conv[0]['value'].replace('<image>', '').strip(), 'answer': conv[1]['value'],
            'answer_sentences': sentences(conv[1]['value']), 'original_caption': c['Original_Caption'].strip(),
            'caption_sentences': sentences(c['Original_Caption']), **link,
            'article_sentences': article_sentences(link)}

def build_pubmed(n_main=2000, n_box=600):
    mapping = load_mapping()
    caps = {r['id'].rsplit('_', 1)[1]: r for r in json.load(open(ROOT / 'PubMedVision_Original_Caption.json'))}
    instr = json.load(open(ROOT / 'PubMedVision_InstructionTuning_VQA.json'))
    pool = []
    for r in instr:
        k = r['id'].rsplit('_', 1)[1]
        c = caps.get(k)
        if len(r['image']) != 1 or r['modality'] not in MODALITIES or not c or c['image'] != r['image']: continue
        pmcid = mapping.get(k)
        if pmcid and (BASE / 'fulltext/xml' / pmcid[-2:] / f'{pmcid}.xml').exists():
            pool.append((pmcid, r, c))
    random.seed(20261024); random.shuffle(pool)
    # the 600-record pilot comes first (the XML pool has grown since it was built, so it is loaded, not rebuilt)
    main = [json.loads(l) for l in open(BASE / 'pilot_needs_context/pilot.jsonl')]
    for r in main: r['article_sentences'] = article_sentences(r)
    seen = {r['pmcid'] for r in main}
    for pmcid, r, c in pool:
        if pmcid in seen: continue
        rec = make_record(r, c, pmcid, seen)
        if rec is None: continue
        main.append(rec)
        if len(main) >= n_main: break
    # box set: MedSAM3 boxes on instruction records, disjoint articles
    by_id = {r['id']: r for r in instr}
    box = []
    recs = [x for x in json.load(open(BOXES)) if 'PubMedVision' in x['image'] and x.get('modality') in MODALITIES]
    random.seed(1); random.shuffle(recs)
    for x in recs:
        r = by_id.get(x['id']); k = x['id'].rsplit('_', 1)[1]; c = caps.get(k); pmcid = mapping.get(k)
        if r is None or c is None or pmcid is None or pmcid in seen or len(r['image']) != 1 or c['image'] != r['image']: continue
        boxes = []
        for ent, hits in x['bbox_results'].items():
            for h in hits:
                bx, by, bw, bh = h['box']
                if h['score'] >= 0.6 and 0.002 <= bw * bh <= 0.35:
                    boxes.append({'entity': ent, 'xywh': h['box'], 'score': h['score']})
        if not boxes: continue
        rec = make_record(r, c, pmcid, seen)
        if rec is None: continue
        rec['boxes'] = boxes; box.append(rec)
        if len(box) >= n_box: break
    return main, box

SECTION = re.compile(r'^\s*([A-Z][A-Z /&]+):', re.M)

def report_sections(text):
    ms = list(SECTION.finditer(text))
    return {m.group(1).strip(): re.sub(r'\s+', ' ', text[m.end():ms[i + 1].start() if i + 1 < len(ms) else len(text)]).strip()
            for i, m in enumerate(ms)}

def build_mimic(n_prior=600, n_first=300):
    studies = defaultdict(dict)
    for r in csv.DictReader(open(MIMIC_META)):
        studies[r['subject_id']][r['study_id']] = r['StudyDate'] + r['StudyTime'].split('.')[0].zfill(6)
    def report(subj, study):
        p = MIMIC / 'reports_files' / f'p{subj[:2]}' / f'p{subj}' / f's{study}.txt'
        return report_sections(p.read_text()) if p.exists() else None
    out = {'prior': [], 'first': []}
    paths = json.load(open(MIMIC / 'sampled_paths_train.json'))
    random.seed(2); random.shuffle(paths)
    for p in paths:
        _, subj, st, _ = p.split('/'); s, t = subj[1:], st[1:]
        img = MIMIC / 'sampled_files_train' / p
        d = studies[s].get(t); cur = report(s, t)
        if d is None or cur is None or not img.exists(): continue
        target = ' '.join(x for x in (cur.get('FINDINGS', ''), cur.get('IMPRESSION', '')) if x)
        if len(target.split()) < 8: continue
        earlier = sorted((v, k) for k, v in studies[s].items() if v < d)
        rec = {'id': f'{s}_{t}', 'image': str(img), 'target': target, 'target_sentences': sentences(target),
               'indication': cur.get('INDICATION', '') or cur.get('HISTORY', ''), 'has_prior': bool(earlier)}
        if earlier:
            pr = report(s, earlier[-1][1])
            if pr is None: continue
            rec['prior_report'] = ' '.join(x for x in (pr.get('FINDINGS', ''), pr.get('IMPRESSION', '')) if x)[:1500]
            if not rec['prior_report']: continue
            key = 'prior'
        else:
            key = 'first'
        if len(out[key]) < (n_prior if key == 'prior' else n_first): out[key].append(rec)
    return out['prior'] + out['first']

def dump(name, rows):
    with open(OUT / name, 'w') as f:
        for r in rows: f.write(json.dumps(r) + '\n')

if __name__ == '__main__':
    if not (OUT / 'mimic.jsonl').exists():
        mimic = build_mimic(); dump('mimic.jsonl', mimic)
        print({'mimic': len(mimic), 'with_prior': sum(r['has_prior'] for r in mimic)}, flush=True)
    main, box = build_pubmed()
    dump('d2k.jsonl', main); dump('box.jsonl', box)
    print({'d2k': len(main), 'box': len(box), 'article_sentences': sum(len(r['article_sentences']) for r in main)})
