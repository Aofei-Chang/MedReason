"""Map PubMedVision original captions to PMCIDs by normalized exact caption match against PMC-OA."""
import json, re, collections
from pathlib import Path

ROOT = Path('/data/aofei/Medical_datasets/PubMedVision')
UP = Path('/data/aofei/output/ContextMLLM/upstream/pmc_oa')
OUT = Path('/data/aofei/output/ContextMLLM/mapping'); OUT.mkdir(parents=True, exist_ok=True)

def norm(s): return re.sub(r'[^a-z0-9]', '', s.lower())[:300]

index = collections.defaultdict(set)
for fn in ['pmc_oa.jsonl', 'pmc_oa_beta.jsonl']:
    for line in open(UP / fn):
        r = json.loads(line)
        index[norm(r['caption'])].add(r['image'].split('_')[0])

caps = json.load(open(ROOT / 'PubMedVision_Original_Caption.json'))
status = collections.Counter()
with open(OUT / 'pmc_oa_match.jsonl', 'w') as f:
    for r in caps:
        ids = sorted(index.get(norm(r['Original_Caption']), ()))
        status[min(len(ids), 2)] += 1
        f.write(json.dumps({'id': r['id'], 'image': r['image'], 'pmcids': ids}) + '\n')
print({'n': len(caps), 'no_match': status[0], 'unique': status[1], 'ambiguous': status[2]})
