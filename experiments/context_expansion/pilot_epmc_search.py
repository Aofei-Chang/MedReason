"""Pilot: can PubMedVision original captions be traced to a PMC article via Europe PMC phrase search?"""
import json, random, re, sys, time, urllib.parse, urllib.request
from pathlib import Path

ROOT = Path('/data/aofei/Medical_datasets/PubMedVision')
OUT = Path('/data/aofei/output/ContextMLLM/pilot_epmc'); OUT.mkdir(parents=True, exist_ok=True)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200

def phrase(cap, n_words=14):
    # longest run of plain words, avoids quotes/symbols that break the query
    words = [w for w in re.sub(r'[^A-Za-z0-9\- ]', ' ', cap).split()]
    best = max((words[i:i+n_words] for i in range(0, max(1, len(words)-n_words+1))), key=lambda w: sum(len(x) for x in w))
    return ' '.join(best)

def search(q):
    url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?' + urllib.parse.urlencode(
        {'query': f'"{q}" AND OPEN_ACCESS:Y', 'format': 'json', 'resultType': 'lite', 'pageSize': 5})
    with urllib.request.urlopen(url, timeout=30) as r:
        d = json.load(r)
    return [x.get('pmcid') for x in d.get('resultList', {}).get('result', []) if x.get('pmcid')], d.get('hitCount', 0)

caps = json.load(open(ROOT / 'PubMedVision_Original_Caption.json'))
random.seed(0); sample = random.sample(caps, N)
hits = 0
with open(OUT / 'pilot.jsonl', 'w') as f:
    for i, r in enumerate(sample):
        q = phrase(r['Original_Caption'])
        try: pmcids, n = search(q)
        except Exception as e: pmcids, n = [], f'err:{e}'
        hits += len(pmcids) == 1
        f.write(json.dumps({'id': r['id'], 'image': r['image'], 'query': q, 'pmcids': pmcids, 'hitCount': n}) + '\n')
        time.sleep(0.12)
print(f'unique-hit {hits}/{N}')
