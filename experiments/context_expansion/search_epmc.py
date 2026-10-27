"""Map PubMedVision captions that PMC-OA could not match, via Europe PMC exact-phrase search.

Resumable. Output rows match mapping/pmc_oa_match.jsonl: {id, image, pmcids, query, hitCount}.
"""
import json, re, time, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path('/data/aofei/Medical_datasets/PubMedVision')
BASE = Path('/data/aofei/output/ContextMLLM')
OUT = BASE / 'mapping/epmc_search.jsonl'

def phrase(cap, n_words=14):
    words = re.sub(r'[^A-Za-z0-9\- ]', ' ', cap).split()
    windows = [words[i:i + n_words] for i in range(0, max(1, len(words) - n_words + 1))]
    return ' '.join(max(windows, key=lambda w: sum(map(len, w))))

def search(rec):
    q = phrase(rec['Original_Caption'])
    url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?' + urllib.parse.urlencode(
        {'query': f'"{q}" AND OPEN_ACCESS:Y', 'format': 'json', 'resultType': 'lite', 'pageSize': 5})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                d = json.load(r)
            ids = [x['pmcid'] for x in d.get('resultList', {}).get('result', []) if x.get('pmcid')]
            return {'id': rec['id'], 'image': rec['image'], 'pmcids': ids, 'query': q, 'hitCount': d.get('hitCount', 0)}
        except Exception:
            time.sleep(2 ** attempt)
    return {'id': rec['id'], 'image': rec['image'], 'pmcids': [], 'query': q, 'hitCount': 'error'}

if __name__ == '__main__':
    unmatched = {json.loads(l)['id'] for l in open(BASE / 'mapping/pmc_oa_match.jsonl') if not json.loads(l)['pmcids']}
    done = {json.loads(l)['id'] for l in open(OUT)} if OUT.exists() else set()
    todo = [r for r in json.load(open(ROOT / 'PubMedVision_Original_Caption.json')) if r['id'] in unmatched and r['id'] not in done]
    print(f'{len(todo)} to search ({len(done)} done)', flush=True)
    with open(OUT, 'a') as f, ThreadPoolExecutor(6) as ex:
        for i, row in enumerate(ex.map(search, todo)):
            f.write(json.dumps(row) + '\n')
            if i % 2000 == 0: print(i, flush=True); f.flush()
    print('done', flush=True)
