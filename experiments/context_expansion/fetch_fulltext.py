"""Fetch PMC full-text JATS XML for mapped PubMedVision articles.

Primary source: PMC Open Access S3 bucket (pmc-oa-opendata, anonymous HTTPS).
Fallback: Europe PMC fullTextXML. Resumable; files that already exist are skipped.
Usage: python fetch_fulltext.py MAPPING.jsonl [MAPPING2.jsonl ...]
"""
import json, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = Path('/data/aofei/output/ContextMLLM')
XML = BASE / 'fulltext/xml'
LOG = BASE / 'fulltext/fetch_log.jsonl'

def get(url, timeout=60):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()

def fetch(pmcid):
    path = XML / pmcid[-2:] / f'{pmcid}.xml'
    if path.exists():
        return pmcid, 'cached'
    path.parent.mkdir(parents=True, exist_ok=True)
    for v in (1, 2, 3, 4):
        try:
            data = get(f'https://pmc-oa-opendata.s3.amazonaws.com/{pmcid}.{v}/{pmcid}.{v}.xml')
            path.write_bytes(data); return pmcid, f's3.v{v}'
        except urllib.error.HTTPError as e:
            if e.code not in (403, 404): break
        except Exception:
            break
    for attempt in range(3):
        try:
            data = get(f'https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML')
            path.write_bytes(data); return pmcid, 'epmc'
        except urllib.error.HTTPError as e:
            if e.code == 404: return pmcid, 'missing'
            time.sleep(2 ** attempt)
        except Exception:
            time.sleep(2 ** attempt)
    return pmcid, 'error'

if __name__ == '__main__':
    pmcids = set()
    for fn in sys.argv[1:]:
        for line in open(fn):
            r = json.loads(line)
            if len(r['pmcids']) == 1: pmcids.add(r['pmcids'][0])
    pmcids = sorted(pmcids)
    print(f'{len(pmcids)} articles', flush=True)
    with open(LOG, 'a') as log, ThreadPoolExecutor(24) as ex:
        for i, (pmcid, status) in enumerate(ex.map(fetch, pmcids)):
            if status != 'cached': log.write(json.dumps({'pmcid': pmcid, 'status': status}) + '\n')
            if i % 5000 == 0: print(i, pmcid, status, flush=True); log.flush()
    print('done', flush=True)
