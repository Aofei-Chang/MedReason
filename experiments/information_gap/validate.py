"""Audit completed artifacts, scoring matrices, manifests and report links."""
import hashlib,json,math,re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
HERE=Path(__file__).resolve().parent;OUT=HERE/'results'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def finite(obj):
 if isinstance(obj,float):assert math.isfinite(obj)
 elif isinstance(obj,dict):
  for v in obj.values():finite(v)
 elif isinstance(obj,list):
  for v in obj:finite(v)
def main():
 results={};totals=Counter()
 for stage,folder,runner,logprefix in [('primary',OUT,'run_primary.py',''),('adaptive',OUT/'adaptive_prompt','run.py','adaptive_')]:
  expected=dict(ladder=216,fact=1296,anatomy=864,features=12,generation=24) if stage=='primary' else dict(fact=648,anatomy=432,generation=12)
  for model in ['huatuo','medsight']:
   meta=json.loads((folder/f'{model}.meta.json').read_text())
   assert meta['runner_sha256']==sha(HERE/runner)
   for n,h in meta['manifest_hashes'].items():assert sha(folder/n)==h,(stage,model,n)
   rows=[json.loads(l) for l in (folder/f'{model}.jsonl').read_text().splitlines()]
   assert len(rows)==len({r['key'] for r in rows})
   counts=Counter(r['task'] for r in rows);assert counts==expected
   finite(rows);totals.update(counts)
   log=(HERE/f'{logprefix}{model}.log').read_text()
   assert 'COMPLETE' in log and 'Traceback' not in log
   assert 'choice_forward_check' in log and 'greedy_cache_check PASS' in log
   if stage=='primary':assert 'native_loss_check' in log
   results[f'{stage}/{model}']=dict(rows=len(rows),task_counts=dict(counts),manifest_hashes_match=True,runner_hash_matches=True,native_checks_passed=True)
 class Links(HTMLParser):
  def __init__(self):super().__init__();self.targets=[]
  def handle_starttag(self,tag,attrs):
   for k,v in attrs:
    if k in ['href','src'] and not v.startswith(('http:','https:','#','data:')):self.targets.append(v)
 parser=Links();parser.feed((OUT/'report.html').read_text())
 for target in parser.targets:assert (OUT/target).exists(),target
 for p in [OUT/'summary.json',OUT/'adaptive_prompt/summary.json',OUT/'conclusions.json',OUT/'theory_check.json']:finite(json.loads(p.read_text()))
 assert totals==dict(ladder=432,fact=3888,anatomy=2592,features=24,generation=72)
 check=json.loads((OUT/'theory_check.json').read_text());assert abs(check['decomposition_residual'])<1e-12
 report=dict(status='passed',runs=results,total_task_counts=dict(totals),local_report_links_checked=len(parser.targets),theory_identity_residual=check['decomposition_residual'],limitations='Mechanical integrity checks do not validate clinical labels, independence of subjects, or experimental generalization.')
 (OUT/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
