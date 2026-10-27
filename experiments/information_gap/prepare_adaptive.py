"""Record the exploratory second prompt after the primary analysis."""
import hashlib,json,shutil
from pathlib import Path
p=Path(__file__).resolve().parent/'results';o=p/'adaptive_prompt';o.mkdir(exist_ok=True)
for n in ['facts.jsonl','anatomy.jsonl']:shutil.copyfile(p/n,o/n)
protocol={'version':1,'adaptive':True,'reason':'Primary provenance prompt improved missing-fact abstention but damaged negative-evidence acceptance and visual anatomy responses. Test an explicit supported/contradicted/unknown decision rule and affirmative image inspection instruction.','selection':'All primary records, no outcome-based exclusions.','tasks':['fact','anatomy','generation'],'policy':'evidence_first','expected_per_model':{'fact':648,'anatomy':432,'generation':12},'primary_summary_sha256':hashlib.sha256((p/'summary.json').read_bytes()).hexdigest(),'interpretation':'Exploratory prompt robustness check chosen after primary results; not preregistered confirmation. Compare against unchanged primary baseline; original failures retained.'}
(o/'protocol.json').write_text(json.dumps(protocol,indent=2))
