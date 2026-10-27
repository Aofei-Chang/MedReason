"""Sensitivity analysis on pre-inspected nonvisual patient-fact spans."""
import csv
import json
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from analyze import interval

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'

def main():
    annotation=json.loads((HERE/'fact_spans.json').read_text())
    manifest={r['id']:r for r in map(json.loads,(OUT/'manifest.jsonl').read_text().splitlines())}
    rows=[]
    for model in ['huatuo','medsight']:
        meta=json.loads((OUT/f'{model}.meta.json').read_text())
        tok=AutoTokenizer.from_pretrained(meta['checkpoint'],local_files_only=True)
        raw=[json.loads(l) for l in (OUT/f'{model}.jsonl').read_text().splitlines()]
        for rid,span in annotation['spans'].items():
            r=manifest[rid]; start=r['target'].index(span);end=start+len(span)
            enc=tok(r['target'],add_special_tokens=False,return_offsets_mapping=True)
            mask=np.array([b>start and a<end for a,b in enc['offset_mapping']])
            assert mask.sum()>0
            d={(x['condition'],x['prefix_mode']):x for x in raw if x['id']==rid}
            assert len(d)==14
            for x in d.values(): assert x['n_tokens']==len(mask)
            for scope in ['full_sentence','fact_span']:
                def s(c,m='isolated'):
                    a=np.array(d[c,m]['token_logp'])
                    return float(a[mask].mean() if scope=='fact_span' else a.mean())
                v=s('correct')-(s('swap_1')+s('swap_2'))/2
                row=dict(id=rid,model=model,scope=scope,subset=r['subset'],span=span,
                         scored_tokens=int(mask.sum()) if scope=='fact_span' else len(mask),
                         image_identity_gain=v,image_vs_none=s('correct')-s('no_image'),
                         prefix_gain_no_image=s('no_image','preceding_answer')-s('no_image'),
                         prefix_specificity_no_image=s('no_image','preceding_answer')-s('no_image','shuffled_answer'),
                         caption_specificity=s('caption')-s('shuffled_caption'),
                         source_over_image=s('caption')-s('shuffled_caption')-v)
                rows.append(row)
    with (OUT/'fact_span_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    sums=[]
    metrics=['image_identity_gain','image_vs_none','prefix_gain_no_image','prefix_specificity_no_image','caption_specificity','source_over_image']
    for model in ['huatuo','medsight']:
        for scope in ['full_sentence','fact_span']:
            rr=[r for r in rows if r['model']==model and r['scope']==scope]
            sums.append(dict(model=model,scope=scope,metrics={k:interval([r[k] for r in rr]) for k in metrics}))
    result=dict(annotation=annotation,summaries=sums,
                caveat='Exploratory selected-subset sensitivity, not a prevalence estimate or validated clinical ground truth. Fact-span tokens still condition on earlier tokens inside the target sentence.')
    (OUT/'fact_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(sums,indent=2))

if __name__=='__main__':main()
