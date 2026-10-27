"""Paired contrasts, record bootstrap, completeness validation, and figures."""
import csv
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'
METRICS={
    'image_identity_gain': 'Correct image − mean of two matched swaps',
    'image_vs_none': 'Correct image − no image',
    'image_vs_blank': 'Correct image − gray image',
    'prefix_gain': 'Preceding answer benefit with correct image',
    'prefix_gain_no_image': 'Preceding answer benefit without image',
    'prefix_specificity_no_image': 'True preceding answer − length-near, category/modality-matched donor prefix, no image',
    'prefix_image_interaction': 'Image identity gain with prefix − without prefix',
    'caption_rescue': 'True caption − no caption (correct image)',
    'caption_specificity': 'True caption − matched donor caption (correct image)',
    'source_over_image': 'Caption specificity − image identity gain',
    'image_top1': 'Correct image ranks above both donors (ties get fractional credit)',
}

def interval(values, seed=20261018):
    a=np.array(values,dtype=float)
    rng=np.random.default_rng(seed)
    boot=a[rng.integers(len(a),size=(5000,len(a)))].mean(axis=1)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),
                ci95=[float(x) for x in np.quantile(boot,[.025,.975])])

def main():
    manifest={r['id']:r for r in map(json.loads,(OUT/'manifest.jsonl').read_text().splitlines())}
    allrows=[]
    for model in ['huatuo','medsight']:
        raw=[json.loads(l) for l in (OUT/f'{model}.jsonl').read_text().splitlines()]
        keys=[(r['id'],r['condition'],r['prefix_mode']) for r in raw]
        assert len(keys)==len(set(keys)), 'Duplicate scores'
        assert set(r['id'] for r in raw)==set(manifest), 'Missing/extraneous records'
        assert len(raw)==14*len(manifest), 'Incomplete run'
        grouped={rid:{(r['condition'],r['prefix_mode']):r for r in raw if r['id']==rid} for rid in manifest}
        for rid,d in grouped.items():
            assert len(d)==14
            assert len({r['n_tokens'] for r in d.values()})==1
            assert all(np.isfinite(r['mean_logp']) for r in d.values())
            def s(c,m='isolated'): return d[c,m]['mean_logp']
            gain=s('correct')-(s('swap_1')+s('swap_2'))/2
            pg=s('correct','preceding_answer')-s('correct')
            gp=s('correct','preceding_answer')-(s('swap_1','preceding_answer')+s('swap_2','preceding_answer'))/2
            cs=s('caption')-s('shuffled_caption')
            candidates=[s('correct'),s('swap_1'),s('swap_2')]
            top=float(candidates[0]==max(candidates))/candidates.count(max(candidates))
            r=manifest[rid]
            row=dict(id=rid,model=model,category=r['category'],subset=r['subset'],
                     modality=r['modality'],n_tokens=d['correct','isolated']['n_tokens'],
                     sentence_index=r['sentence_index'],prefix_words=len(r['prefix'].split()),
                     image_identity_gain=gain,image_vs_none=s('correct')-s('no_image'),
                     image_vs_blank=s('correct')-s('blank'),prefix_gain=pg,
                     prefix_gain_no_image=s('no_image','preceding_answer')-s('no_image'),
                     prefix_specificity_no_image=s('no_image','preceding_answer')-s('no_image','shuffled_answer'),
                     prefix_image_interaction=gp-gain,caption_rescue=s('caption')-s('correct'),
                     caption_specificity=cs,source_over_image=cs-gain,image_top1=top,
                     correct_logp=s('correct'),target=r['target'])
            allrows.append(row)
    with (OUT/'per_record.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
    queue=[]
    for rid,r in manifest.items():
        row=dict(id=rid,screening_category=r['category'],subset=r['subset'],
                 review_status='unadjudicated',question=r['question'],target=r['target'],
                 caption=r['caption'],image_path=r['image_path'])
        for model in ['huatuo','medsight']:
            x=next(x for x in allrows if x['id']==rid and x['model']==model)
            for k in ['image_identity_gain','prefix_specificity_no_image','caption_specificity','source_over_image']:
                row[model+'_'+k]=x[k]
        queue.append(row)
    queue.sort(key=lambda r:r['huatuo_source_over_image']+r['medsight_source_over_image'],reverse=True)
    with (OUT/'review_queue.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(queue[0]));w.writeheader();w.writerows(queue)
    summaries=[]
    for model in ['huatuo','medsight']:
        for subset in ['all','medsight_60k','full_only']:
            for category in ['all','external_context','diagnostic_inference','visual_description']:
                rr=[r for r in allrows if r['model']==model and (subset=='all' or r['subset']==subset) and (category=='all' or r['category']==category)]
                summaries.append(dict(model=model,subset=subset,category=category,
                                      metrics={k:interval([r[k] for r in rr]) for k in METRICS}))
    contrasts=[]
    # Stratified (within subset) resampling of different-record category groups.
    rng=np.random.default_rng(20261018)
    for model in ['huatuo','medsight']:
        for metric in ['image_identity_gain','caption_specificity','source_over_image','prefix_gain_no_image','prefix_specificity_no_image','prefix_image_interaction']:
            vals=[]; boots=[]
            for subset in ['medsight_60k','full_only']:
                groups=[np.array([r[metric] for r in allrows if r['model']==model and r['subset']==subset and r['category']==cat]) for cat in ['external_context','visual_description']]
                a,b=groups
                vals.append(a.mean()-b.mean())
                boots.append(a[rng.integers(len(a),size=(5000,len(a)))].mean(1)-b[rng.integers(len(b),size=(5000,len(b)))].mean(1))
            boot=np.mean(boots,axis=0)
            contrasts.append(dict(model=model,metric=metric,contrast='external_context_minus_visual_description',mean=float(np.mean(vals)),ci95=np.quantile(boot,[.025,.975]).tolist()))
    result=dict(metric_definitions=METRICS,summaries=summaries,category_contrasts=contrasts,
                validation=dict(n_records=len(manifest),n_models=2,n_scores=len(allrows)*14,
                    unique_keys=True,fixed_target_token_counts=True,finite_scores=True),
                uncertainty='5000 record-bootstrap draws; exploratory pointwise CIs; no multiple-testing correction; heuristic strata, not clinical truth')
    result['group_descriptors']={}
    for category in ['external_context','diagnostic_inference','visual_description']:
        rr=[r for r in manifest.values() if r['category']==category]
        result['group_descriptors'][category]=dict(
            mean_target_words=float(np.mean([len(r['target'].split()) for r in rr])),
            mean_prefix_words=float(np.mean([len(r['prefix'].split()) for r in rr])),
            mean_sentence_index=float(np.mean([r['sentence_index'] for r in rr])),
            mean_donor_prefix_word_difference=float(np.mean([abs(len(r['prefix'].split())-len(r['shuffled_prefix'].split())) for r in rr])))
    (OUT/'summary.json').write_text(json.dumps(result,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(13,4.4))
    cats=['visual_description','diagnostic_inference','external_context']
    for ax,metric in zip(axs,['image_identity_gain','prefix_specificity_no_image','caption_specificity']):
        for m,offset,color in [('huatuo',-.1,'#226fa3'),('medsight',.1,'#c46b23')]:
            sums=[next(s for s in summaries if s['model']==m and s['subset']=='all' and s['category']==c)['metrics'][metric] for c in cats]
            y=np.array([s['mean'] for s in sums]);ci=np.array([s['ci95'] for s in sums])
            ax.errorbar(np.arange(3)+offset,y,yerr=np.maximum(0,np.array([y-ci[:,0],ci[:,1]-y])),fmt='o',capsize=4,label=m,color=color)
        ax.axhline(0,color='gray',linewidth=.8)
        ax.set_xticks(range(3),['Visual','Diagnostic','External'])
        ax.set_title(metric.replace('_',' '));ax.set_ylabel('nats / target token')
    axs[0].legend();fig.suptitle('96 screened training records · frozen models · 95% record-bootstrap intervals')
    fig.tight_layout();fig.savefig(OUT/'pilot_results.png',dpi=180);fig.savefig(OUT/'pilot_results.pdf')
    for s in summaries:
        if s['subset']=='all':
            print(s['model'],s['category'],{k:round(v['mean'],4) for k,v in s['metrics'].items()})
    print(json.dumps(contrasts,indent=2))

if __name__=='__main__': main()
