"""Paired, record-level analysis; permutations are repeated measures, not n=6N."""
import csv,itertools,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;OUT=HERE/'results'
RNG=np.random.default_rng(20261019)

def stat(x):
 x=np.array(x,dtype=float);assert len(x) and np.isfinite(x).all()
 boot=x[RNG.integers(0,len(x),(5000,len(x)))].mean(1)
 result=dict(n=len(x),mean=float(x.mean()),ci95=np.quantile(boot,[.025,.975]).tolist())
 if np.all((x==0)|(x==1)):
  n=len(x);p=x.mean();z=1.95996398454;d=1+z*z/n
  center=(p+z*z/(2*n))/d;radius=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
  result.update(successes=int(x.sum()),wilson95=[float(center-radius),float(center+radius)])
 return result
def csvwrite(name,rows):
 with (OUT/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 summary={'units':'Target log-likelihood contrasts: nats/token. Choice probabilities normalized over A/B/C only; not clinical confidence.','models':{}}
 allchoices=[];allladder=[];allgen=[]
 protocol=json.loads((OUT/'protocol.json').read_text())
 for model in ['huatuo','medsight']:
  rows=[json.loads(s) for s in (OUT/f'{model}.jsonl').read_text().splitlines()]
  assert len({r['key'] for r in rows})==len(rows)
  expected=protocol['expected_per_model'];counts=Counter(r['task'] for r in rows)
  assert counts==dict(ladder=expected['ladder'],fact=expected['fact_choices'],anatomy=expected['anatomy_choices'],features=expected['features'],generation=expected['generation']),counts
  m={'counts':dict(counts)};summary['models'][model]=m
  ladder={(r['id'],r['context'],r['image']):r for r in rows if r['task']=='ladder'}
  ids=sorted({k[0] for k in ladder})
  for rid in ids:
   assert len({ladder[k]['n_tokens'] for k in ladder if k[0]==rid})==1
   assert len({ladder[k]['fact_tokens'] for k in ladder if k[0]==rid})==1
  contrasts={
   'original_minus_visual_only':('original','correct','visual_only','correct'),
   'positive_paraphrase_minus_visual_only':('paraphrase_positive','correct','visual_only','correct'),
   'positive_minus_negative_paraphrase':('paraphrase_positive','correct','paraphrase_negative','correct'),
   'original_caption_gain':('original','correct','none','correct'),
   'visual_context_gain':('visual_only','correct','none','correct'),
   'image_identity_without_context':('none','correct','none','swap'),
   'image_identity_with_positive_context':('paraphrase_positive','correct','paraphrase_positive','swap'),
   'image_gain_without_context':('none','correct','none','none'),
   'image_gain_with_positive_context':('paraphrase_positive','correct','paraphrase_positive','none')}
  m['ladder']={}
  for metric in ['mean_logp','fact_logp']:
   m['ladder'][metric]={}
   for name,(ca,ia,cb,ib) in contrasts.items():
    vals=[ladder[rid,ca,ia][metric]-ladder[rid,cb,ib][metric] for rid in ids]
    m['ladder'][metric][name]=stat(vals)
    allladder.extend(dict(model=model,id=rid,metric=metric,contrast=name,value=v) for rid,v in zip(ids,vals))
  # Aggregate across the six label-order permutations first.
  groups=defaultdict(list)
  for r in rows:
   if r['task'] in ['fact','anatomy']:groups[r['task'],r['id'],r['context'],r['image'],r['policy']].append(r)
   if r['task']=='generation':allgen.append(r)
  aggregate=[]
  for key,rs in groups.items():
   assert len(rs)==6 and len({r['order_index'] for r in rs})==6
   names=list(rs[0]['option_logp']);probs=[];logps=[]
   for r in rs:
    lp=np.array([r['option_logp'][n] for n in names]);assert np.isfinite(lp).all()
    pp=np.exp(lp-lp.max());pp/=pp.sum();probs.append(pp);logps.append(lp)
   pp=np.mean(probs,axis=0);pred=names[np.argmax(pp)];exp=rs[0]['expected']
   task,rid,ctx,im,policy=key
   row=dict(model=model,task=task,id=rid,context=ctx,image=im,policy=policy,expected=exp,prediction=pred,correct=int(pred==exp),expected_probability=float(pp[names.index(exp)]),unknown_probability=float(pp[names.index('unknown')]),order_accuracy=float(np.mean([r['prediction']==exp for r in rs])),order_agreement=float(np.mean([r['prediction']==pred for r in rs])),option_mass=float(np.mean([r['option_mass'] for r in rs])),probabilities=dict(zip(names,map(float,pp))))
   if task=='anatomy':row.update(label=rs[0]['label'],pair=rs[0]['pair'])
   aggregate.append(row)
  allchoices.extend(aggregate)
  lookup={(r['task'],r['id'],r['context'],r['image'],r['policy']):r for r in aggregate}
  m['choice_groups']=[]
  for task in ['fact','anatomy']:
   for ctx,im,pol in sorted({(r['context'],r['image'],r['policy']) for r in aggregate if r['task']==task}):
    rs=[r for r in aggregate if (r['task'],r['context'],r['image'],r['policy'])==(task,ctx,im,pol)]
    m['choice_groups'].append(dict(task=task,context=ctx,image=im,policy=pol,accuracy=stat([r['correct'] for r in rs]),expected_probability=stat([r['expected_probability'] for r in rs]),order_accuracy=stat([r['order_accuracy'] for r in rs]),mean_option_mass=float(np.mean([r['option_mass'] for r in rs]))))
  m['fact_joint_context_update']=[]
  for im,pol in itertools.product(['correct','swap','none'],['baseline','provenance']):
   good=[lookup['fact',rid,'positive',im,pol]['correct']*lookup['fact',rid,'negative',im,pol]['correct'] for rid in ids]
   m['fact_joint_context_update'].append(dict(image=im,policy=pol,accuracy=stat(good)))
  m['policy_effects']=[]
  for task in ['fact','anatomy']:
   taskids=sorted({r['id'] for r in aggregate if r['task']==task})
   contexts=['missing','positive','negative'] if task=='fact' else ['neutral','congruent','conflicting']
   for ctx in contexts:
    vals=[];acc=[]
    for rid in taskids:
     if ctx in ['congruent','conflicting']:
      label=lookup[task,rid,'neutral','correct','baseline']['label'];context=label if ctx=='congruent' else ('head' if label=='chest' else 'chest')
     else:context=ctx
     a=lookup[task,rid,context,'correct','baseline'];b=lookup[task,rid,context,'correct','provenance']
     vals.append(b['expected_probability']-a['expected_probability']);acc.append(b['correct']-a['correct'])
    m['policy_effects'].append(dict(task=task,context=ctx,probability_difference=stat(vals),accuracy_difference=stat(acc)))
  m['anatomy_regimes']=[]
  for pol,regime in itertools.product(['baseline','provenance'],['neutral','congruent','conflicting']):
   rs=[]
   for r in aggregate:
    if r['task']!='anatomy' or r['image']!='correct' or r['policy']!=pol:continue
    select=r['context']=='neutral' if regime=='neutral' else (r['context']==r['label'] if regime=='congruent' else r['context'] not in ['neutral',r['label']])
    if select:rs.append(r)
   m['anatomy_regimes'].append(dict(policy=pol,regime=regime,accuracy=stat([r['correct'] for r in rs]),probability=stat([r['expected_probability'] for r in rs]),order_accuracy=stat([r['order_accuracy'] for r in rs])))
  m['anatomy_pairs']=[]
  for pol,ctx in itertools.product(['baseline','provenance'],['neutral','head','chest']):
   rs=[r for r in aggregate if r['task']=='anatomy' and r['context']==ctx and r['policy']==pol and r['image']=='correct']
   pairvals=[]
   for pair in range(6):
    pairrows=[r for r in rs if r['pair']==pair];assert len(pairrows)==2
    pairvals.append(int(all(r['correct'] for r in pairrows)))
   m['anatomy_pairs'].append(dict(context=ctx,policy=pol,joint_accuracy=stat(pairvals)))
  m['features']=[]
  fs=[r for r in rows if r['task']=='features'];assert len(fs)==12
  y=np.array([r['label']=='head' for r in fs]);assert sum(y)==6
  archives=[np.load(OUT/r['file']) for r in fs];assert all(a.files==archives[0].files for a in archives)
  for name in archives[0].files:
   x=np.stack([a[name] for a in archives]);assert np.isfinite(x).all()
   x/=np.linalg.norm(x,axis=1,keepdims=True).clip(1e-10)
   sim=x@x.T;np.fill_diagonal(sim,-np.inf);nn=sim.argmax(1)
   good=y[nn]==y;acc=good.mean();null=[]
   for comb in itertools.combinations(range(12),6):
    yp=np.zeros(12,dtype=bool);yp[list(comb)]=True;null.append(float((yp[nn]==yp).mean()))
   m['features'].append(dict(layer=name,accuracy=float(acc),correct=int(sum(good)),n=12,permutation_p=float(np.mean(np.array(null)>=acc-1e-10)),dimension=x.shape[1],neighbors=[dict(id=fs[i]['id'],neighbor=fs[nn[i]]['id'],correct=bool(good[i])) for i in range(12)]))
  m['generation_truncated']=sum(r['truncated'] for r in allgen if r['model']==model)
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
 (OUT/'choice_records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in allchoices))
 csvwrite('ladder_contrasts.csv',allladder)
 (OUT/'generations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in allgen))
 plot(summary)
 print(json.dumps(summary,indent=2))

def plot(s):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,3,figsize=(15,4.4));colors=['#2166ac','#b35806']
 for j,(name,m) in enumerate(s['models'].items()):
  metric=m['ladder']['fact_logp'];keys=['image_identity_without_context','positive_paraphrase_minus_visual_only','positive_minus_negative_paraphrase']
  for i,k in enumerate(keys):
   v=metric[k];ci=v['ci95'];axes[0].errorbar(i+(j-.5)*.17,v['mean'],yerr=[[v['mean']-ci[0]],[ci[1]-v['mean']]],fmt='o',color=colors[j],capsize=4,label=name if i==0 else None)
  for i,pol in enumerate(['baseline','provenance']):
   v=next(r for r in m['choice_groups'] if (r['task'],r['context'],r['image'],r['policy'])==('fact','missing','correct',pol))
   axes[1].bar(i+(j-.5)*.32,v['accuracy']['mean'],width=.3,color=colors[j],label=name if i==0 else None)
  vals=[next(r for r in m['anatomy_regimes'] if r['policy']=='baseline' and r['regime']==reg)['accuracy']['mean'] for reg in ['neutral','congruent','conflicting']]
  axes[2].plot(range(3),vals,'o-',color=colors[j],label=name)
 axes[0].axhline(0,color='gray',lw=.8);axes[0].set_xticks(range(3),['Image identity','Fact added\n(paraphrase)','Fact polarity\n(+ vs −)']);axes[0].set_ylabel('Fact-span Δ log likelihood (nats/token)');axes[0].set_title('A. Source availability: 12 claims')
 axes[1].set_xticks(range(2),['Baseline','Provenance prompt']);axes[1].set_title('B. Missing-fact abstention');axes[1].set_ylabel('Order-averaged choice accuracy');axes[1].set_ylim(0,1.08)
 axes[2].set_xticks(range(3),['No note','Matching note','Conflicting note']);axes[2].set_ylim(0,1.08);axes[2].set_ylabel('Anatomy choice accuracy');axes[2].set_title('C. Visual control: 12 CT figures')
 for a in axes:a.spines[['top','right']].set_visible(False);a.legend(frameon=False,fontsize=8)
 fig.suptitle('Exploratory frozen-model pilot: provenance, abstention and visual use',fontsize=13)
 fig.tight_layout();fig.savefig(OUT/'followup_results.png',dpi=180);fig.savefig(OUT/'followup_results.pdf');plt.close(fig)
if __name__=='__main__':main()
