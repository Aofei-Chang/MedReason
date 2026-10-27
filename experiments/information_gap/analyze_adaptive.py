"""Analyze the adaptive second prompt without replacing primary results."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np
from analyze import stat
OUT=Path(__file__).resolve().parent/'results';AD=OUT/'adaptive_prompt'
def main():
 base=[json.loads(l) for l in (OUT/'choice_records.jsonl').read_text().splitlines()]
 result={'adaptive':True,'models':{}};new=[];gens=[]
 for model in ['huatuo','medsight']:
  rows=[json.loads(l) for l in (AD/f'{model}.jsonl').read_text().splitlines()]
  assert len({r['key'] for r in rows})==len(rows)
  assert Counter(r['task'] for r in rows)==dict(fact=648,anatomy=432,generation=12)
  groups=defaultdict(list)
  for r in rows:
   if r['task']=='generation':gens.append(r);continue
   groups[r['task'],r['id'],r['context'],r['image'],r['policy']].append(r)
  agg=[]
  for (task,rid,ctx,im,pol),rs in groups.items():
   assert len(rs)==6 and len({r['order_index'] for r in rs})==6
   names=list(rs[0]['option_logp']);probs=[]
   for r in rs:
    lp=np.array([r['option_logp'][n] for n in names]);assert np.isfinite(lp).all()
    pp=np.exp(lp-lp.max());pp/=pp.sum();probs.append(pp)
   pp=np.mean(probs,0);pred=names[np.argmax(pp)];exp=rs[0]['expected']
   a=dict(model=model,task=task,id=rid,context=ctx,image=im,policy=pol,expected=exp,prediction=pred,correct=int(pred==exp),expected_probability=float(pp[names.index(exp)]),unknown_probability=float(pp[names.index('unknown')]),probabilities=dict(zip(names,map(float,pp))),order_accuracy=float(np.mean([r['prediction']==exp for r in rs])),option_mass=float(np.mean([r['option_mass'] for r in rs])))
   if task=='anatomy':a.update(label=rs[0]['label'],pair=rs[0]['pair'])
   agg.append(a)
  new.extend(agg)
  allrows=agg+[r for r in base if r['model']==model]
  m={'fact':[],'anatomy':[],'fact_joint':[],'anatomy_pairs':[],'paired_effects':[],'no_image':[]};result['models'][model]=m
  for pol in ['baseline','provenance','evidence_first']:
   for ctx in ['missing','positive','negative']:
    rs=[r for r in allrows if (r['task'],r['context'],r['image'],r['policy'])==('fact',ctx,'correct',pol)]
    m['fact'].append(dict(policy=pol,context=ctx,accuracy=stat([r['correct'] for r in rs]),unknown_count=sum(r['prediction']=='unknown' for r in rs),yes_count=sum(r['prediction']=='yes' for r in rs),no_count=sum(r['prediction']=='no' for r in rs),order_accuracy=float(np.mean([r['order_accuracy'] for r in rs]))))
   pos={r['id']:r['correct'] for r in allrows if (r['task'],r['context'],r['image'],r['policy'])==('fact','positive','correct',pol)}
   neg={r['id']:r['correct'] for r in allrows if (r['task'],r['context'],r['image'],r['policy'])==('fact','negative','correct',pol)}
   m['fact_joint'].append(dict(policy=pol,accuracy=stat([pos[k]*neg[k] for k in pos])))
   for regime in ['neutral','congruent','conflicting']:
    rs=[]
    for r in allrows:
     if (r['task'],r['image'],r['policy'])!=('anatomy','correct',pol):continue
     select=r['context']=='neutral' if regime=='neutral' else (r['context']==r['label'] if regime=='congruent' else r['context'] not in ['neutral',r['label']])
     if select:rs.append(r)
    m['anatomy'].append(dict(policy=pol,regime=regime,accuracy=stat([r['correct'] for r in rs]),unknown_count=sum(r['prediction']=='unknown' for r in rs),order_accuracy=float(np.mean([r['order_accuracy'] for r in rs]))))
   for ctx in ['neutral','head','chest']:
    rs=[r for r in allrows if (r['task'],r['context'],r['image'],r['policy'])==('anatomy',ctx,'correct',pol)]
    successes=[all(r['correct'] for r in rs if r['pair']==p) for p in range(6)]
    m['anatomy_pairs'].append(dict(policy=pol,context=ctx,accuracy=stat(successes)))
    rs=[r for r in allrows if (r['task'],r['context'],r['image'],r['policy'])==('anatomy',ctx,'none',pol)]
    # These 12 records repeat exactly the same no-image prompt. They are
    # a single control input, not 12 independent anatomy trials.
    predictions={r['prediction'] for r in rs};assert len(predictions)==1
    m['no_image'].append(dict(policy=pol,context=ctx,prediction=next(iter(predictions)),expected='unknown',distinct_inputs=1,repeated_records=len(rs)))
  # Paired records for new prompt relative to the unchanged primary baseline.
  for ctx in ['missing','positive','negative']:
   rs=[r for r in allrows if r['task']=='fact' and r['image']=='correct' and r['context']==ctx]
   a={r['id']:r for r in rs if r['policy']=='baseline'};b={r['id']:r for r in rs if r['policy']=='evidence_first'}
   m['paired_effects'].append(dict(task='fact',context=ctx,accuracy_difference=stat([b[k]['correct']-a[k]['correct'] for k in a]),probability_difference=stat([b[k]['expected_probability']-a[k]['expected_probability'] for k in a])))
 (AD/'summary.json').write_text(json.dumps(result,indent=2))
 (AD/'choice_records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in new))
 (AD/'generations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in gens))
 plot(result)
 print(json.dumps(result,indent=2))
def plot(s):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,3,figsize=(14,4.3))
 for j,(model,m) in enumerate(s['models'].items()):
  for i,(task,field,ctx,title) in enumerate([
   ('fact','context','missing','Missing fact: choose unknown'),
   ('fact','context','negative','Explicit negative fact: choose no'),
   ('anatomy','regime','conflicting','Visible anatomy: override wrong note')]):
   vals=[next(r for r in m[task] if r['policy']==p and r[field]==ctx)['accuracy']['mean'] for p in ['baseline','provenance','evidence_first']]
   axes[i].plot(range(3),vals,marker=['o','s'][j],linestyle=['-','--'][j],color=['#2166ac','#b35806'][j],label=model)
   axes[i].set_title(title,fontsize=11);axes[i].set_ylim(-.03,1.06)
   axes[i].set_xticks(range(3),['Baseline','Caution\nprompt','Evidence-first\n(adaptive)'])
   axes[i].set_ylabel('Correct aggregate choice / 12');axes[i].legend(frameon=False,fontsize=9)
   axes[i].spines[['top','right']].set_visible(False)
 fig.suptitle('More abstention does not imply better evidence use',fontsize=14)
 fig.tight_layout();fig.savefig(OUT/'prompt_tradeoffs.png',dpi=180);fig.savefig(OUT/'prompt_tradeoffs.pdf');plt.close(fig)
if __name__=='__main__':main()
