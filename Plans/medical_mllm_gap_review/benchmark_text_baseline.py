"""Train-split-only lookup baselines, without pixels, pretrained LMs, or test labels.

SLAKE uses English entries in the named train/test JSON files. Exact match uses
lowercasing/space/punctuation normalization only. CIs cluster by test image.
PMC-VQA only uses the training answer-letter mode and evaluates that on test.
"""
from collections import Counter, defaultdict
from pathlib import Path
import csv
import json
import re
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=Path('/data/aofei/hallucination/Slake')

def norm(x):return re.sub(r'\s+',' ',re.sub(r'[^a-z0-9 ]',' ',str(x).lower())).strip()
def mode(xs):return sorted(Counter(xs).items(),key=lambda x:(-x[1],x[0]))[0][0]

def main():
    train=[r for r in json.loads((ROOT/'train.json').read_text()) if r['q_lang']=='en']
    test=[r for r in json.loads((ROOT/'test.json').read_text()) if r['q_lang']=='en']
    byq=defaultdict(list);bytype=defaultdict(list)
    for r in train:
        byq[norm(r['question'])].append(norm(r['answer']))
        bytype[r['content_type']].append(norm(r['answer']))
    qmode={k:mode(v) for k,v in byq.items()};tmode={k:mode(v) for k,v in bytype.items()}
    gmode=mode([norm(r['answer']) for r in train]);rows=[]
    for r in test:
        q=norm(r['question']);gold=norm(r['answer']);typ=r['content_type'];fallback=tmode.get(typ,gmode)
        rows.append({'qid':r['qid'],'image':r['img_name'],'question':r['question'],'gold':gold,'base_type':r.get('base_type'),
             'answer_type':r.get('answer_type'),'content_type':typ,'question_seen_in_train':q in qmode,
             'global_prediction':gmode,'type_prediction':fallback,'question_prediction':qmode.get(q,fallback),
             'question_only_prediction':qmode.get(q,gmode)})
    grouped=defaultdict(list)
    for r in rows:grouped[r['image']].append(r)
    keys=list(grouped);rng=np.random.default_rng(20261019);boot=rng.integers(0,len(keys),(2000,len(keys)))
    models={}
    for name in ['global','type','question','question_only']:
        ncorrect=sum(r[name+'_prediction']==r['gold'] for r in rows)
        hits=np.array([sum(r[name+'_prediction']==r['gold'] for r in grouped[k]) for k in keys]);ns=np.array([len(grouped[k]) for k in keys])
        b=hits[boot].sum(1)/ns[boot].sum(1)
        models[name]={'correct':ncorrect,'n':len(rows),'accuracy':ncorrect/len(rows),'image_cluster_bootstrap95':np.quantile(b,[.025,.975]).tolist()}
    strata={}
    for key in ['base_type','answer_type','content_type']:
        strata[key]={}
        for v in sorted({str(r[key]) for r in rows}):
            subset=[r for r in rows if str(r[key])==v]
            strata[key][v]={'n':len(subset),'question_lookup_correct':sum(r['question_prediction']==r['gold'] for r in subset)}
    seen=[r for r in rows if r['question_seen_in_train']]
    result={'seed':20261019,'SLAKE':{'train_english':len(train),'test_english':len(test),'test_unique_images':len(keys),
       'train_test_image_overlap':len({r['img_name'] for r in train}&set(keys)),'test_questions_seen_in_train':len(seen),
       'seen_question_correct':sum(r['question_prediction']==r['gold'] for r in seen),'models':models,'strata':strata,
       'definition':'question lookup uses exact normalized train question; fallback is content-type answer mode. Type metadata is supplied, so this is a metadata-aware text baseline, not a question-only language model.',
       'limits':'Repeated question templates and class frequency explain some success. Correct text-only predictions do not establish individual visual claims are unsupported. Existing local split may differ from published protocols.'}}
    root=Path('/data/aofei/Medical_datasets/PMC-VQA')
    counts={}
    for split in ['train','test']:
        with (root/f'{split}.csv').open() as f:counts[split]=Counter(r['Answer_label'].strip() for r in csv.DictReader(f))
    letter=sorted(counts['train'].items(),key=lambda x:(-x[1],x[0]))[0][0]
    result['PMC_VQA']={'train_letter_counts':dict(counts['train']),'test_letter_counts':dict(counts['test']),
        'train_mode_letter':letter,'blind_test_accuracy':counts['test'][letter]/sum(counts['test'].values()),
        'invalid_train_answer_labels':sum(v for k,v in counts['train'].items() if k not in 'ABCD'),
        'invalid_test_answer_labels':sum(v for k,v in counts['test'].items() if k not in 'ABCD'),
        'definition':'Training most frequent answer letter, no question or image. Four-choice nominal random baseline=25%. Raw-file denominator includes malformed labels, listed separately.'}
    (OUT/'text_baseline.json').write_text(json.dumps(result,indent=2))
    with (OUT/'text_baseline_predictions.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
