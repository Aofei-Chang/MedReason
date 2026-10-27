"""Read-only dataset audit. Run from anywhere; writes only beside this script.

Requires ijson, Pillow. Counts are lexical screens, never clinical error labels.
No model inference is run. Source data and existing experiments are not modified.
"""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import json
import random
import re
import ijson

OUT = Path(__file__).resolve().parent
MED = Path('/data/aofei/Medical_datasets')
BENCH = Path('/data/aofei/hallucination')
SEED = 20261019
PATTERNS = {
    'exact_age': r'\b(?:\d{1,3}[ -]years?[ -]old|aged?\s+\d{1,3})\b',
    'external_test_event': r'\b(?:biopsy|histopatholog\w*|immunohistochem\w*|genetic testing|laboratory (?:test|result)\w*)\b',
    'longitudinal': r'\b(?:compared (?:to|with)|previous (?:scan|study|radiograph|exam\w*)|prior (?:scan|study|radiograph|exam\w*)|follow[ -]up|unchanged|interval (?:change|increase|decrease)|stable)\b',
    'clinical_history': r'\b(?:history of|presented with|patient (?:had|underwent)|after (?:surgery|treatment)|chemotherapy)\b',
}
RX = {k: re.compile(v, re.I) for k,v in PATTERNS.items()}
STRONG_TEMPORAL = re.compile(r'\b(?:compared (?:to|with)|in comparison with|previous (?:radiograph|exam\w*|study|image)|prior (?:radiograph|exam\w*|study|image))\b',re.I)

def stream(path):
    with open(path,'rb') as f:
        yield from ijson.items(f,'item')

def qa(record):
    conv=record.get('conversations',[])
    return ('\n'.join(x.get('value','') for x in conv if x.get('from') in ['human','user']),
            '\n'.join(x.get('value','') for x in conv if x.get('from') in ['gpt','assistant']))

def image_keys(record):
    v=record.get('image',[])
    if isinstance(v,str):v=[v]
    return [Path(x).name for x in v]

def reservoir_add(pool,record,n,limit,rng):
    if len(pool)<limit:pool.append(record)
    else:
        j=rng.randrange(n)
        if j<limit:pool[j]=record

def fingerprint(path):
    p=Path(path)
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return {'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()}

def main():
    result={'seed':SEED,'interpretation':'Lexical screens of local copies; no clinical sufficiency or hallucination prevalence estimates.',
            'patterns':PATTERNS,'inputs':[],'datasets':{}}
    candidates=[]
    path=MED/'PubMedVision/PubMedVision_InstructionTuning_VQA.json'
    counts=Counter(); pools=defaultdict(list); seen=Counter(); age_records=[]; sample=[]
    rng=random.Random(SEED); strata_rng={k:random.Random(SEED+i) for i,k in enumerate(RX)}
    selected_ids={'Instruction-Tuning_1','Instruction-Tuning_394494','Instruction-Tuning_202483'}
    prior=Path(__file__).resolve().parents[2]/'experiments/information_gap/results/facts.jsonl'
    if prior.exists():selected_ids.update(json.loads(s)['id'] for s in prior.read_text().splitlines())
    selected=[]
    for n,r in enumerate(stream(path),1):
        q,a=qa(r)
        reservoir_add(sample,r,n,4000,rng)
        if r['id'] in selected_ids:selected.append(r)
        for k,rx in RX.items():
            if rx.search(a):
                counts[k]+=1;seen[k]+=1
                reservoir_add(pools[k],r,seen[k],30,strata_rng[k])
        if RX['exact_age'].search(a):
            age_records.append(r)
            if not RX['exact_age'].search(q):counts['answer_age_no_age_pattern_in_question']+=1
    result['datasets']['PubMedVision_instruction']={'n_records':n,'answer_screen_counts':dict(counts),'source':str(path),'random_sample_n':len(sample)}
    result['inputs'].append(fingerprint(path))
    keep={r['id']:r for r in sample+age_records+selected+sum(pools.values(),[])}
    needed={x for r in keep.values() for x in image_keys(r)}
    cap_path=MED/'PubMedVision/PubMedVision_Original_Caption.json'
    captions={}
    for cap_n,r in enumerate(stream(cap_path),1):
        for k in image_keys(r):
            if k in needed:captions[k]=r.get('Original_Caption','')
    result['datasets']['PubMedVision_original_captions']={'n_records':cap_n,'source':str(cap_path)}
    result['inputs'].append(fingerprint(cap_path))
    def enrich(r,origin):
        q,a=qa(r);keys=image_keys(r)
        imgs=[str(MED/'PubMedVision/images_all'/k) for k in keys]
        return {'dataset':'PubMedVision','id':r['id'],'question':q,'answer':a,'caption':'\n'.join(captions.get(k,'') for k in keys),
                'image_paths':imgs,'selection':origin,'source':str(path),'screen_flags':[k for k,rx in RX.items() if rx.search(a)]}
    for k,pool in pools.items():candidates.extend(enrich(r,'reservoir within lexical stratum: '+k) for r in pool)
    candidates.extend(enrich(r,'pre-existing pilot case or first record') for r in selected)
    # Replicate exact numeric-age provenance screening; not a validated mismatch rate.
    age_rx=re.compile(r'\b(\d{1,3})[ -]years?[ -]old\b|\baged?\s+(\d{1,3})\b',re.I)
    ages=lambda t:{int(a or b) for a,b in age_rx.findall(t)}
    age_counts=Counter()
    for r in age_records:
        q,a=qa(r);aa=ages(a);qq=ages(q);cc=ages(' '.join(captions.get(k,'') for k in image_keys(r)))
        if aa:age_counts['answer_numeric_age']+=1
        if aa-qq:age_counts['answer_age_absent_question']+=1
        if (aa-qq)&cc:age_counts['source_shared_age_absent_question']+=1
        if (aa-qq)-cc:age_counts['absent_question_not_found_in_caption']+=1
    result['datasets']['PubMedVision_instruction']['numeric_age_provenance']=dict(age_counts)
    result['datasets']['PubMedVision_instruction']['age_caveat']='Exact numeric matches only. Image OCR, hypotheticals, patient identity, and clinical derivability are not verified by this count.'
    with (OUT/'pubmedvision_random_sample.jsonl').open('w') as f:
        for r in sample:f.write(json.dumps(enrich(r,'uniform reservoir sample'),ensure_ascii=False)+'\n')

    # Reports: local task packaging is the object of this audit, not all of MIMIC-CXR/IU.
    for dataset in ['MIMIC_CXR','IU_Xray']:
        for split in ['training','test']:
            p=BENCH/dataset/'data_report'/f'{split}.json'
            if not p.exists():continue
            records=json.loads(p.read_text()); c=Counter(); examples=[]
            for i,r in enumerate(records):
                q,a=qa(r); imgs=image_keys(r)
                if len(imgs)==1:c['one_image_record']+=1
                for k,rx in RX.items():
                    if rx.search(a):c[k]+=1
                if STRONG_TEMPORAL.search(a):
                    c['explicit_comparison_reference']+=1
                    if len(examples)<8:
                        raw_img=r.get('image',[]);raw_img=raw_img if isinstance(raw_img,list) else [raw_img]
                        examples.append({'dataset':dataset,'split':split,'id':str(r.get('id',i)),'row_index':i,'question':q,'answer':a,'image_relative':raw_img,'source':str(p),'selection':'first explicit comparison hits; illustrative only','screen_flags':['explicit_comparison_reference']})
            result['datasets'][dataset+'_'+split]={'n_records':len(records),'counts':dict(c),'source':str(p),'unit':'record, not deduplicated patient'}
            result['inputs'].append(fingerprint(p));candidates.extend(examples)
    # SLAKE: base_type=KG questions are deliberately knowledge-based, not necessarily failures.
    for split in ['train','validate','test']:
        p=BENCH/'Slake'/f'{split}.json';records=json.loads(p.read_text());en=[r for r in records if r.get('q_lang')=='en']
        result['datasets']['SLAKE_'+split]={'n_records_all_languages':len(records),'n_english':len(en),'english_base_types':dict(Counter(r.get('base_type') for r in en)),
            'english_content_types':dict(Counter(r.get('content_type') for r in en)),'unique_english_images':len({r['img_name'] for r in en}),'source':str(p)}
        result['inputs'].append(fingerprint(p))
        if split=='test':
            for typ in ['kvqa','vqa']:
                picked=[r for r in en if str(r.get('base_type')).lower()==typ.lower()][:12]
                for r in picked:candidates.append({'dataset':'SLAKE','id':str(r['qid']),'question':r['question'],'answer':r['answer'],'image_paths':[str(BENCH/'Slake/imgs'/r['img_name'])],
                    'base_type':r.get('base_type'),'content_type':r.get('content_type'),'selection':'first examples by base_type','source':str(p)})
    p=BENCH/'VQA_RAD/MED_RAD_test.json';records=json.loads(p.read_text())
    closed=[r for r in records if r.get('answer_type')=='CLOSED'];answer_counts=Counter(str(r['answer']).strip().lower() for r in closed)
    result['datasets']['VQA_RAD_local_file']={'n_records':len(records),'n_closed':len(closed),'closed_answer_counts':dict(answer_counts),'unique_images':len({r['image_name'] for r in records}),
         'source':str(p),'split_warning':'Filename says test, but 2248 rows resembles the complete original corpus; do not assume official held-out split.'}
    result['inputs'].append(fingerprint(p))
    for r in records:
        if any(x in r['question'].lower() for x in ['which lung','what modality','what imaging','what plane']):
            candidates.append({'dataset':'VQA_RAD','id':str(r['qid']),'question':r['question'],'answer':r['answer'],'image_paths':[str(BENCH/'VQA_RAD/images'/r['image_name'])],'source':str(p),'selection':'question structure examples'})
            if sum(x['dataset']=='VQA_RAD' for x in candidates)>=14:break
    # PMC-VQA: deterministic 4000-row reservoir from test.csv, screen only.
    p=MED/'PMC-VQA/test.csv';pool=[];rng=random.Random(SEED);cnt=Counter();examples=[]
    with p.open() as f:
        for n,r in enumerate(csv.DictReader(f),1):
            reservoir_add(pool,r,n,4000,rng)
            q=r.get('Question','');a=r.get('Answer','');s=q+' '+a
            for k,rx in RX.items():
                if rx.search(s):cnt[k]+=1
            if RX['external_test_event'].search(s) and len(examples)<20:
                examples.append({'dataset':'PMC_VQA','id':'test.csv:'+str(n),'question':q,'answer':a,'options':{k:v for k,v in r.items() if k.startswith('Choice')},'image_paths':[str(MED/'PMC-VQA/images'/r.get('Figure_path',''))],'source':str(p),'selection':'first external-test keyword hits'})
    result['datasets']['PMC_VQA_test']={'n_records':n,'question_answer_screen_counts':dict(cnt),'source':str(p)}
    result['inputs'].append(fingerprint(p));candidates.extend(examples)
    # Empty training directories: do not mistake repo evaluation JSONs for training data.
    root=MED/'llava-med/data'
    result['datasets']['LLaVA_Med_local']={'data_instruct_files':sum(p.is_file() for p in (root/'instruct').rglob('*')),
        'data_pmc_articles_files':sum(p.is_file() for p in (root/'pmc_articles').rglob('*')),
        'note':'Code repository exists; these expected training-data directories are empty. No local training QA audit is claimed.'}
    unique={(r['dataset'],r.get('split',''),r['id']):r for r in candidates}
    (OUT/'candidate_examples.json').write_text(json.dumps(list(unique.values()),indent=2,ensure_ascii=False))
    (OUT/'dataset_audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps(result['datasets'],indent=2))

if __name__=='__main__':main()
