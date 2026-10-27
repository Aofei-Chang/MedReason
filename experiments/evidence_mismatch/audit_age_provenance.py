"""Exact numeric age transfer from source caption to supervised answer.

An interpretable source-provenance screen, not a pixel-adjudicated mismatch label.
"""
import collections
import json
import random
import re
from pathlib import Path

ROOT=Path('/data/aofei/Medical_datasets/PubMedVision')
OUT=Path(__file__).resolve().parent/'results'
AGE=re.compile(r'\b(\d{1,3})[ -]years?[ -]old\b|\baged?\s+(\d{1,3})\b',re.I)
def ages(s): return {int(a or b) for a,b in AGE.findall(s)}

def main():
    captions={r['id'].rsplit('_',1)[-1]:r['Original_Caption'] for r in json.load(open(ROOT/'PubMedVision_Original_Caption.json'))}
    sub={r['id'] for r in json.load(open(ROOT/'60k_PubMedVision_InstructionTuning_VQA.json'))}
    counts={s:collections.Counter() for s in ['full','medsight_60k']}
    rows=[]
    for r in json.load(open(ROOT/'PubMedVision_InstructionTuning_VQA.json')):
        q,a=[c['value'] for c in r['conversations'][:2]]
        aq,aa=ages(q),ages(a)
        if not aa: continue
        caption=captions.get(r['id'].rsplit('_',1)[-1],'')
        ac=ages(caption)
        shared=(aa&ac)-aq
        missing=aa-aq
        for group in ['full']+(['medsight_60k'] if r['id'] in sub else []):
            c=counts[group]
            c['answer_numeric_age']+=1
            c['answer_age_absent_question']+=bool(missing)
            c['source_shared_age_absent_question']+=bool(shared)
            c['absent_question_not_found_in_caption']+=bool(missing-ac)
        if shared:
            rows.append(dict(id=r['id'],subset='medsight_60k' if r['id'] in sub else 'full_only',
                question=q,answer=a,caption=caption,shared_ages=sorted(shared),image=r['image'],
                modality=r['modality'],body_part=r['body_part']))
    random.Random(20261018).shuffle(rows)
    result=dict(pattern=AGE.pattern,counts=counts,
                caveat='Numeric exact-match provenance only. Does not check image text, approximate-age questions, hypothetical examples, or whether the same age refers to the patient.',examples=rows[:40])
    (OUT/'age_provenance.json').write_text(json.dumps(result,indent=2))
    (OUT/'age_provenance_all.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    print(json.dumps(counts,indent=2))

if __name__=='__main__':main()
