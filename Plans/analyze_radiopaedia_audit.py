"""Derive candidate pools and quality checks from the completed census."""
from pathlib import Path
import collections
import csv
import hashlib
import json
import re
import random
import statistics

OUT=Path(__file__).parent/'radiopaedia_audit'
ROOT=Path('/data/lingwei/radiopaedia_0826')

def norm(s): return re.sub(r'[^a-z0-9]+',' ',s.lower()).strip()

def main():
    rows=[json.loads(l) for l in (OUT/'metadata.jsonl').open()]
    cases=[r for r in rows if r['folder']=='cases' and not r['error']]
    articles=[r for r in rows if r['folder']=='articles' and not r['error']]
    title_index=collections.defaultdict(list)
    for r in articles:title_index[norm(r['title'])].append(r['rid'])
    edges=[]
    for r in cases:
        for title in r['related_titles']:
            ids=title_index.get(norm(title),[])
            if len(ids)==1: edges.append({'case_id':r['rid'],'article_id':ids[0],'link_title':title})
    with (OUT/'case_article_title_matches.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['case_id','article_id','link_title']);w.writeheader();w.writerows(edges)
    pools={}
    for name,pat in [('parotid',r'parotid'),('intracranial_lesions',r'brain abscess|cerebral abscess|glioblastoma|cerebral metast|brain metast|ring.enhancing'),('adrenal',r'adrenal'),('scaphoid',r'scaphoid')]:
        cc=[r for r in cases if re.search(pat,r['title'],re.I)]
        pools[name]={'title_matched':len(cc),'confirmation_mention':sum(r['confirmation_mention'] for r in cc),'uncertainty_mention':sum(r['uncertainty_mention'] for r in cc), 'both':sum(r['confirmation_mention'] and r['uncertainty_mention'] for r in cc)}
    candidates=[]
    for name,pat in [('parotid',r'parotid'),('intracranial_lesions',r'brain abscess|cerebral abscess|glioblastoma|cerebral metast|brain metast|ring.enhancing'),('adrenal',r'adrenal'),('scaphoid',r'scaphoid')]:
        cc=[r for r in cases if re.search(pat,r['title'],re.I)]
        for r in sorted(cc,key=lambda r:(-r['confirmation_mention'],-r['uncertainty_mention'],-r['text_chars'],r['rid']))[:25]:
            candidates.append({**{k:r[k] for k in ['rid','title','certainty','confirmation_mention','uncertainty_mention','followup_mention','text_chars']},'pool':name,'metadata_path':str(ROOT/'cases'/r['rid']/'data.json'),'status':'lexical candidate; not adjudicated'})
    with (OUT/'candidate_cases.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(candidates[0]));w.writeheader();w.writerows(candidates)
    sample=json.loads((OUT/'image_sample.json').read_text())
    checks=[]
    for s in sample:
        p=ROOT/s['folder']/s['rid']; vv=[]
        for path in sorted((p/'images').glob('*/viewer_info.json')):
            d=json.loads(path.read_text());vv.append({'viewer':path.parent.name,**d})
        nonempty=[v for v in vv if v.get('text','').strip()]
        repeated=len(nonempty)>1 and len(set(v['text'] for v in nonempty))<len(nonempty)
        across=any(a['text']==b['text'] and a.get('modality')!=b.get('modality') for i,a in enumerate(nonempty) for b in nonempty[i+1:])
        checks.append({'folder':s['folder'],'rid':s['rid'],'viewer_count':len(vv),'repeated_nonempty_viewer_text':repeated,'same_text_different_modality':across})
    (OUT/'viewer_text_checks.json').write_text(json.dumps(checks,indent=2))
    summary={'pools':pools,'article_unique_normalized_titles':sum(len(v)==1 for v in title_index.values()),'case_article_links':len(edges),'cases_with_resolved_title_link':len(set(r['case_id'] for r in edges)), 'total_related_titles':sum(len(r['related_titles']) for r in cases), 'lexical_confirmation_and_uncertainty':sum(r['confirmation_mention'] and r['uncertainty_mention'] for r in cases)}
    for sub in ['cases','cases_new']:
        cc=[r for r in checks if r['folder']==sub]
        summary[sub+'_viewer_checks']={k:sum(bool(r[k]) for r in cc) for k in ['repeated_nonempty_viewer_text','same_text_different_modality']}
        summary[sub+'_viewer_checks']['multiple_viewers']=sum(r['viewer_count']>1 for r in cc)
    # Audit identical filenames across re-downloads on a bounded sample, preserving both versions.
    compare=[]
    for s in [r for r in sample if r['folder']=='cases_new'][:32]:
        p=ROOT/'cases_new'/s['rid']; old=ROOT/'cases'/s['rid']
        fs=sorted(f for f in (p/'images').rglob('*') if f.suffix in ['.jpg','.png'])[:2]
        for f in fs:
            other=old/f.relative_to(p)
            compare.append({'rid':s['rid'],'relative_path':str(f.relative_to(p)),'exists_in_base':other.exists(),'identical_bytes':other.exists() and hashlib.sha256(f.read_bytes()).digest()==hashlib.sha256(other.read_bytes()).digest()})
    summary['new_vs_base_image_check']={'files_checked':len(compare),'exist_in_base':sum(r['exists_in_base'] for r in compare),'identical_bytes':sum(r['identical_bytes'] for r in compare)}
    (OUT/'new_vs_base_image_check.json').write_text(json.dumps(compare,indent=2))
    article_sample=random.Random(20261015).sample(articles, min(256,len(articles)))
    article_inventory=[{'rid':r['rid'],'entries':sorted(x.name for x in (ROOT/'articles'/r['rid']).iterdir())} for r in article_sample]
    (OUT/'article_asset_sample.json').write_text(json.dumps(article_inventory,indent=2))
    summary['article_asset_sample']={'n':len(article_inventory),'only_data_json':sum(r['entries']==['data.json'] for r in article_inventory)}
    (OUT/'analysis.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    selected=['198678','32560','45969','87566','33598','78687','194727','37511','100001']
    evidence=[]
    for rid in selected:
        p=ROOT/'cases'/rid;d=json.loads((p/'data.json').read_text())
        views=[]
        for f in sorted((p/'images').glob('*/viewer_info.json')):
            vi=json.loads(f.read_text());ims=sorted(x for x in f.parent.iterdir() if x.suffix.lower() in ['.jpg','.png','.mp4'])
            views.append({'viewer_id':f.parent.name,'path':str(f),**vi,'image_file_count':len(ims)})
        evidence.append({'rid':rid,'source_path':str(p/'data.json'),'title':d['title'],'certainty':d['certainty'],'presentation':d['presentation'],'discussion':d['discussion'],'qa_pairs':d['qa_pairs'],'viewers':views,'annotation_status':'source facts only; not clinician-adjudicated'})
    (OUT/'selected_case_evidence.json').write_text(json.dumps(evidence,indent=2,ensure_ascii=False))

if __name__=='__main__': main()
