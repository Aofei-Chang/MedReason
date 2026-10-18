"""Build review-only information-state manifests; no generated clinical gold labels.

Full source frames are listed as candidates, not asserted to be sufficient or
affordable for a particular model. Reviewer must select and validate model inputs.
"""
from pathlib import Path
import hashlib
import json
import re

ROOT=Path('/data/lingwei/radiopaedia_0826/cases')
OUT=Path(__file__).parent/'radiopaedia_audit'

SPECS={
 '198678': {'phenomenon':'Specific histology exceeds the explicitly described imaging differential',
  'article_ids':['9149','1893'],
  'states':[
   ('images','All saved MRI series; no report or final diagnosis', ['0'], None),
   ('history','Same MRI plus presenting symptoms', ['0'], 'presentation'),
   ('confirmation','Same evidence plus the source statement of excision-histology confirmation', ['0'], 'confirmation_excerpt')],
  'confirmation':'An acinic cell carcinoma of the left superficial and deep parotid gland is definitively confirmed on excision histology in this case.',
  'review_questions':['Which findings can be localized in the selected frames?', 'Which differential and specificity are defensible before histology?', 'Does revealing histology change justified specificity while leaving image findings unchanged?'],
  'synthetic_bad_claim':'The MRI definitively establishes acinic cell carcinoma histology.',
  'contrast_note':'Candidate overclaim based on the source-described broad differential; independent clinician annotation required.'},
 '32560': {'phenomenon':'More imaging can refine characterization without resolving tumor histology',
  'article_ids':['9149','1893'],
  'states':[
   ('ct','CT and presenting symptoms', ['0'], 'presentation'),
   ('ct_mri','CT plus MRI and same history', ['0','1'], 'presentation'),
   ('confirmation','Same evidence plus source core-biopsy diagnosis', ['0','1'], 'confirmation_excerpt')],
  'confirmation':'On the core biopsy, a diagnosis of acinic cell carcinoma was made, a rare diagnosis.',
  'review_questions':['Does CT-to-MRI improve characterization without making the histologic label certain?', 'Keep ultrasound biopsy-procedure images out of pre-biopsy inputs unless the task specifically includes them.'],
  'synthetic_bad_claim':'Additional MRI proves the exact histologic subtype without tissue confirmation.',
  'contrast_note':'This is a proposed test statement, not a recorded model output.'},
 '194727': {'phenomenon':'Report information may come from an undisplayed sequence',
  'article_ids':[],
  'states':[
   ('displayed','All saved MRI series and history, no viewer description', ['0'], 'presentation'),
   ('report','Same evidence plus the full source viewer report, explicitly labeled as a report', ['0'], 'viewer_report'),
   ('confirmation','Same evidence plus pathology statement', ['0'], 'confirmation_excerpt')],
  'confirmation':'Pathology showed a large adrenal hematoma with microscopic myelolipoma and negative for carcinoma.',
  'review_questions':['Verify series identities, including which contrast series is subtraction.', 'Does the image-only response invent an opposed-phase finding?', 'When the report is supplied, does the response attribute that finding to the report rather than to displayed pixels?'],
  'synthetic_bad_claim':'The displayed opposed-phase images demonstrate no signal dropout.',
  'contrast_note':'Source viewer text says opposed-phase sequences are not shown. Check saved images independently before finalizing this label.'},
 '87566': {'phenomenon':'Sequence availability should constrain diffusion claims',
  'article_ids':['6679','25074'],
  'states':[
   ('without_diffusion','MRI excluding source series indices 4 and 5; no annotated viewer', ['0'], 'presentation'),
   ('with_diffusion','Same MRI with DWI and ADC restored; no annotated viewer', ['0'], 'presentation')],
  'confirmation':None,
  'review_questions':['Validate DWI/ADC mapping against pixels, not filenames alone.', 'Other retained sequences may already support abscess: do not force a diagnostic flip.', 'Distinguish reported imaging impression from patient-specific microbiological or tissue confirmation.'],
  'synthetic_bad_claim':'Restricted diffusion is directly demonstrated on the provided DWI and ADC images.',
  'contrast_note':'False availability claim in the withheld state, subject to verification that no retained montage embeds diffusion images.'},
 '33598': {'phenomenon':'Atypical morphology and later microbiology limit image-only etiologic specificity',
  'article_ids':['6679'],
  'states':[
   ('images','Saved MRI and presenting history', ['0'], 'presentation'),
   ('confirmation','Same evidence with the source microbiology result', ['0'], 'confirmation_excerpt')],
  'confirmation':"Unfortunately, the microbiological test performed a few hours prior to the patient's death revealed that the causative organism was Mycobacterium tuberculosis.",
  'review_questions':['What etiologic specificity is justified before microbiology?', 'The source discussion contains ambiguous wording; adjudicate it rather than copying it as a reasoning trace.'],
  'synthetic_bad_claim':'MRI alone identifies Mycobacterium tuberculosis as the organism.',
  'contrast_note':'Proposed over-specific explanation, not a model result.'},
 '37511': {'phenomenon':'Adequately assessable normal images are a control against indiscriminate abstention',
  'article_ids':[],
  'states':[('complete','All three saved thumb views and history', ['0'], 'presentation')],
  'confirmation':None,
  'review_questions':['Validate the normal-image target at the task-specific finding level.', 'Avoid universal claims that every possible pathology is excluded.'],
  'synthetic_bad_claim':None,
  'contrast_note':'Source says normal thumb series / NAD; this still requires independent review.'},
}

def main():
    records=[]
    for rid,spec in SPECS.items():
        p=ROOT/rid;data=json.loads((p/'data.json').read_text());inventory=[]
        for vf in sorted((p/'images').glob('*/viewer_info.json')):
            v=json.loads(vf.read_text());seen={}
            for f in sorted(vf.parent.iterdir()):
                if f.suffix.lower() not in ['.jpg','.jpeg','.png']:continue
                m=re.search(r'_s(\d+)_(\d+)\.',f.name)
                digest=hashlib.sha256(f.read_bytes()).hexdigest()
                key=(int(m[1]) if m else None,digest)
                inventory.append({'path':str(f),'viewer_id':vf.parent.name,'series_index':int(m[1]) if m else None,'frame_index':int(m[2]) if m else None,'series_title':v.get('titles',[])[int(m[1])] if m and int(m[1])<len(v.get('titles',[])) else None,'sha256':digest,'exact_duplicate_of':seen.get(key)})
                seen.setdefault(key,str(f))
        states=[]
        for sid,description,views,text in spec['states']:
            selected=[i for i in inventory if i['viewer_id'] in views and not i['exact_duplicate_of']]
            if rid=='87566' and sid=='without_diffusion':selected=[i for i in selected if i['series_index'] not in [4,5]]
            states.append({'state_id':sid,'description':description,'candidate_images':[i['path'] for i in selected],
                           'include_presentation':text in ['presentation','confirmation_excerpt','viewer_report'],
                           'include_viewer_report':text=='viewer_report','include_confirmation_excerpt':text=='confirmation_excerpt',
                           'clinician_selected_images':None,'gold_claims':None,'gold_differential':None,'gold_specificity':None,'gold_evidence_pointers':None})
        if spec['confirmation'] is not None:assert spec['confirmation'] in data['discussion']
        records.append({'case_id':rid,'split_group':rid,'status':'DRAFT_FOR_REVIEW_NOT_TRAINING_GOLD',
          'phenomenon':spec['phenomenon'],'source_path':str(p/'data.json'),
          'source_sha256':hashlib.sha256((p/'data.json').read_bytes()).hexdigest(),
          'reviewer_only_title':data['title'],'reviewer_only_article_ids':spec['article_ids'],
          'source_confirmation_excerpt':spec['confirmation'],'source_certainty':data['certainty'],
          'inventory':inventory,'states':states,'review_questions':spec['review_questions'],
          'synthetic_bad_claim':spec['synthetic_bad_claim'],'contrast_note':spec['contrast_note'],
          'default_hidden_fields':['title','certainty','citation_info','discussion','qa_pairs','related-article','article_feature_image','public_playlist','viewer_text','series_titles_with_diagnoses'],
          'input_policy':'Source paths, IDs, titles, answer-derived links and reviewer-only fields must not be sent as model prompt text. Use neutral IDs. Text availability is explicit per state.'})
    (OUT/'pilot_state_drafts.json').write_text(json.dumps(records,indent=2,ensure_ascii=False))
    print([(r['case_id'],len(r['inventory']),sum(i['exact_duplicate_of'] is None for i in r['inventory']),len(r['states'])) for r in records])

if __name__=='__main__':main()
