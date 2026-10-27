"""Freeze reviewed examples and a prospective protocol before model scoring."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'evidence_mismatch/results'
# Assistant-reviewed patient facts. Counterfactuals are constructed interventions,
# not claims that the original source is false. No clinician adjudication.
FACTS=[
(394494,'The woman whose right knee is shown is 31 years old.',
 'The participant with the right-knee findings was aged thirty-one at the examination.',
 'The participant with the right-knee findings was aged sixty-one, not thirty-one, at the examination.',
 'What is the exact age of the woman whose right knee is shown?',
 'Axial proton-density fat-saturated MR images show changes in subchondral bone marrow oedema-like signal of the patella across three examinations. Arrows indicate bone marrow oedema-like signal.'),
(294756,'The patient has a history of recurrent urinary tract infections.',
 'Before these scans, this transplant recipient had experienced repeated infections of the urinary tract.',
 'Before these scans, this transplant recipient had never experienced a urinary tract infection.',
 'Did this patient have recurrent urinary tract infections before these scans?',
 'Diffusion-weighted MRI and 68Ga-pentixafor PET image the kidney allograft. Areas of reduced ADC correspond to foci with upregulated CXCR4 expression. The allograft is in the right lower abdomen.'),
(421261,'Renal function remained stable after the intervention.',
 'Laboratory follow-up documented preserved kidney function with no decline after treatment.',
 'Laboratory follow-up documented a substantial decline in kidney function after treatment.',
 'Did renal function remain stable after this intervention?',
 'A three-dimensional volume-rendered computed tomography scan shows patent visceral grafts and exclusion of the aneurysm.'),
(512675,'The patient is 55 years old.',
 'At the time of imaging, this man was aged fifty-five.',
 'At the time of imaging, this man was aged thirty-five, not fifty-five.',
 'What is the exact age of this patient?',
 'Volumetric brain tumor mapping shows FLAIR hyperintensity, necrosis and contrast enhancement in a right temporal lesion. Colored segmentations are overlaid on axial images.'),
(68061,'The patient previously had ovarian serous adenocarcinoma.',
 'The medical record documents a prior diagnosis of serous adenocarcinoma of the ovary.',
 'The medical record explicitly documents that this patient has never had ovarian cancer.',
 'Did this patient previously have ovarian serous adenocarcinoma?',
 'PET, CT and DWI depict lesions near the hepatoduodenal ligament, splenic hilum and right paracolic gutter. DWI depicts adjacent small lesions.'),
(502647,'The patient received BRAF- and MEK-inhibitor treatment.',
 'Medication records document administration of inhibitors targeting both BRAF and MEK.',
 'Medication records document that no BRAF inhibitor or MEK inhibitor was ever administered.',
 'Did this patient receive BRAF- and MEK-inhibitor treatment?',
 'Sagittal brain MRI panels show the lesion on serial examinations, including residual disease and later reduction in lesion extent.'),
(43585,'The patient had fever and seizures before developing movement disorders.',
 'The clinical history records a febrile illness with convulsions followed years later by dystonia and choreoathetoid movements.',
 'The clinical history explicitly records no fever and no seizures before the onset of dystonia and choreoathetoid movements.',
 'Did this patient have fever and seizures before developing movement disorders?',
 'T2-weighted MRI shows hyperintensities in both thalami.'),
(422538,'A biopsy confirmed fibroadenoma with lactational changes.',
 'A subsequent percutaneous tissue sample was examined and established fibroadenoma with lactational changes.',
 'No tissue sampling was performed; there is no biopsy confirmation of fibroadenoma with lactational changes.',
 'Was fibroadenoma with lactational changes confirmed by biopsy in this patient?',
 'T2 and post-contrast T1 MRI show a complex solid and cystic mass in the upper quadrants of the left breast.'),
(210842,'The patient has a history of alcoholism.',
 'The clinical record documents longstanding alcohol dependence in this patient.',
 'The clinical record documents lifelong abstinence from alcohol and explicitly excludes any history of alcohol dependence.',
 'Does this patient have a history of alcoholism?',
 'Axial CT demonstrates diffuse symmetric unencapsulated fatty accumulation of the neck and upper back.'),
(141171,'A biopsy histologically confirmed non-small cell adenocarcinoma.',
 'A needle tissue sample was obtained under CT guidance; microscopy established non-small cell adenocarcinoma.',
 'No tissue sample was obtained and no histological diagnosis has been made.',
 'Was non-small cell adenocarcinoma confirmed by a biopsy for this patient?',
 'Axial CT at the upper lobes using a lung window shows a cavitary lesion with an irregular thick wall.'),
(160931,'Focal adenomyosis was histologically confirmed in this patient.',
 'Microscopic examination of a tissue specimen established focal adenomyosis in this patient.',
 'No tissue specimen was examined and focal adenomyosis has not been histologically confirmed.',
 'Was focal adenomyosis histologically confirmed in this patient?',
 'Axial fat-suppressed T2 imaging shows a predominantly hypointense oval mass occupying the myometrium and junctional zone. Diffusion maps show the lesion.'),
(572961,'Histopathological examination confirmed massive choroidal invasion.',
 'Microscopic examination of the eye specimen established extensive invasion of the choroid.',
 'No eye specimen was examined histopathologically, so no tissue confirmation of choroidal invasion exists.',
 'Was massive choroidal invasion confirmed by histopathological examination?',
 'Postcontrast transaxial T1 MRI shows an enhancing intraocular tumor, focal choroidal thickening and discontinuity of the linear choroidal enhancement pattern.')]
HEAD=[202483,542571,351561,392502,455491,363837]
CHEST=[219192,298378,555935,220158,357717,241269]

def main():
 out=HERE/'results';out.mkdir(exist_ok=True)
 old={r['id']:r for r in map(json.loads,(OLD/'manifest.jsonl').read_text().splitlines())}
 facts=[]
 for n,claim,pos,neg,q,visual in FACTS:
  r=old[f'Instruction-Tuning_{n}'].copy()
  r.update(claim=claim,positive=pos,negative=neg,fact_question=q,visual_context=visual)
  facts.append(r)
 anatomy=[]
 for j,(a,b) in enumerate(zip(HEAD,CHEST)):
  for n,label in [(a,'head'),(b,'chest')]:
   r=old[f'Instruction-Tuning_{n}'].copy()
   r.update(anatomy_label=label,pair=j,review='Assistant visually checked CT head/chest anatomy; not a lesion label. Panels and acquisition styles remain confounded.')
   anatomy.append(r)
 for name,rows in [('facts',facts),('anatomy',anatomy)]:
  (out/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
 from PIL import Image,ImageOps,ImageDraw
 canvas=Image.new('RGB',(1600,1200),'white');draw=ImageDraw.Draw(canvas);checks=[]
 for j,r in enumerate(anatomy):
  im=Image.open(r['image_path']).convert('RGB')
  checks.append(dict(id=r['id'],label=r['anatomy_label'],pixel_sha256=hashlib.sha256(im.tobytes()).hexdigest(),size=im.size))
  im=ImageOps.contain(im,(390,365));x=j%4*400;y=j//4*400
  canvas.paste(im,(x+(400-im.width)//2,y+25));draw.text((x+5,y+5),r['id']+' '+r['anatomy_label'],fill='black')
 assert len({r['pixel_sha256'] for r in checks})==12
 (out/'anatomy_image_integrity.json').write_text(json.dumps(checks,indent=2))
 canvas.save(out/'selected_anatomy.jpg',quality=92)
 protocol={
  'seed':20261019,'version':1,'pre_scoring':True,
  'source_pdf_sha256':hashlib.sha256((HERE.parent.parent/'med_mllm_gap_notes.pdf').read_bytes()).hexdigest(),
  'selection':'Reuse 12 previously reviewed explicit facts; visually select 6 head and 6 chest CT figures from original pilot, without examining new model scores.',
  'H1':'External-fact target likelihood gains should shrink when off-image facts are removed from source context, and recover under independent paraphrases. This is availability sensitivity, not MI or proof of impossibility.',
  'H2':'Correct provenance behavior follows positive/negative supplied context, but chooses unknown when patient facts are absent; an explicit provenance instruction may help without retraining.',
  'H3':'Visual anatomy decisions should follow the image when an unverified textual label conflicts. Compare to no-image and congruent-text conditions; measure both members correct within fixed-context image pairs.',
  'H4':'Coarse anatomy may be retrievable from encoder, projector and middle LLM representations. Frozen leave-one-out 1NN is a pipeline sanity test, not a lesion oracle or pre/post boundary estimate.',
  'source_ladder':{'n':12,'contexts':['none','original','visual_only','paraphrase_positive','paraphrase_negative','donor'],'images':['correct','swap','none'],'score':'Original target mean logp and reviewed fact-span mean logp; targets fixed within model.'},
  'fact_choices':{'n':12,'contexts':['missing','positive','negative'],'images':['correct','swap','none'],'policies':['baseline','provenance'],'orders':'All 6 permutations of yes/no/unknown; one forward scores all 3 single-token labels.'},
  'anatomy_choices':{'n':12,'contexts':['neutral','head','chest'],'images':['correct','none'],'policies':['baseline','provenance'],'orders':'All 6 permutations of head/chest/unknown.'},
  'generation':{'n':12,'contexts':['missing'],'images':['correct'],'policies':['baseline','provenance'],'max_new_tokens':64,'purpose':'Qualitative check of natural output; no automated hallucination-rate claim.'},
  'statistics':'Average option permutations within record. Bootstrap records (5000) for paired effects; anatomy pair bootstrap for joint-correctness. Exploratory intervals, no multiplicity adjustment. Retrieval uses exact balanced-label permutation null.',
  'expected_per_model':{'ladder':216,'fact_choices':1296,'anatomy_choices':864,'features':12,'generation':24},
  'not_tested':['Unreported small lesions','Causal effect of supervision or training scale','Pretraining versus post-training interventions','Clinical calibration or conformal guarantees','Population mismatch prevalence']}
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
 print(json.dumps(protocol['expected_per_model']))
if __name__=='__main__':main()
