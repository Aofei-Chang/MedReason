"""Summarize the completed primary and adaptive experiments."""
import json
from pathlib import Path
OUT=Path(__file__).resolve().parent/'results'
def main():
 s=json.loads((OUT/'summary.json').read_text());a=json.loads((OUT/'adaptive_prompt/summary.json').read_text())
 conclusion={
 'summary':'The follow-up supports missing-context attribution as a measurable problem, but also demonstrates the surface-compliance risk in the updated plan: a prompt can improve abstention while suppressing valid negative or visual evidence. Paraphrased facts influence target likelihood beyond verbatim caption copying. Coarse anatomy is both retained and used; these results do not establish a coverage gap or a pretraining/post-training boundary.',
 'scored_prompt_evaluations':6912,'feature_extractions':24,'natural_generations':72,
 'independent_selection_units':{'reviewed_fact_records':12,'visually_checked_anatomy_figures':12,'anatomy_pairs':6},
 'models':{},
 'measurement_recommendation':'Evaluate missing-fact abstention, positive-and-negative context update consistency, and fixed-context image-pair consistency together. Use target likelihood gains as model-dependence measurements, alongside structural or expert evidence labels; do not convert them into inferred Venn-region labels.',
 'theory_corrections':['The four-term identity needs a defined joint distribution and nested information assumptions (e.g., deterministic Z=g(I)).','Fixed-model ablation gains do not estimate the four entropy/KL terms. The exact counterexample has positive true image MI but negative frozen-model ablation gain.','Fano bounds apply to a specified frozen representation, not training chronology; updating the encoder or input can change the bound.','A low probe score can reflect pooling, task, probe capacity or finite data rather than missing information.','Report omission does not imply zero indirect learning signal.','Conformal guarantees still require suitable calibration labels, loss construction and exchangeability.'],
 'not_established':['Population mismatch prevalence','Causal effect of PubMedVision supervision or scaling','Weak perception of unreported medical findings','A validated source-groundedness ratio or mismatch detector','Superiority of pretraining or post-training','A general model ranking','Clinical safety or conformal guarantees'],
 'next_decisive_experiment':'Independently label reported and unreported lesions on held-out patients, match anatomy/acquisition/lesion size, then compare frozen encoder/projector/image-token readouts and actual counterfactual lesion decisions. Add explicit history/pathology availability manipulations. A matched frozen-versus-updated-encoder intervention is needed to test the pre/post boundary.'}
 for name,m in s['models'].items():
  am=a['models'][name]
  def fact(pol,ctx):return next(r['accuracy']['successes'] for r in am['fact'] if r['policy']==pol and r['context']==ctx)
  def anat(pol,regime):return next(r['accuracy']['successes'] for r in am['anatomy'] if r['policy']==pol and r['regime']==regime)
  conclusion['models'][name]={
   'paraphrase_fact_gain_nats':m['ladder']['fact_logp']['positive_paraphrase_minus_visual_only'],
   'positive_minus_negative_fact_gain_nats':m['ladder']['fact_logp']['positive_minus_negative_paraphrase'],
   'choice_scores_out_of_12':{pol:{'missing_fact_abstention':fact(pol,'missing'),'positive_context_correct':fact(pol,'positive'),'negative_context_correct':fact(pol,'negative'),'conflicting_note_anatomy_correct':anat(pol,'conflicting')} for pol in ['baseline','provenance','evidence_first']},
   'encoder_nearest_neighbor_correct_out_of_12':next(r['correct'] for r in m['features'] if r['layer']=='encoder_selected'),
   'projector_nearest_neighbor_correct_out_of_12':next(r['correct'] for r in m['features'] if r['layer']=='projector')}
 (OUT/'conclusions.json').write_text(json.dumps(conclusion,indent=2));print(json.dumps(conclusion,indent=2))
if __name__=='__main__':main()
