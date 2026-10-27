"""Training-free evidence interventions and frozen representation extraction.

Separate native MedProbe model processes avoid llava namespace collisions.
No optimizer, fitted probe, API generation or weight updates.
"""
import argparse,hashlib,itertools,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
POLICY='Use only the supplied evidence. Do not infer patient history, exact age, or whether a test was performed from appearance alone. If the requested fact is not provided, say that it cannot be determined. '
VIS_POLICY='Identify the anatomy from the actual image. An unverified text label may be wrong. If no image is supplied, say that the anatomy cannot be determined. '

def main():
 global POLICY,VIS_POLICY
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['huatuo','medsight'],required=True)
 ap.add_argument('--tasks',default='ladder,fact,anatomy,features,generation');ap.add_argument('--limit',type=int,default=0)
 ap.add_argument('--output-dir',type=Path,default=HERE/'results')
 ap.add_argument('--policies',default='baseline,provenance')
 ap.add_argument('--policy-variant',choices=['original','evidence_first'],default='original')
 args=ap.parse_args();tasks=set(args.tasks.split(','))
 policies=args.policies.split(',')
 if args.policy_variant=='evidence_first':
  POLICY='Use all evidence that is actually supplied. An explicit patient record can establish that a statement is true or false. Choose Yes when the evidence supports the statement, No when it contradicts the statement, and Cannot determine only when neither conclusion is supported. '
  VIS_POLICY='Inspect the supplied image and identify its body region from visible anatomy. Give a specific answer when the anatomy is clear, even if the unverified note disagrees. Use Cannot determine only when the available evidence does not establish the region. '
 import numpy as np
 import torch,transformers
 from PIL import Image
 torch.set_num_threads(4);torch.manual_seed(20261019)
 sys.path.insert(0,'/home/avc6555/research/MedProbe')
 from common import load_model
 if args.model=='huatuo':
  ckpt='/data/aofei/output/MedProbe/models/huatuogpt-vision-7b';adapter=load_model('huatuo',ckpt,device='cuda')
 else:
  from huggingface_hub import snapshot_download
  ckpt=snapshot_download('Caf0914/MedSIGHT-Comp',local_files_only=True)
  adapter=load_model('medsight',ckpt,device='cuda',template='huatuo')
 model,bot,tok=adapter.model,adapter._bot,adapter.tokenizer;model.eval()
 assert all(not p.requires_grad or p.grad is None for p in model.parameters())
 out=args.output_dir;path=out/f'{args.model}.jsonl'
 manifests=['facts.jsonl','anatomy.jsonl','protocol.json']
 hashes={n:hashlib.sha256((out/n).read_bytes()).hexdigest() for n in manifests}
 meta=dict(model=args.model,checkpoint=ckpt,manifest_hashes=hashes,started=time.time(),torch=torch.__version__,transformers=transformers.__version__,visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),gpu=torch.cuda.get_device_name(),template='huatuo',medprobe_commit=subprocess.check_output(['git','-C','/home/avc6555/research/MedProbe','rev-parse','HEAD'],text=True).strip(),runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 mp=out/f'{args.model}.meta.json'
 meta['policy_variant']=args.policy_variant
 meta['policies']=policies
 meta['fact_policy_text']=POLICY
 meta['visual_policy_text']=VIS_POLICY
 if path.exists():
  prior=json.loads(mp.read_text())
  for k in ['checkpoint','manifest_hashes','template','runner_sha256','policy_variant','policies']:assert prior[k]==meta[k],k
 else:mp.write_text(json.dumps(meta,indent=2))
 done={r['key'] for r in map(json.loads,path.read_text().splitlines())} if path.exists() else set()
 facts=[json.loads(s) for s in (out/'facts.jsonl').read_text().splitlines()]
 anatomy=[json.loads(s) for s in (out/'anatomy.jsonl').read_text().splitlines()]
 if args.limit:facts=facts[:args.limit];anatomy=anatomy[:args.limit]
 spans=json.loads((HERE.parent/'evidence_mismatch/fact_spans.json').read_text())['spans']
 pixels_cache={};verified=False;choice_verified=False;generation_verified=False
 fn=model.prepare_inputs_labels_for_multimodal_new if args.model=='huatuo' else model.prepare_inputs_labels_for_multimodal
 def pixels(image):
  if image is None:return None
  if image not in pixels_cache:pixels_cache[image]=torch.stack(bot.get_image_tensors([image])).to(device='cuda',dtype=torch.bfloat16)
  return pixels_cache[image]
 def prompt_ids(q,image):
  text=bot.insert_image_placeholder(bot.input_moderation(q),int(image is not None))
  return bot.preprocess_huatuo(bot.get_conv_without_history(text),return_tensors='pt').to('cuda')
 def prep(ids,labels,image):return fn(ids,None,None,None,labels,pixels(image))[:6]
 def forward(p,hidden=False):
  ids,pos,mask,_,emb,_=p
  return model.get_model()(input_ids=ids,inputs_embeds=emb,position_ids=pos,attention_mask=mask,use_cache=False,return_dict=True,output_hidden_states=hidden)
 @torch.inference_mode()
 def score(q,target,image,span):
  nonlocal verified
  p=prompt_ids(q,image);t=tok(target,add_special_tokens=False,return_offsets_mapping=True)
  tail=torch.tensor(t.input_ids,device='cuda');ids=torch.cat([p,tail])[None]
  labels=torch.full_like(ids,-100);labels[0,-len(tail):]=tail
  prepared=prep(ids,labels,image);expanded=prepared[-1];ix=torch.where(expanded[0]!=-100)[0]
  assert len(ix)==len(tail) and torch.equal(expanded[0,ix],tail)
  h=forward(prepared).last_hidden_state
  lp=torch.log_softmax(model.lm_head(h[0,ix-1]).float(),-1).gather(1,tail[:,None]).squeeze(1)
  assert torch.isfinite(lp).all()
  if not verified:
   loss=model(input_ids=ids,images=pixels(image),labels=labels,use_cache=False,return_dict=True).loss
   err=abs(loss.item()+lp.mean().item());assert err<0.015
   print('native_loss_check',err,flush=True);verified=True
  start=target.index(span);end=start+len(span)
  si=[i for i,(a,b) in enumerate(t.offset_mapping) if a<end and b>start]
  assert si
  return dict(mean_logp=lp.mean().item(),fact_logp=lp[si].mean().item(),n_tokens=len(tail),fact_tokens=len(si),token_logp=lp.cpu().tolist())
 letters=['A','B','C'];letterids=[tok(s,add_special_tokens=False).input_ids for s in letters]
 assert all(len(x)==1 for x in letterids);letterids=[x[0] for x in letterids]
 @torch.inference_mode()
 def choose(q,image,order):
  nonlocal choice_verified
  ids=prompt_ids(q,image)[None];p=prep(ids,torch.full_like(ids,-100),image)
  h=forward(p).last_hidden_state;logits=model.lm_head(h[0,-1]).float();lp=torch.log_softmax(logits,-1)
  if not choice_verified:
   native=model(input_ids=ids,images=pixels(image),use_cache=False,return_dict=True).logits[0,-1].float()
   error=(torch.log_softmax(native,-1)[letterids]-lp[letterids]).abs().max().item()
   assert error<0.015;print('choice_forward_check',error,flush=True);choice_verified=True
  chosen=lp[letterids];assert torch.isfinite(chosen).all()
  return dict(option_logp={x:chosen[j].item() for j,x in enumerate(order)},option_mass=chosen.exp().sum().item(),prediction=order[chosen.argmax().item()])
 @torch.inference_mode()
 def features(image):
  # Input to mm_projector = selected encoder features actually passed downstream.
  captured={}
  def pool(t):
   assert isinstance(t,torch.Tensor)
   return t.float().reshape(-1,t.shape[-1]).mean(0).cpu().numpy()
  def before(m,inputs):captured['encoder_selected']=pool(inputs[0])
  def after(m,inputs,output):captured['projector']=pool(output)
  m=model.get_model().mm_projector
  hs=[m.register_forward_pre_hook(before),m.register_forward_hook(after)]
  ids=prompt_ids('Describe the supplied image briefly.',image)[None]
  try:p=prep(ids,torch.full_like(ids,-100),image)
  finally:
   for h in hs:h.remove()
  result=forward(p,True)
  locations=torch.where(ids[0]<0)[0];assert len(locations)==1
  start=locations[0].item();nimg=result.last_hidden_state.shape[1]-ids.shape[1]+1;assert nimg>1
  depth=len(result.hidden_states)-1
  for layer in sorted({depth//4,depth//2,3*depth//4,depth}):
   hidden=result.hidden_states[layer][0]
   captured[f'llm_{layer}_image_mean']=hidden[start:start+nimg].float().mean(0).cpu().numpy()
   captured[f'llm_{layer}_last_prompt']=hidden[-1].float().cpu().numpy()
  assert all(np.isfinite(v).all() for v in captured.values())
  return captured,dict(image_tokens=nimg,llm_depth=depth)
 def options(q,order,names):
  return q+'\n'+'\n'.join(f'{letters[i]}. {names[x]}' for i,x in enumerate(order))+'\nRespond with only A, B, or C.'
 @torch.inference_mode()
 def generate(q,image):
  nonlocal generation_verified
  # Native Huatuo's legacy generation wrapper rejects cache_position under
  # this transformers release. Greedy decoding through the same Qwen core
  # avoids that wrapper; no scoring or model weights are changed.
  ids=prompt_ids(q,image)[None];p=prep(ids,torch.full_like(ids,-100),image)
  input_ids,pos,mask,_,emb,_=p
  result=model.get_model()(input_ids=input_ids,inputs_embeds=emb,position_ids=pos,
    attention_mask=mask,use_cache=True,return_dict=True)
  output=[]
  eos=model.generation_config.eos_token_id
  stops=set(eos if isinstance(eos,list) else [eos,tok.eos_token_id])
  for _ in range(64):
   nxt=model.lm_head(result.last_hidden_state[:,-1]).argmax(-1)
   if len(output)==2 and not generation_verified:
    full=torch.cat([ids[0],torch.tensor(output,device='cuda')])[None]
    full_h=forward(prep(full,torch.full_like(full,-100),image)).last_hidden_state
    ref=model.lm_head(full_h[:,-1]).argmax(-1)
    assert torch.equal(nxt,ref),'Cached decoding differs from full-prefix decoding'
    generation_verified=True;print('greedy_cache_check PASS',flush=True)
   if nxt.item() in stops:break
   output.append(nxt.item())
   result=model.get_model()(input_ids=nxt[:,None],past_key_values=result.past_key_values,
     use_cache=True,return_dict=True)
  return tok.decode(output,skip_special_tokens=True),len(output)
 started=time.time()
 with path.open('a',buffering=1) as f:
  def save(key,row):
   assert key not in done
   f.write(json.dumps(dict(key=key,model=args.model,**row))+'\n');done.add(key)
  for j,r in enumerate(facts):
   rid=r['id'];ims={'correct':r['image_path'],'swap':r['donors'][0]['image_path'],'none':None}
   if 'ladder' in tasks:
    contexts={'none':'','original':r['caption'],'visual_only':r['visual_context'],'paraphrase_positive':r['visual_context']+'\n'+r['positive'],'paraphrase_negative':r['visual_context']+'\n'+r['negative'],'donor':r['shuffled_caption']}
    for ctx,im in itertools.product(contexts,ims):
     key=f'ladder/{rid}/{ctx}/{im}'
     if key in done:continue
     q=r['question']+ ('\nAdditional source context:\n'+contexts[ctx] if contexts[ctx] else '')
     save(key,dict(task='ladder',id=rid,context=ctx,image=im,**score(q,r['target'],ims[im],spans[rid])))
   if 'fact' in tasks:
    for ctx,im,policy in itertools.product(['missing','positive','negative'],ims,policies):
     q=(POLICY if policy!='baseline' else '')
     if ctx!='missing':q+='Additional patient record:\n'+r[ctx]+'\n'
     q+='Based on the supplied material, is this patient-specific statement true?\n'+r['claim']
     for oi,order in enumerate(itertools.permutations(['yes','no','unknown'])):
      key=f'fact/{rid}/{ctx}/{im}/{policy}/{oi}'
      if key in done:continue
      query=options(q,order,dict(yes='Yes.',no='No.',unknown='Cannot determine from the supplied material.'))
      save(key,dict(task='fact',id=rid,context=ctx,image=im,policy=policy,order_index=oi,order=order,expected={'missing':'unknown','positive':'yes','negative':'no'}[ctx],**choose(query,ims[im],order)))
   if 'generation' in tasks:
    for policy in policies:
     key=f'generation/{rid}/{policy}'
     if key in done:continue
     q=(POLICY if policy!='baseline' else '')+r['fact_question']+' Answer in one sentence.'
     answer,ntokens=generate(q,r['image_path'])
     save(key,dict(task='generation',id=rid,policy=policy,prompt=q,answer=answer,n_generated_tokens=ntokens,truncated=ntokens==64))
   print(f'facts {j+1}/{len(facts)} elapsed={time.time()-started:.1f}s',flush=True)
  for j,r in enumerate(anatomy):
   rid=r['id']
   if 'features' in tasks:
    key=f'features/{rid}';fp=out/f'{args.model}_features_{rid}.npz'
    if key not in done:
     vectors,info=features(r['image_path']);np.savez_compressed(fp,**vectors)
     save(key,dict(task='features',id=rid,label=r['anatomy_label'],file=fp.name,**info))
   if 'anatomy' in tasks:
    for ctx,im,policy in itertools.product(['neutral','head','chest'],['correct','none'],policies):
     image=r['image_path'] if im=='correct' else None
     q=(VIS_POLICY if policy!='baseline' else '')
     if ctx!='neutral':q+=f'An earlier unverified note labels this as a {ctx} scan.\n'
     q+='Which body region actually appears in the supplied image?'
     for oi,order in enumerate(itertools.permutations(['head','chest','unknown'])):
      key=f'anatomy/{rid}/{ctx}/{im}/{policy}/{oi}'
      if key in done:continue
      query=options(q,order,dict(head='Head.',chest='Chest.',unknown='Cannot determine from the supplied material.'))
      save(key,dict(task='anatomy',id=rid,label=r['anatomy_label'],pair=r['pair'],context=ctx,image=im,policy=policy,order_index=oi,order=order,expected=r['anatomy_label'] if image else 'unknown',**choose(query,image,order)))
   print(f'anatomy {j+1}/{len(anatomy)} elapsed={time.time()-started:.1f}s',flush=True)
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
