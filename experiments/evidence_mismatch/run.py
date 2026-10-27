"""Frozen-model, target-only likelihood audit using native MedProbe wrappers.

Each model runs in its own process because their llava packages conflict.
Scores are mean log probabilities (nats/token), not clinical correctness.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['huatuo', 'medsight'], required=True)
    ap.add_argument('--medprobe', default='/home/avc6555/research/MedProbe')
    ap.add_argument('--manifest', type=Path, default=HERE/'results/manifest.jsonl')
    ap.add_argument('--output', type=Path)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--checkpoint')
    args = ap.parse_args()
    records = [json.loads(l) for l in args.manifest.read_text().splitlines()]
    if args.limit:
        records = records[:args.limit]
    import numpy as np
    import torch
    import transformers
    from PIL import Image
    torch.manual_seed(20261018)
    torch.set_num_threads(4)
    sys.path.insert(0, args.medprobe)
    from common import load_model
    if args.model == 'huatuo':
        checkpoint = args.checkpoint or '/data/aofei/output/MedProbe/models/huatuogpt-vision-7b'
        adapter = load_model('huatuo', checkpoint, device='cuda')
    else:
        from huggingface_hub import snapshot_download
        checkpoint = args.checkpoint or snapshot_download('Caf0914/MedSIGHT-Comp', local_files_only=True)
        adapter = load_model('medsight', checkpoint, device='cuda', template='huatuo')
    model, bot, tok = adapter.model, adapter._bot, adapter.tokenizer
    model.eval()
    out = args.output or HERE/f'results/{args.model}.jsonl'
    metadata = dict(model=args.model, checkpoint=checkpoint, torch=torch.__version__,
                    transformers=transformers.__version__, python=sys.version,
                    visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    gpu=torch.cuda.get_device_name(), template='huatuo',
                    manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                    medprobe_commit=subprocess.check_output(['git','-C',args.medprobe,'rev-parse','HEAD'],text=True).strip(),
                    config=model.config.to_dict(), started_unix=time.time())
    meta_path = out.with_suffix('.meta.json')
    if out.exists() and out.stat().st_size:
        prior = json.loads(meta_path.read_text())
        for k in ['manifest_sha256','checkpoint','template']:
            assert prior[k] == metadata[k], f'Resume mismatch: {k}'
    else:
        meta_path.write_text(json.dumps(metadata, indent=2, default=str))
    done = {(r['id'], r['condition'], r['prefix_mode']) for r in map(json.loads, out.read_text().splitlines())} if out.exists() else set()
    verified = False

    @torch.inference_mode()
    def score(question, target, prefix, image):
        nonlocal verified
        text = bot.insert_image_placeholder(bot.input_moderation(question), int(image is not None))
        conv = bot.get_conv_without_history(text)
        prompt = bot.preprocess_huatuo(conv, return_tensors='pt').to('cuda')
        pids = tok(prefix+'\n', add_special_tokens=False).input_ids if prefix else []
        tids = tok(target, add_special_tokens=False).input_ids
        # Fixed target tokens across every intervention; preceding answer
        # sentences are conditioning only, never included in the score.
        tail = torch.tensor(pids+tids, device='cuda', dtype=torch.long)
        ids = torch.cat([prompt,tail])[None]
        labels = torch.full_like(ids, -100)
        labels[0,-len(tids):] = torch.tensor(tids,device='cuda')
        pixels = None
        if image is not None:
            pixels = torch.stack(bot.get_image_tensors([image])).to(device='cuda',dtype=torch.bfloat16)
        fn = (model.prepare_inputs_labels_for_multimodal_new if args.model=='huatuo'
              else model.prepare_inputs_labels_for_multimodal)
        prepared = fn(ids,None,None,None,labels,pixels)
        input_ids, position_ids, attention_mask, _, embeds, expanded = prepared[:6]
        assert expanded is not None
        positions = torch.where(expanded[0] != -100)[0]
        assert len(positions) == len(tids), 'Target truncated or incorrectly masked'
        assert torch.equal(expanded[0,positions],labels[0,-len(tids):])
        h = model.get_model()(input_ids=input_ids, inputs_embeds=embeds,
            position_ids=position_ids, attention_mask=attention_mask,
            use_cache=False, return_dict=True).last_hidden_state
        logits = model.lm_head(h[0, positions-1]).float()
        lp = torch.log_softmax(logits, dim=-1).gather(1,expanded[0,positions,None]).squeeze(1)
        assert torch.isfinite(lp).all()
        mean_lp = lp.mean().item()
        if not verified:
            check = model(input_ids=ids, images=pixels, labels=labels, use_cache=False,return_dict=True)
            error = abs(mean_lp + check.loss.item())
            assert error < 0.015, (mean_lp,check.loss.item(),error)
            print(json.dumps(dict(validation='native_forward_loss_agreement', error=error, n_tokens=len(tids))),flush=True)
            verified = True
        return dict(mean_logp=mean_lp,n_tokens=len(tids), token_logp=lp.cpu().tolist(),
                    input_length=int(expanded.shape[1]), prefix_tokens=len(pids))

    start=time.time()
    with out.open('a', buffering=1) as f:
        for i,r in enumerate(records):
            image=Image.open(r['image_path']).convert('RGB')
            blank=Image.new('RGB',image.size,(127,127,127))
            conditions=[('correct',image),('swap_1',r['donors'][0]['image_path']),
                        ('swap_2',r['donors'][1]['image_path']),('blank',blank),('no_image',None)]
            for condition, im in conditions:
                modes=['isolated','preceding_answer']+(['shuffled_answer'] if condition in ['correct','no_image'] else [])
                for mode in modes:
                    key=(r['id'],condition,mode)
                    if key in done:
                        continue
                    prefix={'isolated':'','preceding_answer':r['prefix'],'shuffled_answer':r['shuffled_prefix']}[mode]
                    s=score(r['question'],r['target'],prefix,im)
                    f.write(json.dumps(dict(id=r['id'],model=args.model,category=r['category'],subset=r['subset'],
                                            condition=condition,prefix_mode=mode,**s))+'\n')
            # Caption release is a deliberately privileged, potentially
            # answer-revealing condition. It cannot prove visual grounding.
            for condition,caption in [('caption',r['caption']),('shuffled_caption',r['shuffled_caption'])]:
                if (r['id'],condition,'isolated') in done:
                    continue
                if not caption:
                    raise ValueError(f'Missing caption for {r["id"]}/{condition}')
                q=r['question']+'\nAdditional source context:\n'+caption
                s=score(q,r['target'],'',image)
                f.write(json.dumps(dict(id=r['id'],model=args.model,category=r['category'],subset=r['subset'],
                                        condition=condition,prefix_mode='isolated',**s))+'\n')
            print(f'{args.model} {i+1}/{len(records)} {r["id"]} {r["category"]} elapsed={time.time()-start:.1f}s',flush=True)
    print('COMPLETE',flush=True)

if __name__=='__main__':
    main()
