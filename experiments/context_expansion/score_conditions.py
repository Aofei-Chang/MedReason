"""Teacher-forced per-sentence log-likelihood of PubMedVision targets under image/context ablations.

Target A = instruction answer;  context = original caption (what GPT-4V saw when writing it).
Target B = original caption;    context = article (title, abstract, figure-citing paragraphs),
           plus a redacted variant with sentences that cite this figure removed.
Conditions: I (image only), IC (image+context), C (context only), 0 (neither); B also IR, R (redacted).
Also stores hidden states (decision point + in-sentence mean, image-only condition) and a zero-shot
self-report p(Yes) to "can this be verified from the image alone?".
No weights are updated.
"""
import argparse, json, re
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODELS = {
    'qwen25vl': '/data/aofei/cache/huggingface/models--Qwen--Qwen2.5-VL-7B-Instruct',
    'lingshu': '/data/aofei/cache/huggingface/models--lingshu-medical-mllm--Lingshu-7B',
    'huatuo': '/data/aofei/cache/huggingface/hub/models--FreedomIntelligence--HuatuoGPT-Vision-7B-Qwen2.5VL',
}
BASE = Path('/data/aofei/output/ContextMLLM/pilot_needs_context')
LAYERS = [7, 14, 21, 28]
CAPTION_Q = 'Write the figure caption for this medical image as it would appear in the source article.'
SELF_Q = ('Statement about this image: "{s}"\nCan this statement be fully verified by looking at the image alone, '
          'without any other information about the patient or the study? Answer Yes or No.')

def snapshot(repo_dir):
    return str(next(Path(repo_dir, 'snapshots').iterdir()))

def article_context(r, redact):
    paras = ' '.join(p['text'] for p in r['citing_paragraphs'])
    if redact:
        num = re.search(r'\d+', r['fig_label'] or '')
        pat = re.compile(rf'\bfig(ure)?s?\.?\s*{num.group(0)}(?![0-9])', re.I) if num else None
        paras = ' '.join(s for s in re.split(r'(?<=[.!?])\s+', paras) if not (pat and pat.search(s)))
    return (f"Article title: {r['title']}\nAbstract: {r['abstract'][:800]}\n"
            f"Article text: {paras[:1500]}")

class Scorer:
    def __init__(self, name, device='cuda'):
        path = snapshot(MODELS[name])
        self.proc = AutoProcessor.from_pretrained(path, min_pixels=64 * 28 * 28, max_pixels=1024 * 28 * 28)
        self.tok = self.proc.tokenizer
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            path, torch_dtype=torch.bfloat16, attn_implementation='sdpa').to(device).eval()
        self.device = device
        self.yes = self.tok.encode('Yes', add_special_tokens=False)[0]
        self.no = self.tok.encode('No', add_special_tokens=False)[0]

    def prompt_inputs(self, text, image):
        content = ([{'type': 'image'}] if image is not None else []) + [{'type': 'text', 'text': text}]
        prompt = self.proc.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True)
        return self.proc(text=[prompt], images=[image] if image is not None else None, return_tensors='pt')

    @torch.no_grad()
    def score(self, text, image, target, spans, hidden=False):
        inp = self.prompt_inputs(text, image)
        enc = self.tok(target, add_special_tokens=False, return_offsets_mapping=True)
        tgt = torch.tensor([enc['input_ids']])
        P = inp['input_ids'].shape[1]
        ids = torch.cat([inp['input_ids'], tgt], 1).to(self.device)
        kw = {k: v.to(self.device) for k, v in inp.items() if k not in ('input_ids', 'attention_mask')}
        out = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=hidden, **kw)
        logp = torch.log_softmax(out.logits[0, P - 1:-1].float(), -1)
        tok_lp = logp.gather(1, tgt[0].to(self.device)[:, None])[:, 0].cpu().numpy()
        ent = -(logp.exp() * logp).sum(-1).cpu().numpy()
        res, hs = [], []
        offs = enc['offset_mapping']
        for a, b in spans:
            idx = [i for i, (s, _) in enumerate(offs) if a <= s < b]
            if not idx:
                res.append(None); hs.append(None); continue
            res.append({'sum': float(tok_lp[idx].sum()), 'n': len(idx), 'mean_entropy': float(ent[idx].mean()),
                        'max_entropy': float(ent[idx].max())})
            if hidden:
                pos = [P - 1 + i for i in idx]
                hs.append(np.stack([np.concatenate([out.hidden_states[l][0, pos[0]].float().cpu().numpy(),
                                                    out.hidden_states[l][0, pos].float().mean(0).cpu().numpy()])
                                    for l in LAYERS]).astype(np.float16))
        return res, hs

    @torch.no_grad()
    def self_report(self, sentence, image):
        inp = self.prompt_inputs(SELF_Q.format(s=sentence), image)
        inp = {k: v.to(self.device) for k, v in inp.items()}
        logits = self.model(**inp).logits[0, -1].float()
        return float(torch.log_softmax(logits[[self.yes, self.no]], 0)[0].exp())

def spans_of(text, sents):
    out, cur = [], 0
    for s in sents:
        a = text.find(s, cur)
        if a < 0: a = text.find(s)
        out.append((a, a + len(s)) if a >= 0 else (0, 0)); cur = max(cur, a + len(s))
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', choices=MODELS, required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--start', type=int, default=0); ap.add_argument('--end', type=int, default=None)
    args = ap.parse_args()
    torch.set_num_threads(4)
    recs = [json.loads(l) for l in open(BASE / 'pilot.jsonl')][:args.limit or None][args.start:args.end]
    hs_dir = BASE / f'hidden_{args.model}'; hs_dir.mkdir(exist_ok=True)
    done = {json.loads(l)['id'] for p in BASE.glob(f'scores_{args.model}*.jsonl') for l in open(p)}
    out_path = BASE / (f'scores_{args.model}.jsonl' if args.start == 0 else f'scores_{args.model}_s{args.start}.jsonl')
    sc = Scorer(args.model)
    with open(out_path, 'a') as f:
        for i, r in enumerate(recs):
            if r['id'] in done: continue
            img = Image.open(r['image']).convert('RGB')
            ans, cap = r['answer'], r['original_caption']
            sa, sb = spans_of(ans, r['answer_sentences']), spans_of(cap, r['caption_sentences'])
            ctx_a = f"Figure caption from the source article: {cap}"
            ctx_b, ctx_r = article_context(r, False), article_context(r, True)
            q = r['question']
            row = {'id': r['id'], 'A': {}, 'B': {}}
            row['A']['I'], ha = sc.score(q, img, ans, sa, hidden=True)
            row['A']['IC'], _ = sc.score(f'{ctx_a}\n\n{q}', img, ans, sa)
            row['A']['C'], _ = sc.score(f'{ctx_a}\n\n{q}', None, ans, sa)
            row['A']['0'], _ = sc.score(q, None, ans, sa)
            row['B']['I'], hb = sc.score(CAPTION_Q, img, cap, sb, hidden=True)
            for cond, ctx, im in [('IC', ctx_b, img), ('C', ctx_b, None), ('IR', ctx_r, img), ('R', ctx_r, None)]:
                row['B'][cond], _ = sc.score(f'{ctx}\n\n{CAPTION_Q}', im, cap, sb)
            row['B']['0'], _ = sc.score(CAPTION_Q, None, cap, sb)
            row['A']['self_yes'] = [sc.self_report(s, img) for s in r['answer_sentences']]
            row['B']['self_yes'] = [sc.self_report(s, img) for s in r['caption_sentences']]
            np.savez_compressed(hs_dir / f"{r['id']}.npz",
                                A=np.stack([h if h is not None else np.zeros((len(LAYERS), 7168), np.float16) for h in ha]) if ha else np.zeros((0,)),
                                B=np.stack([h if h is not None else np.zeros((len(LAYERS), 7168), np.float16) for h in hb]) if hb else np.zeros((0,)))
            f.write(json.dumps(row) + '\n'); f.flush()
            if i % 25 == 0: print(i, flush=True)

if __name__ == '__main__':
    main()
