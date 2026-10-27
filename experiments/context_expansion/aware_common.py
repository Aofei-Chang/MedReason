"""Shared pieces for the awareness / privilege-decomposition run (V0-V6, privilege_aware_design.md).

Context split, three ways (labels.jsonl for V/P/K/M, labels_time.jsonl for when a P sentence is known):
  V     visual context: V/M sentences (findings, arrows, panels). Privileged: never returned by a request.
  pre   non-visual information known when the image is read: history, symptoms, labs, indication, and K
        (general knowledge). This is what a <request> returns.
  post  non-visual information established after the image: pathology, surgery, treatment given, outcome.
        Never available to the deployed model.
  N     pre + post (the "non-visual context" of the 2026-10-25 run).
Mismatched image: another record of the same modality (controls for "an image is present").
"""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from score_conditions import snapshot, MODELS
from score_typed_context import OUT, load_labels

AW = Path('/data/aofei/output/ContextMLLM/aware'); AW.mkdir(exist_ok=True)
VISUAL = (0, 3)
TIME = 'pre post general'.split()

def load_time_labels():
    lab = {}
    p = AW / 'labels_time.jsonl'
    if not p.exists(): return lab
    for line in open(p):
        x = json.loads(line)
        lab[(x['id'], x['field'])] = {i: int(np.argmax(pr)) for i, pr in zip(x['idx'], x['probs'])}
    return lab

def part_of(c, t):
    """Sentence class c (V/P/K/M index) and time label t (pre/post/general or None) -> V / pre / post."""
    if c in VISUAL: return 'V'
    if c == 1 and t == 1: return 'post'
    return 'pre'

def split_sentences(r, lab, tlab, field):
    """[(source, sentence, part)] for caption_sentences or article_sentences."""
    cls = lab[(r['id'], field)]; tl = tlab.get((r['id'], field), {})
    sents = r[field] if field == 'caption_sentences' else r['article_sentences']
    out = []
    for i, (s, c) in enumerate(zip(sents, cls)):
        src, text = ('caption', s) if field == 'caption_sentences' else s
        out.append((src, text, part_of(c, tl.get(i))))
    return out

HEADS = [('caption', 'Figure caption from the source article (excerpt)'), ('title', 'Article title'),
         ('abstract', 'Abstract'), ('citing', 'Article text')]

def fmt_context(items, keep):
    sel = [(src, s) for src, s, p in items if p in keep]
    if not sel: return None
    out = []
    for src, head in HEADS:
        xs = [s for k, s in sel if k == src]
        if xs: out.append(f"{head}: {' '.join(xs)}")
    return '\n'.join(out)

def contexts(r, lab, tlab, field):
    items = split_sentences(r, lab, tlab, field)
    return {'V': fmt_context(items, {'V'}), 'pre': fmt_context(items, {'pre'}), 'post': fmt_context(items, {'post'}),
            'N': fmt_context(items, {'pre', 'post'}), 'VN': fmt_context(items, {'V', 'pre', 'post'})}

def mismatched(recs, key='modality', shift=997):
    """index -> index of another record with the same modality (deterministic)."""
    by = defaultdict(list)
    for i, r in enumerate(recs): by[r.get(key, '')].append(i)
    out = {}
    for g in by.values():
        for j, i in enumerate(g):
            out[i] = g[(j + shift) % len(g)] if len(g) > 1 else (i + shift) % len(recs)
    return out

def load_image(p):
    return Image.open(p).convert('RGB')

class TF:
    """Teacher forcing with Qwen2.5-VL-family models: per-token log-probs of a target given (text, image).

    score() is single-example (long targets, per-token output); score_batch() is batched for short targets.
    Logits are computed only for target positions (logits_to_keep).
    """
    def __init__(self, name, max_pixels=1024 * 28 * 28, device='cuda'):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        path = snapshot(MODELS[name])
        self.proc = AutoProcessor.from_pretrained(path, min_pixels=32 * 28 * 28, max_pixels=max_pixels)
        self.proc.tokenizer.padding_side = 'left'
        self.tok = self.proc.tokenizer
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            path, torch_dtype=torch.bfloat16, attn_implementation='sdpa').to(device).eval()
        self.device = device
        self._cache_vision()

    def _cache_vision(self, size=6):
        """The same images are encoded by many conditions in a row; reuse the vision tower output (no flash-attn here)."""
        vis, fwd, cache = self.model.model.visual, self.model.model.visual.forward, {}
        def cached(pixel_values, grid_thw=None, **kw):
            key = (tuple(grid_thw.flatten().tolist()), pixel_values.shape[0],
                   float(pixel_values[::97].float().sum()), float(pixel_values[1::89].float().sum()))
            if key not in cache:
                if len(cache) >= size: cache.pop(next(iter(cache)))
                cache[key] = fwd(pixel_values, grid_thw=grid_thw, **kw)
            return cache[key]
        vis.forward = cached

    def prompt(self, text, n_images):
        content = [{'type': 'image'} for _ in range(n_images)] + [{'type': 'text', 'text': text}]
        return self.proc.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True)

    @staticmethod
    def _imgs(image):
        return image if isinstance(image, list) else ([image] if image is not None else [])

    @torch.no_grad()
    def score(self, text, image, target, entropy=False):
        imgs = self._imgs(image)
        inp = self.proc(text=[self.prompt(text, len(imgs))], images=imgs or None, return_tensors='pt')
        tgt = self.tok(target, add_special_tokens=False)['input_ids']
        ids = torch.cat([inp['input_ids'], torch.tensor([tgt])], 1).to(self.device)
        kw = {k: v.to(self.device) for k, v in inp.items() if k not in ('input_ids', 'attention_mask')}
        logits = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), logits_to_keep=len(tgt) + 1, **kw).logits[0, :-1]
        logp = torch.log_softmax(logits.float(), -1)
        lp = logp.gather(1, torch.tensor(tgt, device=self.device)[:, None])[:, 0].cpu().numpy()
        if not entropy: return lp
        return lp, (-(logp.exp() * logp).sum(-1)).cpu().numpy()

    @torch.no_grad()
    def score_batch(self, texts, images, targets, hidden_layers=None):
        """Sum log-prob and token count of each target; optional hidden states at the last prompt token."""
        imgs = [self._imgs(x) for x in images]
        tg = [self.tok(t, add_special_tokens=False)['input_ids'] for t in targets]
        full = [self.prompt(t, len(i)) + y for t, i, y in zip(texts, imgs, targets)]
        flat = [x for i in imgs for x in i]
        inp = self.proc(text=full, images=flat or None, padding=True, return_tensors='pt').to(self.device)
        K = max(len(t) for t in tg) + 1
        out = self.model(**inp, logits_to_keep=K, output_hidden_states=hidden_layers is not None)
        logp = torch.log_softmax(out.logits.float(), -1)
        ids = inp['input_ids']; L = ids.shape[1]
        res, hs = [], []
        for b, t in enumerate(tg):
            n = len(t)
            assert ids[b, L - n:].tolist() == t, 'target tokenization differs inside the prompt'
            pos = torch.arange(K - 1 - n, K - 1, device=self.device)
            lp = logp[b, pos].gather(1, ids[b, L - n:][:, None])[:, 0]
            res.append((float(lp.sum()), n))
            if hidden_layers is not None:
                hs.append(np.stack([out.hidden_states[l][b, L - n - 1].float().cpu().numpy() for l in hidden_layers]))
        return res, (np.stack(hs).astype(np.float16) if hidden_layers is not None else None)

    @torch.no_grad()
    def next_token_probs(self, texts, images, choices, log_odds=False):
        """Softmax over the first token of each choice string at the first answer position
        (log_odds: log p(choices[0]) - log p(choices[1]) instead, which does not saturate)."""
        imgs = [self._imgs(x) for x in images]
        full = [self.prompt(t, len(i)) for t, i in zip(texts, imgs)]
        flat = [x for i in imgs for x in i]
        inp = self.proc(text=full, images=flat or None, padding=True, return_tensors='pt').to(self.device)
        ids = [self.tok.encode(c, add_special_tokens=False)[0] for c in choices]
        logits = self.model(**inp, logits_to_keep=1).logits[:, -1].float()
        if log_odds: return (logits[:, ids[0]] - logits[:, ids[1]]).cpu().numpy()
        return torch.softmax(logits[:, ids], -1).cpu().numpy()

    @torch.no_grad()
    def generate(self, texts, images, max_new_tokens=16, n=1, sample=False):
        """Returns per prompt a list of n (text, mean_nll, first_token_entropy)."""
        imgs = [self._imgs(x) for x in images]
        full = [self.prompt(t, len(i)) for t, i in zip(texts, imgs)]
        flat = [x for i in imgs for x in i]
        inp = self.proc(text=full, images=flat or None, padding=True, return_tensors='pt').to(self.device)
        out = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=sample, num_return_sequences=n,
                                  temperature=1.0 if sample else None, top_p=None, top_k=None,
                                  output_scores=True, return_dict_in_generate=True)
        seq = out.sequences[:, inp['input_ids'].shape[1]:]
        logp = torch.log_softmax(torch.stack(out.scores, 1).float(), -1)  # [B*n, T, V]
        lp = logp.gather(2, seq[:, :, None])[:, :, 0]
        eos = {self.tok.eos_token_id, self.tok.pad_token_id, self.tok.convert_tokens_to_ids('<|im_end|>')}
        mask = torch.ones_like(seq, dtype=torch.bool)
        for i in range(seq.shape[0]):
            stop = next((j for j, t in enumerate(seq[i].tolist()) if t in eos), seq.shape[1])
            mask[i, stop + 1:] = False
        nll = (-(lp * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu().numpy()
        ent0 = (-(logp[:, 0].exp() * logp[:, 0]).sum(-1)).cpu().numpy()
        txt = self.tok.batch_decode(seq, skip_special_tokens=True)
        rows = [(t.strip(), float(a), float(e)) for t, a, e in zip(txt, nll, ent0)]
        return [rows[i * n:(i + 1) * n] for i in range(len(texts))]
