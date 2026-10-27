"""Shared pieces for generation (P3, P4, P5, P6) and the answer-or-request protocol (P4, P5).

Episode: turn 1 the model sees the image and question and replies <answer>..</answer> or <request>..</request>.
On a request the environment returns a context and the model answers in turn 2. Two environments:
  full   the article context as found (caption + article excerpt): contains visual restatements, so a request
         can leak the answer to an image question;
  nonvis only the non-visual information sentences of that context (visual sentences removed).
Rollouts are built at the token-id level so the training forward pass scores exactly the generated tokens.
"""
import json, re, string
from collections import Counter
from pathlib import Path
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from score_conditions import MODELS, snapshot

OUT = Path('/data/aofei/output/ContextMLLM/overnight')
PROTOCOL = ('You are shown a medical image from a published case, and a question about it.\n'
            'If the question can be answered from the image, reply with <answer>short answer</answer>.\n'
            'If answering needs information about this patient that is not visible in the image (for example history, '
            'symptoms, laboratory or pathology results, treatment, or outcome), reply with <request>what you need</request> '
            'and the text of the source article will be provided.\nKeep answers to a few words.\n\nQuestion: {q}')
DIRECT = 'Answer the question about this medical image in a few words. Reply with <answer>short answer</answer>.\n\nQuestion: {q}'
GIVEN = ('Text from the source article:\n{ctx}\n\nAnswer the question about this medical image in a few words. '
         'Reply with <answer>short answer</answer>.\n\nQuestion: {q}')
TURN2 = ('\n<|im_start|>user\n<context>\n{ctx}\n</context>\nNow answer the question. '
         'Reply with <answer>short answer</answer>.<|im_end|>\n<|im_start|>assistant\n')

def load_vl(name, max_pixels=256 * 28 * 28, device='cuda'):
    path = snapshot(MODELS[name])
    proc = AutoProcessor.from_pretrained(path, min_pixels=32 * 28 * 28, max_pixels=max_pixels)
    proc.tokenizer.padding_side = 'left'
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        path, torch_dtype=torch.bfloat16, attn_implementation='sdpa').to(device).eval()
    return model, proc

def chat_text(proc, text, n_images):
    content = [{'type': 'image'} for _ in range(n_images)] + [{'type': 'text', 'text': text}]
    return proc.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def generate(model, proc, texts, images, max_new_tokens=256, sample=False, bs=8):
    """texts[i] is the user text; images[i] a PIL image, a list of them, or None (uniform within a call)."""
    outs = []
    for b in range(0, len(texts), bs):
        imgs = [x if isinstance(x, list) else ([x] if x is not None else []) for x in images[b:b + bs]]
        prompts = [chat_text(proc, t, len(i)) for t, i in zip(texts[b:b + bs], imgs)]
        flat = [x for i in imgs for x in i]
        inp = proc(text=prompts, images=flat or None, padding=True, return_tensors='pt').to(model.device)
        seq = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=sample,
                             temperature=1.0 if sample else None, top_p=None, top_k=None)
        outs += proc.batch_decode(seq[:, inp['input_ids'].shape[1]:], skip_special_tokens=True)
    return outs

# ---------------------------------------------------------------- protocol
TAG = re.compile(r'<(answer|request)>(.*?)(?:</\1>|$)', re.S)

def parse(text):
    m = TAG.search(text)
    return (m.group(1), m.group(2).strip()) if m else ('invalid', text.strip())

def normalize(s):
    s = s.lower().translate(str.maketrans('', '', string.punctuation))
    return [w for w in s.split() if w not in {'a', 'an', 'the', 'of', 'in', 'on', 'and', 'is', 'was'}]

def f1(pred, gold):
    p, g = normalize(pred), normalize(gold)
    if not p or not g: return 0.0
    if ' '.join(g) in ' '.join(p) and len(p) <= len(g) + 6: return 1.0
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0: return 0.0
    pr, rc = common / len(p), common / len(g)
    return 2 * pr * rc / (pr + rc)

def load_qa(split):
    return [json.loads(l) for l in open(OUT / 'qa.jsonl') if json.loads(l)['split'] == split]

class Rollout:
    def __init__(self, model, proc):
        self.model, self.proc, self.tok = model, proc, proc.tokenizer
        self.im_end = self.tok.convert_tokens_to_ids('<|im_end|>')
        self.pad = self.tok.pad_token_id

    def prompt_inputs(self, texts, images):
        prompts = [chat_text(self.proc, t, 1) for t in texts]
        return self.proc(text=prompts, images=images, padding=True, return_tensors='pt')

    def _trim(self, ids):
        ids = [t for t in ids if t != self.pad]
        return ids[:ids.index(self.im_end) + 1] if self.im_end in ids else ids

    @torch.no_grad()
    def _gen(self, ids_list, pix, grid, max_new, sample):
        L = max(len(x) for x in ids_list)
        ids = torch.tensor([[self.pad] * (L - len(x)) + x for x in ids_list], device=self.model.device)
        att = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in ids_list], device=self.model.device)
        seq = self.model.generate(input_ids=ids, attention_mask=att, pixel_values=pix.to(self.model.device),
                                  image_grid_thw=grid.to(self.model.device), max_new_tokens=max_new,
                                  do_sample=sample, temperature=1.0 if sample else None, top_p=None, top_k=None)
        return [self._trim(s[L:].tolist()) for s in seq]

    def pixel_slices(self, inp):
        grid = inp['image_grid_thw']; n = grid.prod(-1).tolist(); out, a = [], 0
        for k in n: out.append(inp['pixel_values'][a:a + k]); a += k
        return out

    def run(self, items, images, env, template=PROTOCOL, sample=False, max1=64, max2=48, bs=16):
        """One episode per (item, image). Returns dicts with token ids for training and the parsed outcome."""
        res = []
        for b in range(0, len(items), bs):
            its, ims = items[b:b + bs], images[b:b + bs]
            texts = [template.format(q=it['question'], ctx=it.get('ctx_' + env, '')) for it in its]
            inp = self.prompt_inputs(texts, ims)
            att = inp['attention_mask']
            p_ids = [row[m.bool()].tolist() for row, m in zip(inp['input_ids'], att)]
            pix, grid = self.pixel_slices(inp), inp['image_grid_thw']
            g1 = self._gen(p_ids, inp['pixel_values'], grid, max1, sample)
            eps = []
            for it, p, g, px, gr in zip(its, p_ids, g1, pix, grid):
                kind, val = parse(self.tok.decode(g, skip_special_tokens=True))
                eps.append({'item': it, 'prompt_ids': p, 'gen1': g, 'kind': kind, 'text1': val,
                            'pix': px, 'grid': gr, 't2': None, 'gen2': None, 'answer': val if kind == 'answer' else ''})
            req = [e for e in eps if e['kind'] == 'request' and template is PROTOCOL]
            if req:
                ids2 = []
                for e in req:
                    if e['gen1'] and e['gen1'][-1] != self.im_end: e['gen1'] = e['gen1'] + [self.im_end]
                    e['t2'] = self.tok.encode(TURN2.format(ctx=e['item']['ctx_' + env]), add_special_tokens=False)
                    ids2.append(e['prompt_ids'] + e['gen1'] + e['t2'])
                g2 = self._gen(ids2, torch.cat([e['pix'] for e in req]), torch.stack([e['grid'] for e in req]), max2, sample)
                for e, g in zip(req, g2):
                    e['gen2'] = g
                    k2, v2 = parse(self.tok.decode(g, skip_special_tokens=True))
                    e['answer'] = v2 if k2 == 'answer' else v2[:100]
            for e in eps:
                e['f1'] = f1(e['answer'], e['item']['answer'])
            res += eps
        return res

def summarize(eps):
    out = {}
    for t in ('image', 'context'):
        E = [e for e in eps if e['item']['type'] == t]
        if not E: continue
        out[t] = {'n': len(E), 'request_rate': round(sum(e['kind'] == 'request' for e in E) / len(E), 3),
                  'invalid_rate': round(sum(e['kind'] == 'invalid' for e in E) / len(E), 3),
                  'f1': round(sum(e['f1'] for e in E) / len(E), 3),
                  'f1_if_request': round(sum(e['f1'] for e in E if e['kind'] == 'request') / max(1, sum(e['kind'] == 'request' for e in E)), 3),
                  'f1_if_answer': round(sum(e['f1'] for e in E if e['kind'] == 'answer') / max(1, sum(e['kind'] == 'answer' for e in E)), 3)}
    if 'image' in out and 'context' in out:
        tp = out['context']['request_rate'] * out['context']['n']; fp = out['image']['request_rate'] * out['image']['n']
        out['request_precision'] = round(tp / (tp + fp), 3) if tp + fp else None
        out['request_recall'] = out['context']['request_rate']
    return out

def episode_record(e):
    return {'qid': e['item']['qid'], 'type': e['item']['type'], 'kind': e['kind'], 'text1': e['text1'],
            'answer': e['answer'], 'gold': e['item']['answer'], 'f1': e['f1']}
