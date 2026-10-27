"""P2b crossover on the MedSAM3 box set: a perception aid (zoom) vs. visual and non-visual context.

Target = instruction answer, context pool = original caption (typed into visual / non-visual parts).
Conditions: I, 0, IC (full caption), IV (visual part), IN (non-visual part),
            Z (image + zoomed crop of the entity boxes), ZR (image + random crop of the same size: control).
Predicted interaction: non-visual context helps P sentences, zoom helps V sentences (most on sentences that
mention a boxed entity), the random crop helps neither. The boxes come from entities in the answer, so the
zoom is an oracle "look", the perception-side counterpart of privileged context.
"""
import argparse, json, random
from pathlib import Path
import torch
from PIL import Image
from score_conditions import Scorer, spans_of
from score_typed_context import OUT, caption_parts, load_labels

ZOOM_Q = 'The second image is a zoomed-in view of a region of the first image.\n\n{q}'

def crop_box(r, W, H):
    boxes = sorted(r['boxes'], key=lambda b: -b['score'])
    x1 = min(b['xywh'][0] for b in boxes); y1 = min(b['xywh'][1] for b in boxes)
    x2 = max(b['xywh'][0] + b['xywh'][2] for b in boxes); y2 = max(b['xywh'][1] + b['xywh'][3] for b in boxes)
    if (x2 - x1) * (y2 - y1) > 0.5:
        x, y, w, h = boxes[0]['xywh']; x1, y1, x2, y2 = x, y, x + w, y + h
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = max((x2 - x1) * 1.5, 0.2), max((y2 - y1) * 1.5, 0.2)
    return cx, cy, min(w, 1), min(h, 1)

def crop(img, cx, cy, w, h):
    W, H = img.size
    x1 = min(max(cx - w / 2, 0), 1 - w); y1 = min(max(cy - h / 2, 0), 1 - h)
    c = img.crop((int(x1 * W), int(y1 * H), int((x1 + w) * W), int((y1 + h) * H)))
    s = 448 / max(c.size)
    return c.resize((max(28, int(c.size[0] * s)), max(28, int(c.size[1] * s))))

def random_center(cx, cy, w, h, rng):
    best = None
    for _ in range(50):
        rx, ry = rng.uniform(w / 2, 1 - w / 2), rng.uniform(h / 2, 1 - h / 2)
        ox = max(0, min(cx + w / 2, rx + w / 2) - max(cx - w / 2, rx - w / 2))
        oy = max(0, min(cy + h / 2, ry + h / 2) - max(cy - h / 2, ry - h / 2))
        ov = ox * oy / (w * h)
        if best is None or ov < best[0]: best = (ov, rx, ry)
        if ov < 0.1: break
    return best[1], best[2], best[0]

class ZoomScorer(Scorer):
    def prompt_inputs(self, text, image):
        if not isinstance(image, list): return super().prompt_inputs(text, image)
        content = [{'type': 'image'} for _ in image] + [{'type': 'text', 'text': text}]
        prompt = self.proc.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True)
        return self.proc(text=[prompt], images=image, return_tensors='pt')

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True)
    ap.add_argument('--start', type=int, default=0); ap.add_argument('--end', type=int, default=None)
    args = ap.parse_args()
    torch.set_num_threads(4)
    recs = [json.loads(l) for l in open(OUT / 'box.jsonl')]
    lab = load_labels('box')
    out_path = OUT / f'scores_zoom_{args.model}_s{args.start}.jsonl'
    done = {json.loads(l)['id'] for p in OUT.glob(f'scores_zoom_{args.model}_s*.jsonl') for l in open(p)}
    sc = ZoomScorer(args.model)
    rng = random.Random(0)
    with open(out_path, 'a') as f:
        for i in range(args.start, args.end or len(recs)):
            r = recs[i]
            if r['id'] in done: continue
            img = Image.open(r['image']).convert('RGB')
            cx, cy, w, h = crop_box(r, *img.size)
            rx, ry, overlap = random_center(cx, cy, w, h, rng)
            z, zr = crop(img, cx, cy, w, h), crop(img, rx, ry, w, h)
            ans, q = r['answer'], r['question']
            spans = spans_of(ans, r['answer_sentences'])
            cv, cn = caption_parts(r, lab)
            row = {'id': r['id'], 'crop': [cx, cy, w, h], 'random_overlap': overlap,
                   'mentions_entity': [any(b['entity'].lower() in s.lower() for b in r['boxes']) for s in r['answer_sentences']]}
            row['I'], _ = sc.score(q, img, ans, spans)
            row['0'], _ = sc.score(q, None, ans, spans)
            row['IC'], _ = sc.score(f"Figure caption from the source article: {r['original_caption']}\n\n{q}", img, ans, spans)
            row['IV'] = sc.score(f'{cv}\n\n{q}', img, ans, spans)[0] if cv else None
            row['IN'] = sc.score(f'{cn}\n\n{q}', img, ans, spans)[0] if cn else None
            row['Z'], _ = sc.score(ZOOM_Q.format(q=q), [img, z], ans, spans)
            row['ZR'], _ = sc.score(ZOOM_Q.format(q=q), [img, zr], ans, spans)
            f.write(json.dumps(row) + '\n'); f.flush()
            if i % 50 == 0: print(i, flush=True)

if __name__ == '__main__':
    main()
