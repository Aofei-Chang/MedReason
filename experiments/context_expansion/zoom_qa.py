"""V6b: does a zoomed crop help short-answer accuracy? (P2b found no likelihood effect on long answers.)

Items: aware/zoom_items.jsonl (question about a MedSAM3-boxed structure, answer from the caption).
Conditions: I (image only), Z (image + crop around the boxes), ZR (image + random crop of the same size).
Greedy answer token F1 and teacher-forced reference log-likelihood per condition.
"""
import argparse, json, random
import torch
from aware import ANS
from aware_common import AW, TF, load_image
from protocol import f1
from score_zoom import ZOOM_Q, crop, crop_box, random_center

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    items = [json.loads(l) for l in open(AW / 'zoom_items.jsonl')]
    tf = TF(args.model, max_pixels=512 * 28 * 28)
    rng = random.Random(0)
    with open(AW / f'zoom_qa_{args.model}.jsonl', 'w') as f:
        for b in range(0, len(items), 8):
            its = items[b:b + 8]
            views = {'I': [], 'Z': [], 'ZR': []}
            for it in its:
                img = load_image(it['image'])
                cx, cy, w, h = crop_box(it, *img.size)
                rx, ry, _ = random_center(cx, cy, w, h, rng)
                views['I'].append(img); views['Z'].append([img, crop(img, cx, cy, w, h)]); views['ZR'].append([img, crop(img, rx, ry, w, h)])
            rows = [{'qid': it['qid'], 'answer': it['answer']} for it in its]
            for v, ims in views.items():
                texts = [(ANS if v == 'I' else ZOOM_Q.format(q=ANS)).format(q=it['question']) for it in its]
                for r, g, it in zip(rows, tf.generate(texts, ims), its):
                    r[f'ans_{v}'] = g[0][0]; r[f'f1_{v}'] = f1(g[0][0], it['answer'])
                res, _ = tf.score_batch(texts, ims, [it['answer'] for it in its])
                for r, (s, n) in zip(rows, res): r[f'lp_{v}'] = s / n
            for r in rows: f.write(json.dumps(r) + '\n')
            f.flush(); print(b, len(items), flush=True)

if __name__ == '__main__':
    main()
