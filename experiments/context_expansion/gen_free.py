"""P3 step 1: free generation with no context, on the 600 pilot images.

Prompts: 'q' the original PubMedVision instruction question; 'd' a neutral "describe" request.
Greedy, 320 new tokens. The judge (judge_free.py) then extracts non-visual claims and checks them against the article.
"""
import argparse, json
from PIL import Image
from protocol import OUT, generate, load_vl

DESCRIBE = 'Describe this medical image in detail.'

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True)
    ap.add_argument('--n', type=int, default=600)
    args = ap.parse_args()
    recs = [json.loads(l) for l in open(OUT / 'd2k.jsonl')][:args.n]
    model, proc = load_vl(args.model, max_pixels=1024 * 28 * 28)
    out = OUT / f'gen_free_{args.model}.jsonl'
    done = {(json.loads(l)['id'], json.loads(l)['prompt']) for l in open(out)} if out.exists() else set()
    with open(out, 'a') as f:
        for key in ('q', 'd'):
            todo = [r for r in recs if (r['id'], key) not in done]
            for b in range(0, len(todo), 8):
                rs = todo[b:b + 8]
                texts = [r['question'] if key == 'q' else DESCRIBE for r in rs]
                gens = generate(model, proc, texts, [Image.open(r['image']).convert('RGB') for r in rs], max_new_tokens=320)
                for r, g in zip(rs, gens):
                    f.write(json.dumps({'id': r['id'], 'prompt': key, 'text': g}) + '\n')
                f.flush(); print(key, b, flush=True)

if __name__ == '__main__':
    main()
