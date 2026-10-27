"""P3 step 2: a judge with the article (Qwen3-8B, no thinking) finds non-visual claims and checks them.

Inputs: free generations of each model, plus the PubMedVision reference answer (what training rewards).
Per text: the number of sentences, and a list of claims that state patient information not visible in the
image ('patient': age, sex, symptoms, history, labs, pathology, treatment, outcome, timing, comparison) or
a definitive diagnosis stated as fact ('diagnosis'), each SUPPORTED / CONTRADICTED / UNVERIFIED by the article.
"""
import argparse, json, re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from label_semantic import PATH
from protocol import OUT
from score_conditions import article_context

PROMPT = """You check a description of a medical image against its source article.

ARTICLE EXCERPT (figure caption, abstract, and paragraphs citing the figure):
<<<
Figure caption: {caption}
{article}
>>>

DESCRIPTION:
<<<
{text}
>>>

List every statement in the DESCRIPTION that is not a visual finding, of these two kinds:
- "patient": patient-specific information that cannot be seen in the image: age, sex, symptoms, history, laboratory or pathology results, treatment, outcome, timing, or a comparison with other scans or time points.
- "diagnosis": a definitive diagnosis or cause stated as fact (not hedged with words like "may", "could", "suggests", "consistent with").
Do not list visual findings (location, size, shape, density, signal) or general medical knowledge.
For each statement give a verdict against the ARTICLE EXCERPT: "SUPPORTED" (stated there), "CONTRADICTED" (the excerpt says otherwise), or "UNVERIFIED" (not mentioned).
Return only JSON: {{"n_sentences": <number of sentences in the DESCRIPTION>, "claims": [{{"kind": "patient|diagnosis", "claim": "...", "verdict": "SUPPORTED|CONTRADICTED|UNVERIFIED"}}]}}"""

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--models', nargs='+', default=['qwen25vl', 'lingshu'])
    ap.add_argument('--shard', type=int, default=0); ap.add_argument('--nshards', type=int, default=1)
    args = ap.parse_args()
    recs = {json.loads(l)['id']: json.loads(l) for l in open(OUT / 'd2k.jsonl')}
    texts = []
    for m in args.models:
        for l in open(OUT / f'gen_free_{m}.jsonl'):
            g = json.loads(l); texts.append((m, g['prompt'], g['id'], g['text']))
    pilot_ids = {g[2] for g in texts}
    texts += [('reference', 'q', i, recs[i]['answer']) for i in sorted(pilot_ids)]
    texts = texts[args.shard::args.nshards]
    out = OUT / f'judge_free_s{args.shard}.jsonl'
    done = {(x['source'], x['prompt'], x['id']) for x in map(json.loads, open(out))} if out.exists() else set()
    texts = [t for t in texts if t[:3] not in done]
    tok = AutoTokenizer.from_pretrained(PATH, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.bfloat16).cuda().eval()
    with open(out, 'a') as f:
        for b in range(0, len(texts), 16):
            batch = texts[b:b + 16]
            prompts = [tok.apply_chat_template([{'role': 'user', 'content': PROMPT.format(
                caption=recs[i]['original_caption'], article=article_context(recs[i], False), text=t[:2000])}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False) for _, _, i, t in batch]
            enc = tok(prompts, return_tensors='pt', padding=True).to('cuda')
            with torch.no_grad():
                seq = model.generate(**enc, max_new_tokens=600, do_sample=False, temperature=None, top_p=None, top_k=None)
            for (src, pr, i, _), s in zip(batch, tok.batch_decode(seq[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)):
                m = re.search(r'\{.*\}', s, re.S)
                try: js = json.loads(m.group(0))
                except Exception: js = None
                f.write(json.dumps({'source': src, 'prompt': pr, 'id': i, 'judge': js, 'raw': None if js else s}) + '\n')
            f.flush(); print(b, len(texts), flush=True)

if __name__ == '__main__':
    main()
