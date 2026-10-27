"""Independent semantic label per target sentence with a text-only LLM (Qwen3-8B, no thinking).

V visual finding observable in the image; P patient/clinical information not visible in an image;
K general medical knowledge or explanation; M figure meta-text (panels, arrows, staining, layout).
Scored by next-token probability over the four letters; no generation.
"""
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = Path('/data/aofei/output/ContextMLLM/pilot_needs_context')
PATH = str(next(Path('/data/aofei/cache/huggingface/models--Qwen--Qwen3-8B/snapshots').iterdir()))
PROMPT = """A sentence from a description of a medical image (modality: {modality}) is given. Classify what the sentence mainly states.
A: a visual finding that a radiologist could observe in this image itself (anatomy, appearance, location, size, signal/density, laterality).
B: patient or clinical information that cannot be seen in this single image (age, sex, history, symptoms, lab or pathology results, treatment, outcome, timing, comparison with other scans or time points, diagnosis established by other tests).
C: general medical knowledge or explanation, not specific to this image or patient.
D: figure meta-information (panel letters, arrows, staining, scale bars, image layout) or a generic statement.

Sentence: "{s}"
Answer with a single letter."""

def main():
    tok = AutoTokenizer.from_pretrained(PATH)
    model = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.bfloat16).cuda().eval()
    letters = [tok.encode(x, add_special_tokens=False)[0] for x in 'ABCD']
    out = open(BASE / 'semantic_labels.jsonl', 'w')
    for line in open(BASE / 'pilot.jsonl'):
        r = json.loads(line)
        row = {'id': r['id']}
        for key, sents in [('A', r['answer_sentences']), ('B', r['caption_sentences'])]:
            probs = []
            for s in sents:
                msg = [{'role': 'user', 'content': PROMPT.format(modality=r['modality'], s=s)}]
                text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                ids = tok(text, return_tensors='pt').to('cuda')
                with torch.no_grad():
                    logits = model(**ids).logits[0, -1].float()
                probs.append(torch.softmax(logits[letters], 0).tolist())
            row[key] = probs
        out.write(json.dumps(row) + '\n'); out.flush()

if __name__ == '__main__':
    main()
