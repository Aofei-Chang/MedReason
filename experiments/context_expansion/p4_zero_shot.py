"""P4: zero-shot answer-or-request policy on the QA test split (no training). Also the step-0 point for P5.

Conditions: protocol with env=full / env=nonvis (request precision, recall, F1 per item type), and forced
answers with no context (direct), the full context given, and the non-visual context given.
"""
import argparse, json
from PIL import Image
from protocol import DIRECT, GIVEN, OUT, Rollout, episode_record, load_qa, load_vl, summarize

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True)
    ap.add_argument('--split', default='test')
    args = ap.parse_args()
    items = load_qa(args.split)
    model, proc = load_vl(args.model)
    ro = Rollout(model, proc)
    images = [Image.open(it['image']).convert('RGB') for it in items]
    res, eps_out = {'model': args.model, 'n': len(items)}, open(OUT / f'p4_{args.model}_episodes.jsonl', 'w')
    for name, env, tpl in [('protocol_full', 'full', None), ('protocol_nonvis', 'nonvis', None),
                           ('direct', 'full', DIRECT), ('given_full', 'full', GIVEN), ('given_nonvis', 'nonvis', GIVEN)]:
        eps = ro.run(items, images, env, **({'template': tpl} if tpl else {}))
        res[name] = summarize(eps); print(name, res[name], flush=True)
        for e in eps: eps_out.write(json.dumps({'condition': name, **episode_record(e)}) + '\n')
    (OUT / f'p4_{args.model}.json').write_text(json.dumps(res, indent=1))

if __name__ == '__main__':
    main()
