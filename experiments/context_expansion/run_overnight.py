"""Driver for the validation experiments P1b-P4 and P6: a dependency queue over GPUs 1-3 (2 slots each)
with a hard deadline. No training (the RL pilot P5 is out of scope for now).
Status: overnight/status.json; per-job logs: overnight/logs/<job>.log.
"""
import json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path('/data/aofei/output/ContextMLLM/overnight'); LOGS = OUT / 'logs'
PY = '/data/aofei/conda/env/lingshu/bin/python'
GPUS = [3, 1, 2]; SLOTS = 2
HOURS = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
T0 = time.time(); DEADLINE = T0 + HOURS * 3600

def J(name, cmd, deps=(), units=1, timeout_min=300):
    return {'name': name, 'cmd': cmd, 'deps': list(deps), 'units': units, 'timeout': timeout_min * 60}

def jobs():
    # the labeller was started by hand before the driver; wait for its output, rerun it if it crashed
    js = [J('labels', ['bash', '-c', f'while [ ! -s {OUT}/labels.jsonl ]; do '
                                    f'if grep -q Traceback {LOGS}/label_sentences.log; then {PY} label_sentences.py; break; fi; '
                                    f'sleep 15; done; sleep 20; test -s {OUT}/labels.jsonl'], units=1, timeout_min=100),
          # qa_build.py was also started by hand (on GPU 3, listed first); this job holds its slot until it finishes
          J('qa', ['bash', '-c', f'while [ ! -s {OUT}/qa_stats.json ]; do '
                                f'if grep -q Traceback {LOGS}/qa.log; then {PY} qa_build.py; break; fi; sleep 15; done'],
            ['labels'], timeout_min=90)]
    n, k = 2000, 3
    for m in ('qwen25vl', 'lingshu'):
        for s in range(k):
            a, b = s * n // k, (s + 1) * n // k
            js.append(J(f'typed_{m}_{s}', [PY, 'score_typed_context.py', '--model', m, '--start', str(a), '--end', str(b)], ['labels']))
    for m in ('qwen25vl', 'lingshu'):
        js.append(J(f'zoom_{m}', [PY, 'score_zoom.py', '--model', m], ['labels']))
        js.append(J(f'genfree_{m}', [PY, 'gen_free.py', '--model', m]))
        js.append(J(f'p4_{m}', [PY, 'p4_zero_shot.py', '--model', m], ['qa'], timeout_min=120))
        js.append(J(f'mimic_score_{m}', [PY, 'mimic_p6.py', '--model', m, '--stage', 'score']))
        js.append(J(f'mimic_gen_{m}', [PY, 'mimic_p6.py', '--model', m, '--stage', 'gen']))
    for s in range(2):
        js.append(J(f'judge_{s}', [PY, 'judge_free.py', '--shard', str(s), '--nshards', '2'], ['genfree_qwen25vl', 'genfree_lingshu']))
    return js

def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    todo, running, state = jobs(), {}, {}
    used = {g: 0 for g in GPUS}
    def save():
        (OUT / 'status.json').write_text(json.dumps({'elapsed_min': round((time.time() - T0) / 60, 1),
                                                     'deadline_in_min': round((DEADLINE - time.time()) / 60, 1),
                                                     'jobs': state}, indent=1))
    while todo or running:
        now = time.time()
        for name, (p, j, g, t) in list(running.items()):
            rc = p.poll()
            if rc is None and now - t > j['timeout']: p.kill(); rc = 'timeout'
            if rc is None and now > DEADLINE - 10 * 60: p.kill(); rc = 'deadline'
            if rc is not None:
                state[name].update(status='done' if rc == 0 else f'failed({rc})', min=round((now - t) / 60, 1))
                used[g] -= j['units']; del running[name]
        for j in list(todo):
            st = [state.get(d, {}).get('status') for d in j['deps']]
            if any(s not in (None, 'running', 'done') for s in st):
                state[j['name']] = {'status': 'skipped (dependency failed)'}; todo.remove(j); continue
            if not all(s == 'done' for s in st): continue
            g = next((g for g in GPUS if used[g] + j['units'] <= SLOTS), None)
            if g is None: continue
            cmd = list(j['cmd'])
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), TOKENIZERS_PARALLELISM='false')
            p = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=open(LOGS / f"{j['name']}.log", 'w'), stderr=subprocess.STDOUT)
            running[j['name']] = (p, j, g, now); used[g] += j['units']; todo.remove(j)
            state[j['name']] = {'status': 'running', 'gpu': g, 'start_min': round((now - T0) / 60, 1), 'cmd': ' '.join(cmd[1:])}
        save()
        if time.time() > DEADLINE + 5 * 60:
            for name, (p, *_ ) in running.items(): p.kill(); state[name]['status'] = 'killed (deadline)'
            break
        time.sleep(20)
    rc = subprocess.run([PY, 'analyze_overnight.py'], cwd=HERE, stdout=open(LOGS / 'analysis.log', 'w'), stderr=subprocess.STDOUT).returncode
    state['analysis'] = {'status': 'done' if rc == 0 else f'failed({rc})'}; save()

if __name__ == '__main__':
    main()
