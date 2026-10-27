"""Driver for the awareness / privilege-decomposition run (V0-V6 of privilege_aware_design.md; no training).
Same dependency queue as run_overnight.py: GPUs 1-3, 2 slots each, hard deadline.
Status: aware/status.json; per-job logs: aware/logs/<job>.log; analysis: analyze_aware.py at the end.
"""
import json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
AW = Path('/data/aofei/output/ContextMLLM/aware'); LOGS = AW / 'logs'
PY = '/data/aofei/conda/env/lingshu/bin/python'
GPUS = [1, 2, 3]; SLOTS = 2
HOURS = float(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1][0].isdigit() else 8.0
T0 = time.time(); DEADLINE = T0 + HOURS * 3600
MODELS = ('qwen25vl', 'lingshu', 'huatuo')

def J(name, cmd, deps=(), timeout_min=360):
    return {'name': name, 'cmd': cmd, 'deps': list(deps), 'timeout': timeout_min * 60}

def jobs():
    # label_aware.py --what time was started by hand on GPU 1 (listed first); this job holds that slot
    js = [J('time_labels', ['bash', '-c', f'while [ ! -s {AW}/labels_time.jsonl ]; do '
                                         f'if grep -q Traceback {LOGS}/label_time.log; then {PY} label_aware.py --what time; break; fi; '
                                         f'sleep 15; done; sleep 10; test -s {AW}/labels_time.jsonl'], timeout_min=60),
          J('items', [PY, 'build_aware_items.py', 'items'], ['time_labels'], timeout_min=120)]
    for m in MODELS:  # central experiment first
        for k in range(2):
            js.append(J(f'aware_{m}_{k}', [PY, 'aware.py', '--model', m, '--shard', str(k), '--nshards', '2'], ['items']))
    js.append(J('genfree_huatuo', [PY, 'gen_free.py', '--model', 'huatuo']))
    for m in MODELS:
        for a, b in ((0, 600), (600, 1200)):
            js.append(J(f'decomp_d2k_{m}_{a}', [PY, 'token_decomp.py', '--model', m, '--set', 'd2k', '--start', str(a), '--end', str(b)], ['time_labels']))
    for m in MODELS:
        js.append(J(f'decomp_genfree_{m}', [PY, 'token_decomp.py', '--model', m, '--set', 'genfree'],
                    ['time_labels'] + (['genfree_huatuo'] if m == 'huatuo' else [])))
        js.append(J(f'decomp_mimic_{m}', [PY, 'token_decomp.py', '--model', m, '--set', 'mimic_test']))
        js.append(J(f'mimic_gen_{m}', [PY, 'mimic_p6.py', '--model', m, '--stage', 'gen', '--set', 'mimic_test']))
        js.append(J(f'decomp_box_{m}', [PY, 'token_decomp.py', '--model', m, '--set', 'box']))
        js.append(J(f'zoom_{m}', [PY, 'zoom_qa.py', '--model', m], ['items']))
    js.append(J('label_gen', [PY, 'label_aware.py', '--what', 'gen'], ['genfree_huatuo']))
    js.append(J('label_cmp', [PY, 'label_aware.py', '--what', 'cmp'], [f'mimic_gen_{m}' for m in MODELS]))
    return js

def adopt(todo, running, state, used):
    """--resume: keep jobs a previous driver finished, adopt its still-running jobs, requeue the rest."""
    old = json.loads((AW / 'status.json').read_text())['jobs']
    for j in list(todo):
        o = old.get(j['name'], {})
        if o.get('status') == 'done':
            state[j['name']] = o; todo.remove(j)
        elif o.get('status') == 'running':
            pid = subprocess.run(['pgrep', '-f', '^' + ' '.join(j['cmd']).replace('.', '[.]') + '$'],
                                 capture_output=True, text=True).stdout.split()
            if pid:
                running[j['name']] = (Adopted(int(pid[0])), j, o['gpu'], time.time())
                used[o['gpu']] += 1; state[j['name']] = {**o, 'status': 'running', 'adopted': True}; todo.remove(j)

class Adopted:
    """A job started by an earlier driver: poll by pid; its exit code is unknown (checked from its output)."""
    def __init__(self, pid): self.pid = pid
    def poll(self):
        try: os.kill(self.pid, 0); return None
        except OSError: return 0
    def kill(self):
        try: os.kill(self.pid, 9)
        except OSError: pass

def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    todo, running, state = jobs(), {}, {}
    used = {g: 0 for g in GPUS}
    if '--resume' in sys.argv: adopt(todo, running, state, used)
    def save():
        (AW / 'status.json').write_text(json.dumps({'elapsed_min': round((time.time() - T0) / 60, 1),
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
                used[g] -= 1; del running[name]
        for j in list(todo):
            st = [state.get(d, {}).get('status') for d in j['deps']]
            if any(s not in (None, 'running', 'done') for s in st):
                state[j['name']] = {'status': 'skipped (dependency failed)'}; todo.remove(j); continue
            if not all(s == 'done' for s in st): continue
            g = next((g for g in GPUS if used[g] < SLOTS), None)
            if g is None: break  # keep list order as priority
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), TOKENIZERS_PARALLELISM='false')
            p = subprocess.Popen(j['cmd'], cwd=HERE, env=env, stdout=open(LOGS / f"{j['name']}.log", 'w'), stderr=subprocess.STDOUT)
            running[j['name']] = (p, j, g, now); used[g] += 1; todo.remove(j)
            state[j['name']] = {'status': 'running', 'gpu': g, 'start_min': round((now - T0) / 60, 1), 'cmd': ' '.join(j['cmd'][1:])}
        save()
        if time.time() > DEADLINE + 5 * 60:
            for name, (p, *_) in running.items(): p.kill(); state[name]['status'] = 'killed (deadline)'
            break
        time.sleep(20)
    rc = subprocess.run([PY, 'analyze_aware.py'], cwd=HERE, stdout=open(LOGS / 'analysis.log', 'w'), stderr=subprocess.STDOUT).returncode
    state['analysis'] = {'status': 'done' if rc == 0 else f'failed({rc})'}; save()

if __name__ == '__main__':
    main()
