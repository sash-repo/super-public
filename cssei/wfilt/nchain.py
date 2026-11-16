#!/usr/bin/env python3
"""Maximal subalgebras beyond the exhaustive range: normaliser chains from random lines
<x> (and from n_L(s)), then greedy growth through the minimal submodules until maximal
(every step certified: M is maximal iff each minimal M-submodule generates L).
usage: nchain.py NAME NLINES CAP SEED"""
import sys, os, json, time, random, glob
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wall17 as E
name, nl, cap, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
L = E.load(name)
rng = random.Random(seed)
starts = []
jn = os.path.basename(E.ALGS[name])[:-5]
for f in glob.glob(os.path.join(E.BV, 'sand', 'sand_*.json')):
    for r in json.load(open(f)):
        if r['name'] == jn and r.get('s_basis'):
            sys.path.insert(0, os.path.join(E.BV, 'sand'))
            from sand import Alg, Span
            N = Alg(L.tab).normaliser(Span(r['s_basis']))
            starts.append(('ns', E.rref(N.basis())))
for _ in range(nl):
    starts.append(('line', E.W.closure(L, [rng.randrange(1, 1 << L.n)])))
maxes, t = {}, time.time()
os.makedirs(os.path.join(HERE, 'nchain'), exist_ok=True)
logf = open(os.path.join(HERE, 'nchain', name + '.log'), 'w')
def log(s):
    logf.write(s + '\n'); logf.flush()
skipped = {}
for how, A in starts:
    while True:
        N = [v for v in E.W.normaliser(L, A) if v]
        if len(N) == L.n or len(N) == len(A):
            break
        A = E.rref(N)
    if len(A) == L.n or not A:
        continue
    while True:
        if L.n - len(A) > cap:
            skipped[L.n - len(A)] = skipped.get(L.n - len(A), 0) + 1
            A = None
            break
        mins = E.minimal_submodules(L, A)
        rng.shuffle(mins)
        grown = False
        for N in mins:
            C = E.W.closure(L, N)
            if len(C) != L.n:
                A, grown = C, True
                break
        if not grown:
            break
    if A is not None:
        k = tuple(A)
        if k not in maxes:
            maxes[k] = (how, A)
            log('%s codim %d  [%.0fs]' % (how, L.n - len(A), time.time() - t))
rows = {}
for k, (how, M) in maxes.items():
    for w in E.wgradings_of(L, M):
        key = json.dumps([w['codim'], w['neg'], w['pos'], w['g0'], how])
        e = rows.setdefault(key, {'w': w, 'how': how, 'n': 0})
        e['n'] += 1
for e in sorted(rows.values(), key=lambda e: e['w']['codim']):
    w = e['w']; log('  codim %2d  g_- %s  g_>=0 %s  g0 %s  [%s x%d]' % (w['codim'], w['neg'], w['pos'], w['g0'], e['how'], e['n']))
log('skipped (codim > cap after the chain): %s' % skipped)
json.dump({'name': name, 'rows': list(rows.values()), 'skipped': skipped,
           'codims': sorted(L.n - len(M) for _, M in maxes.values())},
          open(os.path.join(HERE, 'nchain', name + '.json'), 'w'), indent=1)
