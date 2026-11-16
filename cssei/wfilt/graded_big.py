#!/usr/bin/env python3
"""For the big constituents with a known Z-grading (blocks/*.npz, 'deg'): is L_{>=0}
(resp. L_{<=0}) a maximal subalgebra, and which W-gradings does it define (every minimal
L_0-submodule of L/L_0)?  usage: graded_big.py NAME [NPZ]"""
import sys, os, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import wall17 as E

name = sys.argv[1]
npz = sys.argv[2] if len(sys.argv) > 2 else E.ALGS[name]
if len(sys.argv) > 2:
    E.ALGS[name] = npz
L = E.load(name)
deg = np.load(os.path.join(E.BV, npz))['deg'].tolist()
print('==', name, npz, 'dim', L.n, 'Jacobi', L.check_jacobi()[0], flush=True)
res = []
for side in ('>=0', '<=0'):
    t = time.time()
    M = E.rref([1 << i for i in range(L.n) if (deg[i] >= 0 if side == '>=0' else deg[i] <= 0)])
    assert L.is_subalgebra(M)
    if L.n - len(M) > 16:
        print('  L_%s: codim %d, too many cosets, skipped' % (side, L.n - len(M)), flush=True)
        continue
    ok, N = E.is_maximal(L, M)
    print('  L_%s: codim %d, maximal %s [%.0fs]' % (side, L.n - len(M), ok, time.time() - t), flush=True)
    # all maximal subalgebras containing M: recurse over the minimal submodules
    tops, stack, seen = {}, [M], set()
    while stack:
        K = stack.pop()
        if tuple(K) in seen:
            continue
        seen.add(tuple(K))
        grew = False
        for N in E.minimal_submodules(L, K):
            C = E.W.closure(L, N)
            if len(C) != L.n:
                stack.append(C); grew = True
        if not grew:
            tops[tuple(K)] = K
    for K in tops.values():
        ws = E.wgradings_of(L, K)
        print('   maximal above it: codim %d' % (L.n - len(K)), flush=True)
        for w in ws:
            print('    g_- %s  g_>=0 %s  g0 %s' % (w['neg'], w['pos'], w['g0']), flush=True)
        res.append({'side': side, 'codim0': L.n - len(M), 'Lge0_maximal': ok,
                    'codim': L.n - len(K), 'w': ws})
json.dump(res, open(os.path.join(HERE, 'graded_%s.json' % name), 'w'), indent=1)
