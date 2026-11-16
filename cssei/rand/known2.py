#!/usr/bin/env python3
"""More reference fingerprints in W(m;1): S(m;1) (m = 4, 5), Hamiltonian algebras
{H_f = sum_{ij} B_ij d_i f d_j} for B = Pi_{2n} (alternating, m = 4, 6) and B = 1_m
(non-alternating, m = 4, 5, 6), their simple factors.  -> known2.json"""
import json, numpy as np
import ambients as A
from lie import closure, table_of, factors
from known import fpr

def dm(a, i):
    return None if not (a >> i & 1) else a ^ (1 << i)

def ham(m, B, rng):
    amb = A.witt(m); N = amb.N
    gens = []
    for f in range(N):
        v = np.zeros(N * N, np.uint8)
        for i in range(m):
            for j in range(m):
                if B[i][j]:
                    b = dm(f, i)            # coefficient d_i f, field d_j
                    if b is not None: v ^= amb.basis[b * m + j]
        if v.any(): gens.append(v)
    E = closure(amb, gens, 10 ** 6); out = []
    ab = factors(table_of(amb, E), rng, out)
    return E.dim(), ab, out

def special(m, rng):
    amb = A.witt(m); N = amb.N; gens = []
    for a in range(N):
        for i in range(m):
            for j in range(i + 1, m):
                v = np.zeros(N * N, np.uint8)
                b = dm(a, j)
                if b is not None: v ^= amb.basis[b * m + i]
                b = dm(a, i)
                if b is not None: v ^= amb.basis[b * m + j]
                if v.any(): gens.append(v)
    E = closure(amb, gens, 10 ** 6); out = []
    ab = factors(table_of(amb, E), rng, out)
    return E.dim(), ab, out

if __name__ == '__main__':
    rng = np.random.default_rng(3); res = {}
    for m in (4, 5):
        dim, ab, out = special(m, rng)
        for o in out: res['S(%d;1)' % m] = fpr(o[0], rng)
        print('S(%d;1)' % m, dim, ab, res.get('S(%d;1)' % m), flush=True)
    for n in (2, 3):
        m = 2 * n
        B = [[1 if abs(i - j) == n else 0 for j in range(m)] for i in range(m)]
        dim, ab, out = ham(m, B, rng)
        for o in out: res['h_Pi(%d;1)' % m] = fpr(o[0], rng)
        print('h_Pi(%d;1)' % m, dim, ab, res.get('h_Pi(%d;1)' % m), flush=True)
    for m in (4, 5, 6):
        B = [[int(i == j) for j in range(m)] for i in range(m)]
        dim, ab, out = ham(m, B, rng)
        for o in out: res['h_I(%d;1)' % m] = fpr(o[0], rng)
        print('h_I(%d;1)' % m, dim, ab, res.get('h_I(%d;1)' % m), flush=True)
    json.dump(res, open('known2.json', 'w'), indent=0)
