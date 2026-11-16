#!/usr/bin/env python3
r"""dim Der J for the simple core J of a row (route A realization), J graded by
degree and weight, so Der J = sum over (degree, weight) shifts; each shift is a
linear system over F_2.  An isomorphism invariant, used to decide whether rows
of different families give the same algebra at N~ = 1.

usage: python3 der.py ROW [ROW ...]
"""
import os, sys, json
from collections import defaultdict
from f2vf import Space, parse_field, find_weights, kernel_bits
from prolong import prolong, generators, ad_closure, log
from run_row import names_of, needs_height
from absutil import Coords
import rows as R


def core(name):
    spec = R.ROWS[name]
    nm = names_of(spec)
    m = len(spec['coords'])
    complete = spec['g0'] == 'complete'
    sp0 = Space([1] * m, spec['degs'], spec['par'])
    neg = [parse_field(sp0, s, nm) for s in spec['neg']]
    g0s = [] if complete else [s for s in spec['g0'] if needs_height(s) == 1]
    g0 = [parse_field(sp0, s, nm) for s in g0s]
    W = find_weights(sp0, neg + g0)
    sp = Space([1] * m, spec['degs'], spec['par'], W)
    neg = [parse_field(sp, s, nm) for s in spec['neg']]
    g0 = [parse_field(sp, s, nm) for s in g0s]
    G = prolong(sp, neg, None if complete else g0, verbose=False)
    J = ad_closure(sp, generators(G), G.basis(-1))
    return sp, G, J


def derdim(sp, J):
    B = J.basis()
    n = len(B)
    lab = [(sp.kdeg(next(iter(v))), sp.kwt(next(iter(v)))) for v in B]
    co = Coords(B)
    br = {}
    for i in range(n):
        for j in range(i, n):
            c = co(sp.bracket(B[i], B[j]))
            br[(i, j)] = br[(j, i)] = c
    # shifts
    shifts = set()
    for a in lab:
        for b in lab:
            shifts.add((b[0] - a[0], tuple(x - y for x, y in zip(b[1], a[1]))))
    bylab = defaultdict(list)
    for i, l in enumerate(lab):
        bylab[l].append(i)
    total = 0
    detail = {}
    for (s, w) in shifts:
        # unknowns: (i, j) with lab[j] = lab[i] + (s, w)
        unk = []
        for i in range(n):
            tgt = (lab[i][0] + s, tuple(x + y for x, y in zip(lab[i][1], w)))
            for j in bylab.get(tgt, ()):
                unk.append((i, j))
        if not unk:
            continue
        uidx = {u: t for t, u in enumerate(unk)}
        # D(b_i) = sum_j c_ij b_j.  Equation for pair (a,b):
        # D([a,b]) + [D a, b] + [a, D b] = 0, coefficientwise
        # build column images: for each unknown (i,j) its contribution vector
        imgs = [0] * len(unk)
        eq = 0
        rows_of = {}
        for a in range(n):
            for b in range(a + 1, n):
                c = br[(a, b)]
                # contributions to the equation (a,b): vector over basis index k
                # from D([a,b]): for i in bits(c), unknown (i,j) contributes e_j
                contrib = defaultdict(int)
                t = c
                i = 0
                while t:
                    if t & 1:
                        for j in bylab.get((lab[i][0] + s, tuple(x + y for x, y in zip(lab[i][1], w))), ()):
                            contrib[uidx[(i, j)]] ^= 1 << j
                    t >>= 1
                    i += 1
                for j in bylab.get((lab[a][0] + s, tuple(x + y for x, y in zip(lab[a][1], w))), ()):
                    contrib[uidx[(a, j)]] ^= br[(j, b)]
                for j in bylab.get((lab[b][0] + s, tuple(x + y for x, y in zip(lab[b][1], w))), ()):
                    contrib[uidx[(b, j)]] ^= br[(a, j)]
                for u, vec in contrib.items():
                    if vec:
                        imgs[u] |= vec << (eq * n)
                eq += 1
        k = len(kernel_bits(imgs))
        if k:
            detail[str((s, w))] = k
        total += k
    return total, detail


if __name__ == '__main__':
    out = {}
    for name in sys.argv[1:]:
        sp, G, J = core(name)
        d, detail = derdim(sp, J)
        log('%-8s dim J = %d, dim G = %d, dim Der J = %d, dim Der J / ad J = %d'
            % (name, J.dim(), G.dim(), d, d - J.dim()))
        out[name] = {'J': J.dim(), 'G': G.dim(), 'Der J': d}
    os.makedirs('results', exist_ok=True)
    json.dump(out, open('results/der_%s.json' % '_'.join(n.replace('~', 't') for n in sys.argv[1:]), 'w'), indent=1)
