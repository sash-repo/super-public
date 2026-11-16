#!/usr/bin/env python3
r"""The lowest-weight vector v_1 of g_1 printed in [BGLLS, Subsec. 18.1] for the row kas~(7).

1) v_1 has no divided power, so it would lie in g_1 for every shearing vector iff [v_1, d_i]
   lies in g_0 for every i; this is tested (g_0 being that of the row, independent of N).
2) Every modification of v_1 by at most two terms x_a x_b d_k (a < b; a term is deleted if
   present, added if absent) is tested for membership in g_1 at N = (1,...,1); the
   modifications that lie in g_1 are printed.

usage: python3 misprint.py"""
import itertools
from f2vf import Space, parse_field
from prolong import prolong
from run_row import names_of, needs_height
import rows as R

name = 'kas~7'
spec = R.ROWS[name]; P = R.PARTIAL[name]
nm = names_of(spec)
for a, b in P['rename'].items():
    nm[a] = nm[b]
m = len(spec['coords'])
g0s = [s for s in spec['g0'] if needs_height(s) == 1]
sp = Space([1] * m, spec['degs'], spec['par'])          # no weight grading: v_1 may be modified freely
neg = [parse_field(sp, s, nm) for s in spec['neg']]
g0 = [parse_field(sp, s, nm) for s in g0s]
G = prolong(sp, neg, g0, verbose=False)
print('%s: g_0 of dim %d, g_1 of dim %d at N = 1' % (name, G.dim(0), G.dim(1)))

terms = P['vecs']['v1'].split(' + ')
v1 = parse_field(sp, P['vecs']['v1'], nm)
print('printed v_1 lies in g_1:', G.contains(v1))
bad = []
for i in range(m):
    d = parse_field(sp, 'dx%d' % (i + 1), nm)
    if not G.contains(sp.bracket(v1, d)):
        bad.append(i + 1)
print('[v_1, d_i] outside g_0 for i =', bad)

cand = ['x%d x%d dx%d' % (a, b, k) for a in range(1, m + 1) for b in range(a + 1, m + 1)
        for k in range(1, m + 1)]
base = set(terms)
sols = []
for r in (1, 2):
    for ch in itertools.combinations(cand, r):
        new = base ^ set(ch)
        if not new:
            continue
        v = parse_field(sp, ' + '.join(sorted(new)), nm)
        if G.contains(v):
            sols.append(ch)
print('modifications by at most two terms that lie in g_1:', len(sols))
for ch in sols:
    print('   ', ' ; '.join(('delete ' if t in base else 'add ') + t for t in ch))
