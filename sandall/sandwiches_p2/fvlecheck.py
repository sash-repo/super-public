"""does the normalizer of the homogeneous-sandwich subalgebra S (dim 52) of F vle lie
in L_{>=0} of a W-grading of the census?  And the normalizer chain of S."""
import os as _os_runs; _os_runs.makedirs('runs', exist_ok=True)  # logs go to ./runs
import itertools, pickle
import fvle
from fvle import graded_pieces, normalize, irw, echelon, in_span, bracket, wtest
from fvlesand import L4
import ir
from math import gcd

basis = fvle.build(); n = len(basis)
tms = [irw.terms(v) for v in basis]
dat = pickle.load(open('runs/fvlesand.pkl', 'rb'))
S = dat['S']
def normalizer(Sv):
    SE = echelon(Sv)
    rec, ker = [], []
    for j, x in enumerate(basis):
        key = 0
        for s in Sv:
            key = (key << 896) | ir.reduce_vec(bracket(x, s), SE)
        c = 1 << j
        for p, b, cc in rec:
            if key >> p & 1:
                key ^= b; c ^= cc
        if key:
            rec.append((key.bit_length() - 1, key, c)); rec.sort(key=lambda t: -t[0])
        else:
            v = 0
            for i in range(n):
                if c >> i & 1:
                    v ^= basis[i]
            ker.append(v)
    return ker
N = normalizer(S)
print('dim n(S)', len(N))
NN = normalizer(N)
print('dim n(n(S))', len(NN))
# containment in L_{>=0} of the W-gradings
found = 0
for lam in itertools.product(range(-2, 3), repeat=4):
    d = tuple(sum(l * v[i] for l, v in zip(lam, L4)) for i in range(7))
    if not any(d): continue
    p = normalize(graded_pieces(basis, tms, d))
    ok, _ = wtest(p)
    if not ok: continue
    L0 = echelon([v for c, vs in p.items() if c >= 0 for v in vs])
    if all(in_span(x, L0) for x in N):
        found += 1
        print('n(S) inside L_>=0 of d =', d, 'codim', sum(len(vs) for c, vs in p.items() if c < 0))
    sS = all(in_span(x, L0) for x in S)
print('W-gradings whose L_>=0 contains n(S):', found)
