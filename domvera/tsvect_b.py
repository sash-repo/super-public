import sys, random
from fractions import Fraction
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import rank
from tsvect import run, vbr, flat

def study(n, t, funcs):
    s, alg, wt, basis = run(n, t)
    W = {}
    for w, p, D in alg: W.setdefault(w, []).append((p, D))
    zero = tuple([0]*(n-1))
    # h_alpha for odd roots
    hd = {}
    for w in W:
        if w == zero: continue
        mw = tuple(-x for x in w)
        if mw in W and any(p for p, _ in W[w]):
            brs = [flat(vbr(D1, D2, n), basis) for _, D1 in W[w] for _, D2 in W[mw]]
            hd[w] = rank(brs)
    res = [sorted(set(hd.values()))]
    for f in funcs:
        pos = [w for w in W if w != zero and sum(a*b for a, b in zip(f, w)) > 0]
        S = set(pos)
        indec = [w for w in pos if not any(tuple(a-b for a, b in zip(w, u)) in S for u in pos)]
        gens = [D for w in indec for _, D in W[w]]
        span = [flat(D, basis) for D in gens]; cur = list(gens)
        r = rank(span)
        while True:
            new = [vbr(a, b, n) for a in gens for b in cur]
            new = [D for D in new if D]
            span2 = span + [flat(D, basis) for D in new]
            r2 = rank(span2)
            if r2 == r: break
            span, cur, r = span2, new, r2
        npos = sum(len(W[w]) for w in pos)
        res.append((len(indec), r, npos))
    return res

random.seed(7)
for n in (4, 6):
    funcs = [[random.randint(-50, 50) for _ in range(n-1)] for _ in range(6 if n == 4 else 3)]
    for t in (0, 1):
        print(n, t, study(n, t, funcs), flush=True)
