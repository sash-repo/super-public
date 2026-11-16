import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from tsvect import run, vbr, div, vapply, padd
# does ad E (E = sum xi_i d_i) preserve svect~(0|n), n even, t=1?  and the reason
for n in (4, 6):
    s, alg, wt, basis = run(n, 1)
    E = {i: {(i,): 1} for i in range(n)}
    top = tuple(range(n))
    bad = 0
    for w, p, D in alg:
        B = vbr(E, D, n)
        c = div(B)
        for m, x in vapply(B, {top: 1}).items(): padd(c, m, s*x)
        if c: bad += 1
    print(n, "elements D with [E,D] outside svect~:", bad, "of", len(alg))
