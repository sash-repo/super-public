"""Does the span of the generators of kas^L inside k^L(1|6) close, and is it
isotropic for the pairing Res(fg) on k^L(1|6)?  Labels: xi1,xi2,xi3 = 1,2,3; eta3,eta2,eta1 = 4,5,6."""
import sys, itertools
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from stringy import mono, add, smul, mul, parity, d_t, d_odd, eq
from fractions import Fraction as Fr
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import rank
XI = {1: 1, 2: 2, 3: 3}; ETA = {1: 6, 2: 5, 3: 4}
th = [1, 2, 3, 4, 5, 6]
hat = {1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1}
Theta = mono(0, (1, 2, 3, 4, 5, 6))
def E(h):
    out = {}
    for (a, S), c in h.items():
        out = add(out, {(a, S): c * len(S)})
    return out
def kbx(f, g, c):
    nab = lambda h: add(smul(2, h), smul(-1, E(h)))
    out = add(mul(nab(f), d_t(g)), smul(-1, mul(d_t(f), nab(g))))
    s = (-1) ** parity(f)
    for i in (1, 2, 3):
        x, y = XI[i], ETA[i]
        out = add(out, smul(-c * s, add(mul(d_odd(x, f), d_odd(y, g)), mul(d_odd(y, f), d_odd(x, g)))))
    return out
def tpow(f, k):
    return {(a + k, S): v for (a, S), v in f.items()}
def gens(n, sg):
    out = []
    out.append(add(mono(n), smul(sg * n * (n - 1) * (n - 2), tpow(Theta, n - 3))))
    for i in th:
        out.append(add(mono(n, (i,)), smul(sg * n * (n - 1), tpow(d_odd(hat[i], Theta), n - 2))))
    for i, j in itertools.combinations(th, 2):
        out.append(add(mono(n, (i, j)), smul(sg * n, tpow(d_odd(hat[i], d_odd(hat[j], Theta)), n - 1))))
    x1, x2, x3, e1, e2, e3 = 1, 2, 3, 6, 5, 4
    S = [add(mono(0, (x1, x2, e2)), mono(0, (x1, x3, e3))), add(mono(0, (x2, x1, e1)), mono(0, (x2, x3, e3))),
         add(mono(0, (x3, x1, e1)), mono(0, (x3, x2, e2))),
         add(mono(0, (e1, x2, e2)), smul(-1, mono(0, (e1, x3, e3)))), add(mono(0, (e2, x1, e1)), smul(-1, mono(0, (e2, x3, e3)))),
         add(mono(0, (e3, x1, e1)), smul(-1, mono(0, (e3, x2, e2)))),
         mono(0, (x1, x2, x3)), mono(0, (x1, e2, e3)), mono(0, (x2, e1, e3)), mono(0, (x3, e1, e2))]
    out += [tpow(s, n) for s in S]
    return out
from exactlin import solve
def member(h, sg):
    h = dict(h)
    pw = {a for (a, S) in h}
    for a in sorted(pw):
        pass
    # leading monomials with |S|<=2
    for (a, S), c in list(h.items()):
        if len(S) == 0:
            h = add(h, smul(-c, gens(a, sg)[0]))
    for (a, S), c in list(h.items()):
        if len(S) == 1 and c:
            g = gens(a, sg)[1 + th.index(S[0])]
            h = add(h, smul(-h.get((a, S), 0), g))
    pairs = list(itertools.combinations(th, 2))
    for (a, S), c in list(h.items()):
        if len(S) == 2 and h.get((a, S), 0):
            g = gens(a, sg)[7 + pairs.index(S)]
            h = add(h, smul(-h.get((a, S), 0), g))
    for a in sorted({a for (a, S) in h if len(S) == 3}):
        Sg = gens(a, sg)[22:]
        keys3 = sorted({k for g in Sg for k in g} | {k for k in h if len(k[1]) == 3 and k[0] == a})
        M = [[g.get(k, 0) for g in Sg] for k in keys3]
        x = solve(M, [h.get(k, 0) for k in keys3])
        if x is None:
            return False
        for xi, g in zip(x, Sg):
            h = add(h, smul(-xi, g))
    return not h
for sg in (1, -1):
    B = [g for n in range(-3, 5) for g in gens(n, sg)]
    iso = all(mul(f, g).get((-1, (1, 2, 3, 4, 5, 6)), 0) == 0 for f in B for g in B)
    for c in (1, -1):
        small = [g for n in range(-1, 3) for g in gens(n, sg)]
        bad = sum(1 for f in small for g in small if not member(kbx(f, g, c), sg))
        print("sign", sg, "bracket c", c, "isotropic", iso, "non-closing brackets", bad, "of", len(small)**2, flush=True)
