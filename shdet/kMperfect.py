"""Is [g,g] of codimension > 0 in k^M(1|n)?  
Bracket {f,g} = K^M_f(g) - 2 D(f) g (the Ramond bracket).  Only the weight-0 space of ad K_t can be missed."""
import sympy as sp, itertools
from volM import t, mul, add, sc, dodd, dt, par
def setup(n):
    Z = n - 1; gens = list(range(n))
    def D(f): return add(dt(f), sc(-sp.Rational(1, 2) / t, mul({(Z,): 1}, dodd(f, Z))))
    def E(f): return add(*[mul({(g,): 1}, dodd(f, g)) for g in gens])
    def nabla(f): return add(sc(2, f), sc(-1, E(f)))
    def KMapply(f, g):
        s = (-1) ** par(f)
        out = mul(nabla(f), D(g))
        out = add(out, sc(s / t, mul(dodd(f, Z), dodd(g, Z))))
        for i in gens[:-1]: out = add(out, sc(s, mul(dodd(f, i), dodd(g, i))))
        out = add(out, mul(D(f), E(g)))
        return out
    def br(f, g): return add(KMapply(f, g), sc(-2, mul(D(f), g)))
    return gens, br
def run(n, W=7):
    gens, br = setup(n)
    mons = []
    for k in range(-W, W + 1):
        for d in range(n + 1):
            for I in itertools.combinations(gens, d):
                mons.append({tuple(I): t ** k})
    def weight(f):
        b = br({(): t}, f)
        if not b: return 0
        (m, v), = list(f.items())
        return sp.nsimplify(sp.simplify(b.get(m, 0) / v))
    wt = [weight(f) for f in mons]
    zero = [f for f, w in zip(mons, wt) if w == 0]
    keys = sorted({(m, sp.Poly(v, t).degree() if v.is_polynomial(t) else str(v)) for f in zero for m, v in f.items()}, key=str)
    # basis of weight-0 space: the monomials themselves
    basis = [list(f.items())[0] for f in zero]
    def coords(x):
        row = []
        for (m, v) in basis:
            c = x.get(m, 0)
            row.append(sp.simplify(c / v) if c != 0 else 0)
        return row
    rows = []
    for i, f in enumerate(mons):
        for j, g in enumerate(mons):
            if wt[i] + wt[j] == 0 and wt[i] >= 0:
                b = br(f, g)
                if b: rows.append(coords(b))
    M = sp.Matrix(rows) if rows else sp.zeros(1, len(basis))
    return len(basis), M.rank()
for n in (3, 4, 5):
    d, r = run(n, 5)
    print("k^M(1|%d): weight-0 space dim %d, brackets span %d" % (n, d, r))
