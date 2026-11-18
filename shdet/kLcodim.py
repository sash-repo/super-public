"""Control for kMperfect.py: the same test on k^L(1|n) (NS bracket: D = d_t, no 1/t), where k^L(1|4) must show codim 1."""
import sympy as sp, itertools
import kMperfect as derived
from volM import t, mul, add, sc, dodd, dt, par
def setupL(n):
    gens = list(range(n))
    def D(f): return dt(f)
    def E(f): return add(*[mul({(g,): 1}, dodd(f, g)) for g in gens])
    def nabla(f): return add(sc(2, f), sc(-1, E(f)))
    def br(f, g):
        s = (-1) ** par(f)
        out = mul(nabla(f), D(g))
        for i in gens: out = add(out, sc(s, mul(dodd(f, i), dodd(g, i))))
        out = add(out, mul(D(f), E(g)))
        return add(out, sc(-2, mul(D(f), g)))
    return gens, br
derived.setup = setupL
for n in (3, 4, 5):
    d, r = derived.run(n, 5)
    print("k^L(1|%d): weight-0 space dim %d, brackets span %d" % (n, d, r))
