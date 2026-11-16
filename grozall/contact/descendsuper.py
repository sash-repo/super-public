"""The same descent one odd indeterminate down: k(2n+1|m-1) inside k(2n+1|m).

descend.py answers the question in the even direction; this file answers it in the odd
one, for the WEIGHTED DENSITIES, which is what the supercontact paper's censuses are
about.  The subalgebra is that of the generating functions free of xi_m (restrict.py
verified that it is one, and is k(2n+1|m-1)), the submanifold is N = {xi_m = 0}, and

    pi^*   reads a function of the small manifold on the big one,
    iota^* sets xi_m = 0.

The claim to be tested is the one proved in the even case: on F_lam BOTH maps are
equivariant, because the only place xi_m enters the coefficients of K_f, for f free of
xi_m, is the term (1/2) f_t xi_m d_{xi_m}, which is vertical for pi and vanishes on N.
The orthonormal shape is used, so that dropping xi_m leaves the Gram matrix invertible.

Exact rational arithmetic.  Run:  python descendsuper.py
"""

import os, sys
from fractions import Fraction as Fr

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from kmix import KMix, Dens, add, scal  # noqa: E402

LOG = None


def say(s):
    print(s, flush=True)
    if LOG:
        LOG.write(s + "\n")
        LOG.flush()


class SDescent:
    """(pi^*, iota^*) between k(2n+1|m-1) and k(2n+1|m), m >= 1."""

    def __init__(self, n, m):
        assert m >= 1
        self.n, self.m = n, m
        self.big = KMix(n, m)
        self.small = KMix(n, m - 1)

    def up(self, f):
        """A function of the small algebra, read on the big one (same key shape)."""
        return dict(f)

    def down(self, f):
        """iota^*: drop every term containing xi_m."""
        out = {}
        for (e, od), v in f.items():
            if self.m in od:
                continue
            key = (e, od)
            nv = out.get(key, Fr(0)) + v
            if nv:
                out[key] = nv
            elif key in out:
                del out[key]
        return out


def _fields(alg, deg, count, seed0=17):
    """A few dense functions of the algebra's own coordinate ring."""
    import random
    out = []
    mons = []
    for d in range(deg + 1):
        mons += alg.monomials_of_degree(d)
    for s in range(count):
        rnd = random.Random(seed0 + s)
        F = {}
        for _ in range(12):
            mm = mons[rnd.randrange(len(mons))]
            for k, v in mm.items():
                F[k] = F.get(k, Fr(0)) + Fr(rnd.randint(1, 5)) * v
        out.append({k: v for k, v in F.items() if v})
    return out


def check(n, m, lam=Fr(1, 3), deg=3, nfields=3, imax=2):
    D = SDescent(n, m)
    Mb, Ms = Dens(D.big, lam), Dens(D.small, lam)
    Fs = _fields(D.small, deg, nfields)
    FB = _fields(D.big, deg, nfields, seed0=500)
    gens = []
    for i in range(-2, imax + 1):
        gens += D.small.k_piece(i)
    badpi = badio = 0
    for f in gens:
        fb = D.up(f)
        for F in Fs:                                   # pi^* equivariance
            d = dict(Mb.act(fb, D.up(F)))
            for k, v in D.up(Ms.act(f, F)).items():
                d[k] = d.get(k, Fr(0)) - v
            if any(d.values()):
                badpi += 1
        for F in FB:                                   # iota^* equivariance
            d = dict(D.down(Mb.act(fb, F)))
            for k, v in Ms.act(f, D.down(F)).items():
                d[k] = d.get(k, Fr(0)) - v
            if any(d.values()):
                badio += 1
    say("  k(%d|%d) -> k(%d|%d), lam=%-5s: %3d generators; pi^* fails %d, iota^* fails %d"
        % (2 * n + 1, m, 2 * n + 1, m - 1, lam, len(gens), badpi, badio))
    return badpi == 0 and badio == 0


def main():
    global LOG
    LOG = open("descendsuper.log", "w")
    say("=== the descent one odd indeterminate down, on the weighted densities ===")
    ok = True
    for (n, m) in [(0, 1), (0, 2), (0, 3), (0, 4), (1, 1), (1, 2), (1, 3), (2, 1)]:
        for lam in (Fr(0), Fr(1, 3), Fr(-1), Fr(5, 2)):
            ok = check(n, m, lam) and ok
    say("")
    say("ALL PASS: %s" % ok)


if __name__ == "__main__":
    main()
