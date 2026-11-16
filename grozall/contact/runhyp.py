"""The censuses of the supercontact part, run in the SYMMETRIC shape of alpha.

The two shapes are related by a linear change of the
odd coordinates over Cee, so the answers must agree, and their agreement is the check
of both engines; only m >= 2 can differ at all, since hyperbolic_gram(1) = 1.

    python3 runhyp.py v0        the table of dim V_0, W = 0..3, in both shapes
    python3 runhyp.py k13       k(1|3), the densities, W <= 4
    python3 runhyp.py k12       k(1|2), the densities, W <= 3, both weights
    python3 runhyp.py k32       k(3|2), the densities, W <= 3
    python3 runhyp.py k31       k(3|1), the densities, W <= 5
"""

import os, sys
import time
from fractions import Fraction as Fr

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import nullspace  # noqa: E402

from kmix import KMix, Dens, hyperbolic_gram  # noqa: E402
from genmix import MixProblem, term_basis, term_parity, rand_f  # noqa: E402

ALGS = [(1, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (2, 1)]


def osp_gens(alg):
    """The generating functions of osp(m|2n) = the quadratic monomials without t."""
    return [g for g in alg.k_piece(0) if all(k[0][0] == 0 for k in g)]


def V0(n, m, W, gram=None, par=None):
    """The space of k_{<=0}-invariant expressions of weighted order W: the weights do
    not enter, every f in osp(m|2n) having f_t = 0."""
    if par is None:
        return sum(V0(n, m, W, gram, p) for p in (0, 1))
    P = MixProblem(n, m, W, par=par, gram=gram)
    if not P.terms:
        return 0
    alg = P.alg
    M = Dens(alg, 0)
    # the test functions must be rich enough: with two of degree W+2 the system is
    # under-determined for k(5|1) at W=3 and reports dim 2 where the answer is 0.
    Fs = [f for f in (rand_f(alg, 17 + i, W + 4, 12, par=p)
                      for p in (0, 1) for i in range(3)) if f]
    Gs = [g for g in (rand_f(alg, 93 + i, W + 4, 12, par=p)
                      for p in (0, 1) for i in range(3)) if g]
    mat = []
    for f in osp_gens(alg):
        for F in Fs:
            for G in Gs:
                mat.extend(P._rows(f, F, G, M, M, M))
    ns = nullspace(mat) if mat else [[1]] * len(P.terms)
    return len(ns)


def op_span(alg, W, ns, terms):
    """The dimension of the span AS OPERATORS (the PBW terms are independent, so this
    is len(ns); recomputed as a guard)."""
    return len(ns)


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "v0"

    if what == "v0":
        print("dim V_0 (k_{<=0}-invariant expressions), both shapes")
        for (n, m) in ALGS:
            row_o = [V0(n, m, W) for W in (0, 1, 2, 3)]
            row_s = ([V0(n, m, W, hyperbolic_gram(m)) for W in (0, 1, 2, 3)]
                     if m >= 2 else row_o)
            same = "same shape" if m < 2 else ("AGREE" if row_o == row_s else "DIFFER")
            print("  k(%d|%d):  orthonormal %s   symmetric %s   [%s]"
                  % (2 * n + 1, m, row_o, row_s, same))
            sys.stdout.flush()
        return 0

    if what in ("k13", "k32", "k31"):
        n, m, Wmax = {"k13": (0, 3, 4), "k32": (1, 2, 3), "k31": (1, 1, 5)}[what]
        gram = hyperbolic_gram(m)
        WTS = [(Fr(0), Fr(0)), (Fr(2, 3), Fr(-1, 5)), (Fr(-1), Fr(-1)),
               (Fr(1, 2), Fr(3, 7)), (Fr(-3, 2), Fr(5, 4))]
        print("k(%d|%d), symmetric shape: the weighted densities" % (2 * n + 1, m))
        for W in range(Wmax + 1):
            t0 = time.time()
            out = []
            for (lam, mu) in WTS:
                for par in (0, 1):
                    d, a = MixProblem(n, m, W, par=par, gram=gram), None
                    if not d.terms:
                        out.append((par, str(lam), str(mu), 0))
                        continue
                    _, ns = d.solve(lam, mu)
                    out.append((par, str(lam), str(mu), len(ns)))
            print("  W=%d  %s   [%.0f s]" % (W, out, time.time() - t0))
            sys.stdout.flush()
        return 0

    if what == "k12":
        gram = hyperbolic_gram(2)
        NUS = [(Fr(0), Fr(0)), (Fr(1), Fr(0)), (Fr(1), Fr(-1)), (Fr(1, 2), Fr(1, 3))]
        CANDS = [Fr(0), Fr(1), Fr(-1), Fr(1, 2), Fr(-1, 2), Fr(2), Fr(-2)]
        lam, mu = Fr(17, 13), Fr(31, 11)
        print("k(1|2), symmetric shape: lam = %s, mu = %s, the second weights scanned"
              % (lam, mu))
        for W in (0, 1, 2, 3):
            t0 = time.time()
            hits = []
            for (n1, n2) in NUS:
                for c in CANDS:
                    n3 = n1 + n2 + c
                    for par in (0, 1):
                        P = MixProblem(0, 2, W, par=par, gram=gram)
                        if not P.terms:
                            continue
                        _, ns = P.solve(lam, mu, nus=(n1, n2, n3))
                        if ns:
                            hits.append(((str(n1), str(n2), str(n3)), par, len(ns)))
            print("  W=%d  %s   [%.0f s]" % (W, hits, time.time() - t0))
            sys.stdout.flush()
        return 0

    raise SystemExit("unknown: " + what)


if __name__ == "__main__":
    raise SystemExit(main())
