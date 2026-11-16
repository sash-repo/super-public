"""Census: bilinear k(1|n)-invariant operators on weighted densities.

For each n and each weighted order W: the generic dimension, the jump locus in the plane
of the two weights (grid + the gcd of the maximal minors of the pencil), the shape of
the operators, and an INDEPENDENT verification of each operator found -- against the
elements of k_i for i up to 6 and on functions not used in finding it.
"""

import os, sys, random, time
from fractions import Fraction as Fr

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]

from super import (mono, add, scal, mul, parity, Kf, kbracket, k_piece, Dens)  # noqa
from gensuper import pencil, show_op, apply_term, rand_fun, term_basis  # noqa
from affine import dim_at, ops_at  # noqa
from sweep import grid_jumps, minor_gcd, points_on  # noqa


def verify(n, terms, vec, W, lam, mu, nus=(0, 0, 0), maxdeg=6, seed=999):
    """Is sum vec_T T really invariant?  Checked against k_i, i <= maxdeg, on fresh data."""
    nu = Fr(lam) + Fr(mu) + Fr(W, 2)
    n1, n2, n3 = nus
    M1, M2, M3 = Dens(n, lam, n1), Dens(n, mu, n2), Dens(n, nu, n3)
    pB = W % 2

    def B(F, G):
        out = {}
        for c, t in zip(vec, terms):
            if c:
                out = add(out, scal(c, apply_term(t, F, G)))
        return out

    rnd = random.Random(seed)
    gens = [f for i in range(-2, maxdeg + 1) for f in k_piece(n, i)]
    Fs = [rand_fun(n, p, seed + 3 + i, maxdeg=4) for p in (0, 1) for i in range(3)]
    Gs = [rand_fun(n, p, seed + 40 + i, maxdeg=4) for p in (0, 1) for i in range(3)]
    for f in gens:
        pf = parity(f)
        for F in Fs:
            if not F:
                continue
            pF = parity(F)
            for G in Gs:
                if not G:
                    continue
                lhs = add(B(M1.act(f, F), G),
                          scal((-1) ** (pf * pF), B(F, M2.act(f, G))))
                rhs = scal((-1) ** (pB * pf), M3.act(f, B(F, G)))
                if add(lhs, scal(-1, rhs)):
                    return False
    return True


def census(n, Wmax, gdeg=3, grid=True):
    print("=" * 74)
    print("k(1|%d):  weighted densities,  lam3 = lam1 + lam2 + W/2" % n)
    for W in range(Wmax + 1):
        t0 = time.time()
        terms, V0, N0, N1, N2 = pencil(n, W, gdeg=gdeg)
        if not terms:
            print(" W=%d: no term of this weighted order" % W)
            continue
        g = dim_at(V0, N0, N1, N2, Fr(17, 13), Fr(31, 11))
        par = "odd" if W % 2 else "even"
        print(" W=%d (%s):  #terms=%d   generic dim = %d   [%.0fs]"
              % (W, par, len(terms), g, time.time() - t0))
        if g:
            for v in ops_at(terms, V0, N0, N1, N2, Fr(17, 13), Fr(31, 11)):
                ok = verify(n, terms, v, W, Fr(17, 13), Fr(31, 11))
                print("      generic operator %s   [verified: %s]"
                      % (show_op(terms, v), ok))
        r = len(V0) - g
        if r > 0 and grid:
            js = grid_jumps(V0, N0, N1, N2, g, den=4, bound=3)
            if js:
                seen = {}
                for (a, b, d) in js:
                    seen.setdefault(d, []).append((a, b))
                for d, pts in sorted(seen.items()):
                    print("      jump to dim %d at %d grid points, e.g. %s"
                          % (d, len(pts), pts[:6]))
                    a, b = pts[0]
                    for v in ops_at(terms, V0, N0, N1, N2, a, b):
                        ok = verify(n, terms, v, W, a, b)
                        print("         at (%s,%s): %s   [verified: %s]"
                              % (a, b, show_op(terms, v), ok))
            else:
                print("      no jump on the grid (den<=4, |.|<=3): dim is constant")
            if len(V0) <= 12:
                try:
                    gc, facs = minor_gcd(V0, N0, N1, N2, r)
                    print("      gcd of the maximal minors:", gc)
                except Exception as e:                       # noqa
                    print("      (minor gcd skipped: %s)" % e)
        sys.stdout.flush()


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "1"
    if which == "1":
        census(1, 6)
    elif which == "3":
        census(3, 4)
    elif which == "4":
        census(4, 3)
