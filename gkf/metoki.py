"""The Metoki class: H^9(h(2), sl(2))_{N=7} recomputed through wt.invariant_space, and an
explicit cocycle representing it, with a certificate.

runs/cohN-N7.log found dim H^{9,7} = 1 with the relative complex built by
h2.relative_space (the bad-row nullspace).  This driver redoes the three degrees that
decide that class with the invariant basis of wt.py instead -- a different basis of the
same subspace, obtained by a different piece of linear algebra -- and then produces the
cocycle itself:

    dim H^{9,7} = dim C~^{9,7} - rank d^9 - rank d^8,

so only q = 8, 9, 10 are needed.  The cocycle z is picked out of ker(d^9) as a vector
that the image of d^8 does not reach, and it is certified directly:

    d z = 0   exactly (every coordinate, over Q), and   rank(B) < rank(B + z),

where B is the image of d^8.  Neither check knows how z was found.

    ../venv/bin/python metoki.py            (needs FLINT: the ranks are ~10^4 wide)
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wt                                                            # noqa: E402
from cohN import differential_sparse, image_rows                     # noqa: E402
from exactlin import rank                                            # noqa: E402

N = 7


def main():
    t00 = time.time()
    print("Metoki class: N = %d (weight %d), backend = %s"
          % (N, 2 * N, __import__("exactlin").backend()), flush=True)

    # ---------------------------------------------------- the relative complex, q = 8..10
    sp = {}
    for q in (8, 9, 10):
        t0 = time.time()
        sp[q] = wt.invariant_space(q, N)
        print("  C~^{%d,%d}: dim %-5d (of %d condition-1)   %.1fs"
              % (q, N, len(sp[q][0]), len(sp[q][1]), time.time() - t0), flush=True)
        assert len(sp[q][0]) == wt.dim_relative(q, N), "(*) disagrees with ker(e) at q=%d" % q

    # ------------------------------------------------------------------- the two ranks
    ranks, imgs = {}, {}
    for q in (8, 9):
        t0 = time.time()
        vsrc, bsrc = sp[q]
        vtgt, btgt = sp[q + 1]
        trips, src, tgt = differential_sparse(q, N)
        assert src == bsrc and tgt == btgt, "basis mismatch at q=%d" % q
        img = image_rows(vsrc, trips, len(tgt))
        ranks[q] = rank(img)
        imgs[q] = img
        # d must stay inside the relative complex
        assert rank(vtgt) == rank(list(vtgt) + img), "d leaves C~ at q=%d" % q
        print("  rank d^%d = %-5d (nnz %d)   %.1fs"
              % (q, ranks[q], len(trips), time.time() - t0), flush=True)

    dim9 = len(sp[9][0])
    h9 = dim9 - ranks[9] - ranks[8]
    print("  dim H^{9,%d} = %d - %d - %d = %d" % (N, dim9, ranks[9], ranks[8], h9), flush=True)
    if h9 != 1:
        print("  UNEXPECTED: the class is not 1-dimensional; stopping")
        return 1

    # -------------------------------------------------------------------- the cocycle
    # cocycles: combinations of the relative basis killed by d, in condition-1 coordinates
    v9, b9 = sp[9]
    D9 = imgs[9]                                  # row k = d(v9[k]) in cond1 coords of q=10
    from exactlin import nullspace
    # the unknowns are the coefficients over the 113 relative basis vectors, i.e. over the
    # ROWS of D9 -- so the nullspace wanted is that of the transpose, of dimension
    # dim C~^9 - rank d^9.  (nullspace(D9) itself solves D9 x = 0 over its 542 columns.)
    D9t = [[D9[k][r] for k in range(len(D9))] for r in range(len(D9[0]))] if D9 else []
    ker = nullspace(D9t)                          # combinations of v9 that d kills
    print("  cocycles: dim Z^9 = %d (expected %d - %d = %d)"
          % (len(ker), dim9, ranks[9], dim9 - ranks[9]), flush=True)
    assert len(ker) == dim9 - ranks[9], "the cocycle space has the wrong dimension"

    def combine(coeffs):
        out = [0] * len(b9)
        for c, v in zip(coeffs, v9):
            if c:
                for j, a in enumerate(v):
                    if a:
                        out[j] += c * a
        return out

    B = imgs[8]                                   # coboundaries, in cond1 coords of q=9
    rB = rank(B)
    assert rB == ranks[8]
    z = None
    for k, coeffs in enumerate(ker):
        cand = combine(coeffs)
        if rank(B + [cand]) > rB:
            z = cand
            print("  a cocycle outside the image found at basis vector %d of Z^9" % k,
                  flush=True)
            break
    assert z is not None, "every cocycle is a coboundary -- contradicts dim H = 1"

    # ------------------------------------------------------------------- certification
    trips, src, tgt = differential_sparse(9, N)
    assert src == b9
    dz = [0] * len(tgt)
    for r, j, c in trips:
        if z[j]:
            dz[r] += c * z[j]
    ok_closed = all(x == 0 for x in dz)
    ok_exact = rank(B + [z]) == rB + 1
    print("  certificate: d z = 0  %s   z not a coboundary  %s"
          % ("ok" if ok_closed else "FAIL", "ok" if ok_exact else "FAIL"), flush=True)

    # ------------------------------------------------- how far up the class really reaches
    # The support of z is a property of the representative: 87 dimensions' worth of
    # coboundaries may be added.  What IS an invariant of the class is the least d such
    # that SOME representative involves Hamiltonians of degree <= d only.  For each d,
    # that asks for a w in C~^8 with (z + dw)_j = 0 at every basis element j carrying a
    # monomial of degree > d -- a linear system in the 232 coefficients of w.
    from exactlin import solve
    B8 = imgs[8]                                  # row k = d(v8[k]) in cond1 coords of q=9
    alldeg = sorted({m[0] + m[1] for S in b9 for m in S})
    minimal, zmin = None, None
    for d in alldeg:
        rows = [j for j, S in enumerate(b9) if any(m[0] + m[1] > d for m in S)]
        A = [[B8[k][j] for k in range(len(B8))] for j in rows]
        rhs = [-z[j] for j in rows]
        w = solve(A, rhs) if rows else []
        print("     degree <= %-2d : %s" % (d, "attainable" if w is not None else "no"),
              flush=True)
        if w is not None and minimal is None:
            minimal = d
            zmin = list(z)
            for k, c in enumerate(w):
                if c:
                    for j, a in enumerate(B8[k]):
                        if a:
                            zmin[j] += c * a
    print("  the class needs Hamiltonians of degree %s and no less" % minimal, flush=True)

    # certify the low-degree representative on its own terms
    dzm = [0] * len(tgt)
    for r, j, c in trips:
        if zmin[j]:
            dzm[r] += c * zmin[j]
    degs_min = sorted({m[0] + m[1] for j, c in enumerate(zmin) if c for m in b9[j]})
    print("  certificate for it: d z' = 0  %s   not a coboundary  %s   degrees %s"
          % ("ok" if all(x == 0 for x in dzm) else "FAIL",
             "ok" if rank(B + [zmin]) == rB + 1 else "FAIL", degs_min), flush=True)

    # --------------------------------------------------------------------- its shape
    from math import gcd
    from fractions import Fraction
    z = zmin                                      # report the low-degree representative
    zf = [Fraction(c) for c in z]
    lcm = 1
    for c in zf:
        lcm = lcm * c.denominator // gcd(lcm, c.denominator)
    zi = [int(c * lcm) for c in zf]
    g = 0
    for c in zi:
        g = gcd(g, abs(c))
    z = [c // g for c in zi] if g > 1 else zi
    print("  scaled to coprime integers (x%s / %d)" % (lcm, g))
    supp = [(b9[j], c) for j, c in enumerate(z) if c]
    mons = sorted({m for S, _ in supp for m in S}, key=wt.keyof)
    degs = sorted({m[0] + m[1] for m in mons})
    print("  the cocycle has %d terms out of %d basis elements" % (len(supp), len(b9)))
    print("  monomials occurring: %d, of degrees %s" % (len(mons), degs))
    print("  a Hamiltonian of degree %d occurs, so the cocycle is not a function of "
          "%d-jets" % (max(degs), max(degs) - 1))
    print("  monomials: %s" % ", ".join("x^%dy^%d" % m for m in mons))
    print("  first terms:")
    for S, c in supp[:6]:
        print("     %+d  " % c + " ^ ".join("(x^%dy^%d)*" % m for m in S))
    print("total %.1fs" % (time.time() - t00), flush=True)
    return 0 if (ok_closed and ok_exact) else 1


if __name__ == "__main__":
    sys.exit(main())
