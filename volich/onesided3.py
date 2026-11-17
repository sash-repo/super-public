#!/usr/bin/env python3
"""Statement 4.16 of the text: Proposition 2.13 -- "simple epimorphic
Volichenko algebras from Tables 2.12.1 and 2.12.2 have no one-sided ideals" --
for the two families that Statement 4.15 leaves out:

    osp(2m|2n),  osp(2m+1|2n)   with x = the image of x^p_p ,
    spe(n)                      with x = y^p_q .

The argument is the one of onesided.py / onesided2.py: a left ideal containing v
contains A.v, where A is the unital associative subalgebra of End(h) generated
by the left multiplications L_g of the product

    g.h = [g,h] + S(g,h),   S(g,h) = [[g,x],[h,x]] .

If A = End(h) then, by Burnside, h is an irreducible A-module, so A.v = h for
every v /= 0: no proper left ideal.  Same on the right.

The closure is walked mod p = 2^61 - 1, which costs no rigour in the direction
needed: reduction mod p can only LOWER a rank, so rank_p = dim End(h) forces
rank_Q = dim End(h).

The ambients come from vdescr.py, where they are built by solving the defining
linear condition, and where the same x is checked to lie in g_od, to satisfy
[x,x] = 0 and to ensure nontriviality.

Usage: python3 onesided3.py
"""

import sys
import time
from fractions import Fraction as F

import nontriv
import vdescr as V
from onesided import closure_mod_p, to_p
from onesided2 import subalgebra, _coords_of


def build(amb):
    """an Alg on the matrix superalgebra amb, plus the coordinate data."""
    gens = [(A, 0) for A in amb.ev] + [(A, 1) for A in amb.od]
    return subalgebra(amb.name, amb.d, amb.M, gens, center=amb.center)


def run(label, amb, x, verbose=True):
    t0 = time.time()
    alg, basis, rows, N = build(amb)
    xv = _coords_of(rows, N, amb.d, x)
    br = alg.br(xv, xv)
    assert not any(br), f"{label}: [x,x] /= 0"
    h = nontriv.Voli(alg, xv)
    assert h.closure_ok(), f"{label}: h not closed"
    ens = any(any(h.Sm[i][j]) for i in range(h.dim) for j in range(h.dim))
    if not ens:
        print(f"  {label:22s} x does NOT ensure nontriviality -- outside "
              f"the Table's range")
        return None
    n = h.dim
    full = n * n
    L = [[[to_p(h.prod(h.e(i), h.e(j))[k]) for j in range(n)]
          for k in range(n)] for i in range(n)]
    R = [[[to_p(h.prod(h.e(j), h.e(i))[k]) for j in range(n)]
          for k in range(n)] for i in range(n)]
    dL = closure_mod_p(L, n)
    dR = closure_mod_p(R, n)
    ok = (dL == full and dR == full)
    print(f"  {label:22s} dim h = {n:3d}   dim End h = {full:5d}   "
          f"dim A_left = {dL:5d}   dim A_right = {dR:5d}   "
          f"{'A = End h  -- no one-sided ideals' if ok else 'PROPER'}"
          f"   [{time.time() - t0:.1f}s]")
    return ok


def main():
    print("=" * 78)
    print("One-sided ideals: the osp and spe families of Table 2.12.1")
    print("=" * 78)

    verdicts = []

    print("\nspe(n), x = y^p_q  (Table: p, q > 0, p + 2q <= n):")
    for (n, p, q) in [(3, 1, 1), (4, 1, 1), (4, 2, 1), (5, 1, 1), (5, 1, 2),
                      (5, 2, 1), (5, 3, 1)]:
        verdicts.append(run(f"spe({n}) p={p} q={q}", V.spe(n),
                            V.y_spe(n, p, q)))

    print("\nosp(2m|2n), x = image of x^p_p  (Table: p > 0, 2p <= min(m,n)):")
    for (m, n, p) in [(2, 2, 1), (2, 3, 1), (3, 2, 1), (3, 3, 1)]:
        verdicts.append(run(f"osp({2*m}|{2*n}) p={p}", V.osp(m, n, 0),
                            V.x_osp(m, n, p, 0)))

    print("\nosp(2m+1|2n), x = the same under osp(2m|2n) < osp(2m+1|2n):")
    for (m, n, p) in [(2, 2, 1), (2, 3, 1), (3, 2, 1)]:
        verdicts.append(run(f"osp({2*m+1}|{2*n}) p={p}", V.osp(m, n, 1),
                            V.x_osp(m, n, p, 1)))

    done = [v for v in verdicts if v is not None]
    print(f"\n{len(done)} pairs (g, x) run; "
          f"{'all give A = End h' if all(done) else 'SOME FAILED'}")
    return 0 if all(done) else 1


if __name__ == "__main__":
    sys.exit(main())
