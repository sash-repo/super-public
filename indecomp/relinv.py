#!/usr/bin/env python3
"""
relinv.py -- the relative cohomology H^*(g, g_ev; C) for g = pe(n), spe(n),
q(n), sq(n), and the resulting Hochschild--Serre E_2-page.

Why this is enough to pin the structure down
--------------------------------------------
g_ev is reductive in g and acts semisimply, so

  (a) H^*(g;C) = H^*( C^*(g;C)^{g_ev} )              [Casimir homotopy], and
  (b) the relative cochains are C^*(g,g_ev;C) = (S^* g_odd^*)^{g_ev} with
      ZERO differential: for trivial coefficients d only ever feeds a bracket
      [x_i,x_j] of two ODD arguments back in, and that bracket lies in g_ev,
      where a relative cochain vanishes.

Hence H^*(g,g_ev;C) = (S^* g_odd^*)^{g_ev} and the Hochschild--Serre spectral
sequence of the pair g_ev < g has

      E_2 = H^*(g,g_ev;C) (x) H^*(g_ev;C).

The ghosts of g_odd are EVEN, so (S^* g_odd^*)^{g_ev} is an ordinary ring of
polynomial invariants; H^*(g_ev;C) is an exterior algebra.  What is left is
the transgression, and comparing this E_2-page with the answers computed
directly by cohtriv.py identifies it.

Usage:  python3 relinv.py pe 3 --deg 10
        python3 relinv.py --table
"""

import argparse
import sys
from fractions import Fraction

import numpy as np

from defcheck import PRIMES, rank_mod
from cohtriv import ALGS, Ghosts, sparse_mod


def relative_invariants(g, maxdeg, verbose=False):
    """dim (S^k g_odd^*)^{g_ev} for 0 <= k <= maxdeg."""
    G = Ghosts(g)
    odd = [a for a in range(g.dim) if g.par[a] == 1]     # even ghosts
    ev = [a for a in range(g.dim) if g.par[a] == 0]
    rank = G.rank

    # L_x theta^c = - sum_a f_{x a}^c theta^a   (x even)
    act = {}
    for x in ev:
        tab = {}
        for c in range(g.dim):
            row = {}
            for a in range(g.dim):
                coef = g.bracket_coords(x, a)[c]
                if coef:
                    row[a] = -coef
            if row:
                tab[c] = row
        act[x] = tab

    def apply(x, mon):
        """L_x on a monomial (all factors even ghosts) -- no signs needed."""
        res = {}
        tab = act[x]
        for i, c in enumerate(mon):
            if c not in tab:
                continue
            rest = mon[:i] + mon[i + 1:]
            for a, v in tab[c].items():
                r = G.mul(rest, (a,))
                if r is None:
                    continue
                s, mm = r
                res[mm] = res.get(mm, Fraction(0)) + s * v
        return {m: v for m, v in res.items() if v}

    # weight-zero monomials of degree k in the odd-part ghosts
    def mons(k):
        out = []
        wt = G.wt

        def rec(i, left, w, mon):
            if left == 0:
                if all(v == 0 for v in w):
                    out.append(tuple(mon))
                return
            if i == len(odd):
                return
            a = odd[i]
            for m in range(left, -1, -1):
                nw = tuple(w[c] + m * wt[a][c] for c in range(rank))
                rec(i + 1, left - m, nw, mon + [a] * m)

        rec(0, k, tuple(Fraction(0) for _ in range(rank)), [])
        return sorted(out)

    dims = []
    for k in range(maxdeg + 1):
        src = mons(k)
        if not src:
            dims.append(0)
            continue
        cols = {}
        ent = []
        for i, m in enumerate(src):
            for x in ev:
                for mm, v in apply(x, m).items():
                    j = cols.setdefault((x, mm), len(cols))
                    ent.append((i, j, v))
        if not cols:
            dims.append(len(src))
        else:
            r = rank_mod(sparse_mod(ent, len(src), len(cols), PRIMES[0]),
                         PRIMES[0])
            r2 = rank_mod(sparse_mod(ent, len(src), len(cols), PRIMES[1]),
                          PRIMES[1])
            assert r == r2, f"prime disagreement {r} {r2}"
            dims.append(len(src) - r)
        if verbose:
            print(f"    S^{k}: {len(src)} weight-0 monomials, "
                  f"{dims[-1]} invariant", file=sys.stderr)
    return dims


def ev_cohomology_series(g, maxdeg):
    """Poincare series of H^*(g_ev;C): Lambda(xi_1,...) for gl(n)/sl(n)."""
    n = g.n
    name = g.name
    if name.startswith("spe") or name.startswith("sq"):
        degs = list(range(3, 2 * n, 2))          # sl(n): xi_3,...,xi_{2n-1}
    else:
        degs = list(range(1, 2 * n, 2))          # gl(n): xi_1,...,xi_{2n-1}
    s = [0] * (maxdeg + 1)
    s[0] = 1
    for d in degs:
        t = s[:]
        for k in range(maxdeg + 1 - d):
            t[k + d] += s[k]
        s = t
    return s, degs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("alg", choices=sorted(ALGS))
    ap.add_argument("n", type=int)
    ap.add_argument("--deg", type=int, default=10)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    g = ALGS[a.alg](a.n)
    rel = relative_invariants(g, a.deg, verbose=a.verbose)
    ser, degs = ev_cohomology_series(g, a.deg)
    e2 = [0] * (a.deg + 1)
    for i, x in enumerate(rel):
        for j, y in enumerate(ser):
            if i + j <= a.deg:
                e2[i + j] += x * y
    print(f"{g.name}:")
    print("  k                    : " +
          " ".join(f"{k:4d}" for k in range(a.deg + 1)))
    print("  dim H^k(g,g_ev;C)    : " + " ".join(f"{v:4d}" for v in rel))
    print(f"  H^*(g_ev) = Lambda{tuple('xi_%d' % d for d in degs)}")
    print("  dim E_2^k            : " + " ".join(f"{v:4d}" for v in e2))


if __name__ == "__main__":
    main()
