#!/usr/bin/env python3
"""
cohtriv.py -- exact computation of H^*(g; C), trivial coefficients, for the
Lie superalgebras occurring in formula (13) [\\eqref{HofSPE}] of
"Supervarieties of indecomposable modules over Lie superalgebras and
Grassmann superalgebras".

    g = pe(n), spe(n), q(n), sq(n), gl(n), sl(n), ...

Method
------
* Cochain complex: C^*(g;C) = S^*(Pi g^*), the free supercommutative algebra
  on generators theta^a dual to a homogeneous basis e_a of g, with
  p(theta^a) = p(e_a) + 1.  So a ghost coming from an EVEN element of g is
  ODD (exterior, exponent <= 1) and a ghost coming from an ODD element of g
  is EVEN (polynomial, any exponent).  The cochain degree is the polynomial
  degree.
* Differential: the odd derivation determined by
      d theta^c = -1/2 sum_{a,b} (-1)^{p(e_a)} f_{ab}^c theta^a theta^b,
  where [e_a, e_b] = sum_c f_{ab}^c e_c.  The sign (-1)^{p(e_a)} is forced:
  it is the unique parity-only prefactor for which the (a,b) and (b,a) terms
  agree, given f_{ba}^c = -(-1)^{p_a p_b} f_{ab}^c and the supercommutativity
  of the theta's.  d^2 = 0 is verified numerically on every generator.
* Reduction to weight 0: a torus h of g_ev acts semisimply on C^*, and by the
  Cartan homotopy formula L_h = d i_h + i_h d it acts as 0 on cohomology, so
      H^*(g;C) = H^*(g;C)_0 .
  Only weight-zero monomials are enumerated.  (For spe(n) and sl(n) the torus
  is the traceless diagonal one, so "weight 0" is the correct condition
  there too.)
* Ranks over GF(p) for two large primes (and over Q for small cases).

Ground truth used as a self-test (see --selftest)
    H^*(sl(n)), H^*(gl(n))   -- classical (exterior algebras)
    H^*(sl(1|1)) = C[xi_1, eta_1]/(xi_1 eta_1)   -- the (commented-out) first
                                line of formula (13) itself; a super test case
    H^*(q(2)), H^*(q(3)), H^*(spe(2)), H^*(spe(3)), H^*(pe(2)), H^*(pe(3))
                                -- computed here, and independently predicted
                                by relinv.py from the Hochschild--Serre
                                E_2-page; the two agree.

Usage:
    python3 cohtriv.py --selftest
    python3 cohtriv.py pe 2 --deg 10
    python3 cohtriv.py spe 4 --deg 9
"""

import argparse
import sys
from fractions import Fraction

import numpy as np

from defcheck import (Alg, PRIMES, gl, pe, q, sl11, rank_mod, rank_exact,
                      unit, par_idx)


# --------------------------------------------------------------------------
# extra algebras (defcheck has gl, q, sq, pe, spe, sl(1|1))
# --------------------------------------------------------------------------
def sl(n):
    """sl(n), purely even; torus = traceless diagonal."""
    basis, torus = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                basis.append((unit(n, i, j), 0))
    for i in range(n - 1):
        basis.append((unit(n, i, i) - unit(n, i + 1, i + 1), 0))
        torus.append(len(basis) - 1)
    return Alg(f"sl({n})", n, basis, torus, n=n)


def z_grading(g):
    """The ad_D-eigenvalues, D = diag(1_n,-1_n), if every basis vector is an
    eigenvector; otherwise None.  For pe(n)/spe(n) this is the consistent
    Z-grading g_{-1} + g_0 + g_1 (halved eigenvalues); d preserves it, so the
    weight-zero complex splits further along it.  Not available for q(n)."""
    from defcheck import sbr
    n = g.n
    if n is None or g.N != 2 * n:
        return None
    D = np.diag([1] * n + [-1] * n).astype(np.int64)
    out = []
    for x, px in g.basis:
        c = sbr(D, 0, x, px)
        nz = np.nonzero(x)
        a, b = nz[0][0], nz[1][0]
        lam = Fraction(int(c[a, b]), int(x[a, b]))
        if not np.array_equal(c, np.array(lam * x, dtype=np.int64)):
            return None
        if lam.denominator != 1:
            return None
        out.append(int(lam))
    return out


ALGS = {
    "gl": lambda n: gl(n),
    "sl": lambda n: sl(n),
    "q": lambda n: q(n),
    "sq": lambda n: q(n, special=True),
    "pe": lambda n: pe(n),
    "spe": lambda n: pe(n, special=True),
}


# --------------------------------------------------------------------------
# the ghost algebra
# --------------------------------------------------------------------------
class Ghosts:
    """The supercommutative algebra S(Pi g^*) with the CE differential."""

    def __init__(self, g, grading=None):
        self.g = g
        self.dim = g.dim
        self.par = g.par                       # parity of e_a
        self.gpar = [(p + 1) % 2 for p in g.par]  # parity of theta^a
        self.wt = [tuple(-w for w in g.weights[a]) for a in range(g.dim)]
        self.rank = len(g.weights[0]) if g.dim else 0
        # an extra Z-grading of g, preserved by the bracket and hence by d;
        # the complex splits into blocks along it.  For spe(n) this is the
        # ad_D-eigenvalue, D = diag(1_n,-1_n), which is NOT captured by the
        # sl(n)-torus weight and therefore is a genuine extra reduction.
        self.gr = list(grading) if grading is not None else [0] * g.dim
        self._dtheta = self._build_d()

    # -- multiplication ---------------------------------------------------
    def mul(self, m1, m2):
        """Product of two canonical monomials -> (sign, monomial) or None."""
        lst = list(m1) + list(m2)
        sign = 1
        # insertion sort; each adjacent transposition of two ODD ghosts flips
        for i in range(1, len(lst)):
            j = i
            while j > 0 and lst[j - 1] > lst[j]:
                if self.gpar[lst[j - 1]] and self.gpar[lst[j]]:
                    sign = -sign
                lst[j - 1], lst[j] = lst[j], lst[j - 1]
                j -= 1
        for i in range(1, len(lst)):
            if lst[i] == lst[i - 1] and self.gpar[lst[i]]:
                return None                    # square of an odd ghost
        return sign, tuple(lst)

    # -- d on the generators ----------------------------------------------
    def _build_d(self):
        g = self.g
        out = []
        f = [[g.bracket_coords(a, b) for b in range(g.dim)]
             for a in range(g.dim)]
        for c in range(g.dim):
            terms = {}
            for a in range(g.dim):
                for b in range(a, g.dim):
                    coef = f[a][b][c]
                    if coef == 0:
                        continue
                    # the (a,b) and (b,a) terms of the full sum coincide
                    mult = Fraction(1) if a < b else Fraction(1, 2)
                    val = -mult * (-1) ** self.par[a] * coef
                    r = self.mul((a,), (b,))
                    if r is None:
                        continue
                    s, mon = r
                    terms[mon] = terms.get(mon, Fraction(0)) + s * val
            out.append({m: v for m, v in terms.items() if v != 0})
        return out

    def d_mon(self, mon):
        """d of a canonical monomial -> dict monomial -> Fraction."""
        res = {}
        for i, a in enumerate(mon):
            sgn = (-1) ** sum(self.gpar[x] for x in mon[:i])
            rest_l, rest_r = mon[:i], mon[i + 1:]
            for m2, v in self._dtheta[a].items():
                r = self.mul(rest_l, m2)
                if r is None:
                    continue
                s1, mm = r
                r2 = self.mul(mm, rest_r)
                if r2 is None:
                    continue
                s2, mfin = r2
                res[mfin] = res.get(mfin, Fraction(0)) + sgn * s1 * s2 * v
        return {m: v for m, v in res.items() if v != 0}

    # -- weight-zero monomials of a given degree --------------------------
    def blocks(self, deg, weight0=True):
        """Monomials of degree `deg` (weight 0), grouped by the extra grading."""
        out = {}
        for m in self.monomials(deg, weight0=weight0):
            gval = sum(self.gr[a] for a in m)
            out.setdefault(gval, []).append(m)
        return out

    def monomials(self, deg, weight0=True):
        """All canonical monomials of degree `deg` (and weight 0)."""
        N = self.dim
        if not weight0:
            out = []

            def rec0(a, left, mon):
                if left == 0:
                    out.append(tuple(mon))
                    return
                if a == N:
                    return
                maxmult = 1 if self.gpar[a] else left
                for k in range(maxmult, -1, -1):
                    rec0(a + 1, left - k, mon + [a] * k)

            rec0(0, deg, [])
            return sorted(out)
        # suffix bound: max |w[c]| among generators with index >= a
        bound = [[0] * self.rank for _ in range(N + 1)]
        for a in range(N - 1, -1, -1):
            for c in range(self.rank):
                bound[a][c] = max(bound[a + 1][c], abs(self.wt[a][c]))
        out = []
        zero = tuple(Fraction(0) for _ in range(self.rank))

        def rec(a, left, w, mon):
            if left == 0:
                if all(x == 0 for x in w):
                    out.append(tuple(mon))
                return
            if a == N:
                return
            for c in range(self.rank):
                if abs(w[c]) > left * bound[a][c]:
                    return
            maxmult = 1 if self.gpar[a] else left
            for k in range(maxmult, -1, -1):
                if k:
                    nw = tuple(w[c] + k * self.wt[a][c]
                               for c in range(self.rank))
                    rec(a + 1, left - k, nw, mon + [a] * k)
                else:
                    rec(a + 1, left, w, mon)

        rec(0, deg, zero, [])
        return sorted(out)


# --------------------------------------------------------------------------
# cohomology
# --------------------------------------------------------------------------
def sparse_mod(entries, nrow, ncol, p):
    """entries: list of (i, j, Fraction) -> int64 numpy array mod p."""
    M = np.zeros((nrow, ncol), dtype=np.int64)
    cache = {}
    for i, j, v in entries:
        key = (v.numerator, v.denominator)
        r = cache.get(key)
        if r is None:
            r = (int(v.numerator) % p) * pow(int(v.denominator) % p, p - 2, p) % p
            cache[key] = r
        M[i, j] = r
    return M


def rank_sparse(entries, nrow, ncol, exact=False):
    """Rank of the Fraction matrix given by `entries`, over two primes."""
    if nrow == 0 or ncol == 0 or not entries:
        return 0
    ranks = [rank_mod(sparse_mod(entries, nrow, ncol, p), p) for p in PRIMES]
    assert ranks[0] == ranks[1], f"prime disagreement {ranks}"
    if exact:
        from math import lcm
        den = 1
        for _, _, v in entries:
            den = lcm(den, v.denominator)
        M = np.zeros((nrow, ncol), dtype=object)
        for i, j, v in entries:
            M[i, j] = int(v * den)
        re = rank_exact(M)
        assert re == ranks[0], f"GF(p) rank {ranks[0]} != exact rank {re}"
    return ranks[0]


def rank_frac(A, exact=False):
    """Rank of a dense Fraction matrix (list of lists)."""
    if not A or not A[0]:
        return 0
    ent = [(i, j, v) for i, row in enumerate(A) for j, v in enumerate(row) if v]
    return rank_sparse(ent, len(A), len(A[0]), exact=exact)


def cohomology(g, maxdeg, exact=False, verbose=False, weight0=True,
               grading=None):
    """dim H^k(g;C) for 0 <= k <= maxdeg (weight-zero complex by default)."""
    G = Ghosts(g, grading=grading)
    # verify d^2 = 0 on the generators
    for a in range(G.dim):
        acc = {}
        for m, v in G._dtheta[a].items():
            for m2, v2 in G.d_mon(m).items():
                acc[m2] = acc.get(m2, Fraction(0)) + v * v2
        assert not any(v for v in acc.values()), f"d^2 != 0 on theta^{a}"

    bases = {}
    for k in range(maxdeg + 2):
        bases[k] = G.blocks(k, weight0=weight0)
        if verbose:
            tot = sum(len(v) for v in bases[k].values())
            big = max((len(v) for v in bases[k].values()), default=0)
            print(f"    C^{k}: {tot} weight-0 monomials "
                  f"in {len(bases[k])} blocks (largest {big})", file=sys.stderr,
                  flush=True)

    ranks = {}
    dims = {}
    for k in range(maxdeg + 1):
        tot = 0
        for gval, src in sorted(bases[k].items()):
            tgt = bases[k + 1].get(gval, [])
            if not src or not tgt:
                continue
            pos = {m: j for j, m in enumerate(tgt)}
            ent = []
            for i, m in enumerate(src):
                for mm, v in G.d_mon(m).items():
                    ent.append((i, pos[mm], v))  # d preserves weight & grading
            tot += rank_sparse(ent, len(src), len(tgt), exact=exact)
        ranks[k] = tot
        ck = sum(len(v) for v in bases[k].values())
        dims[k] = ck - ranks[k] - (ranks[k - 1] if k else 0)
        print(f"    H^{k} = {dims[k]}   (C^{k} = {ck}, rank d^{k} = {tot})",
              file=sys.stderr, flush=True)

    cdim = {k: sum(len(v) for v in bases[k].values()) for k in bases}
    return dims, cdim


# --------------------------------------------------------------------------
# Poincare series helpers
# --------------------------------------------------------------------------
def series_ext(degs, maxdeg):
    """Poincare series of Lambda(xi_{d} : d in degs), truncated."""
    s = [0] * (maxdeg + 1)
    s[0] = 1
    for d in degs:
        t = s[:]
        for k in range(maxdeg + 1 - d):
            t[k + d] += s[k]
        s = t
    return s


def series_pol(degs, maxdeg):
    """Poincare series of K[x_d : d in degs], truncated."""
    s = [0] * (maxdeg + 1)
    s[0] = 1
    for d in degs:
        t = s[:]
        for k in range(d, maxdeg + 1):
            t[k] += t[k - d]
        s = t
    return s


def mul_series(a, b, maxdeg):
    out = [0] * (maxdeg + 1)
    for i, x in enumerate(a):
        if not x:
            continue
        for j, y in enumerate(b):
            if i + j <= maxdeg and y:
                out[i + j] += x * y
    return out


# --------------------------------------------------------------------------
def selftest():
    ok = True
    cases = [
        ("sl(2)", sl(2), 6, series_ext([3], 6)),
        ("sl(3)", sl(3), 8, series_ext([3, 5], 8)),
        ("gl(2)", gl(2), 6, series_ext([1, 3], 6)),
        ("gl(3)", gl(3), 9, series_ext([1, 3, 5], 9)),
        ("sl(1|1)", sl11(), 8, [1] + [2] * 8),        # C[xi,eta]/(xi eta)
        ("gl(1|1)", gl(1, 1), 6, series_ext([1], 6)),
        ("q(2)", q(2), 7, [1, 1, 1, 1, 0, 0, 0, 0]),  # K[x_1]/(x_1^4)
        ("q(3)", q(3), 7, [1, 1, 1, 2, 1, 1, 1, 0]),  # prod_{i<=3}(1+t^i)
        ("pe(2)", pe(2), 8, series_ext([1], 8)),      # Lambda(xi_1)
        ("pe(3)", pe(3), 8, series_ext([1, 5], 8)),   # Lambda(xi_1, xi_5)
        ("spe(2)", pe(2, special=True), 8,
         [1, 1, 2, 2, 2, 2, 2, 2, 2]),                # K[x_1,y_2]/(x_1^2 y_2)
        ("spe(3)", pe(3, special=True), 8,
         [1, 0, 0, 1, 0, 1, 1, 0, 1]),                # K[u_3] (x) Lambda(xi_5)
    ]
    for name, g, K, expect in cases:
        dims, _ = cohomology(g, K, exact=(g.dim <= 8))
        got = [dims[k] for k in range(K + 1)]
        good = got == expect[:K + 1]
        ok &= good
        print(f"  {name:8s} H^0..H^{K} = {got}"
              f"   expected {expect[:K + 1]}   {'OK' if good else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("alg", nargs="?", choices=sorted(ALGS))
    ap.add_argument("n", nargs="?", type=int)
    ap.add_argument("--deg", type=int, default=8)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--exact", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--noweight", action="store_true")
    ap.add_argument("--nograding", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        print("self-test against known answers:")
        sys.exit(0 if selftest() else 1)

    g = ALGS[a.alg](a.n)
    gr = None if a.nograding else z_grading(g)
    if a.verbose:
        print(f"  extra Z-grading: {'yes' if gr else 'no'}", file=sys.stderr)
    dims, cdim = cohomology(g, a.deg, exact=a.exact, verbose=a.verbose,
                            weight0=not a.noweight, grading=gr)
    print(f"{g.name}:  sdim = "
          f"{sum(1 for p in g.par if p == 0)}|{sum(g.par)}")
    print("  k      : " + " ".join(f"{k:4d}" for k in range(a.deg + 1)))
    print("  dim C^k: " + " ".join(f"{cdim[k]:4d}" for k in range(a.deg + 1)))
    print("  dim H^k: " + " ".join(f"{dims[k]:4d}" for k in range(a.deg + 1)))


if __name__ == "__main__":
    main()
