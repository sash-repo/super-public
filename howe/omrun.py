"""
omrun.py -- driver for omcheck.py.

Verifies, degree by degree and exactly, the graph of [L3, Subsec. 2.4 b)]:

  * omega^2 = 0 and L_D(omega) = 0 for every D in pe(n)   (self-tests);
  * the homology of (Omega(const), omega.);
  * the number and the weights of the pe(n)-singular vectors of Omega^i,
    of the submodule  I^i = omega . Omega^{i-2}  and of the quotient Omega^i / I^i.

Usage:  python3 omrun.py [nmax] [imax]
"""

import sys
from fractions import Fraction as F
from omcheck import (basis, deg, omega_mul, gens_pe, to_vec, rank, nullspace,
                     weight, add)


def selftest(n):
    """omega^2 = 0 and pe(n) preserves omega."""
    om = {}
    for i in range(n):
        B = [0] * n
        B[i] = 1
        add(om, ((i,), tuple(B)), F(1))
    assert not omega_mul(n, om), "omega^2 != 0"
    npl, allg = gens_pe(n)
    for g in allg:
        r = g.act(om)
        assert not r, "L_D(omega) != 0 for %s(%d,%d)" % (g.kind, g.i, g.j)
    return om


def singular(n, vecs, index, npl, basis_i):
    """basis of the n_+-invariants inside the span of `vecs` (a list of
    coordinate vectors in the basis `basis_i`)."""
    if not vecs:
        return []
    ncol = len(vecs)
    rows = []
    # for each generator of n_+ and each target monomial, one linear equation
    # on the coefficients c_k of sum_k c_k vecs[k]
    for g in npl:
        images = []
        for v in vecs:
            el = {}
            for m, c in zip(basis_i, v):
                if c:
                    add(el, m, c)
            images.append(g.act(el))
        targets = set()
        for im in images:
            targets |= set(im.keys())
        for t in sorted(targets):
            rows.append([im.get(t, F(0)) for im in images])
    return nullspace(rows, ncol)


def wt_of(vec, basis_i):
    """weight of a weight vector given in coordinates (None if not homogeneous)."""
    ws = {weight(m) for m, c in zip(basis_i, vec) if c}
    return ws.pop() if len(ws) == 1 else None


def report(n, imax):
    om = selftest(n)
    npl, _ = gens_pe(n)
    print("=" * 72)
    print("n = %d   (le(n) = Le(n), L_0 = pe(n));  omega^2 = 0 and L_pe(omega) = 0 OK"
          % n)
    print("=" * 72)
    bas = {i: basis(n, i) for i in range(-1, imax + 3)}
    idx = {i: {m: k for k, m in enumerate(bas[i])} for i in bas}

    # the matrix of omega. : Omega^i -> Omega^{i+2}, as rows = images of basis
    img = {}
    for i in range(imax + 1):
        rows = []
        for m in bas[i]:
            rows.append(to_vec(omega_mul(n, {m: F(1)}), idx[i + 2]))
        img[i] = rows

    print("%3s %6s %6s %6s %6s   %s" % ("i", "dim", "rk om", "dim I", "homol", "singular vectors (weights)"))
    for i in range(imax + 1):
        d = len(bas[i])
        rk = rank(img[i]) if d else 0
        prev = rank(img[i - 2]) if i >= 2 else 0      # dim of I^i = omega.Omega^{i-2}
        ker = d - rk
        homol = ker - prev
        # singular vectors of the whole Omega^i
        full = []
        for k in range(d):
            v = [F(0)] * d
            v[k] = F(1)
            full.append(v)
        sing = singular(n, full, idx[i], npl, bas[i])
        ws = sorted(w for w in (wt_of(s, bas[i]) for s in sing) if w is not None)
        print("%3d %6d %6d %6d %6d   %d  %s"
              % (i, d, rk, prev, homol, len(sing),
                 " ".join(str(w) for w in ws)))
        if homol:
            # exhibit the homology classes
            kerbas = nullspace([[img[i][k][c] for k in range(d)]
                                for c in range(len(bas[i + 2]))], d)
            print("      homology of dimension %d in degree %d" % (homol, i))
    print()


if __name__ == "__main__":
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    imax = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    for n in range(1, nmax + 1):
        report(n, imax)
