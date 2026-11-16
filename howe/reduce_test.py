"""
reduce_test.py -- validate the reduction lemma.

Lemma (see l1gen.py, l1gen2.py).  Put

    X_0 = D_{xi_{n-2} xi_{n-1} xi_n},   Y_0 = D_{x_{n-1} xi_{n-1} xi_n}   (n >= 3),
    X_0 = D_{x_1 xi_1 xi_2}                                              (n = 2).

If n_+ z = 0 then  L_1 z = 0  <=>  X_0 z = 0  and  Y_0 z = 0.

Reason: A(z) = {X in L_1 : X z = 0} is an ad(n_+)-submodule of L_1 as soon as n_+ z = 0, and
U(n_+).<X_0,Y_0> = L_1.

Here we check the lemma the honest way: for every lambda in a box we compute the singular
vectors twice -- with the full L_1 (dim 12, 38, 88, ... equations) and with the two generators
only -- and compare the two kernels.  They must be equal, not merely of equal dimension.
"""

import sys
from fractions import Fraction as F
from le import nullspace, rref
from imod import Le, Induced
from pemod import mon
from scan import fmtwt, box, weights_of_V


def gens(n):
    if n == 2:
        return [mon(n, [1, 0], [0, 1])]                       # x_1 xi_1 xi_2
    a = [0] * n
    a[n - 2] = 1
    return [mon(n, [0] * n, [n - 3, n - 2, n - 1]),           # xi_{n-2} xi_{n-1} xi_n
            mon(n, a, [n - 2, n - 1])]                        # x_{n-1} xi_{n-1} xi_n


def singular_with(I, m, nu, ops1):
    """copy of Induced.singular, but with L_1 replaced by the list ops1"""
    cols = []
    for u in I.umonos(m):
        w = I.uwt(u)
        mu = tuple(a - b for a, b in zip(nu, w))
        for i in range(len(I.V.wtbasis(mu))):
            cols.append((u, mu, i))
    if not cols:
        return []
    ops = list(ops1) + [I.pe.basis[len(I.pe.nneg) + I.n + t] for t in range(len(I.pe.npos))]
    eqs = []
    for f in ops:
        img = {}
        for key in cols:
            for k2, c2 in I._act_one(f, key).items():
                row = img.setdefault(k2, {})
                row[key] = row.get(key, 0) + c2
        eqs.extend(img.values())
    return nullspace(eqs, cols)


def same_space(A, B):
    if len(A) != len(B):
        return False
    keys = sorted({k for v in A + B for k in v}, key=str)
    idx = {k: i for i, k in enumerate(keys)}
    ra = [{idx[k]: c for k, c in v.items()} for v in A]
    rb = [{idx[k]: c for k, c in v.items()} for v in B]
    pa, _ = rref(ra, len(idx))
    pab, _ = rref(ra + rb, len(idx))
    return len(pa) == len(pab)


def run(n, maxdeg, lo, hi, ht=6):
    le = Le(n, odd_up=False)
    g = gens(n)
    print('n = %d: comparing full L_1 (%d elements) with the %d generator(s), '
          'lambda in [%d,%d]^%d, degrees 1..%d'
          % (n, len(le.pos1), len(g), lo, hi, n, maxdeg))
    bad = 0
    tested = 0
    for lam in box(n, lo, hi):
        I = Induced(le, lam)
        if not I.V.wtbasis(I.lam):
            continue
        wV = weights_of_V(I, ht)
        for m in range(1, maxdeg + 1):
            cand = set()
            for u in I.umonos(m):
                w = I.uwt(u)
                for mu in wV:
                    cand.add(tuple(a + b for a, b in zip(mu, w)))
            for nu in sorted(cand):
                full = I.singular(m, nu)
                red = singular_with(I, m, nu, g)
                tested += 1
                if not same_space(full, red):
                    bad += 1
                    print('   MISMATCH lambda=%s deg %d wt %s : full %d, reduced %d'
                          % (fmtwt(tuple(F(x) for x in lam)), m, fmtwt(nu), len(full), len(red)))
                elif full:
                    print('   ok  lambda=%-12s deg %d wt %-12s : %d singular vector(s), '
                          'both ways' % (fmtwt(tuple(F(x) for x in lam)), m, fmtwt(nu), len(full)))
        sys.stdout.flush()
    print('n = %d: %d weight spaces compared, %d mismatches' % (n, tested, bad))
    sys.stdout.flush()


if __name__ == '__main__':
    n = int(sys.argv[1]); maxdeg = int(sys.argv[2])
    lo, hi = int(sys.argv[3]), int(sys.argv[4])
    run(n, maxdeg, lo, hi)
