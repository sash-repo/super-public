"""
l1gen.py -- is L_1 generated, as a module over n_+ = pe(n)_{S^2-up}, by a single vector?

Why this matters.  For z in I(V) put  A(z) = { X in L_1 : X.z = 0 }.  If n_+ z = 0 then
A(z) is an n_+-submodule of L_1, because for e in n_+ and X in A(z)

    [e,X].z = e.(X.z) -+ X.(e.z) = 0.

So if L_1 = U(n_+).X_0 for a single X_0, then the whole system "L_1 z = 0" collapses to the
ONE equation X_0 z = 0 (given n_+ z = 0).  That is the step which replaces an induction on n
by a computation uniform in n.

This script tests the hypothesis L_1 = U(n_+).X_0 for X_0 the ht-lowest weight vector of L_1,
for n = 2,3,4,5,6, and prints which single generators work.
"""

import sys
from fractions import Fraction as F
from le import abr, rref, pwt
from pemod import Pe
from imod import Le


def span_dim(vecs, idx):
    rows = [{idx[k]: c for k, c in v.items()} for v in vecs if v]
    piv, _ = rref(rows, len(idx))
    return len(piv)


def closure(vecs, ops, n, idx):
    """smallest subspace containing vecs and closed under {e,-} for e in ops"""
    cur = [v for v in vecs if v]
    d = span_dim(cur, idx)
    while True:
        new = cur + [abr(e, v, n) for e in ops for v in cur]
        new = [v for v in new if v]
        d2 = span_dim(new, idx)
        if d2 == d:
            return cur, d
        # keep a spanning set of manageable size: reduce to independent ones
        keep = []
        for v in new:
            if span_dim(keep + [v], idx) > span_dim(keep, idx):
                keep.append(v)
        cur, d = keep, d2


def run(n):
    le = Le(n, odd_up=False)
    pe = Pe(n, odd_up=False)
    L1 = le.pos1                                  # generating functions of degree 3
    monos = sorted({m for f in L1 for m in f}, key=str)
    idx = {m: i for i, m in enumerate(monos)}
    dimL1 = span_dim(L1, idx)
    hts = [(pe.ht(pwt(f, n)), f) for f in L1]
    lo = min(h for h, _ in hts)
    lowest = [f for h, f in hts if h == lo]
    print('n = %d :  dim L_1 = %d,  lowest ht = %d, %d vector(s) of that ht'
          % (n, dimL1, lo, len(lowest)))
    for f in lowest:
        _, d = closure([f], pe.npos, n, idx)
        print('    generator %-28s ->  dim U(n_+).X = %d   %s'
              % (name(n, f), d, 'GENERATES L_1' if d == dimL1 else 'does NOT generate'))
    # for completeness: is there ANY single generator?
    if all(closure([f], pe.npos, n, idx)[1] != dimL1 for f in lowest):
        best = 0
        for f in L1:
            _, d = closure([f], pe.npos, n, idx)
            if d > best:
                best, bf = d, f
        print('    best single generator: %s -> %d of %d' % (name(n, bf), best, dimL1))
    sys.stdout.flush()


def name(n, f):
    out = []
    for (a, S), c in sorted(f.items(), key=str):
        s = ''.join('x%d' % (i + 1) * a[i] for i in range(n)) + ''.join('t%d' % (i + 1) for i in S)
        out.append(('' if c == 1 else str(c)) + s)
    return '+'.join(out)


if __name__ == '__main__':
    for n in (2, 3, 4, 5, 6):
        run(n)
