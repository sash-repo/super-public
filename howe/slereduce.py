"""
slereduce.py -- validation of the reduction for sle(n):

  for z with n_+ z = 0, is  L_1(sle) z = 0  equivalent to  X_0 z = 0,
  where X_0 = D_{xi_{n-2} xi_{n-1} xi_n}?

(Ker Delta|_{S^3 W} = U(n_+) X_0 is verified for 3 <= n <= 6 by l1gen2.py; together with
Lemma "A(z) is an ad n_+-submodule" that gives the equivalence.  Here we check it directly,
subspace by subspace, against the full test.)

Usage: python3 slereduce.py <n> <maxdeg> <box-hi> [ht]
"""
import sys
from fractions import Fraction as F
from le import nullspace
from imod import Le
from sle import InducedSle
from slescan import lift, kobox, top, koweight
from scan import fmtwt, weights_of_V
from pemod import mon


def X0(n):
    return mon(n, [0] * n, [n - 3, n - 2, n - 1])


def sing_with(I, m, nu, ops):
    cols = []
    for u in I.umonos(m):
        w = I.uwt(u)
        mu = tuple(a - b for a, b in zip(nu, w))
        for i in range(len(I.V.wtbasis(mu))):
            cols.append((u, mu, i))
    if not cols:
        return None, []
    eqs = []
    for f in ops:
        img = {}
        for key in cols:
            for k2, c2 in I._act_one(f, key).items():
                img.setdefault(k2, {})[key] = img.setdefault(k2, {}).get(key, 0) + c2
        eqs.extend(img.values())
    return cols, nullspace(eqs, cols)


def span_eq(cols, A, B):
    """do the two lists of vectors span the same subspace?"""
    from le import rref
    idx = {c: i for i, c in enumerate(cols)}
    def red(vs):
        piv, r = rref([{idx[k]: v for k, v in x.items()} for x in vs], len(cols))
        return [(p, tuple(sorted(rr.items()))) for p, rr in zip(piv, r)]
    return red(A) == red(B)


if __name__ == '__main__':
    n = int(sys.argv[1]); maxdeg = int(sys.argv[2]); hi = int(sys.argv[3])
    ht = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    le = Le(n, odd_up=False)
    npos = [le.pe.basis[len(le.pe.nneg) + n + t] for t in range(len(le.pe.npos))]
    tot = bad = 0
    for a in kobox(n, hi):
        I = InducedSle(le, lift(a))
        if not I.V.wtbasis(I.lam):
            continue
        wV = weights_of_V(I, ht)
        full = list(I.pos1) + npos
        red = [X0(n)] + npos
        for m in range(1, maxdeg + 1):
            cand = set()
            for u in I.umonos(m):
                w = I.uwt(u)
                for mu in wV:
                    cand.add(tuple(x + y for x, y in zip(mu, w)))
            for nu in cand:
                cols, A = sing_with(I, m, nu, full)
                if cols is None:
                    continue
                _, B = sing_with(I, m, nu, red)
                tot += 1
                if not span_eq(cols, A, B):
                    bad += 1
                    print('MISMATCH a=%s deg %d wt %s: full %d, reduced %d'
                          % (a, m, fmtwt(nu), len(A), len(B)))
        print('a=%s done, %d weight spaces compared, %d mismatches' % (a, tot, bad))
        sys.stdout.flush()
    print('TOTAL %d weight spaces, %d mismatches' % (tot, bad))
