"""How short can Baranov's degree-5 identity be made?

He writes it with 49 monomials and says he could not simplify it.  An identity
is only defined modulo the ones already known, and that is a large space, so
the length of a representative depends entirely on which monomials one lets
survive the reduction.  Here the reduction is run in the basis of monomials in
the two natural products, [a,b] and a o b, with the columns ordered so that a
chosen family of monomials is the last to be eliminated -- whatever survives is
then the representative, and its support is counted.
"""

import itertools
import random
import sys
import time
from fractions import Fraction as Fr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exactlin import rref as ex_rref, rank as ex_rank  # noqa

from videal import MSpace, identities, mevaluate
from vfind import build_C, oracle, test, P
from veventest import even_identities
from vlucid import cb_monomials


def count_ops(name):
    return name.count(" o "), name.count("[")


def basis_change(ms, mons, p=P):
    """Coordinates on the [,]/o-basis: returns the matrix of that basis."""
    return [[int(x) % p for x in ms.vec(f)] for f, _ in mons]


def solve_coords(B, v, p=P):
    """Write v (tree coordinates) on the [,]/o-basis B (rows)."""
    n = len(B)
    aug = [list(B[i]) + [1 if j == i else 0 for j in range(n)]
           for i in range(n)]
    R, piv = ex_rref(aug, p=p)
    # B is a basis, so the left block reduces to the identity
    raise NotImplementedError


if __name__ == "__main__":
    n = 5
    t0 = time.time()
    ms, C = build_C(n)
    I = identities(n, ms)
    I_rows = [[int(x) % P for x in ms.vec(f)] for f in I]
    known = list(C) + [[int(x) % P for x in ms.vec(f)]
                       for f in even_identities(n)]
    R, _ = ex_rref(known, p=P)
    R = [r for r in R if any(r)]
    print("known: dim %d   (I_5 = %d)" % (len(R), len(I)), flush=True)
    phis = oracle(R, I_rows)

    mons = cb_monomials(tuple(range(1, n + 1)))
    names = [nm for _, nm in mons]
    B = basis_change(ms, mons)              # rows: the new basis, in tree coords

    # Transition: a vector v in tree coordinates has [,]/o-coordinates w with
    # w B = v.  Solve once for the whole of R and for one new identity, by
    # row-reducing the transposed system.
    m = len(B)
    Bt = [[B[i][j] for i in range(m)] for j in range(len(ms.trees))]

    def to_cb(vs):
        """Solve  w B = v  for every v in vs, simultaneously."""
        aug = [row[:] for row in Bt]
        for v in vs:
            for j in range(len(aug)):
                aug[j] = aug[j] + [v[j] % P]
        Rr, piv = ex_rref(aug, p=P)
        out = []
        for k in range(len(vs)):
            w = [0] * m
            for i, c in enumerate(piv):
                if c < m:
                    w[c] = Rr[i][m + k]
            out.append(w)
        return out

    # a new identity: any element of I_5 outside the known space
    newf = None
    for f, row in zip(I, I_rows):
        if test(row, phis):
            newf = row
            break
    print("found a new identity in I_5", flush=True)

    cb = to_cb([newf] + R)
    fcb, Rcb = cb[0], cb[1:]
    print("transition done [%.0fs]" % (time.time() - t0), flush=True)

    def reduce_with_order(order):
        """rref the known space with the given column order, reduce f."""
        pos = {c: i for i, c in enumerate(order)}
        rows = [[r[c] for c in order] for r in Rcb]
        Rr, piv = ex_rref(rows, p=P)
        Rr = [r for r in Rr if any(r)]
        piv = list(piv)[:len(Rr)]
        v = [fcb[c] for c in order]
        for r, c in zip(Rr, piv):
            if v[c]:
                fac = v[c] * pow(r[c], P - 2, P) % P
                v = [(a - fac * b) % P for a, b in zip(v, r)]
        return v, order

    best = None
    orders = {
        "many o's kept": sorted(range(m), key=lambda i: count_ops(names[i])[0]),
        "few o's kept": sorted(range(m), key=lambda i: -count_ops(names[i])[0]),
    }
    for k in range(6):
        idx = list(range(m))
        random.Random(k).shuffle(idx)
        orders["random %d" % k] = idx
    for label, order in orders.items():
        v, order = reduce_with_order(order)
        sup = [(order[i], v[i]) for i in range(m) if v[i]]
        print("  %-16s support %d monomials" % (label, len(sup)), flush=True)
        if best is None or len(sup) < len(best[1]):
            best = (label, sup)
    label, sup = best
    print("\nshortest found (%s): %d monomials in the [,]/o-basis" %
          (label, len(sup)))
    for c, coef in sorted(sup, key=lambda s: names[s[0]]):
        cc = coef - P if coef > P // 2 else coef
        print("    %+d  %s" % (cc, names[c]))
    print("[%.0fs]" % (time.time() - t0))
