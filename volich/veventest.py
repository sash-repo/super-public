"""Do the EVEN elements account for all the identities of degree 5?

Volichenko's conditions (a)-(c) say: the commutator is a Lie bracket, and the
Jordan and Jacobi elements are EVEN, where u is even when

        u o x = 0   for every x,        L_u is a derivation.

Every even element u of degree k yields identities of degree k+1 and k+2 in
just those two ways.  This script generates ALL of them for degree 5, using
not only Volichenko's supply A^(JJ) but the FULL space of even elements of the
free Volichenko algebra, and asks how much of the news of degree 5 they cover.
"""

import itertools
import sys
import time
from fractions import Fraction as Fr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exactlin import rref as ex_rref, rank as ex_rank, nullspace as ex_null  # noqa

from videal import (MSpace, identities, msubst, madd, msmul, mmul, mcomm,
                    mcirc, mjbar, var, mevaluate, set_partitions_ordered)
from volid import bracketings, evaluate, par, Space
from vcand import orbit, IRRDIM
from vchar import module_of
from vfind import build_C, oracle, test, alt, P


def even_exprs(k):
    """A basis of {f in M_k : the value of f is an even element}."""
    ms = MSpace(k)
    vals = [evaluate(t) for t in ms.trees]
    sp = Space(vals)
    okeys = [x for x in sp.keys if par(x) == 1]
    opos = {x: i for i, x in enumerate(okeys)}
    rows = []
    for e in vals:
        r = [Fr(0)] * len(okeys)
        for x, c in e.items():
            if par(x) == 1:
                r[opos[x]] = Fr(c)
        rows.append(r)
    mat = [[rows[j][i] for j in range(len(rows))] for i in range(len(okeys))]
    ker = ex_null(mat)
    return [{t: c for t, c in zip(ms.trees, v) if c} for v in ker]


def relab(f, S):
    sub = {i + 1: S[i] for i in range(len(S))}
    return msubst(f, sub)


def even_identities(n):
    """All degree-n identities produced by even elements of lower degree."""
    idx = tuple(range(1, n + 1))
    out = []
    for k in range(2, n):
        Zk = even_exprs(k)
        for S in itertools.combinations(idx, k):
            rest = tuple(x for x in idx if x not in S)
            Zs = [relab(f, S) for f in Zk]
            # (1)  u o m = 0
            for m in bracketings(rest):
                mm = {m: Fr(1)}
                for u in Zs:
                    out.append(mcirc(u, mm))
            # (2)  u(xy) - (ux)y - x(uy) = 0
            for split in set_partitions_ordered(rest, 2) if len(rest) >= 2 \
                    else []:
                for tx in bracketings(split[0]):
                    for ty in bracketings(split[1]):
                        x, y = {tx: Fr(1)}, {ty: Fr(1)}
                        for u in Zs:
                            out.append(madd(mmul(u, mmul(x, y)),
                                            msmul(-1, mmul(mmul(u, x), y)),
                                            msmul(-1, mmul(x, mmul(u, y)))))
    return out


if __name__ == "__main__":
    n = 5
    t0 = time.time()
    ms, C = build_C(n)
    I = identities(n, ms)
    I_rows = [[int(x) % P for x in ms.vec(f)] for f in I]
    print("dim I_5 = %d,  dim C_5 = %d,  new = %d"
          % (len(I), len(C), len(I) - len(C)), flush=True)

    ev = even_identities(n)
    bad = [f for f in ev if mevaluate(f)]
    print("identities produced by even elements: %d expressions, "
          "all of them identities: %s" % (len(ev), not bad), flush=True)

    rows = list(C) + [[int(x) % P for x in ms.vec(f)] for f in ev]
    d = ex_rank(rows, p=P)
    print("C_5 + (all even-element identities) has dim %d  ->  they contribute"
          " %d of the %d new" % (d, d - len(C), len(I) - len(C)), flush=True)

    R, _ = ex_rref(rows, p=P)
    R = [r for r in R if any(r)]
    dec, ch = module_of(R, ms, n, P)
    dec0, _ = module_of(C, ms, n, P)
    dd = {k: dec.get(k, 0) - dec0.get(k, 0) for k in set(dec) | set(dec0)}
    dd = {k: m for k, m in dd.items() if m}
    print("   what they give, as an S_5-module: %s"
          % " + ".join("%d.%s" % (m, k) if m > 1 else k
                       for k, m in sorted(dd.items())), flush=True)

    # what is left over: the alternating shape
    phis = oracle(R, I_rows)
    print("   left over: %d functionals" % len(phis), flush=True)
    V = []
    for f in I:
        g = alt(alt(f, [1, 2, 3], n), [4, 5], n)
        if g:
            V.append([int(x) % P for x in ms.vec(g)])
    dV = ex_rank(V, p=P) if V else 0
    dVC = ex_rank(list(C) + V, p=P) - len(C)
    dVR = ex_rank(list(R) + V, p=P) - len(R)
    print("alternating in {1,2,3} and in {4,5}:  dim of that part of I_5 = %d,"
          "  new beyond C_5: %d,  beyond the even-element ones: %d"
          % (dV, dVC, dVR), flush=True)
    print("[%.0fs]" % (time.time() - t0))
