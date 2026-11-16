#!/usr/bin/env python3
"""n4shape.py -- WHICH embedding of the N=2 vacuum into the N=4 superfield?

Every search so far (n4vac, n4minv, n4phi) seeds on ONE reading of the
[IvKr4] (20)-(21) reduction, namely

    pi(q) = diag(E, Eb)         (pi = second planes' thetas to zero),

and all of them stall on a theta^4 sector of E1.  n4scale.py has now shown
that the obstruction is visible already at theta^0: the E2 defect of that
base carries -12 i/w in the (1,1) channel and -8 i/w in the (2,2) one, and
neither moves when the four N=2 scalars are rescaled.

But the placement of E and Eb in the 2x2 matrix is an ASSUMPTION.  q_g{}^b
carries a row index of one su(2) and a column index of the other; the
constraint of (IK4) symmetrizes the column index with the plane index of
cD_-, and qbar is built with two epsilons.  Nothing in our own text fixes
which entries the two N=2 fields occupy.  This file tries all of them --
diagonal, antidiagonal, and the constant-matrix conjugates of the diagonal
one -- and reports the residual of each.

Writes n4shape-out.txt.
"""

import sys
import time

from fractions import Fraction as Q

import n4noeth as N
import n4vac
from n4noeth import (mono, add, sub, smul, mul, mzero, madd, msub, msmul,
                     minv, mmul, mid, C, M_MONO, lhs1, lhs2, rhs2,
                     check_constraints, show, eq)

OUT = []


def log(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)
    with open("n4shape-out.txt", "w") as fh:
        fh.write("\n".join(OUT) + "\n")


def base_at(B, slots):
    """Constrained q whose pi-projection puts the N=2 vacuum in the given
    entry slots: slots = ((slotE, slotEb)) with slot in 0..3 = 2*row+col."""
    E1s, E2s, B1s, B2s = n4vac.n2_structures()
    sE, sEb = slots
    nB = len(B)
    cols = []
    for M in B:
        col = {}
        for i in (0, 1):
            for jj in (0, 1):
                for k, v in n4vac.pi(M[i][jj]).items():
                    col[(2 * i + jj, k)] = v
        cols.append(col)
    for (slot, S_) in ((sE, E1s), (sE, E2s), (sEb, B1s), (sEb, B2s)):
        col = {}
        for k, v in S_.items():
            col[(slot, k)] = -v
        cols.append(col)
    null = N.sparse_null(cols)
    sc_vecs = [v for v in null if any(v[nB:])]
    if not sc_vecs:
        return None
    A = [[sc_vecs[j][nB + r] for r in range(4)] for j in range(len(sc_vecs))]
    want = [C(2), C(0, -2), C(2), C(0, -2)]
    x = N.csolve(A, want)
    if x is None:
        return None
    base = [C() for _ in range(nB + 4)]
    for j, c in enumerate(x):
        if c:
            for i in range(nB + 4):
                base[i] = base[i] + c * sc_vecs[j][i]
    q = mzero()
    for j in range(nB):
        if base[j]:
            q = madd(q, msmul(base[j], B[j]))
    return q


def flat(Ms):
    L = []
    for M in Ms:
        for i in (0, 1):
            for j in (0, 1):
                L.append(M[i][j])
    return L


def theta0(f):
    return {k: v for k, v in f.items() if not k[3]}


def report(tag, q):
    if q is None:
        log("%-28s : no base" % tag)
        return None
    nc = check_constraints(q)
    try:
        qi = minv(q)
    except Exception as e:
        log("%-28s : constraints %d, NOT INVERTIBLE (%s)" % (tag, nc, e))
        return None
    R1 = lhs1(q, qi)
    R2 = msub(lhs2(q, qi), rhs2(q))
    f1, f2 = flat([R1]), flat([R2])
    n1 = sum(len(f) for f in f1)
    n2 = sum(len(f) for f in f2)
    t0 = [theta0(f) for f in f1 + f2]
    log("%-28s : constraints %d | E1 keys %-4d E2 keys %-4d | theta^0 %d"
        % (tag, nc, n1, n2, sum(len(f) for f in t0)))
    for i, f in enumerate(t0):
        if f:
            log("      slot %d (%s) theta^0 : %s"
                % (i, "E1" if i < 4 else "E2", show(f, cap=6)))
    return (n1, n2)


def main():
    t0 = time.time()
    ms = n4vac.graded_monos(0, 0, -5, 1, mfree=(-1, -1))
    B = n4vac.constrained_basis(ms)
    log("constrained basis: %d  [%.1fs]" % (len(B), time.time() - t0))
    log("")

    shapes = {
        "diag(E, Eb)   [received]": (0, 3),
        "diag(Eb, E)": (3, 0),
        "antidiag  q01=E q10=Eb": (1, 2),
        "antidiag  q01=Eb q10=E": (2, 1),
        "both in (1,1) and (1,2)": (0, 1),
        "both in (1,1) and (2,1)": (0, 2),
    }
    res = {}
    for tag, sl in shapes.items():
        try:
            q = base_at(B, sl)
        except Exception as e:
            log("%-28s : build failed (%s)" % (tag, e))
            continue
        res[tag] = report(tag, q)

    log("")
    log("[%.1fs]" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
