#!/usr/bin/env python3
r"""
run_regrad.py -- what g_0 is in a W-regrading, and why [ShM] does not reach it.

Table (65) lists the simple W-graded vectorial Lie superalgebras of depth 1
together with the regrading parameter r: in vect(m|n;r) the first r odd
indeterminates are given degree 0.  Item 2) of Subsec. 9.3 runs [ShM]
-- the classification of the maximal subalgebras of the classical MATRIX Lie
superalgebras -- over g_0.  That step needs g_0 to be such a matrix algebra.

For r = 0 it is, by eq. (38).  For r > 0 the program computes dim g_0 (the
number of monomial vector fields of degree 0) and compares it with the dimensions
of gl, sl, osp, pe, q and c(pe) on a superspace of the superdimension of g_{-1};
no match means g_0 is none of them (table (102), Proposition 9.4.1).  It also
prints the derived series of g_0.  (The helper functions for the nilpotent ideal
I = { X in g_0 | every coefficient of X lies in Lambda_+(xi_1..xi_r) } are kept
but not called by main().)

Output: printed to stdout.
"""

import sys
from fractions import Fraction as F

import os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]

from wgrad import VF, bracket, Span, to_vec, monomials             # noqa: E402

LOG = None


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    if LOG:
        LOG.write(s + "\n")
        LOG.flush()


def independent(fields):
    keys = sorted({k for X in fields for k in X.t}, key=str)
    S = Span({k: i for i, k in enumerate(keys)})
    out = []
    for X in fields:
        v = to_vec(X, S.index)
        if v is not None and S.add(v):
            out.append(X)
    return out


def weight_of(key, w, ne):
    (a, S), k = key
    d = sum(w[i] * a[i] for i in range(ne))
    d += sum(w[ne + j] for j in S)
    return d - w[k]


def graded_piece(ne, no, w, deg, maxdeg):
    """the weight-`deg` component of vect(ne|no), as a list of monomial fields."""
    out = []
    for m in monomials(ne, no, maxdeg):
        for k in range(ne + no):
            if weight_of((m, k), w, ne) == deg:
                out.append(VF({(m, k): F(1)}, ne, no))
    return out


def span_dim(fields):
    return len(independent(fields))


def in_span(X, basis):
    keys = sorted({k for Y in list(basis) + [X] for k in Y.t}, key=str)
    S = Span({k: i for i, k in enumerate(keys)})
    for Y in basis:
        S.add(to_vec(Y, S.index))
    d0 = S.dim()
    S.add(to_vec(X, S.index))
    return S.dim() == d0


def nilpotent_ideal(ne, no, r, w, maxdeg):
    """I = the part of g_0 whose coefficients lie in Lambda_+(xi_1..xi_r)."""
    g0 = graded_piece(ne, no, w, 0, maxdeg)
    I = []
    for X in g0:
        (m, k) = next(iter(X.t))
        a, S = m
        if any(j < r for j in S):
            I.append(X)
    return independent(g0), independent(I)


def lower_central(I, g0):
    """the lower central series of I inside itself: I, [I,I], [I,[I,I]], ..."""
    steps = [len(independent(I))]
    cur = list(I)
    for _ in range(12):
        nxt = []
        for X in cur:
            for Y in I:
                C = bracket(X, Y)
                if C:
                    nxt.append(C)
        cur = independent(nxt) if nxt else []
        steps.append(len(cur))
        if not cur:
            break
    return steps


def is_ideal(I, g0):
    bad = 0
    for X in I:
        for Y in g0:
            C = bracket(X, Y)
            if C and not in_span(C, I):
                bad += 1
    return bad


def central(I, g0):
    """how many elements of a basis of I commute with all of g_0."""
    c = 0
    for X in I:
        if all(not bracket(X, Y) for Y in g0):
            c += 1
    return c


def classical_dims(ev, od):
    """dims of the classical matrix Lie superalgebras on V = (ev|od)."""
    N = ev + od
    out = {"gl": N * N, "sl": N * N - 1}
    if ev % 2 == 0:
        # osp^a: sp(ev) + o(od) + odd part
        out["osp^a"] = ev * (ev + 1) // 2 + od * (od - 1) // 2 + ev * od
    if od % 2 == 0:
        out["osp^sy"] = ev * (ev - 1) // 2 + od * (od + 1) // 2 + ev * od
    if ev == od:
        n = ev
        out["pe^a"] = out["pe^sy"] = 2 * n * n
        out["c(pe)"] = 2 * n * n + 1
        out["q"] = 2 * n * n
    return out


def derived_series(g):
    steps = [len(independent(g))]
    cur = list(g)
    for _ in range(8):
        nxt = []
        for i, X in enumerate(cur):
            for Y in cur[i:]:
                C = bracket(X, Y)
                if C:
                    nxt.append(C)
        cur = independent(nxt) if nxt else []
        steps.append(len(cur))
        if len(steps) > 2 and steps[-1] == steps[-2]:
            break
    return steps


def run(m, n, r, maxdeg=3):
    ne, no = m, n
    w = [1] * ne + [0] * r + [1] * (no - r)
    say("-" * 78)
    say(f"vect({m}|{n}; {r}):  deg x_i = 1, deg xi_j = 0 for j <= {r}, "
        f"deg xi_j = 1 for j > {r}")
    gm1 = independent(graded_piece(ne, no, w, -1, maxdeg))
    g0 = independent(graded_piece(ne, no, w, 0, maxdeg))
    ev = sum(1 for X in gm1 if not X.par(next(iter(X.t))))
    od = len(gm1) - ev
    say(f"    sdim g_-1 = ({ev}|{od}),  dim g_0 = {len(g0)}")
    cd = classical_dims(ev, od)
    say("    the classical matrix Lie superalgebras on this g_-1 have dim: "
        + ", ".join(f"{k} {v}" for k, v in sorted(cd.items())))
    hits = [k for k, v in cd.items() if v == len(g0)]
    if r == 0:
        say(f"    r = 0: g_0 = gl({m}|{n}) of dim {(m+n)**2}, the matrix algebra "
            "of eq. (38)")
        return
    ds = derived_series(g0)
    say(f"    derived series of g_0: " + " -> ".join(str(x) for x in ds))
    if hits:
        say(f"    dim g_0 matches {', '.join(hits)}: dimension alone does not "
            "decide -- inspect")
    else:
        say(f"    dim g_0 = {len(g0)} matches NONE of them ==> g_0 is not a "
            "classical matrix Lie superalgebra on g_-1, so [ShM] does not apply")


def main():
    global LOG
    LOG = None
    say("What g_0 is in the regradings of table (65)\n")
    for (m, n, r) in [(1, 1, 0), (1, 1, 1), (1, 2, 0), (1, 2, 1), (1, 2, 2),
                      (2, 2, 1), (2, 2, 2), (0, 3, 1), (2, 1, 1)]:
        if r > n:
            continue
        run(m, n, r)
    say("\nDone.")


if __name__ == "__main__":
    main()
