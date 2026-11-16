#!/usr/bin/env python3
"""n4shapeq.py -- read the shape of the N=4 vacuum off n4rec11-vacuum.pkl.

Prints the entries by theta-level, and tests the closed forms the N=2 vacuum
suggests:  is q a scalar times a constant matrix?  is 1/q (equivalently
(mw)^{-1}A times an even element) of the N=2 shape  2/(m W)  with W an
even, nilpotently corrected w?

Usage: ../venv/bin/python n4shapeq.py   ->  n4shapeq-out.txt
"""
import pickle
import sys

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]

from n4noeth import (C, mono, mul, add, smul, sub, eq, minv, mdet, mtr,
                     is_solution, show, MI, PL)              # noqa: E402

OUT = []


def log(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)
    with open("n4shapeq-out.txt", "w") as fh:
        fh.write("\n".join(OUT) + "\n")


NAMES = {1: "eta1-", 2: "rho-1", 3: "eta2-", 4: "rho-2",
         5: "eta1+", 6: "rho+1", 7: "eta2+", 8: "rho+2"}


def pretty(k):
    nz, nw, nm, S = k
    parts = []
    if nz:
        parts.append("z^%d" % nz)
    if nw:
        parts.append("w^%d" % nw)
    if nm:
        parts.append("m^%d" % nm)
    for s in S:
        parts.append(NAMES[s])
    return "*".join(parts) if parts else "1"


def main():
    with open("n4rec11-vacuum.pkl", "rb") as fh:
        q = pickle.load(fh)
    log("the vacuum is an exact solution: %s" % is_solution(q))
    tot = 0
    for i in (0, 1):
        for j in (0, 1):
            f = q[i][j]
            tot += len(f)
            lv = {}
            for k, v in f.items():
                lv.setdefault(len(k[3]), []).append((k, v))
            log("")
            log("q[%d][%d]: %d monomials, by theta-level %s"
                % (i + 1, j + 1, len(f),
                   {L: len(vs) for L, vs in sorted(lv.items())}))
            for L in sorted(lv):
                for k, v in sorted(lv[L], key=lambda t: str(t[0])):
                    log("    %-30s %s" % (pretty(k), v))
    log("")
    log("total monomials: %d" % tot)

    # --- is q a scalar multiple of a constant matrix? ---------------------
    log("")
    log("is q = f * A for a scalar f and a constant A?")
    a = q[0][0]
    same = eq(q[1][1], a)
    offz = (not q[0][1]) and (not q[1][0])
    log("   q11 == q22: %s;  off-diagonal zero: %s" % (same, offz))

    # --- the N=2 shape: q = (2/m) A * W^{-1} with W even ------------------
    log("")
    log("the N=2 vacuum is E = 2/(m W) with W = w + (even nilpotent).")
    log("test at N=4: put F := (2/m) * q11^{-1} and see whether F is")
    log("w + nilpotent, i.e. whether its leading term is w and the rest")
    log("has positive theta-level.")
    qi = minv(q)
    F = smul(2, mul(mono(nm=-1), qi[0][0]))
    lv = {}
    for k, v in F.items():
        lv.setdefault(len(k[3]), []).append((k, v))
    for L in sorted(lv):
        log("   level %d: %s" % (L, ", ".join(
            "%s*(%s)" % (pretty(k), v) for k, v in
            sorted(lv[L], key=lambda t: str(t[0])))))
    log("")
    log("det q, tr q:")
    log("   det q: %d monomials" % len(mdet(q)))
    log("   tr  q: %d monomials" % len(mtr(q)))


if __name__ == "__main__":
    main()
