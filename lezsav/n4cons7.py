#!/usr/bin/env python3
"""n4cons7.py -- and if the completion is a LOGARITHM of a current?

T = -u = (1/2) log det q is itself a logarithm, and n4cons3.py showed that
the obstruction to completing it inside the ring is a residue: the level-0
term of d_- T is -1/w, and 1/w is the one thing d_w never produces.  A
polynomial composite therefore cannot do it, whatever its shape -- which is
what n4cons5.py and n4cons6.py measured, list after list.

But a logarithm can.  For any invertible composite f,
        d_w log f = (d_w f) f^{-1},
which is of grade -1 whatever the grade of f, i.e. of exactly the grade the
target has; and log f is as local a functional of the field as f is.  So
this run adjoins the columns d_w log f for every invertible scalar composite
f on the list -- the four tr(M Q), the eight even first-order scalars, the
second-order ones -- to the 1332 polynomial candidates of n4cons6.py.

Usage: ../venv/bin/python n4cons7.py   ->  n4cons7-out.txt
"""
import pickle
import sys
import time

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]

from n4noeth import (C, mono, mul, add, smul, eq, minv, dT, csolve,
                     d_xm, d_xp, is_solution, mmul, mtr, mapply,
                     inv_scalar, lead_of)                      # noqa: E402
from n4cons2 import SIG, scalars                               # noqa: E402
from n4cons5 import grades, homog                              # noqa: E402
from n4cons6 import second_order                               # noqa: E402

OUT = []


def log(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)
    with open("n4cons7-out.txt", "w") as fh:
        fh.write("\n".join(OUT) + "\n")


def logcols(fs, dop):
    """d log f = (dop f) f^{-1} for every invertible f."""
    out = {}
    for nm, f in fs.items():
        try:
            fi = inv_scalar(f, lead_of(f))
        except ValueError:
            continue
        c = mul(dop(f), fi)
        if c:
            out[nm] = c
    return out


def main():
    t0 = time.time()
    with open("n4rec11-vacuum.pkl", "rb") as fh:
        q = pickle.load(fh)
    log("n4cons7: logarithms of the currents as candidate completions.")
    log("")
    log("the vacuum is an exact solution: %s" % is_solution(q))
    qi = minv(q)
    ev, od = scalars(q)
    so = second_order(q, qi)
    fs = {}
    for mn, M in SIG.items():
        f = mul(mono(nm=-1), mtr(mmul(M, qi)))
        if f:
            fs[("Q", mn)] = f
    for k, f in ev.items():
        if f:
            fs[("e", k)] = f
    for k, f in so.items():
        fs[("2", k)] = f
    log("scalar composites offered to the logarithm: %d" % len(fs))

    for dop, dopname in ((d_xm, "d_-"), (d_xp, "d_+")):
        target = dT(q, qi, dop)
        lc = logcols(fs, dop)
        cols = dict(lc)
        cols[("log", "w")] = mono(nw=-1)
        names = sorted(cols, key=str)
        keys = sorted(set(target) | {k for f in cols.values() for k in f})
        bylev = {}
        for k in keys:
            bylev.setdefault(len(k[3]), []).append(k)
        log("")
        log("dop = %s: %d invertible composites, so %d log columns"
            % (dopname, len(lc), len(names)))
        log("   rows by theta-level %s"
            % {L: len(ks) for L, ks in sorted(bylev.items())})
        sol = None
        for cut in sorted(bylev):
            rows = [k for L in sorted(bylev) if L <= cut for k in bylev[L]]
            cl = [[cols[n_].get(k, C()) for k in rows] for n_ in names]
            tg = [-target.get(k, C()) for k in rows]
            x = csolve(cl, tg)
            log("   levels <= %d (%3d rows): %s  [%.1fs]"
                % (cut, len(rows),
                   "solvable" if x is not None else "INCONSISTENT",
                   time.time() - t0))
            if x is None:
                break
            sol = (x, names)
        if sol and cut == max(bylev):
            x, names = sol
            nz = [(names[j], x[j]) for j in range(len(names)) if x[j]]
            log("   COMPLETION FOUND: %d nonzero coefficients" % len(nz))
            for n_, v in sorted(nz, key=lambda t: str(t[0])):
                log("      log %-42s %s" % (str(n_), v))
    log("")
    log("[%.1fs] done" % (time.time() - t0))


if __name__ == "__main__":
    main()
