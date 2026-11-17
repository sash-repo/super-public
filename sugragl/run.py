#!/usr/bin/env python3
"""
run.py -- compute the structure functions of a Minkowski-superspace model and
print them in the format of the paper's Table.

    ../../tools/venv/bin/python run.py 00100 001100 ...
    ../../tools/venv/bin/python run.py --calibrate      (the Table's columns)
    ../../tools/venv/bin/python run.py --new            (the new models)

For each column both cases are computed: "Riemannian" (g_0 semisimple) and
"conformal" (g_0 the whole stationary subalgebra).  The Table prints the
Riemannian answer and boxes what survives conformally; a class present only in
the conformal case is what the Table doubleboxes.  Here all three are listed.
"""

import sys
import time

from models import Prolong, backend
from cohom import h2, irreps, fmt

# the columns Grozman computed with the SuperLie package (the Table)
CALIBRATE = ["0000", "00100", "00001", "001100", "100001", "000011",
             "00111100", "00001111", "111100001111"]

# the palindromic all-size-2 models not in the Table (see census.py)
NEW = ["00110011", "0011111100", "1100110011",
       "001111111100", "110011110011"]


def one(par, kinds=("riem", "full"), verbose=True):
    res = {}
    for kind in kinds:
        t0 = time.time()
        P = Prolong(par, kind)
        H, c2d = h2(P)
        res[kind] = (P, H, c2d, time.time() - t0)
    return res


def report(par, res):
    P0 = res["riem"][0]
    M = P0.M
    print("=" * 78)
    print(f"Par = ({par})   N = {M.Nsusy}   blocks "
          + "|".join(f"{s}{'o' if p else 'e'}" for s, p in M.blocks)
          + f"   depth {P0.depth}")
    nev = sum(1 for a in range(P0.ngm) if P0.gm_par[a] == 0)
    print(f"  sdim g_- = ({nev}|{P0.ngm - nev})"
          f"   sl(2) factors: {len(M.sl2)}")
    for kind in res:
        P, H, c2d, dt = res[kind]
        dims = ", ".join(f"g_{m}:{P.dim(m)}" for m in sorted(P.basis))
        print(f"  [{kind:4s}] prolong  {dims}    ({dt:.1f} s)")
    degs = sorted(set(sum([list(res[k][1]) + list(res[k][2]) for k in res], [])))
    hdr = "  %4s  %-40s %-28s %s" % ("deg", "Riemannian", "conf (ss+z)", "full block diag")
    print(hdr)
    for d in degs:
        cells = []
        for k in ("riem", "conf", "full"):
            if k not in res:
                cells.append("")
                continue
            r = irreps(res[k][0], res[k][1].get(d, {}))
            cells.append(fmt(r) if r else ("---" if res[k][2].get(d)
                                           else "not defined"))
        print("  %4d  %-40s %-28s %s" % (d, cells[0], cells[1], cells[2]))
    print()


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        args = ["--calibrate"]
    pars = []
    for a in args:
        if a == "--calibrate":
            pars += CALIBRATE
        elif a == "--new":
            pars += NEW
        else:
            pars.append(a)
    print(f"exact linear algebra backend: {backend()}\n")
    for par in pars:
        t0 = time.time()
        report(par, one(par))
        sys.stdout.flush()
