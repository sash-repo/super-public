#!/usr/bin/env python3
r"""
runbj3.py --- a complete per-degree scan of the sandwich cone of Bj(3; \One|2),
in the standard grading and in each of the two W-gradings, separately for the
even and the odd part, with the normalisers of everything the scan produces.

The scan is complete for the homogeneous elements: the span of the cone is a
graded subspace in every grading of the algebra (t |-> t^i on g_i is an
automorphism, so Vandermonde applies), and every graded component here has
dimension <= 5, so all its F_3-points are run through; the components of
dimension 2 are also run through over F_9.

Output: runs/bj3.log
"""

import os
import sys
import itertools
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bj import (P, N, MONS, mvec, kbracket, parity, rref, inside,            # noqa
                std_component, deg_std, deg_A, deg_B, V_BASIS, prolong)
from runbj import show, submodule, lie_closure, lower_central, normaliser     # noqa

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "bj3.log")


def build():
    k0 = std_component(0)
    W_eta, _ = submodule([mvec(0, 0, 2, 0, 1)], k0)
    g = prolong(list(V_BASIS) + list(W_eta))
    return rref([v for i in sorted(g) for v in g[i]])


def points(dim):
    """one representative per line of F_3^dim."""
    for lead in range(dim):
        for tail in itertools.product(range(P), repeat=dim - lead - 1):
            yield [0] * lead + [1] + list(tail)


def main():
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    log = open(LOG, "w", buffering=1)

    def say(m):
        print(m)
        log.write(m + "\n")

    R, piv = build()
    say("Bj(3; 1|2): dim %d" % len(R))

    ad = {}

    def ad2_zero(v):
        pv = parity(v)
        for b in R:
            if kbracket(v, kbracket(v, b, pf=pv), pf=pv).any():
                return False
        return True

    allsand = {}
    for gname, dg in (("std", deg_std), ("A", deg_A), ("B", deg_B)):
        comp = {}
        for r in R:
            k0_ = [k for k, c in enumerate(r) if c % P][0]
            comp.setdefault((dg(MONS[k0_]), parity(r)), []).append(r)
        say("")
        say("grading %s" % gname)
        found = []
        for key in sorted(comp):
            bs = comp[key]
            hits = []
            for co in points(len(bs)):
                v = np.zeros(N, dtype=np.int64)
                for c, b in zip(co, bs):
                    v = (v + c * b) % P
                if v.any() and ad2_zero(v):
                    hits.append(v)
            if hits:
                Rh, _ = rref(hits)
                found += hits
                say("    g_%-3d %s : dim %d, sandwich lines %d, their span dim %d"
                    % (key[0], "even" if key[1] == 0 else "odd ",
                       len(bs), len(hits), len(Rh)))
        allsand[gname] = found
        Rf, _ = rref(found)
        ev = sum(1 for r in Rf if parity(r) == 0)
        say("    span of all homogeneous sandwiches: sdim (%d|%d)" % (ev, len(Rf) - ev))

    # the union over the three gradings
    total = [v for g in allsand for v in allsand[g]]
    Rt, _ = rref(total)
    ev = sum(1 for r in Rt if parity(r) == 0)
    say("")
    say("union over the three gradings: span sdim (%d|%d)" % (ev, len(Rt) - ev))

    even = [v for v in total if parity(v) == 0]
    Re, _ = rref(even)
    say("even sandwiches: span dim %d" % len(Re))

    for name, gens in (("all sandwiches", total), ("even sandwiches", even)):
        s_rows, s_piv = lie_closure(gens)
        ev = sum(1 for r in s_rows if parity(r) == 0)
        say("")
        say("s from %s: sdim (%d|%d), lcs %s"
            % (name, ev, len(s_rows) - ev, lower_central(s_rows, s_piv)))
        nn, nnp = normaliser(R, piv, s_rows, s_piv)
        ev = sum(1 for r in nn if parity(r) == 0)
        say("    n_L(s): sdim (%d|%d), codim %d" % (ev, len(nn) - ev, len(R) - len(nn)))
        for tag, dg in (("A", deg_A), ("B", deg_B)):
            L0 = rref([r for r in R
                       if dg(MONS[[k for k, c in enumerate(r) if c][0]]) >= 0])
            same = len(L0[0]) == len(nn) and all(inside(nn, nnp, r) for r in L0[0])
            say("        = L_0^%s : %s   (contains L_0^%s: %s)"
                % (tag, same, tag, all(inside(nn, nnp, r) for r in L0[0])))

    log.close()


if __name__ == "__main__":
    main()
