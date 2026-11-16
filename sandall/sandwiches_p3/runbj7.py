#!/usr/bin/env python3
r"""
runbj7.py --- Theorem ThBj under the repaired definition of a sandwich.

runbj6.py established that, for Bj(3; \One|2),

    s(Bj) = <all s with (ad_s)^2 = 0 AND [[g,s],[g,s]] = 0>

is of sdim (8|5), dim 13, and ABELIAN -- so the Zelmanov--Kostrikin theorem
holds for it, and with it the tie-breaker Lemma LemTie.  Here we ask what the
normaliser of that subalgebra is, i.e., what the third reading of the
definition (sa) gives in Theorem ThBj, and whether the tie-breaker separates
the two W-gradings A and B now that it applies.

Output: runs/bj7.log
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bj import (P, N, MONS, parity, rref, inside, deg_std, deg_A, deg_B,      # noqa
                kbracket)
from runbj import show, lie_closure, lower_central, normaliser
from runbj3 import build, points
from runbj6 import ad2_zero, cond8

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "bj7.log")


def sweep(R, parity_wanted):
    out, seen = [], set()
    for dg in (deg_std, deg_A, deg_B):
        comp = {}
        for r in R:
            k0 = [k for k, c in enumerate(r) if c % P][0]
            comp.setdefault((dg(MONS[k0]), parity(r)), []).append(r)
        for (d, par), bs in comp.items():
            if par != parity_wanted:
                continue
            for co in points(len(bs)):
                v = np.zeros(N, dtype=np.int64)
                for c, b in zip(co, bs):
                    v = (v + c * b) % P
                if not v.any() or not ad2_zero(v, R):
                    continue
                k = tuple(int(c) for c in v)
                if k not in seen:
                    seen.add(k)
                    out.append(v)
    return out


def main():
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    log = open(LOG, "w", buffering=1)

    def say(m):
        print(m)
        log.write(m + "\n")

    R, piv = build()
    say("Bj(3; \\One|2): dim %d" % len(R))

    odd = [v for v in sweep(R, 1) if cond8(v, R)]
    even = sweep(R, 0)
    bad = [v for v in even if not cond8(v, R)]
    assert not bad
    say("odd sandwiches (repaired definition) found: %d" % len(odd))
    say("even sandwiches: %d, spanning dim %d" % (len(even), len(rref(even)[0])))

    S, spiv = lie_closure(odd + even)
    lcs = lower_central(S, spiv)
    nev = sum(1 for r in S if parity(r) == 0)
    say("s(Bj): sdim (%d|%d), dim %d, lower central series %s"
        % (nev, len(S) - nev, len(S), lcs))
    say("   a basis: %s" % "; ".join(show(r) for r in S))

    Nrm, npiv = normaliser(R, piv, S, spiv)
    nev = sum(1 for r in Nrm if parity(r) == 0)
    say("")
    say("n_g(s): sdim (%d|%d), dim %d, codim %d"
        % (nev, len(Nrm) - nev, len(Nrm), len(R) - len(Nrm)))

    for tag, dg in (("A", deg_A), ("B", deg_B)):
        L0 = [r for r in R if dg(MONS[[k for k, c in enumerate(r) if c % P][0]]) >= 0]
        L0r, L0p = rref(L0)
        contained = bool(len(Nrm)) and all(inside(Nrm, npiv, r) for r in L0r)
        # the tie-breaker: is s stable under ad of L_0^X ?
        stable = True
        for h in L0r:
            for r in S:
                v = kbracket(h, r, pf=parity(h))
                if v.any() and not inside(S, spiv, v):
                    stable = False
                    break
            if not stable:
                break
        say("   L_0^%s: dim %d; contained in n_g(s): %s; s is ad L_0^%s-stable: %s"
            % (tag, len(L0r), contained, tag, stable))
        rows, pv = lie_closure(list(S), extra=list(L0r))
        d = lower_central(rows, pv)
        nev2 = sum(1 for r in rows if parity(r) == 0)
        say("      the ad L_0^%s-closure of s: sdim (%d|%d), dim %d, lcs %s"
            % (tag, nev2, len(rows) - nev2, len(rows), d))

    say("")
    say("done.")
    log.close()


if __name__ == "__main__":
    main()
