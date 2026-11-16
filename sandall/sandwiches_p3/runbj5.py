#!/usr/bin/env python3
r"""runbj5.py --- three loose ends of the Bj computation:
 1) how few sandwiches already generate a non-nilpotent subalgebra;
 2) the ad L_0^X-closure of the even sandwiches, for X = A, B;
 3) the two maximal subalgebras and their intersection.
Output: runs/bj5.log"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bj import (P, N, MONS, parity, rref, inside, deg_std, deg_A, deg_B, kbracket)
from runbj import show, lie_closure, lower_central, normaliser
from runbj3 import build, points
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "bj5.log")

def main():
    log = open(LOG, "w", buffering=1)
    def say(m):
        print(m); log.write(m + "\n")
    R, piv = build()
    def ad2_zero(v):
        pv = parity(v)
        return not any(kbracket(v, kbracket(v, b, pf=pv), pf=pv).any() for b in R)
    sand = []
    for dg in (deg_std, deg_A, deg_B):
        comp = {}
        for r in R:
            k0 = [k for k, c in enumerate(r) if c % P][0]
            comp.setdefault((dg(MONS[k0]), parity(r)), []).append(r)
        for key, bs in comp.items():
            for co in points(len(bs)):
                v = np.zeros(N, dtype=np.int64)
                for c, b in zip(co, bs):
                    v = (v + c * b) % P
                if v.any() and ad2_zero(v):
                    sand.append(v)
    odd = [v for v in sand if parity(v) == 1]
    even = [v for v in sand if parity(v) == 0]
    seen, uniq = set(), []
    for v in odd:
        k = tuple(int(c) for c in v)
        if k in seen: continue
        seen.add(k); uniq.append(v)
    say("distinct odd sandwich lines (with both scalings): %d" % len(uniq))
    # 1) greedy: grow a set until non-nilpotent
    cur = []
    for v in uniq:
        cur.append(v)
        rows, pv = lie_closure(cur)
        lcs = lower_central(rows, pv)
        if lcs[-1] != 0:
            say("1) a set of %d odd sandwiches generating a NON-nilpotent subalgebra:" % len(cur))
            for w in cur: say("      %s" % show(w))
            say("   they generate dim %d of %d, lower central series %s" % (len(rows), len(R), lcs))
            break
    else:
        say("1) the whole set of %d generates dim %d" % (len(uniq), len(lie_closure(uniq)[0])))
    # 2) closures
    say("")
    L0 = {}
    for tag, dg in (("A", deg_A), ("B", deg_B)):
        L0[tag] = rref([r for r in R if dg(MONS[[k for k,c in enumerate(r) if c][0]]) >= 0])
        cl, clp = lie_closure(even, extra=list(L0[tag][0]))
        ev = sum(1 for r in cl if parity(r) == 0)
        ds = sorted({dg(MONS[[k for k,c in enumerate(r) if c][0]]) for r in cl})
        say("2) X = %s : ad L_0^X-closure of the even sandwiches: sdim (%d|%d), dim %d, degrees %s, lcs %s"
            % (tag, ev, len(cl)-ev, len(cl), ds, lower_central(cl, clp)))
    # 3) intersection
    say("")
    A, Ap = L0["A"]; B, Bp = L0["B"]
    inter = [r for r in R if inside(A, Ap, r) and inside(B, Bp, r)]
    # honest intersection: solve
    M = np.concatenate([A, B])
    from bj import rref as rr
    rk = len(rr(M)[0])
    say("3) dim L_0^A = %d, dim L_0^B = %d, dim(L_0^A + L_0^B) = %d, so dim of the intersection = %d"
        % (len(A), len(B), rk, len(A)+len(B)-rk))
    log.close()

if __name__ == "__main__":
    main()
