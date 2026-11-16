#!/usr/bin/env python3
"""n4prop.py -- on every solution in the orbit, d_+ and d_- are proportional.

The mirror and the boost act on the plane spanned by z and w LINEARLY, and
every element of the graded class is a function of w alone; so every solution
in the orbit is a function of a single light-cone combination, and there
d_+ = kappa d_- for an invertible constant kappa.  That is why T + X came out
killed by both derivatives: not because the completion is pinned, but because
none of these solutions can tell the two halves apart.

This run measures kappa on all three kinds.
"""
import pickle, sys, time
from fractions import Fraction as Q_
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from n4noeth import (C, add, smul, eq, minv, dT, d_xm, d_xp, is_solution, show,
                     mul, inv_scalar, lead_of)
from n4rec import qadd, qsmul
from n4emb import C as CC
import n4mirror as MIR
import n4boost as BST

OUT = []
def log(*a):
    s = " ".join(str(x) for x in a); OUT.append(s); print(s, flush=True)
    open("n4prop-out.txt","w").write("\n".join(OUT) + "\n")

def ratio(f, g):
    """the constant kappa with f = kappa g, or None."""
    if not g: return None
    try:
        gi = inv_scalar(g, lead_of(g))
    except ValueError:
        return None
    r = mul(f, gi)
    # a CONSTANT of the ring: no z, no w, no m, and no coordinate theta --
    # the labels 9..12 are the odd constants, which may occur nilpotently.
    for (a, b, c, S) in r:
        if a or b or c or any(j <= 8 for j in S):
            return ("nonconstant", len(r))
    return show(r, cap=4)

def main():
    t0 = time.time()
    q = pickle.load(open("n4rec11-vacuum.pkl","rb"))
    tans = pickle.load(open("n4cons-tan.pkl","rb"))
    log("n4prop: d_+ versus d_- on the three kinds of solution.")
    log("")
    fam = [("vacuum", q)]
    for d in tans:
        for lam in (1, -1, Q_(1,2), 2, 3, CC(0,1), CC(1,1), Q_(-1,3)):
            qq = qadd(q, qsmul(lam, d))
            if is_solution(qq): fam.append(("q + (%s) d" % lam, qq))
    kinds = [("vacuum (z-free)", [(n,x) for n,x in fam]),
             ("mirror (w-free)", [(n, MIR.sigma(x)) for n,x in fam]),
             ("boosted (z and w)", [(n, BST.boost(x)) for n,x in fam])]
    for kname, sols in kinds:
        log("=" * 66); log(kname); log("=" * 66)
        seen = {}
        for nm, qq in sols:
            qi = minv(qq)
            tm, tp = dT(qq, qi, d_xm), dT(qq, qi, d_xp)
            r = ratio(tp, tm)
            key = str(r)
            seen[key] = seen.get(key, 0) + 1
        for k, v in sorted(seen.items()):
            log("   d_+ T = kappa . d_- T with kappa = %-24s on %d of %d"
                % (k, v, len(sols)))
        # and on a sample, d_z T versus d_w T
        nm, qq = sols[0]
        qi = minv(qq)
        log("   sample: d_- T = %s" % show(dT(qq, qi, d_xm), cap=3))
        log("           d_+ T = %s" % show(dT(qq, qi, d_xp), cap=3))
        log("   [%.1fs]" % (time.time()-t0)); log("")
    log("[%.1fs] done" % (time.time()-t0))

main()
