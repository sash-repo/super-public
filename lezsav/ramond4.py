#!/usr/bin/env python3
"""
ramond4.py -- which Laurent vector fields preserve the Ramond-type N=4
Ivanov-Krivonos system, i.e. (IK4) on the double cover t = s^2 together with
the condition M q = qbar (eq. (IK4R) of the text).

One half, superspace C^{1|4}, frame D_i = d_i - theta_i d_t (n4shift.py).
On the cover, t^{1/2} = s is a function, so generating functions are
t^a theta_S with a in (1/2)Z.  Weight grading: wt(t^a theta_S) = 2a + |S|.

The deck transformation M: s -> -s, theta_4 -> -theta_4 (theta_4 = s zeta),
acts on t^a theta_S by (-1)^{2a} (-1)^{[4 in S]}; k^M(1|4) = its fixed points.

The NS invariance algebra of the constraints, k^Xi cap k^Xibar (eq. (KAplus)),
read on the cover: S_d = ker Xi cap ker Xibar in weight d.
A symmetry of the Ramond-type system lies in S cap k^M.
"""
import itertools
from fractions import Fraction as Q
from n4emb import C, I, mono, add, smul, mul, d_t, kbr, IDX, parity
from n4shift import Xi, XiBar, cnullspace, crank, Lam, cD, bD

def keys_of_weight(d):
    out = []
    for k in range(5):
        for S in itertools.combinations(IDX, k):
            out.append((Q(d - k, 2), S))
    return out

def Msign(key):
    a, S = key
    return (-1) ** int(2 * a) * (-1 if 4 in S else 1)

def tovec(f, keys):
    idx = {k: n for n, k in enumerate(keys)}
    v = [C() for _ in keys]
    for k, c in f.items():
        v[idx[k]] = v[idx[k]] + c
    return v

def kernel(keys, ops):
    cols = []
    for j, k in enumerate(keys):
        col = {}
        for t, op in enumerate(ops):
            for kk, cc in op(mono(k[0], k[1])).items():
                col[(t, kk)] = cc
        cols.append(col)
    rk = sorted({r for col in cols for r in col}, key=str)
    M = [[cols[j].get(r, C()) for j in range(len(keys))] for r in rk]
    if not M:
        M = [[C() for _ in keys]]
    out = []
    for vec in cnullspace(M):
        f = {}
        for j, k in enumerate(keys):
            if vec[j]:
                f = add(f, mono(k[0], k[1], vec[j]))
        out.append(f)
    return out

def Mact(f):
    return {k: (c if Msign(k) > 0 else -c) for k, c in f.items()}

def show(f):
    return " + ".join("(%s)t^%s th%s" % (c, a, "".join(map(str, S))) for (a, S), c in sorted(f.items(), key=str))

def run(lo=-4, hi=6):
    for d in range(lo, hi + 1):
        allk = keys_of_weight(d)
        S = kernel(allk, [Xi, XiBar])
        Mk = [k for k in allk if Msign(k) > 0]
        SM = kernel(Mk, [Xi, XiBar])
        SS = S + [Mact(f) for f in S]
        r_sum = crank([tovec(f, allk) for f in SS]) if SS else 0
        inter = len(S) * 2 - r_sum
        ev = [f for f in SM if parity(f) == 0]
        od = [f for f in SM if parity(f) == 1]
        print("wt %3d: dim S=%d  dim S cap M(S)=%d  dim S^M=%d (%d|%d)"
              % (d, len(S), inter, len(SM), len(ev), len(od)))
        for f in SM:
            print("        ", show(f))

if __name__ == "__main__":
    run()

def weak(lo=-4, hi=6):
    """Only the plane F of one constraint: k^Xi cap k^M."""
    tot = []
    for d in range(lo, hi + 1):
        Mk = [k for k in keys_of_weight(d) if Msign(k) > 0]
        K = kernel(Mk, [Xi])
        print("wt %3d: dim k^Xi cap k^M = %d" % (d, len(K)))
        for f in K:
            print("        ", show(f))
        tot += K
    return tot

def algebra():
    """S^M: closure, multipliers, and the restriction of the NS cocycle."""
    gens = []
    for d in range(-6, 9):
        Mk = [k for k in keys_of_weight(d) if Msign(k) > 0]
        gens += [(d, f) for f in kernel(Mk, [Xi, XiBar])]
    print("S^M basis:", [show(f) for _, f in gens])
    allf = [f for _, f in gens]
    keysall = sorted({k for f in allf for k in f}, key=str)
    # closure
    bad = 0
    for f in allf:
        for g in allf:
            b = kbr(f, g)
            if not b:
                continue
            ks = sorted(set(keysall) | set(b), key=str)
            r1 = crank([tovec(h, ks) for h in allf])
            r2 = crank([tovec(h, ks) for h in allf + [b]])
            bad += (r2 != r1)
            print("  {%s, %s} = %s" % (show(f), show(g), show(b)))
    print("closure failures:", bad)
    # multipliers Lam (gl(2)-block) of each generator
    for f in allf:
        L = [[show(Lam(f, a, b)) or "0" for b in (1, 2)] for a in (1, 2)]
        print("  Lam(%s) = %s   d_t f = %s" % (show(f), L, show(d_t(f)) or "0"))

if __name__ == "__main__":
    algebra()
