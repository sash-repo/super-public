#!/usr/bin/env python3
"""The one-variable factors of the divided power de Rham complex, every p.

Omega^*(m|n;u) is the tensor product of m factors

    F_ev(u_i) = O(1;u_i) (x) <dx_i>,     dx_i odd  (p>2) / divided (p=2)

and n factors

    F_od      = Lambda(xi_j) (x) Gamma(dxi_j),   dxi_j even, divided powers.

Kuenneth then gives H^* of the whole complex.  This script computes H^k of a
single factor for many (p, cap) and fits the closed form the text states.

At p = 2 the two kinds of factor are the SAME object: Lambda(xi) = O(1;1) and
dxi, dx both carry divided powers.  That is the collapse, seen one variable at
a time.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sform as S

LOG = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "sfactor.log"), "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


def factor_cohom(p, kind, u=1, dh=3, kmax=None):
    """H^k of one factor, k = 0..kmax.  kind = 'ev' (x, height p^u) or 'od'."""
    if kind == "ev":
        A = S.Om(1, 0, p, (u,), dh=dh, dxh=dh)
    else:
        A = S.Om(0, 1, p, (), dh=dh, dxh=dh)
    if kmax is None:
        kmax = p + 1
    out = []
    for k in range(kmax + 1):
        h, tot, r1, r0 = S.cohomology(A, k)
        hev = S.cohomology(A, k, 0)[0]
        hod = S.cohomology(A, k, 1)[0]
        out.append((h, hev, hod, tot))
    return out


def predict(p, kind, u=1, kmax=None):
    """The closed form claimed in the text.

    p > 2, kind='ev':  H^0 = K, H^1 = <xbar dx> (xbar = x^{(p^u-1)}), else 0.
    p > 2, kind='od':  H^k = <(dxi)^{(k)}>      if p | k,
                     + <xi (dxi)^{(k)}>         if p | k+1.
    p = 2: BOTH kinds obey the second rule with xbar in place of xi, i.e.
           H^k = <xbar^0 (dx)^{(k)}> if p | k, + <xbar (dx)^{(k)}> if p | k+1,
           where for kind='ev' with height 2^u the class of degree k is
           x^{(2^u-1)}(dx)^{(k)}.
    """
    if kmax is None:
        kmax = p + 1
    out = []
    for k in range(kmax + 1):
        if p > 2 and kind == "ev":
            h = 1 if k in (0, 1) else 0
        else:
            h = (1 if k % p == 0 else 0) + (1 if (k + 1) % p == 0 else 0)
        out.append(h)
    return out


say("(1) one factor, p > 2")
bad = 0
for p in (3, 5, 7):
    for kind in ("ev", "od"):
        for u in ((1, 2) if kind == "ev" else (1,)):
            got = factor_cohom(p, kind, u)
            pred = predict(p, kind, u)
            hs = [g[0] for g in got]
            ok = hs == pred[:len(hs)]
            bad += 0 if ok else 1
            say(f"    p={p} {kind} u={u}: dim H^k = {hs}   predicted {pred[:len(hs)]}"
                f"   {'ok' if ok else 'MISMATCH'}")
            say(f"        parities (ev|od): "
                + " ".join(f"{g[1]}|{g[2]}" for g in got))

say("")
say("(2) one factor, p = 2 -- the two kinds must coincide")
for u in (1, 2):
    gev = factor_cohom(2, "ev", u, dh=3, kmax=5)
    god = factor_cohom(2, "od", 1, dh=3, kmax=5)
    hev = [g[0] for g in gev]
    hod = [g[0] for g in god]
    say(f"    p=2 ev u={u}: dim H^k = {hev}   (dims of Omega^k = "
        f"{[g[3] for g in gev]})")
    if u == 1:
        say(f"    p=2 od     : dim H^k = {hod}   (dims of Omega^k = "
            f"{[g[3] for g in god]})")
        if hev != hod:
            bad += 1
            say("        MISMATCH: the p=2 collapse fails on one variable")
        else:
            say("        the two factors agree -- collapse holds on one variable")
    pred = predict(2, "od", u, kmax=5)
    if hev != pred:
        bad += 1
        say(f"        MISMATCH against the closed form {pred}")
    else:
        say(f"        closed form {pred}: ok")

say("")
say("(3) the degree-2 classes, which are what the classification sees")
say("    factor         p=2      p=3      p=5      p=7")
for kind, u in (("ev", 1), ("ev", 2), ("od", 1)):
    row = []
    for p in (2, 3, 5, 7):
        g = factor_cohom(p, kind, u, kmax=2)
        row.append(f"{g[2][0]}({g[2][1]}|{g[2][2]})")
    say(f"    {kind} u={u}      " + "   ".join(f"{r:8s}" for r in row))

say("")
say("failures:", bad, " (must be 0)")
LOG.close()
