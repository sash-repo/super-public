#!/usr/bin/env python3
"""runseq.py -- the checks of record for the analog of
Hochschild's six-term sequence, for 2-, 2|4- and (2,4)-structures.  Exact over F_2.

Usage:  python runseq.py      (writes runseq.log)
"""

import sys

import dpco
from dpco import bits
from restcx import RLA, Closure
import seq5
from seq5 import Comparison, semilinear_check, theta_check, out, LOG


def mkbr(dim, pairs):
    br = [[0] * dim for _ in range(dim)]
    for (i, j), v in pairs.items():
        br[i][j] = v
        br[j][i] = v
    return br


def bit(*ks):
    v = 0
    for k in ks:
        v |= 1 << k
    return v


def module(dim_alg, Mdim, act):
    return dpco.Module(Mdim, [[act(k, 1 << i) for k in range(dim_alg)]
                              for i in range(Mdim)], [0] * Mdim, "M")


def adj(br, k, v):
    o = 0
    for i in bits(v):
        o ^= br[k][i]
    return o


def run(name, dimp, dimm, br, p2, p4, mods, maxdeg=3):
    """dimm = 0: a 2- or 2|4-structure (L = g); otherwise a (2,4)-structure"""
    out("=" * 78)
    out(name)
    out("=" * 78)
    dim_g = dimp + dimm
    if dimm == 0:
        L = RLA(dim_g, br, 0, list(p2), "L=g")
        cl = None
        dimL = dim_g
        lift_mod = lambda Mdim, act: module(dim_g, Mdim, act)
    else:
        cl = Closure(dimp, dimm, br, p2, p4)
        L = cl.rla()
        dimL = cl.dimL
        out("   restricted closure: dim L = %d, dim u = %d, L closed: %s"
            % (dimL, len(cl.u.basis), "OK" if not cl.problems else "FAILED"))
        lift_mod = lambda Mdim, act: cl.module_L(Mdim, act)
    g_ord = RLA(dim_g, br, dim_g, [], "g")
    bad = L.check()
    out("   L is a Lie algebra with a 2-structure: %s"
        % ("OK" if not bad else "FAILED " + str(bad[:2])))
    for (mn, Mdim, act) in mods:
        Mres = lift_mod(Mdim, act)
        Mord = module(dim_g, Mdim, act)
        cmp_ = Comparison(L, g_ord, dim_g, Mres, Mord, maxdeg=maxdeg)
        cm = cmp_.check_chain_map()
        on = cmp_.check_onto()
        info = cmp_.five_term()
        out("   M = %s" % mn)
        out("      the restriction is a chain map: %s;  onto in every degree: %s"
            % ("OK" if cm == 0 else "FAILED (%d)" % cm,
               "OK" if on == 0 else "FAILED (%d)" % on))
        out("      H^0(N) = %d, H^1(N) = %d  (both must be 0): %s"
            % (info["H0N"], info["H1N"],
               "OK" if info["H0N"] == 0 and info["H1N"] == 0 else "FAILED"))
        out("      0 -> H^1_res(%d) -> H^1(%d) -> H^2(N)(%d) -> H^2_res(%d) -> H^2(%d)"
            % (info["H1res"], info["H1"], info["H2N"], info["H2res"], info["H2"]))
        for (spot, kerdim, imdim) in info["exactness"]:
            out("      exact %-12s dim ker = %2d, dim im = %2d : %s"
                % (spot, kerdim, imdim, "OK" if kerdim == imdim else "FAILED"))
        sl = semilinear_check(cmp_, cl, L, Mres, Mord, dimp, dimm)
        out("      dim M^g = %d;  dim H^2(N) = %d;  expected "
            "(dim g_+ + dim g_-) dim M^g = %d: %s"
            % (sl["dim_inv"], sl["dim_H2N"], sl["expected"],
               "OK" if sl["dim_H2N"] == sl["expected"] else "MISMATCH"))
        ok = (sl["rank_readoff_nice"] == sl["expected"]
              and sl["every_class_has_nice_rep"] and sl["injective"])
        out("      the read-off H^2(N) -> Hom(g_+;M^g) (+) Hom(g_-;M^g): every "
            "class has a\n         representative with values in M^g: %s;  "
            "injective: %s;  rank %d  ==> %s"
            % (sl["every_class_has_nice_rep"], sl["injective"],
               sl["rank_readoff_nice"],
               "AN ISOMORPHISM" if ok else "FAILED"))
        t, b = theta_check(cmp_, cl, L, Mres, Mord, dimp, dimm, br,
                           list(p2), list(p4))
        out("      the connecting map = theta on all %d ordinary 1-cocycles: %s"
            % (t, "OK" if b == 0 else "FAILED (%d values)" % b))
    out()


def main():
    out("runseq.py -- the six-term sequence for p = 2, all restrictednesses")
    out("17 August 2026;  everything exact over F_2")
    out()

    # ---- 2-structures (L = g) ------------------------------------------
    br = mkbr(2, {})
    run("S1  2-structure: abelian dim 2, x^[2] = 0", 2, 0, br, [0, 0], [],
        [("K", 1, lambda k, v: 0), ("g", 2, lambda k, v: adj(br, k, v))])

    br = mkbr(2, {(0, 1): bit(1)})
    run("S2  2-structure: [x,y] = y, x^[2] = x, y^[2] = 0", 2, 0, br,
        [bit(0), 0], [],
        [("K", 1, lambda k, v: 0), ("g", 2, lambda k, v: adj(br, k, v))])

    br = mkbr(3, {(0, 1): bit(2)})
    run("S3  2-structure: Heisenberg dim 3, all [2] = 0", 3, 0, br, [0, 0, 0], [],
        [("K", 1, lambda k, v: 0)])

    # ---- a 2|4-structure -----------------------------------------------
    br = mkbr(4, {(1, 2): bit(2), (1, 3): bit(3), (2, 3): bit(0)})
    run("S4  2|4-structure: gl(1|1), e1^[2] = e1, s(f_i) = 0", 4, 0, br,
        [0, bit(1), 0, 0], [],
        [("K", 1, lambda k, v: 0), ("g", 4, lambda k, v: adj(br, k, v))])

    # ---- (2,4)-structures ----------------------------------------------
    br = mkbr(2, {})
    run("S5  (2,4)-structure: abelian, g_+ = <x>, g_- = <y>", 1, 1, br, [0], [0],
        [("K", 1, lambda k, v: 0), ("g", 2, lambda k, v: adj(br, k, v))])

    br = mkbr(3, {(0, 1): bit(1), (0, 2): bit(2), (1, 2): bit(0)})
    run("S6  (2,4)-structure: o^(1)(3), H^[2] = H, X^[4] = Y^[4] = 0", 1, 2, br,
        [bit(0)], [0, 0],
        [("K", 1, lambda k, v: 0), ("g", 3, lambda k, v: adj(br, k, v))])

    br = mkbr(3, {})
    run("S7  (2,4)-structure: abelian, g_+ = <x1,x2>, g_- = <y>", 2, 1, br,
        [0, 0], [0], [("K", 1, lambda k, v: 0)])

    with open("runseq.log", "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
