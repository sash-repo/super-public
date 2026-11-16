#!/usr/bin/env python3
"""The kernel half of the Problem on three connections, block by form degree.

D_a^- lowers the form degree by exactly 1 and X_a^- by exactly 2, so the whole
question splits by form degree; each block is solved on its own.  Verified:
whether  ker D_1^- (^) ker D_2^-  is contained in  ker X_3^-  (twisted), and the
contrast with  ker X_1^- (^) ker X_2^-  vs  ker X_3^-  (untwisted).
"""
import sys
from fractions import Fraction as F
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
import hk2
from hk2 import build, safe_columns, op_lin, nullspace


def blocks(cols):
    out = {}
    for b in cols:
        out.setdefault(len(b[1]), []).append(b)
    return out


def kernel_in(block, ops):
    M = []
    for A in ops:
        f = A[0]
        imgs, keys = [], set()
        for b in block:
            img = f({b: F(1)})
            imgs.append(img)
            keys |= set(img)
        for k in sorted(keys, key=repr):
            M.append([img.get(k, F(0)) for img in imgs])
    if not M:
        return [[F(1) if i == j else F(0) for i in range(len(block))]
                for j in range(len(block))]
    return nullspace(M)


def run(m, pdeg, hbars):
    B = build(m, pdeg)
    T = B["T"]
    cols = safe_columns(T, margin=2)
    delta, iota, XM = B["delta"], B["iota"], B["XM"]
    bl = blocks(cols)
    h1, h2 = hbars
    Dm = [op_lin([(F(1), delta[a]), (h, iota[a])]) for a, h in ((0, h1), (1, h2))]
    print(f"--- m={m}, pdeg={pdeg}, hbar=({h1},{h2}); "
          f"test space: forms with coefficients of degree <= {pdeg-2}")
    tw_ok = untw_ok = True
    for i in sorted(bl):
        block = bl[i]
        KD = kernel_in(block, Dm)
        bad = 0
        for v in KD:
            vec = {b: v[j] for j, b in enumerate(block) if v[j]}
            if XM[2][0](vec):
                bad += 1
        KX = kernel_in(block, [XM[0], XM[1]])
        KX3 = kernel_in(block, [XM[0], XM[1], XM[2]])
        if bad:
            tw_ok = False
        if len(KX) != len(KX3):
            untw_ok = False
        print(f"   i={i:2d}: dim block={len(block):5d}   "
              f"ker D_1^-(^)D_2^- = {len(KD):5d} (killed by X_3^- too: "
              f"{len(KD)-bad:5d})   "
              f"ker X_1^-(^)X_2^- = {len(KX):5d}  (^)X_3^- = {len(KX3):5d}")
    print(f"   twisted inclusion holds in every degree: {tw_ok};   "
          f"untwisted: {untw_ok}")
    return tw_ok, untw_ok




def run_single(m, pdeg, h1):
    """the one-connection (Kaehler) case: ker D^- degree by degree -- the
    calibration that says whether the polynomial model sees the square root."""
    B = build(m, pdeg)
    T = B["T"]
    cols = safe_columns(T, margin=2)
    delta, iota, XM = B["delta"], B["iota"], B["XM"]
    bl = blocks(cols)
    D1 = op_lin([(F(1), delta[0]), (h1, iota[0])])
    D2 = op_lin([(F(1), delta[1]), (h1, iota[1])])
    print(f"--- one connection: m={m}, pdeg={pdeg}, hbar={h1}")
    for i in sorted(bl):
        block = bl[i]
        K1 = kernel_in(block, [D1])
        KX = kernel_in(block, [XM[0]])
        K12 = kernel_in(block, [D1, D2])
        print(f"   i={i:2d}: dim block={len(block):5d}   ker D_1^- = {len(K1):5d}"
              f"   ker X_1^- = {len(KX):5d}   ker D_1^-(^)D_2^- = {len(K12):5d}")


if __name__ == "__main__":
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    pdeg = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    for hb in ((F(1), F(1)), (F(1), F(3))):
        run(m, pdeg, hb)
    run_single(m, pdeg, F(1))
