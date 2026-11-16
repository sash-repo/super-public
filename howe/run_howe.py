#!/usr/bin/env python3
"""Run every check of howe.py.

    python3 run_howe.py
"""
import sys
from fractions import Fraction as F

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from howe import (Model, PeriModel, ops, peri_ops, mult_hb, bracket, compose,
                  equal, is_zero, rho, gens_po, opvec, span_dim, in_span,
                  products, centralizer, contact_fields, contact_alpha,
                  check_osp, check_commuting, check_homomorphism,
                  check_contraction, Op)

OUT = open(os.devnull, "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.write(s + "\n")
    OUT.flush()


def proportional(A, B, mod, deg):
    """The scalar c with A = c*B, or None."""
    c = None
    for m in mod.basis(deg):
        v, w = A({m: F(1)}), B({m: F(1)})
        if not v and not w:
            continue
        if not v or not w:
            return None
        k = sorted(v)[0]
        if k not in w:
            return None
        cc = v[k] / w[k]
        if c is None:
            c = cc
        if {kk: c * vv for kk, vv in w.items()} != v:
            return None
    return c


say("=" * 78)
say("howe.py -- the flat models of the Howe pairs.  Exact arithmetic (Fraction).")
say("=" * 78)

# ---------------------------------------------------------------- symplectic
for n, deg in ((1, 4), (2, 3)):
    mod = Model(n, 0)
    say("\n### the symplectic flat model, sdim (%d|0), monomials of degree <= %d: %d"
        % (2 * n, deg, len(mod.basis(deg))))
    o = ops(mod, F(0))
    say("  [X+,X-] = (form degree) - n :", equal(o["H"], o["Hexp"], mod, deg)[0])
    for hbar in (F(1), F(2), F(-3, 5)):
        o = ops(mod, hbar)
        say("  hbar = %-5s : [D+,D+] = %s * hbar*X+ ; [D-,D-] = %s * hbar*X- ; "
            "[D+,D-] = %s * hbar*H"
            % (hbar,
               proportional(bracket(o["Dp"], o["Dp"]), o["Xp"].scaled(hbar), mod, deg),
               proportional(bracket(o["Dm"], o["Dm"]), o["Xm"].scaled(hbar), mod, deg),
               proportional(bracket(o["Dp"], o["Dm"]), o["H"].scaled(hbar), mod, deg)))
        say("               [H,D+]=D+ %s ; [H,D-]=-D- %s ; [X+,D-]=D+ %s ; "
            "[X-,D+]=D- %s"
            % (equal(bracket(o["H"], o["Dp"]), o["Dp"], mod, deg)[0],
               equal(bracket(o["H"], o["Dm"]), o["Dm"].scaled(-1), mod, deg)[0],
               equal(bracket(o["Xp"], o["Dm"]), o["Dp"], mod, deg)[0],
               equal(bracket(o["Xm"], o["Dp"]), o["Dm"], mod, deg)[0]))
    say("  => D+/sqrt(-hbar), D-/sqrt(-hbar) satisfy eq. (5) verbatim.")

    for hbar in (F(0), F(1), F(2)):
        bad = check_commuting(mod, hbar, deg, fdeg=4)
        hom, central = check_homomorphism(mod, hbar, deg, fdeg=3)
        say("  hbar = %-3s : rho commutes with X+,X-,H,D+,D- for every f of degree"
            " <= 4 : %s ; rho a homomorphism (f,g of degree <= 3) : %s ; "
            "rho(1) = -hbar : %s"
            % (hbar, not bad, not hom, central))

    say("  hbar = 0 :", check_contraction(mod, min(deg, 3)))

# ------------------------------------------------------- the centralizer, n=1
mod = Model(1, 0)
say("\n### the centralizer of the po(2)-action (coefficient degree <= 2, order <= 2)")
for hbar in (F(1), F(0)):
    sols, nans = centralizer(mod, hbar, cmax=2, omax=2, fdeg=3, deg=4)
    o = ops(mod, hbar)
    gens = ([o["Xp"], o["Xm"], o["H"], o["Dp"], o["Dm"]] if hbar
            else [o["Xp"], o["Xm"], o["H"], o["d"], o["delta"]])
    pv = products(gens, mod, 4)
    dimP, _ = span_dim(pv)
    dimC, _ = span_dim([opvec(s, mod, 4) for s in sols])
    both, _ = span_dim(pv + [opvec(s, mod, 4) for s in sols])
    say("  hbar = %s : ansatz of %d operators ; centralizer of dim %d ; the span of"
        " 1, the five operators and their pairwise products has dim %d ; joint span"
        " %d  => equal: %s" % (hbar, nans, dimC, dimP, both, both == dimC == dimP))

# ------------------------------------------------------------------- contact
say("\n### the contactization: alpha = dt + sum p_i dq_i, fields of degree <= 2")
for n in (1, 2):
    cm = Model(n, 0, contact=True)
    sols, unk = contact_fields(cm, 2)
    strict, mism = 0, 0
    for vec in sols:
        comps, u = {}, {}
        for c, un in zip(vec, unk):
            if not c:
                continue
            if un[0] == "u":
                u[un[1]] = c
            else:
                comps.setdefault(un[0], {})[un[1]] = c
        f = dict(comps.get(cm.t, {}))
        for i in range(cm.n):
            for e, c in comps.get(cm.q[i], {}).items():
                ee = list(e)
                ee[cm.p[i][1]] += 1
                f[tuple(ee)] = f.get(tuple(ee), F(0)) + c
        dft = {}
        for e, c in f.items():
            k = e[cm.t[1]]
            if k:
                ee = list(e)
                ee[cm.t[1]] -= 1
                dft[tuple(ee)] = dft.get(tuple(ee), F(0)) + c * k
        if {k: v for k, v in u.items() if v} != {k: v for k, v in dft.items() if v}:
            mism += 1
        if not any(u.values()):
            strict += 1
            if any(v for k, v in f.items() if k[cm.t[1]]):
                mism += 1
    say("  n = %d : %d contact fields ; the multiplier is df/dt in every one of them"
        " (mismatches: %d) ; those with L_K alpha = 0 : %d, all with df/dt = 0"
        % (n, len(sols), mism, strict))
say("  => alpha^ commutes with po(2n) = {f | df/dt = 0} and with no other contact field.")

# --------------------------------------------------------------- periplectic
say("\n### the periplectic flat model, omegatilde odd, hbar an ODD parameter")
for n in (1, 2):
    pm = PeriModel(n)
    o = peri_ops(pm)
    deg = 3
    Xp, Xm, H, Dp, Dm = o["Xp"], o["Xm"], o["H"], o["Dp"], o["Dm"]
    say("  n = %d : [X+,X+] = 0 : %s ; [X-,X-] = 0 : %s ; H = [X+,X-] commutes with"
        " X+, X- : %s   (this is sl(1|1))"
        % (n, is_zero(bracket(Xp, Xp), pm, deg), is_zero(bracket(Xm, Xm), pm, deg),
           is_zero(bracket(H, Xp), pm, deg) and is_zero(bracket(H, Xm), pm, deg)))
    say("        [D+,D+] = %s * hbar*X+  (NOT zero: the square root exists over"
        " Lambda(hbar))" % proportional(bracket(Dp, Dp), mult_hb(pm, Xp), pm, deg))
    say("        [D+,D-] = %s * hbar*H ; [D-,D-] = 0 : %s ; [X-,D-] = 0 : %s"
        % (proportional(bracket(Dp, Dm), mult_hb(pm, H), pm, deg),
           is_zero(bracket(Dm, Dm), pm, deg), is_zero(bracket(Xm, Dm), pm, deg)))
    cand = [Xp, Xm, H, Dp, Dm] + [mult_hb(pm, A) for A in (Xp, Xm, H, Dp, Dm)]
    dim, bas = span_dim([v for v in (opvec(A, pm, deg) for A in cand) if v])
    closed = all(in_span(bas, opvec(bracket(A, B), pm, deg)) or
                 not opvec(bracket(A, B), pm, deg)
                 for A in cand for B in cand)
    say("        the ten operators A, hbar*A span dim %d ; closed under the bracket:"
        " %s" % (dim, closed))

OUT.close()
