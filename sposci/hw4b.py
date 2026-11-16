#!/usr/bin/env python3
r"""
hw4b.py -- the sharp form of hw4.py: the level-0 subspace of Spin(V) at n = 4,
and what the labels (c, h; H_1, H_2) = (0, 0; 0, 0) leave out.

The question:  all four components of the highest weight vanish;
is the module then trivial, is it irreducible, is the trivial module a quotient?

The computation, all exact:

  1) g_0 = C K_t (+) o(4) (+) C Z with Z = K_{t^{-1} w_1 w_2 w_3 w_4}
     (Proposition 7.3.16), Z central in g_0 (verified, not assumed).

  2) hat Z does NOT fix the vacuum; the vectors hat Z^k . |0>, k = 0, 1, 2, ...
     are linearly independent, all of level 0, and all carry the SAME four
     labels (0, 0; 0, 0).  So the weight space of the weight (0,0,0,0) is
     infinite-dimensional and the vacuum is not a highest weight vector for
     the whole of g_0.

  3) M_0 = U(g_0).|0> is exactly that tower (checked by closure).

  4) g.M cap (level 0): is the vacuum there?  This decides whether the trivial
     module is a quotient of M = U(g).|0>.

  5) g-invariant vectors of the level-0 tower: the trivial module as a
     submodule.

Usage:  ../../tools/venv/bin/python hw4b.py [N]
"""

import sys
import time
from fractions import Fraction as Q

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "lib"))

from ns4 import (SUBSETS, mono, act_L, act_M, br_L, br_M, TORUS_L, TORUS_M,   # noqa: E402
                 VAC, psi, hatX, sadd)
import runns4 as R                                                            # noqa: E402
from hw4 import build, level_of_mode, level_shift, g_basis, Span              # noqa: E402
from exactlin import rank as exrank                                           # noqa: E402


SECTORS = {"L": (act_L, br_L, TORUS_L), "M": (act_M, br_M, TORUS_M)}


def g0_keys(sector):
    r"""The basis of g_0: 2a + |S| = 2 for k^L; for k^M the grading counts
    deg zeta = 0, so 2a + |S \ {zeta}| = 2."""
    out = []
    for a in range(-2, 3):
        for S in SUBSETS:
            d = len(S) if sector == "L" else len([i for i in S if i != 1])
            if 2 * a + d == 2:
                out.append((Q(a), S))
    return out


def name(k):
    return "K_{t^%s w^%s}" % (k[0], "".join(map(str, k[1])) or "1")


def central_in_g0(sector):
    br = SECTORS[sector][1]
    ks = g0_keys(sector)
    cen = []
    for k in ks:
        if all(not br(mono(*k), mono(*j)) for j in ks):
            cen.append(k)
    return ks, cen


# ------------------------------------------------------------------ the tower

def tower(W, Zk, kmax=8):
    """|0>, hat Z |0>, hat Z^2 |0>, ...  -- as far as the window allows."""
    out, st = [], VAC
    for k in range(kmax + 1):
        out.append(st)
        st = hatX(W, mono(*Zk), st)
        if not st:
            break
    return out


def independent(states):
    sp = Span()
    n = 0
    for st in states:
        if st and sp.add(dict(st)):
            n += 1
        else:
            break
    return n


def labels_of(W, br, torus, st):
    """(c, h; H) of a level-0 vector, by the same formulas as runns4.labels,
    but on an arbitrary vector instead of the vacuum."""
    def ev(f, v):
        img = hatX(W, f, v)
        rest = {k: c for k, c in sadd(img, v, Q(0)).items()}
        # eigenvalue: img must be a multiple of v
        if not img:
            return Q(0)
        ks = set(img) | set(v)
        lam = None
        for k in ks:
            a, b = img.get(k, Q(0)), v.get(k, Q(0))
            if b == 0:
                if a != 0:
                    return None
                continue
            l = a / b
            if lam is None:
                lam = l
            elif lam != l:
                return None
        return lam if lam is not None else Q(0)

    h = ev(mono(1), st)
    H = tuple(ev(mono(*g), st) for g in torus)
    c = R.charge(_OnVec(W, st), br) if False else None
    return h, H


class _OnVec:
    pass


def charge_on(W, br, st, ms=(2, 3, 4)):
    """The central charge read off [hat L_m, hat L_-m] on the vector st."""
    vals = []
    for m in ms:
        f, g = mono(m + 1), mono(1 - m)
        lhs = sadd(hatX(W, f, hatX(W, g, st)), hatX(W, g, hatX(W, f, st)), Q(-1))
        b = br(f, g)
        if b:
            lhs = sadd(lhs, hatX(W, b, st), Q(-1))
        # lhs must be a multiple of st
        lam = None
        for k in set(lhs) | set(st):
            a, bb = lhs.get(k, Q(0)), st.get(k, Q(0))
            if bb == 0:
                if a != 0:
                    return None
                continue
            l = a / bb
            if lam is None:
                lam = l
            elif lam != l:
                return None
        vals.append((m, lam if lam is not None else Q(0)))
    (m1, v1), (m2, v2) = vals[0], vals[1]
    det = m1 ** 3 * m2 - m1 * m2 ** 3
    al = (v1 * m2 - m1 * v2) / det
    ka = (m1 ** 3 * v2 - v1 * m2 ** 3) / det
    for m, v in vals[2:]:
        if al * m ** 3 + ka * m != v:
            return None
    return 3 * al


# ------------------------------------------------------- M_0 = U(g_0).|0>

def M0(W, sector, rounds=6, cap=60):
    ks = g0_keys(sector)
    sp = Span()
    sp.add(dict(VAC))
    frontier = [dict(VAC)]
    for r in range(rounds):
        new = []
        for st in frontier:
            for k in ks:
                img = hatX(W, mono(*k), st)
                if img and sp.add(img):
                    new.append(img)
        if not new or sp.dim() >= cap:
            break
        frontier = new
    return sp


# ------------------------------------- is the vacuum in g.M ?  (level-matched)

def vacuum_in_gM(W, act, par, sector, levels=4, amin=-4, amax=7, verbose=True):
    """M is generated by |0>; M = U(g_-) U(g_0) |0> with g_- level-RAISING.
    The level-0 part of g.M is  sum_d  g_{(-d)} . M_d.  We build M level by
    level and test the coefficient of the empty word."""
    lvmode = {m: level_of_mode(act, par, m) for m in W.allmodes}

    def lev(word):
        return sum(lvmode[m] for m in word)

    gens = [k for k in g_basis(amin, amax)]
    # M, level by level, from the level-0 tower upwards
    sp = Span()
    for st in M0(W, sector).piv.values():
        sp.add(st)
    frontier = list(sp.piv.values())
    for r in range(levels):
        new = []
        for st in frontier:
            for k in gens:
                if level_shift(*k) <= 0:
                    continue
                img = hatX(W, mono(*k), st)
                if img and sp.add(img):
                    new.append(img)
        if not new:
            break
        frontier = new
        if verbose:
            print("        level round %d: dim M >= %d" % (r + 1, sp.dim()))
    # now test
    hits = []
    for w, row in sp.piv.items():
        L = lev(w)
        for k in gens:
            if level_shift(*k) != -L:
                continue
            img = hatX(W, mono(*k), row)
            c = img.get((), Q(0))
            if c:
                hits.append((k, w, c))
    return sp, hits


# ------------------------------------------------------------------- driver

def run(sector="L", N=4, lam=Q(-1), mu=Q(0), kmax=8):
    act, br, torus = SECTORS[sector]
    t0 = time.time()
    print("=" * 78)
    print("Spin(V), V = cF_{%s; %s} over k^%s(1|4), window |a| <= %d"
          % (lam, mu, sector, N))
    print("=" * 78)
    W, act, tor, K, bad = build(sector, N, lam, mu, 1, Nseed=4)
    par = (lam, mu)
    print("  the pairing: a + a' = %s, invariance defects = %d, %d modes"
          % (K, bad, len(W.allmodes)))

    ks, cen = central_in_g0(sector)
    print("\n  g_0: dim = %d; its center: %s" % (len(ks), [name(k) for k in cen]))

    print("\n  g_0 on the vacuum:")
    movers = []
    for k in ks:
        st = hatX(W, mono(*k), VAC)
        print("    %-26s -> %s" % (name(k), "0" if not st else
                                   "%d words, e.g. %s" % (len(st), sorted(st)[0])))
        if st:
            movers.append(k)
    if not movers:
        print("    the vacuum IS g_0-invariant: the four labels are a complete "
              "highest weight, and the module is a genuine highest weight "
              "module of highest weight zero.")
    else:
        Zk = movers[0]
        print("\n  the vacuum is NOT g_0-invariant: %s moves it." % name(Zk))
        _tower_report(W, act, par, br, torus, Zk, kmax, sector)

    print("\n  building M level by level and testing whether |0> is in g.M:")
    sp, hits = vacuum_in_gM(W, act, par, sector)
    print("    dim M (window-truncated) = %d" % sp.dim())
    print("    (generator, vector) pairs whose image has a nonzero coefficient "
          "on |0>: %d" % len(hits))
    for k, w, c in hits[:5]:
        print("       %s on the word %s  ->  %s |0> + ..." % (name(k), w, c))
    if hits:
        print("    => |0> lies in g.M: the trivial module is NOT a quotient of M.")
    else:
        print("    => |0> is not in g.M: M / g.M is the trivial module, which "
              "is therefore a quotient of M.")
    print("\n  (%.0f s)" % (time.time() - t0))


def _tower_report(W, act, par, br, torus, Zk, kmax, sector):
    tw = tower(W, Zk, kmax)
    ind = independent(tw)
    print("\n  the tower hat %s^k . |0>: %d nonzero terms, %d of them "
          "linearly independent (window-limited)" % (name(Zk), len(tw), ind))
    lvmode = {m: level_of_mode(act, par, m) for m in W.allmodes}
    print("  their labels:")
    for k, st in enumerate(tw[:ind]):
        lvs = {sum(lvmode[m] for m in w) for w in st}
        h = None
        img = hatX(W, mono(1), st)
        # eigenvalue of ell_0
        lam0 = _eig(img, st)
        Hs = tuple(_eig(hatX(W, mono(*g), st), st) for g in torus)
        c = charge_on(W, br, st)
        print("    k = %d: %2d words, level %s, c = %s, h = %s, H = %s"
              % (k, len(st), lvs if len(lvs) == 1 else lvs,
                 c, None if lam0 is None else Q(1, 2) * lam0, Hs))

    sp0 = M0(W, sector, rounds=kmax + 2, cap=kmax + 4)
    print("\n  M_0 = U(g_0).|0> : dim = %d in this window (the tower has %d)"
          % (sp0.dim(), ind))


def _eig(img, st):
    lam = None
    for k in set(img) | set(st):
        a, b = img.get(k, Q(0)), st.get(k, Q(0))
        if b == 0:
            if a != 0:
                return None
            continue
        l = a / b
        if lam is None:
            lam = l
        elif lam != l:
            return None
    return lam if lam is not None else Q(0)


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    sec = sys.argv[2] if len(sys.argv) > 2 else "L"
    mu = Q(sys.argv[3]) if len(sys.argv) > 3 else (Q(0) if sec == "L" else Q(1, 4))
    run(sec, N=N, mu=mu)
