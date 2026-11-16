#!/usr/bin/env python3
r"""
hw4.py -- what the labels (c, h; H_1, H_2) = (0, 0; 0, 0) of Statement
Statement 7.3.22 do, and do NOT, say about the module Spin(V) at n = 4.

The question:

  "Something strange: if ALL components of the highest weight are equal to 0,
   then the module is trivial, right? ... But is this module with highest
   weight (0,0,0,0) irreducible?  I'd expect the 1-dimensional trivial module
   be its quotient module..."

Everything below is exact rational arithmetic, on top of ns4.py (its engine); nothing is assumed -- the pairing is rebuilt on each window and its
invariance re-verified before use, and every operator is the quadratic lift
hatX of eq. (150), the same one that produced the labels.

The questions answered, one function each:

 1) energy_census   -- the grading.  ell_0 = 1/2 hat K_t; the mode t^a w^S of
    cF_{lambda; mu} has ell_0-weight a + mu + |S|/2 + (lambda + 1)/2, so its
    LEVEL (= minus that weight, the energy above the vacuum) is
    -(a + mu) - |S|/2 - (lambda+1)/2.  At lambda = -1, mu = 0 the level is
    (1 - 2a - |S|)/2 -- ZERO for the 8 modes with 2a + |S| = 1, i.e., for
    (0, {i}) and (-1, {i}^c).  Those are ODD vectors of V, hence BOSONIC in
    Spin (Weyl, not Clifford): the level-0 subspace is infinite-dimensional.

 2) vac_report      -- which elements of g move the vacuum.  All the
    level-lowering ones kill it, so the vacuum IS a lowest-energy vector; but
    g_0 does NOT fix it: the extra central element
    Z = K_{t^{-1} xi_1 eta_1 xi_2 eta_2} of Proposition 7.3.16 -- the one
    that exists only at n = 4 -- sends it to a nonzero level-0 state.
    Hence the four labels are not a complete highest weight.

 3) parity_split    -- hatX preserves the word-length parity, so Spin(V) is the
    direct sum of two g-submodules; it is not irreducible on that ground alone.

 4) submodule       -- M = U(g) . vacuum, computed level by level, and the
    level-0 part M_0 = U(g_0) . vacuum in particular.

 5) quotient_test   -- is the vacuum in g.M?  If not, M / g.M is the trivial
    module, which is what one expects; if yes, the trivial module is not a
    quotient of M.

 6) invariant_vectors -- the dual question: a vector killed by all of g (the
    trivial module as a SUBmodule).

Usage:  tools/venv/bin/python hw4.py [N]        (N = the truncation window)
"""

import sys
import time
from fractions import Fraction as Q

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "lib"))

from ns4 import (SUBSETS, Q as _Q, mono, keys, act_L, act_M, br_L, br_M,   # noqa: E402
                 TORUS_L, TORUS_M, invariant_forms, split_symmetry,
                 check_invariance, VAC, psi, hatX, sadd, parity, clean)
import runns4 as R                                                          # noqa: E402
from exactlin import rank as exrank                                         # noqa: E402


# ============================================================== the pairing
#
# At lambda = -1 the invariant form is the residue pairing
#         B((a, S), (K - a, S^c)) = eps(S),      K = a + a' = -1 - 2 mu,
# whose sign eps(S) does not depend on a (verified at N = 4 against the form
# produced by ns4.invariant_forms, and re-verified for invariance here).

def residue_form(act, par, torus, N, Nseed=4):
    """The invariant form on the window |a| <= N, built from the solved form on
    the seed window and re-verified there and here."""
    lam, mu = par
    seed, _ = invariant_forms(act, par, par, N=Nseed,
                              gens=keys(range(-1, 3)), torus=torus)
    sym, asym = split_symmetry(seed)
    if len(seed) != 1 or len(sym) != 1:
        raise ValueError("expected a single supersymmetric form, got %d (%d sym)"
                         % (len(seed), len(sym)))
    f0 = sym[0]
    eps, K = {}, None
    for (ku, kv), c in f0.items():
        a, S = ku
        b, T = kv
        if K is None:
            K = a + b
        elif K != a + b:
            raise ValueError("the pairing has no constant t-degree")
        if S in eps and eps[S] != c:
            raise ValueError("the sign depends on a as well as on S")
        eps[S] = c
    if len(eps) != len(SUBSETS):
        raise ValueError("the seed window is too small: %d signs" % len(eps))
    lo, hi = -N, N
    form = {}
    for a in range(lo, hi + 1):
        b = K - a
        if not (lo <= b <= hi):
            continue
        for S in SUBSETS:
            T = tuple(sorted(set(range(1, 5)) - set(S)))
            form[((Q(a), S), (Q(b), T))] = eps[S]
    bad = fast_invariance(act, par, form, keys(range(-2, 4)), N)
    return form, K, bad


def fast_invariance(act, par, form, gens, N):
    """B(f.u, v) + (-1)^{p(f)p(u)} B(u, f.v) = 0 on every pair of the window --
    the same test as ns4.check_invariance, with the images hoisted out of the
    inner loop (which is what makes it usable on the big windows)."""
    bas = keys(range(-N + 2, N - 1))
    bad = 0
    for kf in gens:
        f = mono(*kf)
        pf = parity(f)
        img = {ku: act(par, f, {ku: Q(1)}) for ku in bas}
        for ku in bas:
            sgn = (-1) ** (pf * (len(ku[1]) % 2))
            for kv in bas:
                tot = Q(0)
                for kp, co in img[ku].items():
                    tot += co * form.get((kp, kv), Q(0))
                for kp, co in img[kv].items():
                    tot += sgn * co * form.get((ku, kp), Q(0))
                if tot:
                    bad += 1
    return bad


def build(sector="L", N=6, lam=Q(-1), mu=Q(0), shift=1, Nseed=4):
    """The Fock space Spin(V) (shift = 1) on the window |a| <= N."""
    act = act_L if sector == "L" else act_M
    tor = TORUS_L if sector == "L" else TORUS_M
    form, K, bad = residue_form(act, (lam, mu), tor, N, Nseed)
    W = R.build_V(act, (lam, mu), form, N, shift)
    return W, act, tor, K, bad


# ================================================================ the grading

def wt_of_mode(act, par, m):
    """The ell_0 = 1/2 hat K_t weight of the mode m, read off the module."""
    a, S = m
    img = act(par, mono(1), {(a, S): Q(1)})
    if set(img) - {(a, S)}:
        raise ValueError("K_t is not diagonal on %s" % (m,))
    return Q(1, 2) * img.get((a, S), Q(0))


def level_of_mode(act, par, m):
    return -wt_of_mode(act, par, m)


def level_shift(a, S):
    """The level shift of hat K_{t^a w^S}: [ell_0, hat K_f] = -shift . hat K_f."""
    return Q(1) - Q(a) - Q(len(S), 2)


def energy_census(act, par, W, tag):
    print("\n--- 1) the grading of %s" % tag)
    lv, tp = {}, {}
    for m in W.allmodes:
        lv[m] = level_of_mode(act, par, m)
        tp[m] = W.tp(m)
    zero = sorted(m for m in W.allmodes if lv[m] == 0)
    neg = sorted(m for m in W.allmodes if lv[m] < 0 and W.is_creation(m))
    print("    modes: %d;  level 0: %d;  of them bosonic (tp = 0): %d, "
          "fermionic: %d" % (len(W.allmodes), len(zero),
                             sum(1 for m in zero if tp[m] == 0),
                             sum(1 for m in zero if tp[m] == 1)))
    print("    the level-0 modes: %s" % ", ".join(
        "t^%s w^%s%s" % (m[0], "".join(map(str, m[1])) or "1",
                         "[b]" if tp[m] == 0 else "[f]") for m in zero))
    print("    creation modes of NEGATIVE level: %s"
          % (", ".join("t^%s w^%s (level %s)" % (m[0], "".join(map(str, m[1]))
                                                 or "1", lv[m]) for m in neg)
             or "none"))
    return lv


# =============================================== 2) g on the vacuum

def g_basis(amin, amax):
    return [(Q(a), S) for a in range(amin, amax + 1) for S in SUBSETS]


def vac_report(W, amin=-3, amax=6):
    print("\n--- 2) which elements of g move the vacuum")
    moved, killed = [], 0
    for k in g_basis(amin, amax):
        st = hatX(W, mono(*k), VAC)
        if st:
            moved.append((k, level_shift(*k), st))
        else:
            killed += 1
    lowering_bad = [(k, ls) for k, ls, st in moved if ls < 0]
    zero_bad = [(k, ls) for k, ls, st in moved if ls == 0]
    print("    tested %d monomials K_{t^a w^S}, -%d <= a <= %d"
          % (killed + len(moved), -amin, amax))
    print("    move the vacuum: %d, all of them level-RAISING except:" % len(moved))
    print("      level-lowering that move it: %s" % (lowering_bad or "none"))
    print("      level-preserving (g_0) that move it: %s"
          % ([("K_{t^%s w^%s}" % (k[0], "".join(map(str, k[1])) or "1"))
              for k, ls in zero_bad] or "none"))
    return moved, zero_bad


def g0_on_vac(W):
    """The 8-dimensional g_0 = C K_t (+) o(4) (+) C Z and its action on |0>."""
    out = []
    for k in g_basis(-2, 2):
        a, S = k
        if 2 * int(a) + len(S) != 2:
            continue
        st = hatX(W, mono(*k), VAC)
        out.append((k, st))
    return out


# ================================================ 3) the parity splitting

def parity_split(W, amin=-2, amax=3, depth=2):
    """hatX preserves the word-length parity: check it on words of length
    <= depth built from creation modes."""
    print("\n--- 3) the word-parity splitting")
    cre = [m for m in W.allmodes if W.is_creation(m)]
    states = [VAC]
    for m in cre[:24]:
        states.append(psi(W, m, VAC))
    for m in cre[:8]:
        for n in cre[:8]:
            s = psi(W, m, psi(W, n, VAC))
            if s:
                states.append(s)
    bad, tested = 0, 0
    for k in g_basis(amin, amax):
        f = mono(*k)
        for st in states:
            pars = {len(w) % 2 for w in st}
            if len(pars) != 1:
                continue
            p0 = pars.pop()
            img = hatX(W, f, st)
            tested += 1
            if any(len(w) % 2 != p0 for w in img):
                bad += 1
    print("    hat K_f applied to %d homogeneous states x %d generators: "
          "word-parity changed in %d cases"
          % (len(states), len(g_basis(amin, amax)), bad))
    print("    => Spin(V) = Spin(V)_even (+) Spin(V)_odd as g-modules"
          if bad == 0 else "    => the splitting FAILS")
    return bad


# ============================================ 4) the submodule U(g).vacuum

def _key(st):
    return tuple(sorted(st.items()))


class Span:
    """An exact echelon form over Q, for repeated membership tests."""

    def __init__(self):
        self.piv = {}          # leading key -> normalized row

    def reduce(self, st):
        v = dict(st)
        while v:
            k = max(v)
            if k not in self.piv:
                return v, k
            r = self.piv[k]
            c = v[k]
            for kk, vv in r.items():
                v[kk] = v.get(kk, Q(0)) - c * vv
                if v[kk] == 0:
                    del v[kk]
        return {}, None

    def add(self, st):
        v, k = self.reduce(st)
        if not v:
            return False
        c = v[k]
        self.piv[k] = {kk: vv / c for kk, vv in v.items()}
        return True

    def contains(self, st):
        v, _ = self.reduce(st)
        return not v

    def dim(self):
        return len(self.piv)


def submodule(W, act, par, amin=-3, amax=5, rounds=4, cap=4000, verbose=True):
    """M = U(g) . |0>, computed by repeated application of the generators.
    Returns the Span and the level census."""
    gens = [mono(*k) for k in g_basis(amin, amax)]
    lvmode = {m: level_of_mode(act, par, m) for m in W.allmodes}

    def lev(word):
        return sum(lvmode[m] for m in word)

    sp = Span()
    sp.add(VAC)
    frontier = [VAC]
    for r in range(rounds):
        new = []
        for st in frontier:
            for f in gens:
                img = hatX(W, f, st)
                if img and sp.add(img):
                    new.append(img)
                    if sp.dim() >= cap:
                        break
            if sp.dim() >= cap:
                break
        if verbose:
            print("      round %d: dim M >= %d (+%d)" % (r + 1, sp.dim(), len(new)))
        if not new or sp.dim() >= cap:
            break
        frontier = new
    return sp, lvmode, lev


# ==================================== 5) is the vacuum in g.M?

def quotient_test(W, sp, amin=-3, amax=6):
    """The level-0 component of g.M: does it contain the vacuum?

    A basis vector of M is a combination of words; we apply every generator to
    every basis vector of M and collect the coefficient of the empty word.
    """
    print("\n--- 5) is the vacuum in g.M?")
    gens = [mono(*k) for k in g_basis(amin, amax)]
    hits = []
    for k, row in sp.piv.items():
        st = row
        for kf in g_basis(amin, amax):
            img = hatX(W, mono(*kf), st)
            c = img.get((), Q(0))
            if c:
                hits.append((kf, k, c))
    print("    pairs (generator, basis vector of M) whose image has a nonzero "
          "coefficient on the vacuum: %d" % len(hits))
    for kf, k, c in hits[:6]:
        print("       K_{t^%s w^%s} on %s  ->  %s . |0> + ..."
              % (kf[0], "".join(map(str, kf[1])) or "1",
                 " ".join("psi_%s%s" % (m[0], "".join(map(str, m[1])) or "1")
                          for m in k), c))
    if not hits:
        print("    NONE: the vacuum is not in g.M, so M / g.M is the trivial "
              "module -- it IS a quotient of M.")
    else:
        print("    the vacuum IS in g.M, so the trivial module is NOT a "
              "quotient of M.")
    return hits


# ==================================== 6) invariant vectors in the module

def invariant_vectors(W, sp, amin=-3, amax=6, maxdim=400):
    """Vectors of M killed by every generator -- the trivial SUBmodule."""
    print("\n--- 6) g-invariant vectors of M")
    rows = list(sp.piv.values())
    if len(rows) > maxdim:
        rows = rows[:maxdim]
        print("    (only the first %d basis vectors)" % maxdim)
    gens = g_basis(amin, amax)
    cols, mat = [], []
    imgs = []
    for st in rows:
        d = {}
        for kf in gens:
            for w, c in hatX(W, mono(*kf), st).items():
                d[(kf, w)] = d.get((kf, w), Q(0)) + c
        imgs.append(d)
    allk = sorted({k for d in imgs for k in d}, key=str)
    if not allk:
        print("    every generator kills every basis vector -- the action is "
              "trivial (impossible; the window is too small)")
        return None
    M = [[d.get(k, Q(0)) for d in imgs] for k in allk]
    r = exrank(M)
    print("    dim M(tested) = %d, rank of the map M -> (+) g.M = %d, "
          "so dim of the invariants = %d" % (len(rows), r, len(rows) - r))
    return len(rows) - r


# ================================================================== driver

def run(sector="L", N=6, lam=Q(-1), mu=Q(0), Nseed=4, rounds=3, cap=1200):
    t0 = time.time()
    tag = "k^%s(1|4), lambda = %s, mu = %s, window |a| <= %d" % (sector, lam, mu, N)
    print("=" * 78)
    print("Spin(V) over %s" % tag)
    print("=" * 78)
    W, act, tor, K, bad = build(sector, N, lam, mu, 1, Nseed)
    par = (lam, mu)
    print("    the pairing: constant a + a' = %s, invariance defects = %d, "
          "%d modes" % (K, bad, len(W.allmodes)))
    if bad:
        raise SystemExit("the form is not invariant -- abort")
    energy_census(act, par, W, tag)
    print("\n--- 2a) the eight basis vectors of g_0 on the vacuum")
    for k, st in g0_on_vac(W):
        name = "K_{t^%s w^%s}" % (k[0], "".join(map(str, k[1])) or "1")
        print("    %-28s -> %s" % (name, "0" if not st else
                                   " + ".join("%s psi_%s psi_%s"
                                              % (v, w[0], w[1])
                                              for w, v in list(st.items())[:3])))
    vac_report(W)
    parity_split(W)
    print("\n--- 4) M = U(g).|0>")
    sp, lvmode, lev = submodule(W, act, par, rounds=rounds, cap=cap)
    import collections
    cen = collections.Counter()
    for k in sp.piv:
        cen[lev(k)] += 1
    print("    dim M (in this window, after %d rounds) = %d" % (rounds, sp.dim()))
    print("    by level: %s" % dict(sorted(cen.items())))
    quotient_test(W, sp)
    invariant_vectors(W, sp)
    print("\n    (%.0f s)" % (time.time() - t0))
    return W, sp


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    run("L", N=N)
