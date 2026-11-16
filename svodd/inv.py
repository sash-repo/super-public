#!/usr/bin/env python3
"""
inv.py -- the explicit invariant behind a modulus.

T_3 (t3.py) certifies a modulus structurally: over a normalised quotient class
the stabiliser acts on the residual space by ZERO, so every point of that space
is its own orbit (see fibre_certificate).  That is a proof, but it is not an
invariant one can evaluate on a germ.  This module produces the invariant.

Two shapes of answer are looked for, in this order:

  (1) an honest polynomial invariant -- a semi-invariant of weight zero, so
      P(exp(M)g) = P(g) for every M in rho(s_f);
  (2) a ratio P/Q of two semi-invariants of the SAME weight, which is then a
      rational invariant of the group.

Either is verified from scratch here (D_M P = chi(M) P checked generator by
generator, exactly, over the rationals) rather than trusted from the routine
that produced it, and then evaluated along the family v + lambda u whose
direction u T_3 reports, to confirm it separates its members.
"""

import sys
from fractions import Fraction as F

import stab
import orb
import sv
import t3
from probe import TYPES


def invariants_of_degree(mats, n, d, cap=1200):
    """
    A basis of the space of homogeneous polynomials P of degree d on the layer
    with D_M P = 0 for every M -- that is, of the honest invariants of weight
    zero of the connected group.

    This is one nullspace, so it is cheap; the eigenspace splitting needed for
    semi-invariants of nonzero weight is not.  It is also, as it turns out,
    where the invariants behind the moduli live: the pairing of the linear with
    the cubic part of a germ is ANNIHILATED by the whole stabiliser algebra, the
    cube root of unity there coming from the component group, not from the Lie
    algebra.
    """
    mons = stab.deg_monomials(n, d)
    N = len(mons)
    if N == 0 or N > cap:
        return None, N
    pos = {a: i for i, a in enumerate(mons)}
    # rows of the big matrix, kept in reduced row echelon form as we go
    R = []                                   # rref rows
    piv = []                                 # pivot column of each row

    def add(row):
        for i, p in enumerate(piv):
            if row[p]:
                c = row[p] / R[i][p]
                row = [row[j] - c * R[i][j] for j in range(N)]
        p = next((j for j in range(N) if row[j]), None)
        if p is None:
            return False
        R.append(row)
        piv.append(p)
        return True

    for M in mats:
        cols = stab.derivation_matrix(M, mons, pos)
        # D_M as a matrix: column j = image of the j-th monomial
        rows = [[F(0)] * N for _ in range(N)]
        for j, col in enumerate(cols):
            for a, c in col.items():
                rows[pos[a]][j] = c
        for r in rows:
            if any(r):
                add(r)
        if len(piv) == N:
            return [], N                     # no invariant of this degree
    # back-substitute to a genuine rref, then read off the nullspace
    for i in range(len(R) - 1, -1, -1):
        pv = R[i][piv[i]]
        R[i] = [x / pv for x in R[i]]
        for h in range(i):
            if R[h][piv[i]]:
                c = R[h][piv[i]]
                R[h] = [R[h][j] - c * R[i][j] for j in range(N)]
    free = [j for j in range(N) if j not in piv]
    out = []
    for j in free:
        v = [F(0)] * N
        v[j] = F(1)
        for i, p in enumerate(piv):
            v[p] = -R[i][j]
        out.append({mons[i]: c for i, c in enumerate(v) if c})
    return out, N


def bracket(A, B):
    n = len(A)
    return [[sum(A[i][t] * B[t][j] - B[i][t] * A[t][j] for t in range(n))
             for j in range(n)] for i in range(n)]


def deriv_full(M, mons, pos):
    """the matrix of the derivation D_M on degree-d polynomials"""
    N = len(mons)
    cols = stab.derivation_matrix(M, mons, pos)
    D = [[F(0)] * N for _ in range(N)]
    for j, col in enumerate(cols):
        for a, c in col.items():
            D[pos[a]][j] = c
    return D


def semi_invariants_fast(mats, n, d, cap=1200):
    """
    The semi-invariants of degree d: polynomials P with D_M P = chi(M) P.

    Computed in two steps, which is what makes it affordable.  A character chi
    kills the derived algebra, so every semi-invariant is annihilated by
    [L,L]; that is one nullspace, and it cuts the space of candidates down from
    all polynomials of degree d to a space S that is usually of dimension a few.
    Only then is the eigenspace splitting done, on S alone, and only for
    generators completing [L,L] to L -- the operators D_h commute on S, since
    their brackets lie in the derived algebra and so act by zero there.

    Returns a list of (P, weights), weights indexed by those generators.
    """
    mons = stab.deg_monomials(n, d)
    N = len(mons)
    if N == 0 or N > cap:
        return []
    pos = {a: i for i, a in enumerate(mons)}

    der, rows = [], []
    for A in mats:
        for B in mats:
            C = bracket(A, B)
            r = sum(C, [])
            if any(r) and stab.rank_rows(rows + [r]) > len(rows):
                rows.append(r)
                der.append(C)
    S, _ = invariants_of_degree(der, n, d, cap) if der else (
        [{a: F(1) if i == j else F(0) for i, a in enumerate(mons)}
         for j in range(N)], N)
    if not S:
        return []
    # S as rows in the monomial coordinates
    basis = [[P.get(a, F(0)) for a in mons] for P in S]
    # generators completing the derived algebra to the whole of L
    hs, rows2 = [], list(rows)
    for A in mats:
        r = sum(A, [])
        if stab.rank_rows(rows2 + [r]) > len(rows2):
            rows2.append(r)
            hs.append(A)
    blocks = [(basis, ())]
    for M in hs:
        D = deriv_full(M, mons, pos)
        new = []
        for b, wts in blocks:
            if not b:
                continue
            got, ok = orb._split(D, b, wts)
            if ok:
                new.extend(got)
        blocks = new
        if not blocks:
            return []
    out = []
    for b, wts in blocks:
        for r in b:
            out.append(({mons[i]: c for i, c in enumerate(r) if c}, wts))
    return out


def evalpoly(P, g):
    """the value of the polynomial P = {exponents: coeff} at the vector g"""
    tot = F(0)
    for a, c in P.items():
        t = c
        for i, e in enumerate(a):
            if e:
                t *= g[i] ** e
        tot += t
    return tot


def show_poly(P, L, idx=None, maxterms=8):
    """P written in the coordinates dual to the monomial basis of the layer"""
    out = []
    for a, c in sorted(P.items(), key=lambda kv: (-sum(kv[0]), kv[0])):
        fac = []
        for i, e in enumerate(a):
            if e:
                nm = sv.show_mon(L.basis[idx[i] if idx else i], "x", "xi")
                fac.append(f"[{nm}]" + (f"^{e}" if e > 1 else ""))
        term = "*".join(fac) if fac else "1"
        if c == 1:
            out.append(term)
        elif c == -1:
            out.append("-" + term)
        else:
            out.append(f"{c}*{term}")
        if len(out) == maxterms:
            out.append("...")
            break
    return " + ".join(out).replace("+ -", "- ")


def verify_semi(P, mats):
    """
    Recompute the weights from scratch: return the tuple (chi(M))_M if
    D_M P = chi(M) P for every generator M, else None.
    """
    k0 = next(iter(P))
    d, n = sum(k0), len(k0)
    mons = stab.deg_monomials(n, d)
    pos = {a: i for i, a in enumerate(mons)}
    wts = []
    for M in mats:
        cols = stab.derivation_matrix(M, mons, pos)
        img = {}
        for a, c in P.items():
            for b, cc in cols[pos[a]].items():
                img[b] = img.get(b, F(0)) + c * cc
        img = {k: v for k, v in img.items() if v}
        if not img:
            wts.append(F(0))
            continue
        k0 = next(iter(P))
        if k0 not in img:
            return None
        lam = img[k0] / P[k0]
        if any(img.get(k, F(0)) != lam * P.get(k, F(0))
               for k in set(img) | set(P)):
            return None
        wts.append(lam)
    return tuple(wts)


def separates_ratio(Pn, Pd, n, v, u, lams=(0, 1, 2, 3, 5, -1)):
    """as `separates`, for the ratio Pn/Pd"""
    vals = []
    for lam in lams:
        g = [v[i] + F(lam) * u[i] for i in range(n)]
        b = evalpoly(Pd, g)
        if b == 0:
            return None
        vals.append(evalpoly(Pn, g) / b)
    if len(set(vals)) < len(vals):
        return None
    return vals


def verify_zero(P, mats):
    """recomputed from scratch: is D_M P = 0 for every generator M?"""
    return verify_semi(P, mats) == tuple(F(0) for _ in mats)


def separates(P, n, v, u, lams=(0, 1, 2, 3, 5, -1)):
    """
    Does the invariant take distinct values along v + lambda u?  Returns the
    list of values, or None if it does not separate them.
    """
    vals = []
    for lam in lams:
        g = [v[i] + F(lam) * u[i] for i in range(n)]
        vals.append(evalpoly(P, g))
    if len(set(vals)) < len(vals):
        return None
    return vals


def hunt(tname, n=3, m=1, D=None, degrees=(1, 2, 3), verbose=True):
    """the invariant for every family class of one type"""
    p, q, terms = TYPES[tname](n, m)
    if D is None:
        D = 2 * n + 4
    res = stab.analyse(tname, p, q, terms, D, verbose=False)
    L, classes, cert, notes = t3.classify(res)
    fams = [c for c in classes if c["family"] and c["dirs"]]
    label = f"{tname}({n},{m})"
    print(f"\n--- {label}   dim Nt_f = {L.n}   "
          f"{len(classes)} classes, {len(fams)} of them a family"
          + ("   CERTIFIED" if cert else "   NOT CERTIFIED"))
    if not fams:
        print("    no family to explain")
        return True
    # The invariant of a family is not in general a function on the whole
    # layer: it is a function on the piece of the filtration in which the
    # family sits, which is what an invariant of the classification has to be
    # anyway, the lower degrees having already been normalised.  Since the
    # stabiliser raises degree, the span F^e of the basis monomials of degree
    # >= e is invariant for every e, and the hunt is carried out on each F^e in
    # turn, starting with the deepest one that still contains the family.
    pool, semi = [], []
    for e in L.degrees:
        idx = [i for i in range(L.n) if L.degs[i] >= e]
        if not idx:
            continue
        sub = []
        for M in L.span:
            S = [[M[i][j] for j in idx] for i in idx]
            if any(any(r) for r in S):
                sub.append(S)
        if not sub:
            continue
        for d in degrees:
            got, N = invariants_of_degree(sub, len(idx), d)
            if got is None:
                continue
            for P in got:
                if verify_zero(P, sub):
                    pool.append((P, e, idx, d))
        nz = len([p for p in pool if p[1] == e])
        # and, when weight zero gives nothing, the semi-invariants, whose
        # ratios at equal weight are invariants just as good
        for d in degrees:
            for P, w in semi_invariants_fast(sub, len(idx), d):
                if verify_semi(P, sub) == w and any(w):
                    semi.append((P, w, e, idx, d))
        if verbose:
            print(f"    on F^{e} (dim {len(idx)}): {nz} invariant(s) of "
                  f"weight zero, "
                  f"{len([s for s in semi if s[2] == e])} semi-invariant(s)")

    ok = True
    for c in fams:
        vtxt = L.show(c["v"])
        u = c["dirs"][0]
        print(f"    family {vtxt} + t.{L.show(u)}:")
        found = None
        for P, e, idx, d in pool:
            vv = [c["v"][i] for i in idx]
            uu = [u[i] for i in idx]
            if not any(uu):
                continue
            vals = separates(P, len(idx), vv, uu)
            if vals:
                found = (P, e, idx, d, vals)
                break
        if found is not None:
            P, e, idx, d, vals = found
            print(f"      an invariant of degree {d} on F^{e}, "
                  f"annihilated by all {len(L.span)} generators:")
            print(f"        {show_poly(P, L, idx)}")
            print("      values at t = 0,1,2,3,5,-1: "
                  + ", ".join(str(x) for x in vals))
            continue
        # a ratio of two semi-invariants of the same weight
        rat = None
        for a in range(len(semi)):
            for b in range(len(semi)):
                if a == b or semi[a][1] != semi[b][1] or semi[a][2] != semi[b][2]:
                    continue
                idx = semi[a][3]
                vv = [c["v"][i] for i in idx]
                uu = [u[i] for i in idx]
                if not any(uu):
                    continue
                vals = separates_ratio(semi[a][0], semi[b][0], len(idx), vv, uu)
                if vals:
                    rat = (semi[a], semi[b], vals)
                    break
            if rat:
                break
        if rat is None:
            print("      NO explicit invariant found in these degrees")
            ok = False
            continue
        (Pn, w, e, idx, dn), (Pd, _, _, _, dd), vals = rat
        print(f"      an invariant P/Q on F^{e}, degrees {dn}/{dd}, "
              f"P and Q semi-invariant of one and the same weight:")
        print(f"        P = {show_poly(Pn, L, idx)}")
        print(f"        Q = {show_poly(Pd, L, idx)}")
        print("      values at t = 0,1,2,3,5,-1: "
              + ", ".join(str(x) for x in vals))
    return ok


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    bad = 0
    print("=" * 78)
    print("the explicit invariants behind the moduli of corank 1|4")
    print("=" * 78)
    jobs = []
    if which == "vol":
        jobs += [("III_3", 3, 1, 9), ("III_4", 3, 1, 9)]
    if which in ("all", "III"):
        jobs += [("III_3", 3, 1, 9), ("III_4", 3, 1, 9)]
        jobs += [("III_1", nn, 2, None) for nn in range(3, 7)]
    if which in ("all", "IV"):
        jobs += [("IV_1", 4, 2, None), ("IV_1", 4, 3, None),
                 ("IV_4", 4, 2, None)]
    for tn, nn, mm, DD in jobs:
        try:
            if not hunt(tn, nn, mm, DD):
                bad += 1
        except Exception as e:                                  # noqa: BLE001
            print(f"\n--- {tn}({nn},{mm}): FAILED  {type(e).__name__}: {e}")
            bad += 1
    print("\n" + "=" * 78)
    print("all moduli explained by an explicit invariant" if bad == 0
          else f"{bad} types with an unexplained modulus")
    print("=" * 78)
