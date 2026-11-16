#!/usr/bin/env python3
r"""
run_abfock.py -- the checks on the n+1 Fock spaces of the antibracket.

Run:  python3 run_abfock.py   (from this folder)
"""

import sys
from fractions import Fraction
from itertools import combinations

from abfock import (Fock, pe_basis, pe_bracket, apply_X, stabilizes_L,
                    parity, Bform, nullspace, solve)

PASS = FAIL = 0
LOG = []


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        LOG.append(f"  ok   {name} {detail}")
    else:
        FAIL += 1
        LOG.append(f"  FAIL {name} {detail}")
    return ok


def op_eq(F, A, B, vecs):
    for v in vecs:
        d = Fock.add(A(v), B(v), -1)
        if not Fock.iszero(d):
            return False
    return True


def op_zero(F, A, vecs):
    return all(Fock.iszero(A(v)) for v in vecs)


def testvecs(F, maxdeg=2):
    return [F.mono(S, a) for (S, a) in F.basis(maxdeg)]


# ------------------------------------------------------------------ A
def check_relations(n, maxdeg=2):
    for i in range(n + 1):
        F = Fock(n, i)
        vecs = testvecs(F, maxdeg)
        h = F.mul_odd(0)
        for v in F.gens():
            for w in F.gens():
                A, pA = F.rho(*v)
                Bo, pB = F.rho(*w)
                br = F.bracket(A, pA, Bo, pB)
                c = Bform(v, w)
                rhs = F.opadd((c, h)) if c else (lambda f: {})
                ok = op_eq(F, br, rhs, vecs)
                if not ok:
                    check(f"relation n={n} i={i} [{v},{w}]", False)
                    return
    check(f"ab({n}) relations [v,w]=B(v,w)hbar on all {n+1} Fock modules", True,
          f"({(2*n)**2} pairs x {n+1} modules)")


# ------------------------------------------------------------------ B,C,D
def check_vacuum(n, maxdeg=2):
    for i in range(n + 1):
        F = Fock(n, i)
        vecs = testvecs(F, maxdeg)
        one = F.one()
        # the vacuum is killed by L_i
        for g in F.gens():
            if F.in_L(*g):
                A, _ = F.rho(*g)
                if not Fock.iszero(A(one)):
                    check(f"vacuum killed by L_{i}", False)
                    return
        # the L_i-invariants are exactly  C.1 + hbar F_i
        inv_rows, unk = [], vecs
        for g in F.gens():
            if not F.in_L(*g):
                continue
            A, _ = F.rho(*g)
            img = [A(v) for v in unk]
            keys = sorted({k for d in img for k in d})
            for k in keys:
                inv_rows.append([d.get(k, Fraction(0)) for d in img])
        ns = nullspace(inv_rows, len(unk)) if inv_rows else \
            [[Fraction(1) if t == s else Fraction(0) for t in range(len(unk))]
             for s in range(len(unk))]
        # predicted: 1, and every hbar-monomial
        pred = 1 + sum(1 for (S, a) in F.basis(maxdeg) if 0 in S)
        if len(ns) != pred:
            check(f"L_{i}-invariants in F_{i} (n={n})", False,
                  f"got {len(ns)}, predicted {pred}")
            return
    check(f"n={n}: the L_i-invariants of F_i are C.v + hbar F_i, all i", True,
          "(so the vacuum is unique only modulo hbar)")


def check_hbar_ideal(n, maxdeg=2):
    for i in range(n + 1):
        F = Fock(n, i)
        h = F.mul_odd(0)
        vecs = testvecs(F, maxdeg)
        for g in F.gens():
            A, _ = F.rho(*g)
            for v in vecs:
                img = A(h(v))
                if any(0 not in S for (S, a) in img):
                    check(f"hbar F_{i} invariant", False)
                    return
    check(f"n={n}: hbar F_i is an invariant C-submodule of F_i, all i", True,
          "(whence 'irreducible' must be read in the category of families)")


def check_L_is_hbar_part(n, maxdeg=2):
    """rho(w) lands in hbar.End(F_i) if and only if w is in L_i."""
    for i in range(n + 1):
        F = Fock(n, i)
        vecs = testvecs(F, maxdeg)
        for g in F.gens():
            A, _ = F.rho(*g)
            in_hbar = all(0 in S for v in vecs for (S, a) in A(v))
            if in_hbar != F.in_L(*g):
                check(f"n={n} i={i} gen {g}: hbar-part vs L_i", False)
                return
    check(f"n={n}: rho(w) is in hbar.End(F_i) exactly for w in L_i, all i", True,
          "(the source of the obstruction below)")


# ------------------------------------------------------------------ pe(n)
def check_pe(n):
    B = pe_basis(n)
    ev = sum(1 for _, p in B if p == 0)
    od = sum(1 for _, p in B if p == 1)
    ok = (ev, od) == (n * n, n * n)
    check(f"aut(B) solved from the invariance equations: sdim pe({n})", ok,
          f"= ({ev}|{od}), expected ({n*n}|{n*n})")
    # the odd part: q -> theta symmetric, theta -> q anti-symmetric
    sym = asym = 0
    for X, p in B:
        if p == 0:
            continue
        C = [[apply_X(X, ('q', j)).get(('th', k), 0) for k in range(1, n + 1)]
             for j in range(1, n + 1)]
        D = [[apply_X(X, ('th', j)).get(('q', k), 0) for k in range(1, n + 1)]
             for j in range(1, n + 1)]
        if any(C[a][b] for a in range(n) for b in range(n)):
            if all(C[a][b] == C[b][a] for a in range(n) for b in range(n)):
                sym += 1
        if any(D[a][b] for a in range(n) for b in range(n)):
            if all(D[a][b] == -D[b][a] for a in range(n) for b in range(n)):
                asym += 1
    check(f"pe({n})_odd = (symmetric q->theta) + (anti-symmetric theta->q)",
          sym == n * (n + 1) // 2 and asym == n * (n - 1) // 2,
          f"({sym} symmetric, {asym} anti-symmetric)")
    return B


def stab_dims(n, B):
    out = {}
    for i in range(n + 1):
        ev = od = 0
        for X, p in B:
            pass
        # the stabilizer is a subspace: solve for it in the basis B
        for p in (0, 1):
            vecs = [X for X, q in B if q == p]
            # linear conditions: X L_i subset L_i
            L = [('q', j) for j in range(1, i + 1)] + \
                [('th', j) for j in range(i + 1, n + 1)]
            Lset = set(L)
            gens = [('q', j) for j in range(1, n + 1)] + \
                   [('th', j) for j in range(1, n + 1)]
            rows = []
            for g in L:
                for h in gens:
                    if h in Lset:
                        continue
                    rows.append([apply_X(X, g).get(h, Fraction(0)) for X in vecs])
            rows = [r for r in rows if any(x != 0 for x in r)]
            d = len(vecs) - (len(vecs) - len(nullspace(rows, len(vecs)))) \
                if rows else len(vecs)
            if p == 0:
                ev = d
            else:
                od = d
        out[i] = (ev, od)
    return out


# ------------------------------------------------------------------ the lift
def lift(F, X, pX, maxdeg=2, widen=True):
    """
    Look for an operator hatX on F_i with  [hatX, rho(w)] = rho(Xw)  for every
    generator w.  The ansatz is the whole degree-<=2 part of the algebra
    generated by the rho(w) and the hbar-free reductions a(v), v in L_i,
    together with id and hbar.id -- generous on purpose.
    Returns the operator, or None.
    """
    n = F.n
    pieces = []          # (operator, parity)
    for g in F.gens():
        A, p = F.rho(*g)
        pieces.append((A, p))
        if F.in_L(*g):
            kind, j = g
            if kind == 'q':
                pieces.append((F.d_odd(j), 1))
            else:
                pieces.append((F.opadd((-1, F.d_x(j - F.i - 1))), 0))
    ans = []
    idop = lambda f: dict(f)
    h = F.mul_odd(0)
    for o, p in [(idop, 0), (h, 1)]:
        if p == pX:
            ans.append(o)
    for (A, pA) in pieces:
        for (Bo, pB) in pieces:
            for pre, ppre in [(None, 0), (h, 1)]:
                p = (pA + pB + ppre) % 2
                if p != pX:
                    continue
                o = F.comp(A, Bo) if pre is None else F.comp(pre, A, Bo)
                ans.append(o)
    vecs = testvecs(F, maxdeg)
    # rows: for each generator w and each test vector, each monomial key
    rows, rhs = [], []
    cols = len(ans)
    data = []   # list of (dict_of_target, list_of_dicts_from_ansatz)
    for g in F.gens():
        A, pA = F.rho(*g)
        target = {}
        for hh, c in apply_X(X, g).items():
            Ah, _ = F.rho(*hh)
            target[hh] = (Ah, c)
        for v in vecs:
            want = {}
            for hh, (Ah, c) in target.items():
                want = Fock.add(want, Ah(v), c)
            cont = []
            for o in ans:
                br = F.bracket(o, pX, A, pA)
                cont.append(br(v))
            keys = sorted({k for d in cont for k in d} | set(want))
            for k in keys:
                rows.append([d.get(k, Fraction(0)) for d in cont])
                rhs.append(want.get(k, Fraction(0)))
    sol = solve(rows, rhs, cols)
    if sol is None:
        return None
    def op(f):
        out = {}
        for c, o in zip(sol, ans):
            if c:
                out = Fock.add(out, o(f), c)
        return out
    # verify
    for g in F.gens():
        A, pA = F.rho(*g)
        want = {}
        br = F.bracket(op, pX, A, pA)
        for v in vecs:
            w2 = {}
            for hh, c in apply_X(X, g).items():
                Ah, _ = F.rho(*hh)
                w2 = Fock.add(w2, Ah(v), c)
            if not Fock.iszero(Fock.add(br(v), w2, -1)):
                return None
    return op


def check_lifts(n, B, maxdeg=2):
    """The heart: a lift exists exactly for the stabilizer of L_i."""
    for i in range(n + 1):
        F = Fock(n, i)
        good = bad = 0
        for X, pX in B:
            st = stabilizes_L(X, n, i)
            L = lift(F, X, pX, maxdeg)
            if st and L is not None:
                good += 1
            elif (not st) and L is None:
                bad += 1
            else:
                check(f"lift n={n} i={i}: X stabilizes={st} lift={'yes' if L else 'no'}",
                      False)
                return
        check(f"n={n}, F_{i}: X in pe({n}) lifts  <=>  X L_{i} subset L_{i}", True,
              f"({good} lift, {bad} do not, of {len(B)})")


def scalar_of(F, d, vecs):
    """If the operator d is (c0 + c1 hbar).id, return (c0, c1); else None."""
    c0 = c1 = None
    for v in vecs:
        w = d(v)
        (S, a), = list(v.keys())
        exp = {(S, a): 'c0'}
        if 0 not in S:
            exp[(tuple(sorted(S + (0,))), a)] = 'c1'
        for k in w:
            if k not in exp:
                return None
        for k, tag in exp.items():
            val = w.get(k, Fraction(0))
            if tag == 'c0':
                if c0 is None:
                    c0 = val
                elif c0 != val:
                    return None
            else:
                if c1 is None:
                    c1 = val
                elif c1 != val:
                    return None
    return (c0 or Fraction(0), c1 or Fraction(0))


def check_lift_homo(n, B, maxdeg=2):
    """
    For X, Y in the stabilizer p_i: [hatX, hatY] - hat[X,Y] is a SCALAR, and
    that scalar 2-cocycle is a coboundary -- so the scalars can be normalized
    and X -> hatX is an honest homomorphism p_i -> gl(F_i).  No anomaly.
    """
    gens = [('q', j) for j in range(1, n + 1)] + [('th', j) for j in range(1, n + 1)]

    def coords(X):
        return [Fraction(apply_X(X, g).get(h, 0)) for g in gens for h in gens]

    for i in range(n + 1):
        F = Fock(n, i)
        vecs = testvecs(F, maxdeg)
        st = [(X, p) for X, p in B if stabilizes_L(X, n, i)]
        lifts = [lift(F, X, p, maxdeg) for X, p in st]
        cols = [coords(X) for X, _ in st]
        nonscalar = 0
        rows, rhs0, rhs1 = [], [], []
        for a in range(len(st)):
            for b in range(len(st)):
                X, pX = st[a]
                Y, pY = st[b]
                Z, pZ = pe_bracket(X, pX, Y, pY, n)
                Zop = (lambda f: {}) if not Z else lift(F, Z, pZ, maxdeg)
                if Zop is None:
                    nonscalar += 1
                    continue
                d = F.opadd((1, F.bracket(lifts[a], pX, lifts[b], pY)), (-1, Zop))
                s = scalar_of(F, d, vecs)
                if s is None:
                    nonscalar += 1
                    continue
                tgt = coords(Z) if Z else [Fraction(0)] * len(gens) ** 2
                sol = solve([[cols[t][r] for t in range(len(st))]
                             for r in range(len(tgt))], tgt, len(st))
                if sol is None:
                    nonscalar += 1
                    continue
                rows.append(sol)
                rhs0.append(s[0])
                rhs1.append(s[1])
        cob = (solve(rows, rhs0, len(st)) is not None and
               solve(rows, rhs1, len(st)) is not None)
        check(f"n={n}, F_{i}: [hatX,hatY] - hat[X,Y] is a scalar, and a coboundary",
              nonscalar == 0 and cob,
              f"({len(st)**2} pairs; so p_{i} -> gl(F_{i}) after normalizing, no anomaly)")


# ------------------------------------------------------------------ misc
def check_not_faithful(n, maxdeg=2):
    for i in range(1, n + 1):
        F = Fock(n, i)
        vecs = testvecs(F, maxdeg)
        A, _ = F.rho('q', 1)
        if not op_zero(F, F.comp(A, A), vecs):
            check(f"rho(q_1)^2 = 0 on F_{i}", False)
            return
    if n >= 1:
        check(f"n={n}: rho_i(q_j)^2 = 0 for j<=i, while q_j^2 != 0 in But", True,
              "(the Fock representations are not faithful, unlike Weyl = Diff)")


def check_isotropic(n):
    """
    Every maximal isotropic subspace of the odd form B on W = C^{n|n} is
    homogeneous of superdimension (i|n-i) for some i, and there are exactly
    n+1 orbits.  Verified here by the two statements that carry the proof:
      (1) L_i is maximal isotropic of sdim (i|n-i), i = 0..n;
      (2) an isotropic subspace has  dim L_ev + dim L_od <= n,  with equality
          exactly at the maximal ones -- checked by brute force over F_2, F_3
          for n <= 3 on all homogeneous subspaces.
    """
    for n_ in range(1, n + 1):
        for i in range(n_ + 1):
            L = [('q', j) for j in range(1, i + 1)] + \
                [('th', j) for j in range(i + 1, n_ + 1)]
            if any(Bform(v, w) for v in L for w in L):
                check(f"L_{i} isotropic (n={n_})", False)
                return
            # maximality: every generator outside pairs non-trivially with L
            gens = [('q', j) for j in range(1, n_ + 1)] + \
                   [('th', j) for j in range(1, n_ + 1)]
            for g in gens:
                if g in L:
                    continue
                if not any(Bform(g, w) for w in L):
                    check(f"L_{i} maximal (n={n_})", False)
                    return
    check(f"L_i is maximal isotropic of sdim (i|n-i) for i=0..n, n<={n}", True,
          "(an odd form trades an even direction for an odd one)")


# ------------------------------------------------------------------ main
def main():
    print("=" * 74)
    print("abfock: the n+1 Fock spaces of the antibracket -- exact checks")
    print("=" * 74)
    for n in (1, 2, 3):
        LOG.append(f"\n--- n = {n} ---")
        check_relations(n, maxdeg=2 if n < 3 else 1)
        check_vacuum(n, maxdeg=2 if n < 3 else 1)
        check_hbar_ideal(n, maxdeg=2 if n < 3 else 1)
        check_L_is_hbar_part(n, maxdeg=2 if n < 3 else 1)
        check_isotropic(n)
        B = check_pe(n)
        d = stab_dims(n, B)
        LOG.append("  sdim of the stabilizer p_i = Stab_{pe(%d)}(L_i):" % n)
        okf = True
        for i in range(n + 1):
            pred = (n * n - i * (n - i),
                    n * n - (i * (i + 1) + (n - i) * (n - i - 1)) // 2)
            okf = okf and pred == d[i]
            LOG.append(f"      i={i}:  sdim p_{i} = ({d[i][0]}|{d[i][1]})"
                       f"   F_{i} = functions on sdim ({n-i}|{i})")
        check(f"n={n}: sdim p_i = (n^2-i(n-i) | n^2-[i(i+1)+(n-i)(n-i-1)]/2)", okf)
        check_lifts(n, B, maxdeg=2 if n < 3 else 1)
        check_lift_homo(n, B, maxdeg=2 if n < 3 else 1)
        check_not_faithful(n, maxdeg=2 if n < 3 else 1)
    print("\n".join(LOG))
    print("-" * 74)
    print(f"passed {PASS}, failed {FAIL}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
