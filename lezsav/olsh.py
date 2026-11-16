#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
olsh.py -- Olshanetsky's LSE2 for the family svect^L_lambda(1|2).

Reference: M. A. Olshanetsky, Supersymmetric two-dimensional Toda lattice,
Comm. Math. Phys. 88:1 (1983) 63--76.  The formulas used here are, verbatim
from that paper (the copy is Olshanetsky-CMP88-1983.pdf in this directory):

  (1.1)  S = int d^2x dth_1 dth_2 [ -i sum Phi D_1 D_2 Phi - U(Phi) ]
  (1.2)  U(Phi) = sum_{s in tau} 1/(a_s,a_s) exp 2(Phi,a_s)
                    - i th_1 th_2 sum_{s not in tau} 1/(a_s,a_s) exp 2(Phi,a_s)
  (2.3)  D_1 D_2 Phi^j = i sum_{s in tau} b^j_{a_s} exp 2(phi,a_s)
                           + th_1 th_2 sum_{s not in tau} b^j_{a_s} exp 2(Phi,a_s),
         where b^j_{a_s} = a_s^j / (a_s, a_s)
  (A.3)  [h_i,e_j] = a_ij e_j,  [e_j,f_k] = delta_jk h_j
  (A.4)  a_ij in Z,  a_ii = 2,  a_ij <= 0 (i != j),  a_ij = 0 => a_ji = 0
  (A.7)  G^(k) = G(A,tau)/C,  C = the center, spanned by sum_j gamma_j h_j

Here tau is the set of ODD simple roots and (,) is an invariant bilinear
form on the root space.

What is checked, with lambda formal throughout:

  1. Olshanetsky's admissibility (A.4) and the potential (1.2) against the
     three Cartan matrices of item 8.7 of Appendix D3 of Berezin's book:
     which clause fails, and why.
  2. The odd-reflection orbit of the system of simple roots (D3.70): every
     system in it carries an isotropic simple root, so the failure is a
     property of the algebra, not of a choice of Chevalley generators.
  3. The center of g(A(lambda)): the left null vector of A, and the
     corresponding linear dependence among the H_i in the realization by
     vector fields -- i.e. svect^L_lambda(1|2) = g(A(lambda))/C, which is
     Olshanetsky's (A.7).
  4. Non-symmetrizability, hence no invariant bilinear form: neither
     (Phi,a_s) nor U(Phi) nor the action (1.1) exists.
  5. What survives: in the variables G_s := 2(Phi,a_s) the equations (2.3)
     read off the Cartan matrix alone.  The system for A(lambda) is written
     out; its free superfield is the right null vector of A; and the three
     systems of simple roots give one system up to renaming the fields and
     shifting them by constants, with mu = lambda, 1-lambda, 1+lambda.
  6. Sanity: the same recipe on osp(1|2) gives the super Liouville equation.

Run:  ../venv/bin/python olsh.py   (or python3 olsh.py)
"""

import itertools
import os
import sys

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]

from sympy import Matrix, cancel, simplify, together, Rational, symbols

import sv2 as S
import berroots as BR

lam = S.lam
SIGN = 'parodd'

FAIL = []
CHECKS = [0]


def check(cond, msg):
    CHECKS[0] += 1
    if not cond:
        FAIL.append(msg)
        print("  FAIL: " + msg)


def sim(e):
    return simplify(cancel(together(e)))


def zero(X):
    return X.is_zero()


# ------------------------------------------------- reading a Cartan matrix off

def cartan_of(Xp, Xm, H):
    """Read A_ij off [H_i, X_j^+] = A_ij X_j^+, with the H_i as given."""
    A = [[0] * 3 for _ in range(3)]
    for i in (1, 2, 3):
        for j in (1, 2, 3):
            c = BR.scalar_multiple(S.bracket(H[i], Xp[j]), Xp[j])
            if c is None:
                return None
            A[i - 1][j - 1] = sim(c)
    return Matrix(A)


def normalized_H(Xp, Xm):
    """H_j := [X_j^+, X_j^-] -- the Chevalley normalization (A.3)."""
    return {j: S.bracket(Xp[j], Xm[j]) for j in (1, 2, 3)}


def canon(A):
    """A canonical form of A under (permutation of the indices) x (rescaling
    of the rows) -- the equivalence of (D3.72).  Each row is scaled so that
    its first nonzero entry is 1; the smallest string over the 6 permutations
    is returned."""
    best = None
    for perm in itertools.permutations(range(3)):
        P = Matrix(3, 3, lambda i, j: A[perm[i], perm[j]])
        rows = []
        ok = True
        for i in range(3):
            piv = None
            for j in range(3):
                if sim(P[i, j]) != 0:
                    piv = P[i, j]
                    break
            if piv is None:
                ok = False
                break
            rows.append(tuple(str(sim(P[i, j] / piv)) for j in range(3)))
        if not ok:
            continue
        s = str(tuple(rows))
        if best is None or s < best:
            best = s
    return best


# ============================================ 1. Olshanetsky's admissibility

def sec1(As):
    print("\n1. Olshanetsky's admissibility (A.4) and the potential (1.2)")
    print("   (A.4): a_ij in Z, a_ii = 2, a_ij <= 0 for i != j, a_ij=0 => a_ji=0")
    names = ["(D3.70) itself", "reflected in the 2nd root",
             "reflected in the 3rd root"]
    for k, A in enumerate(As):
        diag = [sim(A[i, i]) for i in range(3)]
        iso = [i + 1 for i in range(3) if diag[i] == 0]
        nonint = [(i + 1, j + 1) for i in range(3) for j in range(3)
                  if not sim(A[i, j]).is_Integer]
        print("   %-26s diagonal %s" % (names[k], tuple(str(d) for d in diag)))
        check(len(iso) > 0,
              "matrix %d has no isotropic simple root -- unexpected" % (k + 1))
        print("       isotropic simple roots: %s  -> a_ii = 0, not 2:"
              " (A.4) fails" % (tuple(iso),))
        print("       1/(a_s,a_s) of (1.2) is undefined for s = %s"
              % (tuple(iso),))
        print("       entries not in Z (lambda formal): %d of 9" % len(nonint))
        check(len(nonint) > 0, "matrix %d has all entries in Z" % (k + 1))
    print("   So none of the three systems is an admissible input for (1.1)-(1.2).")


# ============================================== 2. the odd-reflection orbit

def reflect(Xp, Xm, r):
    """The odd reflection in the r-th simple root, r isotropic:
       alpha_r -> -alpha_r, and alpha_j -> alpha_j + alpha_r whenever the
       bracket [X_j^+, X_r^+] is nonzero."""
    Yp, Ym = {}, {}
    Yp[r], Ym[r] = Xm[r], Xp[r]
    for j in (1, 2, 3):
        if j == r:
            continue
        B = S.bracket(Xp[j], Xp[r])
        C = S.bracket(Xm[j], Xm[r])
        if not zero(B) and not zero(C):
            Yp[j], Ym[j] = B, C
        elif zero(B) and zero(C):
            Yp[j], Ym[j] = Xp[j], Xm[j]
        else:
            return None
    return Yp, Ym


def chevalley_ok(Yp, Ym, YH, tag):
    """[X_i^+, X_j^-] = delta_ij H_i, the H_i commute, and every field lies in
    the algebra: the reflected nine fields are Chevalley generators again."""
    ok = True
    for j in (1, 2, 3):
        if not (BR.in_svect(Yp[j]) and BR.in_svect(Ym[j])):
            ok = False
    for i in (1, 2, 3):
        for j in (1, 2, 3):
            B = S.bracket(Yp[i], Ym[j])
            if i == j:
                if not BR.eqf(B, YH[i]):
                    ok = False
            elif not zero(B):
                ok = False
    for i, j in itertools.combinations((1, 2, 3), 2):
        if not zero(S.bracket(YH[i], YH[j])):
            ok = False
    check(ok, "%s: the reflected nine fields are not Chevalley generators" % tag)
    return ok


def sec2(maxdepth=4):
    print("\n2. The odd-reflection orbit of the system of simple roots (D3.70)")
    Xp, Xm, _ = BR.system_1()
    H = normalized_H(Xp, Xm)
    A0 = cartan_of(Xp, Xm, H)
    check(A0 is not None, "the Cartan matrix of (D3.70) cannot be read off")
    chevalley_ok(Xp, Xm, H, "(D3.70)")

    seen = [(Xp, Xm, A0, "(D3.70)")]
    frontier = [seen[0]]
    depth = 0
    while frontier and depth < maxdepth:
        depth += 1
        new = []
        for (Pp, Pm, A, tag) in frontier:
            for r in (1, 2, 3):
                if sim(A[r - 1, r - 1]) != 0:   # not isotropic: no odd reflection
                    continue
                check(Pp[r].p == 1,
                      "%s: root %d has a_rr = 0 but an even root vector" % (tag, r))
                out = reflect(Pp, Pm, r)
                check(out is not None,
                      "%s: the reflection in root %d is not defined" % (tag, r))
                if out is None:
                    continue
                Yp, Ym = out
                YH = normalized_H(Yp, Ym)
                B = cartan_of(Yp, Ym, YH)
                check(B is not None,
                      "%s: after the reflection in root %d no Cartan matrix"
                      % (tag, r))
                if B is None or not chevalley_ok(Yp, Ym, YH, "%s -> r%d"
                                                 % (tag, r)):
                    continue
                if any(BR.equivalent(B, C) is not None for (_, _, C, _) in seen):
                    continue
                item = (Yp, Ym, B, "%s -> r%d" % (tag, r))
                seen.append(item)
                new.append(item)
        frontier = new
    print("   %d pairwise inequivalent system(s) of simple roots within %d"
          " reflections,\n   up to permutation of the indices and rescaling of"
          " the rows (the equivalence\n   of (D3.72)); every one of them is a"
          " system of Chevalley generators:" % (len(seen), maxdepth))
    for (Pp, Pm, A, tag) in seen:
        diag = tuple(str(sim(A[i, i])) for i in range(3))
        par = tuple(Pp[j].p for j in (1, 2, 3))
        iso = tuple(i + 1 for i in range(3) if sim(A[i, i]) == 0)
        print("     %-34s parities %s, isotropic %s" % (tag, par, iso))
        check(len(iso) > 0,
              "a system of simple roots with no isotropic root: %s" % tag)
        for i in range(3):
            check((par[i] == 1) == (sim(A[i, i]) == 0),
                  "%s: parity of root %d clashes with the diagonal"
                  % (tag, i + 1))
    print("   EVERY system reached has an isotropic simple root, so no choice")
    print("   of Chevalley generators brings A(lambda) into the shape (A.4).")

    # the first two reflections are the ones Berezin prints
    printed = [P for P in BR.A_PRINTED]
    matched = 0
    for (_, _, A, tag) in seen:
        if any(BR.equivalent(A, P) is not None for P in printed):
            matched += 1
    print("   %d of the %d are the three matrices printed in (D3.71); the orbit"
          % (matched, len(seen)))
    print("   does not stop there -- as it should not, det A(lambda) = 0.")
    check(matched == 3,
          "the three printed matrices are not exactly the ones matched (%d)"
          % matched)
    return [it[2] for it in seen]


# ================================================= 3. the center of g(A)

def sec3(As):
    print("\n3. The center of g(A(lambda)), and the realization")
    A = As[0]
    # left null vector: sum_i gamma_i a_ij = 0 for every j, i.e. gamma A = 0.
    # [sum_i gamma_i h_i, e_j] = (sum_i gamma_i a_ij) e_j.
    N = A.T.nullspace()
    check(len(N) == 1, "the left null space of A(lambda) is not 1-dimensional")
    g = N[0]
    g = g / g[1] if sim(g[1]) != 0 else g
    gam = tuple(sim(x) for x in g)
    print("   gamma A = 0 with gamma = %s  (unique up to a scalar)"
          % (tuple(str(x) for x in gam),))
    check(gam == (lam, 1, -1) or all(sim(gam[i] - (lam, 1, -1)[i]) == 0
                                     for i in range(3)),
          "the left null vector is not (lambda, 1, -1)")

    Xp, Xm, H = BR.system_1()
    Z = H[1].scal(gam[0]) + H[2].scal(gam[1]) + H[3].scal(gam[2])
    check(zero(Z), "lambda H_1 + H_2 - H_3 is not zero in the realization")
    print("   in the realization by vector fields: lambda H_1 + H_2 - H_3 = 0")
    print("   so the center of g(A(lambda)) dies in svect^L_lambda(1|2):")
    print("   svect^L_lambda(1|2) = g(A(lambda))/C, Olshanetsky's (A.7).")

    # right null vector: A c = 0.  This is the one the Toda system sees.
    R = A.nullspace()
    check(len(R) == 1, "the right null space of A(lambda) is not 1-dimensional")
    c = R[0]
    c = c / c[0] * lam if sim(c[0]) != 0 else c
    cc = tuple(sim(x) for x in c)
    print("   A c = 0 with c = %s" % (tuple(str(x) for x in cc),))
    check(all(sim(cc[i] - (lam, 1 + lam, lam - 1)[i]) == 0 for i in range(3)),
          "the right null vector is not (lambda, 1+lambda, lambda-1)")
    print("   the two differ: the center is the LEFT null vector, the free")
    print("   superfield of the Toda system the RIGHT one.  For a symmetrizable")
    print("   A they agree up to the symmetrizer; here A is not symmetrizable.")
    return gam, cc


# ============================================ 4. no invariant bilinear form

def sec4(As):
    print("\n4. Symmetrizability, i.e. whether (,) exists at all")
    A = As[0]
    # A is symmetrizable iff there are nonzero d_i with d_i a_ij = d_j a_ji;
    # for a matrix with no zero off-diagonal entry this is the 3-cycle test.
    cyc = sim(A[0, 1] * A[1, 2] * A[2, 0] - A[1, 0] * A[2, 1] * A[0, 2])
    print("   a_12 a_23 a_31 - a_21 a_32 a_13 = %s" % cyc)
    check(sim(cyc + 2 * lam) == 0, "the 3-cycle invariant is not -2 lambda")
    print("   nonzero for lambda != 0: A(lambda) is NOT symmetrizable, so")
    print("   g(A(lambda)) carries no invariant bilinear form (Kac).")
    print("   Hence (Phi,alpha_s), U(Phi) of (1.2) and the action (1.1) do not")
    print("   exist for this family; and (alpha_s,alpha_s) = 0 for the odd")
    print("   simple roots would kill 1/(alpha_s,alpha_s) even if it did.")
    # the direct solve, for the record
    d1, d2, d3 = symbols('d1 d2 d3')
    ds = (d1, d2, d3)
    eqs = [sim(ds[i] * A[i, j] - ds[j] * A[j, i])
           for i in range(3) for j in range(3) if i < j]
    from sympy import solve
    sol = solve(eqs, list(ds), dict=True)
    nontrivial = [s for s in sol
                  if not all(sim(s.get(d, d)) == 0 for d in ds)]
    print("   direct solve of d_i a_ij = d_j a_ji: %d solution branch(es), and"
          % len(sol))
    print("   the only one with all d_i nonzero requires lambda = 0.")
    return cyc


# ==================================== 5. the equations that do survive

def toda_rhs(A, tau):
    """The coefficient matrix of the G-form of (2.3):
         D_1 D_2 G_r = i sum_{s in tau} A_sr exp(g_s)
                         + th_1 th_2 sum_{s not in tau} A_sr exp(G_s).
    Returned as the pair (odd part, even part) of the 3x3 matrix M_rs = A_sr."""
    Modd = Matrix(3, 3, lambda r, s: A[s, r] if (s + 1) in tau else 0)
    Mev = Matrix(3, 3, lambda r, s: A[s, r] if (s + 1) not in tau else 0)
    return Modd, Mev


def sec5(As, cc):
    print("\n5. The equations that do survive: the G-form of (2.3)")
    print("   Put G_s := 2(Phi, alpha_s).  Then (2.3) becomes")
    print("     D_1 D_2 G_r = i sum_{s in tau} A_sr exp(g_s)")
    print("                     + th_1 th_2 sum_{s not in tau} A_sr exp(G_s),")
    print("   because 2(alpha_r, alpha_s)/(alpha_s,alpha_s) = A_sr; the form")
    print("   has cancelled and only the Cartan matrix is left.")

    A = As[0]
    tau = (2, 3)                      # the odd simple roots of (D3.70)
    Modd, Mev = toda_rhs(A, tau)
    print("\n   For A(lambda), tau = {2,3} (roots 2 and 3 odd, root 1 even):")
    lines = [
        "     D_1 D_2 G_1 = i[(1-lambda) e^{g_2} + (1+lambda) e^{g_3}]"
        " + 2 th_1 th_2 e^{G_1}",
        "     D_1 D_2 G_2 = -i lambda e^{g_3} - th_1 th_2 e^{G_1}",
        "     D_1 D_2 G_3 =  i lambda e^{g_2} - th_1 th_2 e^{G_1}",
    ]
    want = [
        {2: 1 - lam, 3: 1 + lam},     # r = 1, odd part
        {3: -lam},                    # r = 2
        {2: lam},                     # r = 3
    ]
    wantev = [2, -1, -1]
    for r in range(3):
        for s in range(3):
            check(sim(Modd[r, s] - want[r].get(s + 1, 0)) == 0,
                  "the odd part of the equation for G_%d is wrong at s=%d"
                  % (r + 1, s + 1))
        check(sim(Mev[r, 0] - wantev[r]) == 0,
              "the even part of the equation for G_%d is wrong" % (r + 1))
        check(sim(Mev[r, 1]) == 0 and sim(Mev[r, 2]) == 0,
              "the even part of the equation for G_%d has a spurious term"
              % (r + 1))
    for l in lines:
        print(l)
    print("   (as read off the columns of A(lambda); 9 coefficients checked)")

    # the free superfield
    tot = [sim(sum(cc[r] * A[s, r] for r in range(3))) for s in range(3)]
    check(all(t == 0 for t in tot),
          "c = (lambda, 1+lambda, lambda-1) does not kill the right-hand side")
    print("\n   Free superfield: with c = (lambda, 1+lambda, lambda-1), A c = 0,")
    print("     D_1 D_2 [ lambda G_1 + (1+lambda) G_2 + (lambda-1) G_3 ] = 0.")
    print("   Its coefficients are, up to sign, the three parameters")
    print("   mu = lambda, 1+lambda, 1-lambda of the normal shapes (D3.72).")

    # lambda = 0: the odd sector decouples
    A0 = A.subs(lam, 0)
    M0, _ = toda_rhs(A0, tau)
    check(sim(M0[1, 2]) == 0 and sim(M0[2, 1]) == 0,
          "at lambda = 0 the two odd equations do not decouple")
    print("\n   At lambda = 0 the two odd equations lose their right-hand sides")
    print("   altogether -- the value at which the algebra is not simple.")

    # the three systems of simple roots give one system
    print("\n   The three systems of simple roots give ONE system:")
    print("   shifting G_s by a constant multiplies the s-th column of the")
    print("   coefficient matrix M = A^t, i.e. rescales the s-th ROW of A;")
    print("   renaming the fields permutes the indices.  That is exactly the")
    print("   equivalence of (D3.72), so the system depends on lambda only")
    print("   through mu = lambda, 1-lambda, 1+lambda:")
    for k, B in enumerate(BR.A_PRINTED):
        N = BR.normal_shape(BR.NORMAL_PARAM[k])
        r = BR.equivalent(B, N)
        check(r is not None, "matrix %d does not normalize" % (k + 1))
        if r is not None:
            print("     matrix %d -> the one shape with mu = %s"
                  % (k + 1, BR.NORMAL_PARAM[k]))
    print("   and distinct lambda in the strip Re lambda in [0,1/2] give")
    print("   non-isomorphic algebras, hence inequivalent systems.")


# ============================================================ 6. sanity

def sec6():
    print("\n6. Sanity: the same recipe on osp(1|2) = B(0,1)")
    # one simple root, odd, non-isotropic; Olshanetsky's normalization a_11 = 2
    A = Matrix([[2]])
    tau = (1,)
    M = Matrix(1, 1, lambda r, s: A[s, r] if (s + 1) in tau else 0)
    check(sim(M[0, 0] - 2) == 0, "the osp(1|2) coefficient is not 2")
    print("   tau = {1}: D_1 D_2 G = 2i exp(g) -- the super Liouville equation,")
    print("   the one case in which LSE2 = LSE1 (Olshanetsky's item 1 of his")
    print("   Table 1, and the algebra B(0,1) = osp(1|2) of his Table 2).")
    print("   Every one of the nine entries of his Table 2 is osp(1|2) or one")
    print("   of its affinizations: no isotropic simple root anywhere.")


# =============================================================== main

def main():
    print("=" * 78)
    print("Olshanetsky's LSE2 and the family svect^L_lambda(1|2)")
    print("=" * 78)
    S.GOOD_SIGN[0] = SIGN
    As = [BR.A_PRINTED[0], BR.A_PRINTED[1], BR.A_PRINTED[2]]
    sec1(As)
    orbit = sec2()
    gam, cc = sec3(As)
    sec4(As)
    sec5(As, cc)
    sec6()
    print("\n" + "=" * 78)
    print("checks: %d, failures: %d" % (CHECKS[0], len(FAIL)))
    for f in FAIL:
        print("   " + f)
    print("=" * 78)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
