#!/usr/bin/env python3
r"""Pairs of non-degenerate bilinear forms on a superspace (Subsec. 6.3).

Setting.  V = C^{m|n} with basis v_1,...,v_{m+n}, parities p_i.  A bilinear
form B of parity p(B) is given by its matrix G_{ij} = B(v_i,v_j), which
vanishes unless p_i + p_j = p(B); its (anti)symmetry is

    G_{ji} = eps * (-1)^{p_i p_j} G_{ij},     eps = +1 (symmetric), -1 (anti).

An X in gl(m|n) preserves B iff  B(Xx,y) + (-1)^{p(X)p(x)} B(x,Xy) = 0, i.e.

    (X.G)_{jk} := (X^t G)_{jk} + (-1)^{p(X) p_j} (G X)_{jk} = 0.        (act)

X.G is again a form of the same symmetry eps and of parity p(X)+p(G) -- this
is asserted in the code, not assumed.

What is computed, exactly:
  1) the sdim of the superspace of eps-symmetric forms, and of aut(omega);
  2) the census of which (m|n, parity, eps) carry a non-degenerate form;
  3) for a pair (omega,h) of non-degenerate forms: sdim aut(omega,h), the
     sdim of the orbit of the pair, and hence the number of moduli as a pair
     (even | odd);
  4) the operator R = omega^{-1} h of the pair, its parity, and the identity
     it satisfies; the check gl(m|n) = aut(omega) (+) S_omega.

Run (from this folder):  ../venv/bin/python pairs.py
"""

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
import sys, os, itertools
from fractions import Fraction

from exactlin import rank, nullspace, backend                  # noqa: E402


# ---------------------------------------------------------------- superspace

def parities(m, n):
    return [0] * m + [1] * n


def sign(k):
    return -1 if k % 2 else 1


# ------------------------------------------------------------------- forms

def form_basis(m, n, p, eps):
    """Basis of the space of eps-symmetric forms of parity p on C^{m|n}.

    Returned as a list of matrices (lists of rows of ints)."""
    par = parities(m, n)
    N = m + n
    basis = []
    for i in range(N):
        for j in range(i, N):
            if (par[i] + par[j]) % 2 != p:
                continue
            c = eps * sign(par[i] * par[j])          # G_ji = c * G_ij
            if i == j and c == -1:
                continue                              # G_ii = -G_ii => 0
            G = [[0] * N for _ in range(N)]
            G[i][j] = 1
            G[j][i] = c if i != j else 1
            basis.append(G)
    return basis


def standard_form(m, n, p, eps):
    """A non-degenerate eps-symmetric form of parity p, or None if none exists.

    p = 0: block diag(A, D) with A eps-symmetric (m x m), D (-eps)-symmetric.
    p = 1: off-diagonal blocks P (m x n) and Q = eps * P^t; needs m = n.
    """
    N = m + n
    G = [[0] * N for _ in range(N)]
    if p == 0:
        if not _fill_square(G, 0, m, eps):
            return None
        if not _fill_square(G, m, n, -eps):
            return None
        return G
    if m != n:
        return None
    for i in range(m):
        G[i][m + i] = 1
        G[m + i][i] = eps          # (-1)^{p_i p_j} = 1 here (one index even)
    return G


def _fill_square(G, off, k, s):
    """Put a non-degenerate s-symmetric k x k block at offset off."""
    if s == 1:
        for i in range(k):
            G[off + i][off + i] = 1
        return True
    if k % 2:
        return False
    for i in range(0, k, 2):
        G[off + i][off + i + 1] = 1
        G[off + i + 1][off + i] = -1
    return True


def is_nondeg(G):
    N = len(G)
    return rank([[Fraction(x) for x in row] for row in G]) == N


# ------------------------------------------------------------ the gl-action

def gl_basis(m, n, q):
    """Basis of the parity-q part of gl(m|n) as elementary matrices E_ij."""
    par = parities(m, n)
    N = m + n
    return [(i, j) for i in range(N) for j in range(N)
            if (par[i] + par[j]) % 2 == q]


def act(m, n, q, ij, G):
    """X.G for X = E_ij of parity q; see eq. (act)."""
    par = parities(m, n)
    N = m + n
    i, j = ij
    out = [[0] * N for _ in range(N)]
    # (X^t G)_{ab} = sum_c X_{ca} G_{cb} = [a == j] G_{ib}
    for b in range(N):
        out[j][b] += G[i][b]
    # (-1)^{q p_a} (G X)_{ab} = (-1)^{q p_a} [b == j] G_{ai}
    for a in range(N):
        out[a][j] += sign(q * par[a]) * G[a][i]
    return out


def flatten(mats):
    return [x for G in mats for row in G for x in row]


def act_matrix(m, n, q, forms):
    """Rows = images X.(forms) for X running over the parity-q basis of gl."""
    rows = []
    for ij in gl_basis(m, n, q):
        rows.append(flatten([act(m, n, q, ij, G) for G in forms]))
    return rows


def sdim_stab(m, n, forms):
    """sdim of the subalgebra of gl(m|n) annihilating every form in `forms`."""
    out = []
    for q in (0, 1):
        rows = act_matrix(m, n, q, forms)
        cols = len(rows[0]) if rows else 0
        # kernel of X -> X.forms : nullspace of the transpose
        M = [[Fraction(rows[r][c]) for r in range(len(rows))]
             for c in range(cols)]
        out.append(len(nullspace(M)) if cols else len(rows))
    return tuple(out)


def sdim_orbit(m, n, forms):
    """sdim of the tangent space to the GL(m|n)-orbit of the tuple `forms`."""
    return tuple(rank([[Fraction(x) for x in r]
                       for r in act_matrix(m, n, q, forms)] or [[Fraction(0)]])
                 for q in (0, 1))


# ------------------------------------------------- self-consistency of (act)

def check_action(m, n):
    """X.G is a form of the same eps and of parity p(X)+p(G)."""
    for p in (0, 1):
        for eps in (1, -1):
            for G in form_basis(m, n, p, eps):
                for q in (0, 1):
                    for ij in gl_basis(m, n, q):
                        H = act(m, n, q, ij, G)
                        assert is_form(m, n, H, (p + q) % 2, eps), \
                            (m, n, p, eps, q, ij)


def is_form(m, n, G, p, eps):
    par = parities(m, n)
    N = m + n
    for i in range(N):
        for j in range(N):
            if (par[i] + par[j]) % 2 != p and G[i][j] != 0:
                return False
            if G[j][i] != eps * sign(par[i] * par[j]) * G[i][j]:
                return False
    return True


# --------------------------------------------------------------- the pencil

def inverse(G):
    """Exact inverse of an integer matrix, as Fractions."""
    N = len(G)
    A = [[Fraction(G[i][j]) for j in range(N)]
         + [Fraction(int(i == k)) for k in range(N)]
         for i in range(N)]
    r = 0
    for c in range(N):
        piv = next((i for i in range(r, N) if A[i][c] != 0), None)
        if piv is None:
            raise ValueError("singular")
        A[r], A[piv] = A[piv], A[r]
        pv = A[r][c]
        A[r] = [x / pv for x in A[r]]
        for i in range(N):
            if i != r and A[i][c] != 0:
                f = A[i][c]
                A[i] = [a - f * b for a, b in zip(A[i], A[r])]
        r += 1
    return [row[N:] for row in A]


def mul(A, B):
    N = len(A)
    return [[sum(A[i][k] * B[k][j] for k in range(N)) for j in range(N)]
            for i in range(N)]


def operator_R(om, h):
    """R = omega^{-1} h as a matrix acting on V (columns = images)."""
    return mul(inverse(om), h)


# ------------------------------------------------------------------- report

NAME = {(0, 1): "even sym", (0, -1): "even anti",
        (1, 1): "odd sym", (1, -1): "odd anti"}


def census(mmax=4, nmax=4):
    print("== 1) the superspaces of forms and their non-degenerate elements ==")
    print(f"{'m|n':>6} {'type':>10} {'dim':>5}  non-deg?   sdim aut(omega)")
    for m in range(0, mmax + 1):
        for n in range(0, nmax + 1):
            if m + n == 0:
                continue
            for p in (0, 1):
                for eps in (1, -1):
                    b = form_basis(m, n, p, eps)
                    G = standard_form(m, n, p, eps)
                    ok = G is not None and is_nondeg(G)
                    if not ok and not b:
                        continue
                    aut = sdim_stab(m, n, [G]) if ok else None
                    print(f"{m}|{n:<4} {NAME[(p, eps)]:>10} {len(b):>5}"
                          f"  {'yes' if ok else 'no ':>7}   "
                          f"{('(%d|%d)' % aut) if aut else '-':>10}")


COEF = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59,
        61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]


def generic_form(m, n, p, eps, shift=0):
    """A non-degenerate form of the given type in general position, or None."""
    b = form_basis(m, n, p, eps)
    if not b:
        return None
    N = m + n
    for t in range(8):
        G = [[0] * N for _ in range(N)]
        for k, Gb in enumerate(b):
            c = COEF[(k + shift + t) % len(COEF)]
            for i in range(N):
                for j in range(N):
                    G[i][j] += c * Gb[i][j]
        if is_nondeg(G):
            return G
    return None


def pair_data(m, n, po, eo, ph, eh, h):
    om = standard_form(m, n, po, eo)
    if om is None or h is None or not is_nondeg(om) or not is_nondeg(h):
        return None
    aut = sdim_stab(m, n, [om, h])
    orb = sdim_orbit(m, n, [om, h])
    sp = [0, 0]
    for (p, eps) in ((0, eo), (1, eo), (0, eh), (1, eh)):
        sp[p] += len(form_basis(m, n, p, eps))
    mod = (sp[0] - orb[0], sp[1] - orb[1])
    R = operator_R(om, h)
    return aut, orb, tuple(sp), mod, R


def best_pair(m, n, po, eo, ph, eh, tries=12):
    """The pair with the LARGEST orbit among several choices of h = the generic
    one.  Randomizing guards against an accidentally special h."""
    import random
    rnd = random.Random(20260817)
    best = None
    b = form_basis(m, n, ph, eh)
    N = m + n
    for t in range(tries):
        if t == 0:
            h = generic_form(m, n, ph, eh)
        else:
            h = [[0] * N for _ in range(N)]
            for Gb in b:
                c = rnd.randint(-9, 9)
                for i in range(N):
                    for j in range(N):
                        h[i][j] += c * Gb[i][j]
            if not is_nondeg(h):
                continue
        d = pair_data(m, n, po, eo, ph, eh, h)
        if d is None:
            continue
        if best is None or (d[1][0] + d[1][1]) > (best[1][0] + best[1][1]):
            best = d
    return best


def pair_table(cases):
    print()
    print("== 3) pairs (omega,h) in general position: stabilizer and moduli ==")
    print("   (h generic in its space; 'std' = the pair with R = 1 or R = J)")
    print(f"{'m|n':>6} {'omega':>10} {'h':>10} {'aut(o,h)':>10}"
          f" {'orbit':>10} {'pairs':>10} {'moduli':>9} {'p(R)':>5}"
          f" {'R^2 scal':>9} {'aut std':>9}")
    for (m, n, po, eo, ph, eh) in cases:
        d = best_pair(m, n, po, eo, ph, eh)
        if d is None:
            continue
        aut, orb, sp, mod, R = d
        pR = (po + ph) % 2
        ds = pair_data(m, n, po, eo, ph, eh, standard_form(m, n, ph, eh))
        std = ('(%d|%d)' % ds[0]) if ds else '-'
        print(f"{m}|{n:<4} {NAME[(po, eo)]:>10} {NAME[(ph, eh)]:>10}"
              f" {'(%d|%d)' % aut:>10} {'(%d|%d)' % orb:>10}"
              f" {'(%d|%d)' % sp:>10} {'(%d|%d)' % mod:>9} {pR:>5}"
              f" {('yes' if is_scalar(mul(R, R)) else 'no'):>9} {std:>9}")


def is_scalar(A):
    N = len(A)
    c = A[0][0]
    return all(A[i][j] == (c if i == j else 0)
               for i in range(N) for j in range(N))


def split_check(cases):
    """gl(m|n) = aut(omega) (+) S_omega, where S = image of h |-> ... .

    Concretely: the map gl -> {forms of omega's type} , X |-> X.omega, has
    kernel aut(omega); so sdim gl = sdim aut + sdim image, and the image is
    the whole (parity-graded) space of forms of that symmetry iff GL acts
    transitively on the non-degenerate ones -- the normal shape theorem."""
    print()
    print("== 2) normal shapes: is the GL(m|n)-orbit of omega open? ==")
    seen = set()
    for (m, n, po, eo, _, _) in cases:
        if (m, n, po, eo) in seen:
            continue
        seen.add((m, n, po, eo))
        om = standard_form(m, n, po, eo)
        if om is None or not is_nondeg(om):
            continue
        orb = sdim_orbit(m, n, [om])
        sp = tuple(len(form_basis(m, n, p, eo)) for p in (0, 1))
        aut = sdim_stab(m, n, [om])
        gl = ((m * m + n * n), 2 * m * n)
        assert (aut[0] + orb[0], aut[1] + orb[1]) == gl, (m, n, po, eo)
        print(f"{m}|{n:<4} {NAME[(po, eo)]:>10}: sdim aut = ({aut[0]}|{aut[1]}),"
              f" orbit = ({orb[0]}|{orb[1]}), forms = ({sp[0]}|{sp[1]})"
              f"  ->  {'transitive' if orb == sp else 'NOT transitive'}")


def formula_check(mmax=6, nmax=6):
    """The moduli counts against closed formulas, on a grid."""
    print()
    print("== 4) the moduli counts against closed formulas ==")
    print(f"{'m|n':>6} {'types':>22} {'moduli':>9} {'formula':>9}  ok")
    bad = 0
    for m in range(0, mmax + 1):
        for n in range(0, nmax + 1):
            if m + n == 0 or m + n > 8:
                continue
            for (po, eo, ph, eh, name, f) in (
                (0, -1, 0, -1, "2-form, 2-form",
                 lambda m, n: (m // 2 + n, 0)),
                (0, 1, 0, 1, "sym, sym",
                 lambda m, n: (m + n // 2, 0)),
                (0, -1, 0, 1, "2-form, sym",
                 lambda m, n: (m // 2 + n // 2, 0)),
                (1, 1, 1, 1, "odd sym, odd sym",
                 lambda m, n: (n, n)),
                (1, -1, 1, -1, "odd anti, odd anti",
                 lambda m, n: (n, n)),
                (1, 1, 1, -1, "odd sym, odd anti",
                 lambda m, n: (n, 0)),
                (0, -1, 1, 1, "2-form, odd sym",
                 lambda m, n: (n // 2, n // 2)),
                (0, 1, 1, 1, "sym, odd sym",
                 lambda m, n: (n // 2, n // 2)),
            ):
                om = standard_form(m, n, po, eo)
                hh = standard_form(m, n, ph, eh)
                if (om is None or hh is None
                        or not is_nondeg(om) or not is_nondeg(hh)):
                    continue
                d = best_pair(m, n, po, eo, ph, eh)
                if d is None:
                    continue
                mod, pred = d[3], f(m, n)
                ok = (mod == pred)
                bad += 0 if ok else 1
                print(f"{m}|{n:<4} {name:>22} {'(%d|%d)' % mod:>9}"
                      f" {'(%d|%d)' % pred:>9}  {'ok' if ok else 'MISMATCH'}")
    print(f"\n  mismatches: {bad}")


def kaehler_obstruction():
    r"""Mixed parity: R = omega^{-1}h is odd, and R^2 = lambda*1 forces lambda=0.

    With omega = diag(A,D) even, R = [[0,b],[c,0]] odd, the symmetry of
    h = omega.R gives c = eps_h eps_omega D^{-1} b^t A, whence

        (R^2)|_{V_0} = b c = eps_h eps_omega (b D^{-1} b^t) A .

    b D^{-1} b^t has the symmetry of D^{-1}, while A^{-1} has that of A, and
    for an even form the two are opposite; so b c = lambda*1 forces lambda=0.
    Verified below on random b: the matrix b D^{-1} b^t is always of D's
    symmetry type, and Sym /\ Anti = 0."""
    import random
    rnd = random.Random(17082026)
    print()
    print("== 5) mixed parity: no complex structure (the p2a obstruction) ==")
    for (m, n) in ((2, 2), (4, 4), (2, 4), (4, 2), (6, 2), (2, 6)):
        for eo in (-1, 1):
            om = standard_form(m, n, 0, eo)
            if om is None or not is_nondeg(om) or m != n:
                continue
            A = [row[:m] for row in om[:m]]
            D = [row[m:] for row in om[m:]]
            Dinv = inverse(D)
            symD = all(D[i][j] == D[j][i] for i in range(n) for j in range(n))
            symA = all(A[i][j] == A[j][i] for i in range(m) for j in range(m))
            worst = 0
            for _ in range(50):
                b = [[Fraction(rnd.randint(-9, 9)) for _ in range(n)]
                     for _ in range(m)]
                M = [[sum(b[i][k] * Dinv[k][l] * b[j][l]
                          for k in range(n) for l in range(n))
                      for j in range(m)] for i in range(m)]
                # M must have D^{-1}'s symmetry
                s = all(M[i][j] == (1 if symD else -1) * M[j][i]
                        for i in range(m) for j in range(m))
                worst += 0 if s else 1
            print(f"  {m}|{n}, omega {NAME[(0, eo)]:>10}: "
                  f"A {'sym' if symA else 'anti'}, D "
                  f"{'sym' if symD else 'anti'}"
                  f"  ->  b D^-1 b^t is {'sym' if symD else 'anti'} in "
                  f"{50 - worst}/50 random b; A^-1 is the opposite type"
                  f"  =>  lambda = 0")


def main():
    print(f"exactlin backend: {backend()}\n")
    for m, n in ((1, 1), (2, 2), (2, 1), (1, 2), (2, 0), (0, 2)):
        check_action(m, n)
    print("action (act) verified: X.G is a form of the right parity/symmetry\n")

    census()

    cases = []
    for (m, n) in ((1, 1), (2, 2), (3, 3), (4, 4), (2, 0), (0, 2), (4, 2),
                   (2, 4), (4, 0), (0, 4)):
        for (po, eo) in ((0, -1), (0, 1), (1, -1), (1, 1)):
            for (ph, eh) in ((0, -1), (0, 1), (1, -1), (1, 1)):
                cases.append((m, n, po, eo, ph, eh))
    split_check(cases)
    pair_table(cases)
    formula_check()
    kaehler_obstruction()
    kaehler_stratum()




# ------------------------------------------------ the Kaehler stratum, R^2=-1

def bracket(X, Y, par):
    """Supercommutator of two matrices with the given index parities."""
    pX = mat_parity(X, par)
    pY = mat_parity(Y, par)
    XY, YX = mul(X, Y), mul(Y, X)
    s = sign(pX * pY)
    return [[XY[i][j] - s * YX[i][j] for j in range(len(X))]
            for i in range(len(X))]


def mat_parity(X, par):
    for i in range(len(X)):
        for j in range(len(X)):
            if X[i][j] != 0:
                return (par[i] + par[j]) % 2
    return 0


def stab_basis(m, n, forms):
    """Basis of aut(forms) inside gl(m|n), split by parity."""
    out = {}
    N = m + n
    for q in (0, 1):
        gb = gl_basis(m, n, q)
        rows = act_matrix(m, n, q, forms)
        if not rows:
            out[q] = []
            continue
        cols = len(rows[0])
        M = [[Fraction(rows[r][c]) for r in range(len(rows))]
             for c in range(cols)]
        ker = nullspace(M)
        mats = []
        for v in ker:
            X = [[Fraction(0)] * N for _ in range(N)]
            for c, (i, j) in enumerate(gb):
                X[i][j] += v[c]
            mats.append(X)
        out[q] = mats
    return out


def kaehler_stratum():
    """The pair (omega even 2-form, h even sym) with omega = h(J.,.), J^2=-1.

    Classically the stabilizer of such a triple in Sp(2k) is u(k); the
    computation below identifies its superization."""
    print()
    print("== 6) the Kaehler pair: its stabilizer is gl(k|l) ==")
    print(f"{'m|n':>6} {'aut(omega)':>12} {'aut(omega,h)':>13}"
          f" {'gl(k|l)':>10} {'derived':>10} {'center':>7}  ok")
    for (m, n) in ((2, 0), (4, 0), (6, 0), (0, 2), (0, 4),
                   (2, 2), (4, 2), (2, 4), (4, 4), (6, 2)):
        om = standard_form(m, n, 0, -1)
        h = standard_form(m, n, 0, 1)
        if (om is None or h is None
                or not is_nondeg(om) or not is_nondeg(h)):
            continue
        k, l = m // 2, n // 2
        aut_o = sdim_stab(m, n, [om])
        aut = sdim_stab(m, n, [om, h])
        pred = (k * k + l * l, 2 * k * l)
        par = parities(m, n)
        b = stab_basis(m, n, [om, h])
        allb = b[0] + b[1]
        # derived subalgebra and center of aut(omega,h)
        brs = [flatten([bracket(X, Y, par)]) for X in allb for Y in allb]
        der = rank([[Fraction(x) for x in r] for r in brs]) if brs else 0
        # the center is a SUBSPACE: kernel of X |-> ([X,Y_1],...,[X,Y_r])
        rows = [flatten([bracket(X, Y, par) for Y in allb]) for X in allb]
        cen = (len(allb) - rank([[Fraction(x) for x in r] for r in rows])
               if rows else 0)
        # J itself: R = omega^{-1}h squares to -1, lies in aut(omega,h), and
        # spans its center
        R = operator_R(om, h)
        Rsq = mul(R, R)
        Jok = all(Rsq[i][j] == (-1 if i == j else 0)
                  for i in range(m + n) for j in range(m + n))
        Jstab = all(all(x == 0 for row in F for x in row)
                    for F in (act_gen(m, n, R, om), act_gen(m, n, R, h)))
        Jcen = all(all(x == 0 for row in bracket(R, Y, par) for x in row)
                   for Y in allb)
        print(f"{m}|{n:<4} {'(%d|%d)' % aut_o:>12} {'(%d|%d)' % aut:>13}"
              f" {'(%d|%d)' % pred:>10} {der:>10} {cen:>7}"
              f"  {'ok' if aut == pred else 'MISMATCH'}"
              f"  J^2=-1:{'y' if Jok else 'N'}"
              f" J in aut:{'y' if Jstab else 'N'}"
              f" J central:{'y' if Jcen else 'N'}")


def act_gen(m, n, X, G):
    """X.G for an arbitrary EVEN matrix X (linear in X, so expand in E_ij)."""
    N = m + n
    out = [[Fraction(0)] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if X[i][j] == 0:
                continue
            A = act(m, n, 0, (i, j), G)
            for a in range(N):
                for b in range(N):
                    out[a][b] += X[i][j] * A[a][b]
    return out


if __name__ == "__main__":
    main()
