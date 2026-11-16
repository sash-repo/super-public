#!/usr/bin/env python3
r"""Normal shapes of a pair of non-degenerate forms on a superspace (Subsec. 6.3),
on top of the machinery of pairs.py (same conventions).

What is computed, exactly (Fraction / FLINT arithmetic, no floating point):

  1) the law of signs  Ad_{B1}(M) = eps1*eps2*M  for M = B1^{-1} B2, in all six
     combinations of parities and symmetries (Lemma 6.3.3);
  2) Theorem 6.3.1, verified constructively: an explicit change of basis bringing
     a random (symmetric, antisymmetric) pair to the claimed shape;
  3) Corollary 6.3.4 item 1) -- both forms even: the pair splits;
  4) Corollary 6.3.4 item 3) -- both forms odd: the invariant is the conjugacy
     class of an arbitrary invertible N, with no constraint on the spectrum;
  5) Corollary 6.3.4 item 2) -- forms of different parities: the invariant is
     the Aut(A)-orbit of S = A^{-1} P D^{-1} P^t, and S lies in aut(A);
  6) mu_odd(pair) = dim aut(pair)_odd;
  7) the conditions on g_0 in Lemma 6.4.1 and its case analysis 2A), 2B), 3A), 3B), 3C).

Run (from this folder):  ../venv/bin/python normshape.py
"""

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
import sys, os, random
from fractions import Fraction


from exactlin import rank, nullspace, backend                       # noqa: E402
from pairs import (parities, sign, form_basis, standard_form,       # noqa: E402
                   is_nondeg, is_form, inverse, mul, operator_R,
                   sdim_stab, sdim_orbit, gl_basis, act, NAME)

RND = random.Random(20260817)


# ------------------------------------------------------------------ helpers

def zeros(r, c):
    return [[Fraction(0)] * c for _ in range(r)]


def eye(k):
    return [[Fraction(int(i == j)) for j in range(k)] for i in range(k)]


def tr(A):
    return [list(row) for row in zip(*A)] if A else []


def matmul(A, B):
    if not A or not B:
        return []
    n, p, q = len(A), len(B), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(p)) for j in range(q)]
            for i in range(n)]


def add(A, B, s=1):
    return [[A[i][j] + s * B[i][j] for j in range(len(A[0]))]
            for i in range(len(A))]


def eq(A, B):
    return len(A) == len(B) and all(
        all(A[i][j] == B[i][j] for j in range(len(A[0]))) for i in range(len(A)))


def is_zero(A):
    return all(all(x == 0 for x in row) for row in A)


def frac(A):
    return [[Fraction(x) for x in row] for row in A]


def block(m, n, G, i, j):
    """The (i,j) block of an (m+n)x(m+n) matrix, i,j in {0,1}."""
    rs = range(0, m) if i == 0 else range(m, m + n)
    cs = range(0, n if j == 0 else n) if False else (
        range(0, m) if j == 0 else range(m, m + n))
    return [[Fraction(G[a][b]) for b in cs] for a in rs]


def assemble(m, n, blocks):
    """Inverse of `block`: build a matrix from the four blocks."""
    N = m + n
    G = zeros(N, N)
    for (i, j), Bl in blocks.items():
        rs = list(range(0, m) if i == 0 else range(m, N))
        cs = list(range(0, m) if j == 0 else range(m, N))
        for a, ra in enumerate(rs):
            for b, cb in enumerate(cs):
                G[ra][cb] = Fraction(Bl[a][b])
    return G


def charpoly(A):
    """Characteristic polynomial of a square rational matrix, Faddeev--LeVerrier.

    Returns [c_0, ..., c_k] with det(xI - A) = sum c_i x^i, c_k = 1."""
    k = len(A)
    if k == 0:
        return [Fraction(1)]
    M = zeros(k, k)
    coeffs = [Fraction(0)] * (k + 1)
    coeffs[k] = Fraction(1)
    for i in range(1, k + 1):
        M = add(matmul(A, M), [[coeffs[k - i + 1] * Fraction(int(a == b))
                                for b in range(k)] for a in range(k)])
        AM = matmul(A, M)
        c = -sum(AM[j][j] for j in range(k)) / i
        coeffs[k - i] = c
    return coeffs


def poly_of_neg(c):
    """Coefficients of p(-x) given those of p(x)."""
    return [c[i] * (-1) ** i for i in range(len(c))]


def is_square_poly(c):
    """Is the monic polynomial with coefficients c a perfect square in Q[x]?"""
    k = len(c) - 1
    if k % 2:
        return False
    h = k // 2
    q = [Fraction(0)] * (h + 1)
    q[h] = Fraction(1)
    # match coefficients of q^2 from the top down
    for d in range(k - 1, h - 1, -1):
        s = sum(q[i] * q[d - i] for i in range(max(0, d - h), h + 1)
                if 0 <= d - i <= h and i != h and d - i != h)
        # coefficient of x^d in q^2 is 2*q[h]*q[d-h] + (terms with both < h)
        q[d - h] = (c[d] - s) / 2
    return all(sum(q[i] * q[d - i] for i in range(max(0, d - h), min(d, h) + 1))
               == c[d] for d in range(k + 1))


# --------------------------------------------- the B-adjoint of an operator

def Ad(m, n, B, X, q):
    """The B-adjoint of X: Ad_B(X) = B^{-1} D_q X^t B, where D_q = diag((-1)^{q p_a}).

    By eq. (act) of pairs.py, X of parity q preserves B iff Ad_B(X) = -X."""
    N = m + n
    par = parities(m, n)
    Bi = inverse(B)
    Xt = tr(frac(X))
    XtB = matmul(Xt, frac(B))
    D = [[sign(q * par[a]) * XtB[a][b] for b in range(N)] for a in range(N)]
    return matmul(Bi, D)


def check_Ad_against_stab(m, n):
    """Self-check: Ad_B(X) = -X cuts out exactly the aut(B) that pairs.py finds."""
    for p in (0, 1):
        for eps in (1, -1):
            B = standard_form(m, n, p, eps)
            if B is None or not is_nondeg(B):
                continue
            got = []
            for q in (0, 1):
                gb = gl_basis(m, n, q)
                # X = sum t_ij E_ij ; Ad is linear in X, so build the matrix
                rows = []
                for ij in gb:
                    X = zeros(m + n, m + n)
                    X[ij[0]][ij[1]] = Fraction(1)
                    A = add(Ad(m, n, B, X, q), X)
                    rows.append([A[a][b] for a in range(m + n)
                                 for b in range(m + n)])
                if not rows:
                    got.append(0)
                    continue
                cols = len(rows[0])
                Mt = [[rows[r][c] for r in range(len(rows))] for c in range(cols)]
                got.append(len(nullspace(Mt)) if cols else len(rows))
            assert tuple(got) == sdim_stab(m, n, [B]), (m, n, p, eps, got)
    return True


# ----------------------------------------------------------- 1) the adjoint law

def generic_pair(m, n, p1, e1, p2, e2, tries=40):
    """B1 standard, B2 a random non-degenerate form of its type."""
    B1 = standard_form(m, n, p1, e1)
    if B1 is None or not is_nondeg(B1):
        return None
    b = form_basis(m, n, p2, e2)
    if not b:
        return None
    N = m + n
    for _ in range(tries):
        B2 = zeros(N, N)
        for Gb in b:
            c = RND.randint(-9, 9)
            for i in range(N):
                for j in range(N):
                    B2[i][j] += c * Gb[i][j]
        if is_nondeg(B2):
            return frac(B1), B2
    return None


CASES = [(0, -1, 0, -1, "both even 2-forms"),
         (0, 1, 0, 1, "both even symmetric"),
         (0, -1, 0, 1, "even 2-form and even symmetric"),
         (1, -1, 1, -1, "both odd, same symmetry"),
         (1, 1, 1, 1, "both odd, same symmetry"),
         (1, -1, 1, 1, "both odd, opposite symmetries"),
         (0, -1, 1, -1, "different parities"),
         (0, 1, 1, 1, "different parities"),
         (0, -1, 1, 1, "different parities"),
         (0, 1, 1, -1, "different parities")]


def adjoint_law(sdims):
    """The law, as the machine found it (the parity sign was NOT guessed right
    at first: for an odd M the sign flips):

        Ad_{B1}(M) = (-1)^{p(M)} eps1 eps2 M,   p(M) = p(B1) + p(B2),

    equivalently:  M lies in aut(B1)  iff  eps1 eps2 = (-1)^{p(M)+1}."""
    print("== 1) the adjoint law:  Ad_{B1}(M) = (-1)^{p(M)} eps1 eps2 M ==")
    print("   (Ad_B(X) = -X is membership in aut(B), so M in aut(B1) iff")
    print("    eps1*eps2 = (-1)^{p(M)+1}: the sign depends on the PARITY of M.)")
    print(f"{'m|n':>6} {'B1':>10} {'B2':>10} {'p(M)':>4} {'e1e2':>5}"
          f" {'law holds':>10} {'M in aut(B1)':>13}")
    bad = 0
    for (m, n) in sdims:
        assert check_Ad_against_stab(m, n)
        for (p1, e1, p2, e2, _name) in CASES:
            gp = generic_pair(m, n, p1, e1, p2, e2)
            if gp is None:
                continue
            B1, B2 = gp
            M = operator_R(B1, B2)
            q = (p1 + p2) % 2
            A = Ad(m, n, B1, M, q)
            s = e1 * e2 * sign(q)
            law = eq(A, [[s * x for x in row] for row in M])
            inaut = eq(A, [[-x for x in row] for row in M])
            if not law or (inaut != (s == -1)):
                bad += 1
            print(f"{m}|{n:<4} {NAME[(p1, e1)]:>10} {NAME[(p2, e2)]:>10}"
                  f" {q:>4} {e1 * e2:>5} {'yes' if law else 'NO':>10}"
                  f" {'yes' if inaut else 'no':>13}")
    print(f"   -> violations of the law: {bad}")
    return bad


# ------------------------------------------- 2) Theorem 6.3.1, constructively

def companion(c):
    """Companion matrix of the monic polynomial with coefficients c."""
    k = len(c) - 1
    C = zeros(k, k)
    for i in range(1, k):
        C[i][i - 1] = Fraction(1)
    for i in range(k):
        C[i][k - 1] = -c[i]
    return C


def cyclic_basis(A):
    """A basis v, Av, ..., A^{k-1}v for a cyclic (regular) A, or None."""
    k = len(A)
    for _ in range(30):
        v = [[Fraction(RND.randint(-9, 9))] for _ in range(k)]
        cols, w = [], v
        for _ in range(k):
            cols.append([r[0] for r in w])
            w = matmul(A, w)
        P = tr(cols)
        if rank([row[:] for row in P]) == k:
            return P
    return None


def theorem31(examples=None):
    """Theorem 6.3.1, verified by reduction.

    Method: build the normal-shape pair from a chosen J, scramble it by a random
    change of basis, then RUN the reduction of the proof of Theorem 6.3.1 on the scrambled
    pair and check that it returns the shape claimed, with a J' having the same
    characteristic polynomial as J.  The eigenvalues are chosen rational so that
    the whole run stays inside Q; the theorem itself is over C."""
    print()
    print("== 2) Theorem 6.3.1 (sym, antisym), verified by reduction ==")
    print("   normal shape:  B1 = [[0,1],[1,0]],  B2 = [[0,J],[-J^t,0]],")
    print("   M = B1^{-1}B2 = diag(-J^t, J), so spec M = spec J u (-spec J).")
    print(f"{'2n':>4} {'J':>24} {'M in o(B1)':>11} {'p(-x)=p(x)':>11}"
          f" {'reduced back':>13} {'same J':>7} {'moduli':>7}")
    if examples is None:
        examples = [diagm([2]), diagm([2, 5]), jordan_cell(3, 2),
                    diagm([2, 5, 7]), diagm([2, 5, 7, 11]),
                    blockdiag(jordan_cell(3, 2), diagm([7]))]
    for J in examples:
        n = len(J)
        d = 2 * n
        B1 = assemble(n, n, {(0, 0): zeros(n, n), (0, 1): eye(n),
                             (1, 0): eye(n), (1, 1): zeros(n, n)})
        B2 = assemble(n, n, {(0, 0): zeros(n, n), (0, 1): J,
                             (1, 0): [[-J[b][a] for b in range(n)]
                                      for a in range(n)],
                             (1, 1): zeros(n, n)})
        assert is_form(d, 0, B1, 0, 1) and is_form(d, 0, B2, 0, -1)
        M = operator_R(B1, B2)
        inaut = eq(Ad(d, 0, B1, M, 0), [[-x for x in r] for r in M])
        cp = charpoly(M)
        even = (cp == poly_of_neg(cp))
        g0 = random_invertible(d)
        C1 = matmul(tr(g0), matmul(B1, g0))
        C2 = matmul(tr(g0), matmul(B2, g0))
        g = reduce_thm31(C1, C2, d)
        ok = same = False
        if g is not None:
            D1 = matmul(tr(g), matmul(C1, g))
            D2 = matmul(tr(g), matmul(C2, g))
            Jn = block(n, n, D2, 0, 1)
            want1 = assemble(n, n, {(0, 0): zeros(n, n), (0, 1): eye(n),
                                    (1, 0): eye(n), (1, 1): zeros(n, n)})
            want2 = assemble(n, n, {(0, 0): zeros(n, n), (0, 1): Jn,
                                    (1, 0): [[-Jn[b][a] for b in range(n)]
                                             for a in range(n)],
                                    (1, 1): zeros(n, n)})
            ok = eq(D1, want1) and eq(D2, want2)
            # the reduction may return the OTHER half first, i.e. -J^t for J;
            # the invariant is the pair {spec J, -spec J}, so accept either
            negJ = [[-x for x in row] for row in J]
            same = charpoly(Jn) in (charpoly(J), charpoly(negJ))
        print(f"{d:>4} {shortJ(J):>24} {'yes' if inaut else 'NO':>11}"
              f" {'yes' if even else 'NO':>11} {'yes' if ok else 'NO':>13}"
              f" {'yes' if same else 'NO':>7} {n:>7}")
    return True


def reduce_thm31(B1, B2, d):
    """The reduction of the proof of Theorem 6.3.1, run over Q.

    M = B1^{-1}B2 is B1-skew, so spec M = -spec M.  Collect the root subspaces
    of M with a positive eigenvalue into W+ and the opposite ones into W-, check
    both are isotropic for BOTH forms, and pair them by B1."""
    n = d // 2
    M = operator_R(B1, B2)
    roots = small_rational_roots(charpoly(M))
    pos = [r for r in roots if r > 0]
    if not pos:
        return None
    Wp = root_space(M, pos, d)
    Wm = root_space(M, [-r for r in pos], d)
    if len(Wp) != n or len(Wm) != n:
        return None
    if not (isotropic(Wp, B1) and isotropic(Wp, B2)
            and isotropic(Wm, B1) and isotropic(Wm, B2)):
        return None
    return dual_pairing(Wp, Wm, B1, d)


def small_rational_roots(c, bound=40):
    """Rational roots of a monic polynomial, searched in a small box: enough
    here, since every example is built with small integer eigenvalues."""
    out = []
    for r in [Fraction(k) for k in range(-bound, bound + 1) if k != 0]:
        if sum(c[i] * r ** i for i in range(len(c))) == 0:
            out.append(r)
    return out


def root_space(M, lams, d):
    """A basis of the sum of the root subspaces of M for the eigenvalues lams."""
    basis = []
    for lam in lams:
        A = add(M, [[lam * Fraction(int(a == b)) for b in range(d)]
                    for a in range(d)], -1)
        Ak = eye(d)
        for _ in range(d):
            Ak = matmul(Ak, A)
        for v in nullspace([row[:] for row in Ak]):
            trial = basis + [v]
            if rank(tr(trial)) == len(trial):
                basis = trial
    return basis


def random_invertible(d):
    while True:
        g = [[Fraction(RND.randint(-4, 4)) for _ in range(d)] for _ in range(d)]
        if rank([row[:] for row in g]) == d:
            return g


def jordan_cell(lam, k):
    J = [[Fraction(0)] * k for _ in range(k)]
    for i in range(k):
        J[i][i] = Fraction(lam)
        if i + 1 < k:
            J[i][i + 1] = Fraction(1)
    return J


def blockdiag(*Bs):
    k = sum(len(B) for B in Bs)
    out = zeros(k, k)
    o = 0
    for B in Bs:
        for i in range(len(B)):
            for j in range(len(B)):
                out[o + i][o + j] = B[i][j]
        o += len(B)
    return out


def shortJ(J):
    n = len(J)
    dg = ",".join(str(J[i][i]) for i in range(n))
    cell = any(J[i][i + 1] != 0 for i in range(n - 1))
    return f"diag({dg})" + ("+cell" if cell else "")


def column_space(A):
    """A basis of the column space of A, as a list of column vectors."""
    d = len(A)
    cols = [[A[i][j] for i in range(d)] for j in range(d)]
    basis = []
    for c in cols:
        trial = basis + [c]
        if rank(tr(trial)) == len(trial):
            basis = trial
    return basis


def isotropic(vecs, B):
    for u in vecs:
        for v in vecs:
            s = sum(u[i] * B[i][j] * v[j] for i in range(len(u))
                    for j in range(len(v)))
            if s != 0:
                return False
    return True


def dual_pairing(Wp, Wm, B, d):
    """g whose first half is Wp and second half the B-dual basis inside Wm."""
    n = len(Wp)
    Gram = [[sum(Wp[a][i] * B[i][j] * Wm[b][j] for i in range(d)
                 for j in range(d)) for b in range(n)] for a in range(n)]
    if rank([row[:] for row in Gram]) != n:
        return None
    Gi = inverse(Gram)
    # new second half: Wm' _b = sum_a (G^{-1})_{ab}... solve <Wp_a, Wm'_b> = delta
    Wm2 = []
    for b in range(n):
        v = [sum(Gi[c][b] * Wm[c][i] for c in range(n)) for i in range(d)]
        Wm2.append(v)
    cols = Wp + Wm2
    g = tr(cols)
    if rank([row[:] for row in g]) != d:
        return None
    return g


# ------------------------------------- 3-5) Corollary 6.3.4: the three super cases

def case_both_even(sdims):
    print()
    print("== 3) Cor. 3.2 item 1): both forms EVEN -- the pair splits ==")
    print("   M = B1^{-1}B2 is even, so M = diag(M_ev, M_od) and the pair is the")
    print("   pair (A1,A2) on V_ev together with (D1,D2) on V_od.  The symmetry")
    print("   types on V_od are the OPPOSITE ones, so the two halves are of the")
    print("   same kind (sym,antisym) resp. of the two 'like' kinds interchanged.")
    print(f"{'m|n':>6} {'B1':>10} {'B2':>10} {'M even':>7} {'M=diag':>7}"
          f" {'char.poly splits':>17} {'moduli ev':>10}")
    for (m, n) in sdims:
        for (p1, e1, p2, e2, _nm) in CASES:
            if p1 or p2:
                continue
            gp = generic_pair(m, n, p1, e1, p2, e2)
            if gp is None:
                continue
            B1, B2 = gp
            M = operator_R(B1, B2)
            offdiag = all(M[a][b] == 0 for a in range(m) for b in range(m, m + n)) \
                and all(M[a][b] == 0 for a in range(m, m + n) for b in range(m))
            Mev, Mod = block(m, n, M, 0, 0), block(m, n, M, 1, 1)
            cp, cev, cod = charpoly(M), charpoly(Mev), charpoly(Mod)
            prod = [Fraction(0)] * (len(cev) + len(cod) - 1)
            for i, x in enumerate(cev):
                for j, y in enumerate(cod):
                    prod[i + j] += x * y
            print(f"{m}|{n:<4} {NAME[(p1, e1)]:>10} {NAME[(p2, e2)]:>10}"
                  f" {'yes':>7} {'yes' if offdiag else 'NO':>7}"
                  f" {'yes' if prod == cp else 'NO':>17}"
                  f" {classical_moduli(m, e1, e2) + classical_moduli(n, -e1, -e2):>10}")
    return True


def classical_shapes(ks=(2, 4, 6)):
    """The three classical normal shapes quoted in the text, verified generic.

    (sym, sym)      : 1_k  and  diag(l_1,...,l_k)                     k moduli
    (anti, anti)    : k/2 blocks [[0,1],[-1,0]] and l_i times them   k/2 moduli
    (sym, anti)     : the shape of Thm 6.3.1              k/2 moduli

    "Verified generic" = the orbit of the claimed shape has the same dimension
    as the orbit of a random pair of that type, and the stabilizers agree."""
    print()
    print("== 3a) the three classical normal shapes are the generic ones ==")
    print(f"{'k':>3} {'types':>14} {'dim stab (shape)':>17}"
          f" {'dim stab (random)':>18} {'moduli':>7} {'formula':>8}")
    for k in ks:
        for (e1, e2, nm) in ((1, 1, "sym, sym"), (-1, -1, "anti, anti"),
                             (1, -1, "sym, anti")):
            sh = classical_shape(k, e1, e2)
            if sh is None:
                continue
            S1, S2 = sh
            if not (is_form(k, 0, S1, 0, e1) and is_form(k, 0, S2, 0, e2)
                    and is_nondeg(S1) and is_nondeg(S2)):
                print(f"{k:>3} {nm:>14}   BAD SHAPE")
                continue
            ds = sdim_stab(k, 0, [S1, S2])[0]
            gp = generic_pair(k, 0, 0, e1, 0, e2)
            dr = sdim_stab(k, 0, [gp[0], gp[1]])[0] if gp else -1
            sp = len(form_basis(k, 0, 0, e1)) + len(form_basis(k, 0, 0, e2))
            mod = sp - sdim_orbit(k, 0, [S1, S2])[0]
            print(f"{k:>3} {nm:>14} {ds:>17} {dr:>18} {mod:>7}"
                  f" {classical_moduli(k, e1, e2):>8}")
    return True


def classical_shape(k, e1, e2):
    """The claimed normal shape of a generic classical pair of types (e1,e2)."""
    lam = [Fraction(2 + 3 * i) for i in range(k)]
    if e1 == 1 and e2 == 1:
        return eye(k), diagm([2 + 3 * i for i in range(k)])
    if k % 2:
        return None
    kap = k // 2
    if e1 == -1 and e2 == -1:
        J2 = [[Fraction(0), Fraction(1)], [Fraction(-1), Fraction(0)]]
        S1 = blockdiag(*[J2] * kap)
        S2 = blockdiag(*[[[Fraction(0), lam[i]], [-lam[i], Fraction(0)]]
                         for i in range(kap)])
        return S1, S2
    if e1 == 1 and e2 == -1:
        J = diagm([2 + 3 * i for i in range(kap)])
        S1 = assemble(kap, kap, {(0, 0): zeros(kap, kap), (0, 1): eye(kap),
                                 (1, 0): eye(kap), (1, 1): zeros(kap, kap)})
        S2 = assemble(kap, kap, {(0, 0): zeros(kap, kap), (0, 1): J,
                                 (1, 0): [[-J[b][a] for b in range(kap)]
                                          for a in range(kap)],
                                 (1, 1): zeros(kap, kap)})
        return S1, S2
    return None


def classical_moduli(k, e1, e2):
    """Moduli of a classical pair of types (e1,e2) in dimension k."""
    if k == 0:
        return 0
    if e1 * e2 == -1:
        return k // 2
    if e1 == 1:
        return k                       # (sym, sym): k eigenvalues
    return k // 2                      # (antisym, antisym): each twice


def case_both_odd(ns=(1, 2, 3, 4)):
    print()
    print("== 4) Cor. 3.2 item 3): both forms ODD -- N arbitrary up to conjugacy ==")
    print("   B_i = [[0,P_i],[eps_i P_i^t,0]] on C^{n|n};  even g = diag(g0,g1) acts")
    print("   by P_i -> g0^t P_i g1, so P_1 -> 1 and N := P_1^{-1}P_2 -> g1^{-1}N g1.")
    print("   Hence the complete invariant is the conjugacy class of N: NO condition")
    print("   on its spectrum whatsoever -- the sharpest contrast with Thm 6.3.1.")
    print(f"{'n':>3} {'e1':>3} {'e2':>3} {'P1 -> 1':>8} {'N -> companion':>15}"
          f" {'moduli(ev|od)':>14} {'aut(pair)':>10} {'odd stab = mu_od':>17}")
    for n in ns:
        for (e1, e2) in ((-1, -1), (1, 1), (-1, 1), (1, -1)):
            gp = generic_pair(n, n, 1, e1, 1, e2)
            if gp is None:
                continue
            B1, B2 = gp
            P1, P2 = block(n, n, B1, 0, 1), block(n, n, B2, 0, 1)
            N = matmul(inverse(P1), P2)
            # explicit reduction: g0^t = P1^{-1}, g1 = cyclic basis of N
            g1 = cyclic_basis(N)
            ok1 = ok2 = False
            if g1 is not None:
                g0 = tr(inverse(matmul(P1, g1)))       # so that g0^t P1 g1 = 1
                g = assemble(n, n, {(0, 0): g0, (1, 1): g1,
                                    (0, 1): zeros(n, n), (1, 0): zeros(n, n)})
                C1 = matmul(tr(g), matmul(B1, g))
                C2 = matmul(tr(g), matmul(B2, g))
                ok1 = eq(block(n, n, C1, 0, 1), eye(n))
                ok2 = eq(block(n, n, C2, 0, 1), companion(charpoly(N)))
                assert is_form(n, n, C1, 1, e1) and is_form(n, n, C2, 1, e2)
            aut = sdim_stab(n, n, [B1, B2])
            orb = sdim_orbit(n, n, [B1, B2])
            sp = [0, 0]
            for (p, eps) in ((0, e1), (1, e1), (0, e2), (1, e2)):
                sp[p] += len(form_basis(n, n, p, eps))
            mod = (sp[0] - orb[0], sp[1] - orb[1])
            print(f"{n:>3} {e1:>3} {e2:>3} {'yes' if ok1 else 'NO':>8}"
                  f" {'yes' if ok2 else 'NO':>15} {'(%d|%d)' % mod:>14}"
                  f" {'(%d|%d)' % aut:>10}"
                  f" {'yes' if mod[1] == aut[1] else 'NO':>17}")
    return True


def odd_symmetries_dichotomy(ns=(2, 3, 4)):
    """Why two odd forms of OPPOSITE symmetries carry no odd parameters.

    With P1 = 1 and P2 = N, an odd X = [[0,b],[c,0]] preserves both forms iff
        b^t = eps1 b,  c^t = -eps1 c,  N^t b = delta b N,  N c = delta c N^t
    with delta = eps1*eps2.  For N with distinct eigenvalues the last equations
    force b, c to be supported on the pairs (i,j) with nu_i = delta nu_j.  For
    delta = +1 that is the diagonal (n solutions); for delta = -1 it is empty
    unless the spectrum of N meets its own negative."""
    print()
    print("== 5) the exception explained: odd symmetries of a pair of ODD forms ==")
    print("   b, c solve  N^t b = delta b N,  N c = delta c N^t  (delta = eps1 eps2)")
    print(f"{'n':>3} {'delta':>6} {'spec N':>22} {'dim odd stab':>13}"
          f" {'predicted':>10}")
    for n in ns:
        for (e1, e2) in ((-1, -1), (-1, 1)):
            delta = e1 * e2
            for kind in ("generic", "spectrum meets its negative"):
                if kind == "generic":
                    N = [[Fraction(0)] * n for _ in range(n)]
                    for i in range(n):
                        N[i][i] = Fraction(2 + 3 * i)          # 2,5,8,...
                else:
                    N = [[Fraction(0)] * n for _ in range(n)]
                    for i in range(n):
                        N[i][i] = Fraction((-1) ** i * (2 + 3 * (i // 2)))
                B1 = assemble(n, n, {(0, 0): zeros(n, n), (0, 1): eye(n),
                                     (1, 0): [[Fraction(e1 * int(a == b))
                                               for b in range(n)]
                                              for a in range(n)],
                                     (1, 1): zeros(n, n)})
                B2 = assemble(n, n, {(0, 0): zeros(n, n), (0, 1): N,
                                     (1, 0): [[Fraction(e2) * N[b][a]
                                               for b in range(n)]
                                              for a in range(n)],
                                     (1, 1): zeros(n, n)})
                if not (is_nondeg(B1) and is_nondeg(B2)):
                    continue
                assert is_form(n, n, B1, 1, e1) and is_form(n, n, B2, 1, e2)
                aut = sdim_stab(n, n, [B1, B2])
                sp = spectrum_pairs(N, delta)
                pred = predicted_odd(N, delta, e1)
                print(f"{n:>3} {delta:>6} {kind[:22]:>22} {aut[1]:>13}"
                      f" {pred:>10}")
    return True


def spectrum_pairs(N, delta):
    n = len(N)
    return sum(1 for i in range(n) for j in range(n) if N[i][i] == delta * N[j][j])


def predicted_odd(N, delta, e1):
    """For diagonal N: b supported on {nu_i = delta nu_j} with b^t = eps1 b,
    c likewise with c^t = -eps1 c."""
    n = len(N)
    pairs = [(i, j) for i in range(n) for j in range(n)
             if N[i][i] == delta * N[j][j]]
    tot = 0
    for s in (e1, -e1):                       # b has s = e1, c has s = -e1
        cnt = 0
        seen = set()
        for (i, j) in pairs:
            if (j, i) in seen:
                continue
            if i == j:
                if s == 1:
                    cnt += 1                  # symmetric: diagonal entry free
            else:
                if (j, i) in pairs:
                    cnt += 1                  # one free parameter per {i,j}
                seen.add((i, j))
        tot += cnt
    return tot


def case_mixed(ns=(2, 4)):
    print()
    print("== 6) Cor. 3.2 item 2): forms of DIFFERENT parities ==")
    print("   omega = diag(A,D) even, h = [[0,P],[eps_h P^t,0]] odd; the complete")
    print("   invariant is the Aut(A)-orbit of S := A^{-1} P D^{-1} P^t, and")
    print("   S lies in aut(A) -- an infinitesimal isometry, never a scalar.")
    print(f"{'n':>3} {'omega':>10} {'h':>10} {'S in aut(A)':>12}"
          f" {'spec(M^2) = -spec':>18} {'M^2 scalar':>11} {'moduli':>9}")
    for n in ns:
        for (e_om, e_h) in ((-1, -1), (-1, 1), (1, 1), (1, -1)):
            gp = generic_pair(n, n, 0, e_om, 1, e_h)
            if gp is None:
                continue
            om, h = gp
            A, D = block(n, n, om, 0, 0), block(n, n, om, 1, 1)
            P = block(n, n, h, 0, 1)
            S = matmul(inverse(A), matmul(P, matmul(inverse(D), tr(P))))
            # S in aut(A): S^t A + A S = 0
            inaut = is_zero(add(matmul(tr(S), A), matmul(A, S)))
            M = operator_R(om, h)
            M2 = matmul(M, M)
            cp2 = charpoly(block(n, n, M2, 0, 0))
            pm = (cp2 == poly_of_neg(cp2))
            scal = all(M2[a][b] == (M2[0][0] if a == b else 0)
                       for a in range(2 * n) for b in range(2 * n))
            aut = sdim_stab(n, n, [om, h])
            orb = sdim_orbit(n, n, [om, h])
            sp = [0, 0]
            for (p, eps) in ((0, e_om), (1, e_om), (0, e_h), (1, e_h)):
                sp[p] += len(form_basis(n, n, p, eps))
            mod = (sp[0] - orb[0], sp[1] - orb[1])
            print(f"{n:>3} {NAME[(0, e_om)]:>10} {NAME[(1, e_h)]:>10}"
                  f" {'yes' if inaut else 'NO':>12} {'yes' if pm else 'NO':>18}"
                  f" {'yes' if scal else 'no':>11} {'(%d|%d)' % mod:>9}")
    return True


# --------------------------------------------- 6) mu_odd = dim aut(pair)_odd

def mu_odd_law(sdims):
    print()
    print("== 7) mu_odd(omega,h) = dim aut(omega,h)_odd, in every case ==")
    print("   Reason: the odd part of the space of pairs and gl(m|n)_odd have the")
    print("   same dimension 2mn, so the odd moduli ARE the odd symmetries.")
    ok = bad = 0
    for (m, n) in sdims:
        for (p1, e1, p2, e2, _nm) in CASES:
            gp = generic_pair(m, n, p1, e1, p2, e2)
            if gp is None:
                continue
            B1, B2 = gp
            aut = sdim_stab(m, n, [B1, B2])
            orb = sdim_orbit(m, n, [B1, B2])
            sp = [0, 0]
            for (p, eps) in ((0, e1), (1, e1), (0, e2), (1, e2)):
                sp[p] += len(form_basis(m, n, p, eps))
            mu_od = sp[1] - orb[1]
            if mu_od == aut[1]:
                ok += 1
            else:
                bad += 1
                print(f"   MISMATCH {m}|{n} {NAME[(p1,e1)]} {NAME[(p2,e2)]}:"
                      f" mu_od={mu_od} aut_od={aut[1]}")
    print(f"   -> {ok} cases confirmed, {bad} mismatches")
    return bad


# ------------------------------------------- 7) Lemma 6.4.1

def g0_of_J(J):
    """Solve eq. (X): X = [[A,B],[C,-A^t]] with
       A J^t = J^t A,   B J + J^t B = 0, B^t = -B,   C J^t + J C = 0, C^t = -C.
    Returns (dim A-part, dim B-part, dim C-part) and the full basis."""
    n = len(J)
    Jt = tr(J)

    def solve(cond, sym):
        rows, idx = [], []
        for i in range(n):
            for j in range(n):
                idx.append((i, j))
        eqs = []
        for i in range(n):
            for j in range(n):
                r = [Fraction(0)] * len(idx)
                for k in range(n):
                    r[idx.index((i, k))] += cond[0][k][j]
                    r[idx.index((k, j))] -= cond[1][i][k]
                eqs.append(r)
        if sym is not None:
            for i in range(n):
                for j in range(n):
                    r = [Fraction(0)] * len(idx)
                    r[idx.index((i, j))] += 1
                    r[idx.index((j, i))] -= sym
                    eqs.append(r)
        return len(nullspace(eqs)) if eqs else len(idx)

    # A: A J^t - J^t A = 0
    dA = solve((Jt, Jt), None)
    # B: B J + J^t B = 0  ->  B J - (-J^t) B = 0 ; and B^t = -B
    dB = solve((J, [[-x for x in row] for row in Jt]), -1)
    # C: C J^t + J C = 0 ; C^t = -C
    dC = solve((Jt, [[-x for x in row] for row in J]), -1)
    return dA, dB, dC


def lemma41():
    print()
    print("== 8) g_0 = aut(omega,h) for the classical mixed pair in normal shape ==")
    print("   g_0 = {[[A,B],[C,-A^t]] | A J^t = J^t A, BJ + J^t B = 0 = B^t + B,")
    print("                              C J^t + J C = 0 = C^t + C}")
    print(f"{'case':>34} {'dim A':>6} {'dim B':>6} {'dim C':>6} {'dim g_0':>8}"
          f" {'direct':>7}")
    cases = []
    for lam in (Fraction(3),):
        cases.append(("n=1, J=(lambda)", [[lam]]))
    cases.append(("n=2, J=lambda*1", [[Fraction(3), Fraction(0)],
                                      [Fraction(0), Fraction(3)]]))
    cases.append(("n=2, J = one Jordan cell", [[Fraction(3), Fraction(1)],
                                               [Fraction(0), Fraction(3)]]))
    cases.append(("2A) n=2, l1+l2 != 0", [[Fraction(3), Fraction(0)],
                                          [Fraction(0), Fraction(5)]]))
    cases.append(("2B) n=2, l1+l2 = 0", [[Fraction(3), Fraction(0)],
                                         [Fraction(0), Fraction(-3)]]))
    cases.append(("3A) n=3, all li+lj != 0", diagm([2, 5, 11])))
    cases.append(("3A') n=3, l1+l2 = 0", diagm([3, -3, 7])))
    cases.append(("3B) n=3, cell at 0, l1 = 0", jordan3(0, 0)))
    cases.append(("3Ba) n=3, l1+l2=0, l2 = 0", jordan3(0, 0)))
    cases.append(("3Bb) n=3, l1+l2=0, l2 != 0", jordan3(-3, 3)))
    cases.append(("3Ca) n=3, l1+l2!=0, l2 = 0", jordan3(5, 0)))
    cases.append(("3Cb) n=3, l1+l2!=0, l2 != 0", jordan3(5, 3)))
    for (name, J) in cases:
        dA, dB, dC = g0_of_J(J)
        n = len(J)
        d = 2 * n
        B1 = standard_form(d, 0, 0, 1)
        # the actual pair, for a direct count with pairs.py
        B1h = assemble(n, n, {(0, 0): zeros(n, n), (0, 1): eye(n),
                              (1, 0): eye(n), (1, 1): zeros(n, n)})
        B2h = assemble(n, n, {(0, 0): zeros(n, n), (0, 1): J,
                              (1, 0): [[-J[b][a] for b in range(n)]
                                       for a in range(n)],
                              (1, 1): zeros(n, n)})
        direct = sdim_stab(d, 0, [B1h, B2h])[0] if is_nondeg(B2h) else -1
        print(f"{name:>34} {dA:>6} {dB:>6} {dC:>6} {dA + dB + dC:>8}"
              f" {direct:>7}")
    print("   NB 2B): dim g_0 = 4 = dim gl(2), NOT 6 = dim o(4).")
    print("       Indeed g_0 = o(4) cap sp(4) = gl(2), the complexified u(2).")
    print("   NB 'direct = -1' marks a DEGENERATE B_2: det B_2 = +-(det J)^2, so")
    print("       J must be invertible, so no eigenvalue of J can vanish and the")
    print("       sub-cases 3Ba), 3Ca) with an eigenvalue 0 are empty.")
    identify_2B()
    kaehler_in_thm31_shape()
    prolongs()
    return True


def kaehler_in_thm31_shape():
    """Case 2B) of the proof of Lemma 6.4.1 is the KAEHLER pair, written with a J
    that the normalization of Theorem 6.3.1 excludes.

    Theorem 6.3.1 allows one to take all eigenvalues of J in one half-plane; the
    eigenvalues of M = diag(-J^t, J) then come in pairs +-lambda, one from each
    pair being an eigenvalue of J.  So J = diag(lambda, -lambda) is NOT
    normalized, and the normalized representative of the same pair is
    J = lambda*1_n.  Both are checked here to reduce to the same normal shape,
    and both have g_0 = gl(n) -- which is our Theorem on Kaehler pairs at l = 0."""
    print()
    print("   the case l_i + l_j = 0 versus the normalization of Thm 6.3.1:")
    print(f"{'J':>22} {'dim g_0':>8} {'M^2 scalar':>11}"
          f" {'normalized J':>26}")
    for J in (diagm([3, -3]), diagm([3, 3]),
              diagm([3, 3, -3, -3]), diagm([3, 3, 3, 3])):
        n = len(J)
        d = 2 * n
        B1 = assemble(n, n, {(0, 0): zeros(n, n), (0, 1): eye(n),
                             (1, 0): eye(n), (1, 1): zeros(n, n)})
        B2 = assemble(n, n, {(0, 0): zeros(n, n), (0, 1): J,
                             (1, 0): [[-J[b][a] for b in range(n)]
                                      for a in range(n)],
                             (1, 1): zeros(n, n)})
        M = matmul(operator_R(B1, B2), operator_R(B1, B2))
        scal = all(M[a][b] == (M[0][0] if a == b else 0)
                   for a in range(d) for b in range(d))
        g = reduce_thm31(B1, B2, d)
        cp = "-"
        if g is not None:
            Jn = block(n, n, matmul(tr(g), matmul(B2, g)), 0, 1)
            cp = "charpoly " + ",".join(str(x) for x in charpoly(Jn))
        print(f"{shortJ(J):>22} {sdim_stab(d, 0, [B1, B2])[0]:>8}"
              f" {'yes' if scal else 'no':>11} {cp:>26}")
    print("   -> diag(l,-l) and l*1_2 are the SAME pair; g_0 = gl(n), our")
    print("      Theorem on Kaehler pairs at l = 0.  So the branches of the")
    print("      cases l_i + l_j = 0 are outside the normalization of Thm 6.3.1.")


def prolongs():
    """The first Cartan prolong g_1 of (g_{-1}, g_0) for the g_0's of Lemma 6.4.1.

    g_1 = {X in Hom(g_{-1}, g_0) | X(v)w = X(w)v}.  Lemma 6.4.1 says
    g_1 = 0, so that g_* = g_{-1} (+) g_0 is FINITE-dimensional, a proper
    subalgebra of vect(n) (+) vect(n), eq. (propersub)."""
    print()
    print("== 9) the first Cartan prolong g_1 of (g_{-1}, g_0) ==")
    print(f"{'case':>34} {'dim g_0':>8} {'dim g_1':>8} {'g_* finite?':>12}")
    cases = [("n=1, J=(lambda)", diagm([3])),
             ("n=2, J = lambda*1  (g_0 = gl(2))", diagm([3, 3])),
             ("n=2, J = one Jordan cell", [[Fraction(3), Fraction(1)],
                                           [Fraction(0), Fraction(3)]]),
             ("2A) n=2, l1+l2 != 0", diagm([3, 5])),
             ("2B) n=2, l1+l2 = 0  (g_0 = gl(2))", diagm([3, -3])),
             ("n=3, J = lambda*1  (g_0 = gl(3))", diagm([3, 3, 3])),
             ("3A) n=3, all li+lj != 0", diagm([2, 5, 11])),
             ("3A') n=3, l1+l2 = 0", diagm([3, -3, 7])),
             ("3Bb) n=3, cell, l1+l2 = 0", jordan3(-3, 3)),
             ("3Cb) n=3, cell, l1+l2 != 0", jordan3(5, 3))]
    for (name, J) in cases:
        n = len(J)
        d = 2 * n
        B1 = assemble(n, n, {(0, 0): zeros(n, n), (0, 1): eye(n),
                             (1, 0): eye(n), (1, 1): zeros(n, n)})
        B2 = assemble(n, n, {(0, 0): zeros(n, n), (0, 1): J,
                             (1, 0): [[-J[b][a] for b in range(n)]
                                      for a in range(n)],
                             (1, 1): zeros(n, n)})
        g0 = stab_matrices(d, B1, B2)
        d1 = prolong_dim(g0, d)
        print(f"{name:>34} {len(g0):>8} {d1:>8}"
              f" {'yes' if d1 == 0 else 'no':>12}")


def prolong_dim(g0, d):
    """dim of g_1 = {X: V -> g_0 | X(v)w = X(w)v} for V = K^d."""
    k = len(g0)
    if k == 0:
        return 0
    # unknowns t_{a,s}: X(e_a) = sum_s t_{a,s} g0[s]
    idx = [(a, s) for a in range(d) for s in range(k)]
    eqs = []
    for a in range(d):
        for b in range(a + 1, d):
            for c in range(d):
                r = [Fraction(0)] * len(idx)
                for s in range(k):
                    r[idx.index((a, s))] += g0[s][c][b]
                    r[idx.index((b, s))] -= g0[s][c][a]
                eqs.append(r)
    return len(nullspace(eqs)) if eqs else len(idx)


def diagm(v):
    n = len(v)
    return [[Fraction(v[i]) if i == j else Fraction(0) for j in range(n)]
            for i in range(n)]


def jordan3(l1, l2):
    return [[Fraction(l1), Fraction(0), Fraction(0)],
            [Fraction(0), Fraction(l2), Fraction(1)],
            [Fraction(0), Fraction(0), Fraction(l2)]]


def identify_2B():
    """The 4-dimensional g_0 of case 2B) is gl(2): derived = sl(2), center = 1."""
    n, d = 2, 4
    J = diagm([3, -3])
    B1 = assemble(n, n, {(0, 0): zeros(n, n), (0, 1): eye(n),
                         (1, 0): eye(n), (1, 1): zeros(n, n)})
    B2 = assemble(n, n, {(0, 0): zeros(n, n), (0, 1): J,
                         (1, 0): [[-J[b][a] for b in range(n)] for a in range(n)],
                         (1, 1): zeros(n, n)})
    bas = stab_matrices(d, B1, B2)
    k = len(bas)
    der = []
    for X in bas:
        for Y in bas:
            der.append(flat(add(matmul(X, Y), matmul(Y, X), -1)))
    dder = rank(der) if der else 0
    cen = 0
    rows = []
    for X in bas:
        r = []
        for Y in bas:
            r += flat(add(matmul(X, Y), matmul(Y, X), -1))
        rows.append(r)
    cen = len(nullspace(tr(rows))) if rows else 0
    print(f"   2B) identified: dim g_0 = {k}, dim [g_0,g_0] = {dder},"
          f" dim z(g_0) = {cen}  ->  {'gl(2)' if (k, dder, cen) == (4, 3, 1) else '?'}")


def stab_matrices(d, *forms):
    """A basis of the even stabilizer of the given forms, as matrices."""
    q = 0
    gb = gl_basis(d, 0, q)
    rows = []
    for ij in gb:
        rows.append([x for G in forms for row in act(d, 0, q, ij, G) for x in row])
    cols = len(rows[0])
    Mt = [[Fraction(rows[r][c]) for r in range(len(rows))] for c in range(cols)]
    out = []
    for v in nullspace(Mt):
        X = zeros(d, d)
        for t, (i, j) in enumerate(gb):
            X[i][j] += v[t]
        out.append(X)
    return out


def flat(A):
    return [x for row in A for x in row]


# ------------------------------------------------------------------- main

def main():
    print("normal shapes of a pair of forms on a superspace -- exact arithmetic")
    print("linear algebra backend:", backend())
    print()
    sdims = [(2, 0), (0, 2), (1, 1), (2, 2), (4, 0), (0, 4), (2, 4), (4, 2),
             (3, 3), (4, 4)]
    bad1 = adjoint_law(sdims)
    theorem31()
    classical_shapes()
    case_both_even([(2, 2), (4, 2), (2, 4), (4, 4)])
    case_both_odd()
    odd_symmetries_dichotomy()
    case_mixed()
    bad2 = mu_odd_law(sdims)
    lemma41()
    print()
    print(f"== summary: adjoint-law violations {bad1}, mu_odd mismatches {bad2} ==")


if __name__ == "__main__":
    main()
