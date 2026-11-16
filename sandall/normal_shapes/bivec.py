#!/usr/bin/env python3
"""The dual bivector, and the bracket it defines, for the non-alternating
symplectic forms at p = 2.

Not only the normal shapes of the forms are classified, but also the brackets
corresponding to the dual bivectors.

For a non-degenerate omega with Gram matrix G the dual bivector is C = G^{-1},
the Hamiltonian field is  H_f = sum_i (C grad f)_i partial_i  (the solution of
iota_{H_f} omega = df), and the bracket is

    {f,g} = H_f(g) = sum_{i,j} C_ij  partial_i f  partial_j g .

Checked here, exactly over F_2:

  A) det G is a CONSTANT for every closed form (not only for the normal shapes);
  B) the diagonal of C is constant too -- so the dual bivector obeys the same
     law as the form, Theorem "the diagonal of a closed form is constant";
  C) C is alternating iff G is: the dichotomy is preserved by duality;
  D) {f,f} = sum_i C_ii (partial_i f)^2 lies in K -- the bracket is
     non-alternating on O and alternating on O/K, which is where it lives;
  E) {f,g} = H_f(g) = omega(H_f,H_g), Leibniz, Jacobi, [H_f,H_g] = H_{f,g};
  F) the closed form of C, hence of the bracket, for both normal shapes of the
     census, for 1 <= m <= 6.
"""
import sys
import itertools

from nalt2 import (Alg, gram, det, nondeg, alternating, bracket, lie,
                   closed_space, g2_basis, preserving_algebra, span_dim)
from run1 import show_form
from hamcheck import inv_gram, invert, hamiltonian


# --------------------------------------------------------------- the shapes

def standard(alg):
    """the standard normal shape of table (tabshapesna), in m indeterminates.

    m odd  : (du_1)^(2) + du_2du_3 + du_4du_5 + ...
    m even : (du_1)^(2) + du_1du_2 + du_3du_4 + ...
    """
    m = alg.m
    om = {('b', 0): alg.one}
    start = 1 if m % 2 else 0
    for i in range(start, m - 1, 2):
        om[('a', i, i + 1)] = alg.one
    return om


def partner(m):
    """the index paired with u_1 in omega' -- only for m even."""
    assert m % 2 == 0
    return 1


def second(alg):
    """the second normal shape: alpha^(2) + omega' with

    m odd  : alpha = du_1 + u_1 du_d, any d != 1        (we take d = 2)
    m even : alpha = du_1 + u_z du_z, u_z the partner of u_1  (z = 2)
    """
    m = alg.m
    om = dict(standard(alg))
    if m % 2:
        d, f = 1, alg.gen(0)                      # alpha = du_1 + u_1 du_2
    else:
        d, f = partner(m), alg.gen(partner(m))    # alpha = du_1 + u_2 du_2
    # alpha^(2) = (du_1)^(2) + f^2 (du_d)^(2) + f du_1 du_d, and f^2 = 0
    key = ('a', 0, d) if 0 < d else ('a', d, 0)
    om[key] = om.get(key, 0) ^ f
    return om


# ------------------------------------------------------------ the bivector

def dual_bivector(alg, om):
    return inv_gram(alg, gram(alg, om))


def poisson(alg, C, f, g):
    out = 0
    for i in range(alg.m):
        df = alg.partial(i, f)
        if not df:
            continue
        for j in range(alg.m):
            if C[i][j]:
                dg = alg.partial(j, g)
                if dg:
                    out ^= alg.mul(C[i][j], alg.mul(df, dg))
    return out


def show_matrix(alg, C, name="C"):
    m = len(C)
    w = max(len(alg.show(C[i][j])) for i in range(m) for j in range(m))
    lines = []
    for i in range(m):
        lines.append("   " + "  ".join(alg.show(C[i][j]).rjust(w)
                                       for j in range(m)))
    return "%s =\n%s" % (name, "\n".join(lines))


def show_bracket(alg, C):
    """the bracket as a readable formula."""
    m, parts = len(C), []
    for i in range(m):
        for j in range(i, m):
            c = C[i][j]
            if not c:
                continue
            s = alg.show(c)
            s = "" if s == "1" else "(%s)" % s
            if i == j:
                parts.append("%sd%df d%dg" % (s, i + 1, i + 1))
            else:
                parts.append("%s(d%df d%dg + d%df d%dg)"
                             % (s, i + 1, j + 1, j + 1, i + 1))
    return "{f,g} = " + " + ".join(parts)


def is_const(alg, f):
    return f == 0 or f == alg.one


# --------------------------------------------------------------- the checks

def check_shape(alg, om, name):
    """the whole battery on one form."""
    m = alg.m
    G = gram(alg, om)
    D = det(alg, G)
    assert nondeg(alg, om), "%s is degenerate" % name
    C = dual_bivector(alg, om)

    print("\n--- %s,  m = %d" % (name, m))
    print("omega  = %s" % show_form(alg, om))
    print("det G  = %s   (constant: %s)" % (alg.show(D), is_const(alg, D)))
    print(show_matrix(alg, C))
    diagC = [C[i][i] for i in range(m)]
    print("diag C = (%s)   (all constant: %s)"
          % (", ".join(alg.show(c) for c in diagC),
             all(is_const(alg, c) for c in diagC)))
    print("dual bivector alternating: %s   (the form: %s)"
          % (all(c == 0 for c in diagC), alternating(om)))
    print(show_bracket(alg, C))

    # E) the bracket is the one of the Hamiltonian fields
    bad_Hg = bad_om = bad_leib = bad_jac = bad_hom = bad_self = 0
    Hs = [hamiltonian(alg, C, 1 << t) for t in range(alg.dim)]
    for s in range(alg.dim):
        f = 1 << s
        # D) self-bracket is a constant
        if not is_const(alg, poisson(alg, C, f, f)):
            bad_self += 1
        for t in range(alg.dim):
            g = 1 << t
            br = poisson(alg, C, f, g)
            if br != act_field(alg, Hs[s], g):                 # {f,g} = H_f(g)
                bad_Hg += 1
            if br != omega_pair(alg, G, Hs[s], Hs[t]):         # = omega(H_f,H_g)
                bad_om += 1
            if bracket(alg, Hs[s], Hs[t]) != hamiltonian(alg, C, br):
                bad_hom += 1
    print("{f,g} = H_f(g)                 : %d failures (must be 0)" % bad_Hg)
    print("{f,g} = omega(H_f,H_g)         : %d failures (must be 0)" % bad_om)
    print("[H_f,H_g] = H_{f,g}            : %d failures (must be 0)" % bad_hom)
    print("{f,f} in K                     : %d failures (must be 0)" % bad_self)

    # Leibniz and Jacobi on the monomial basis (Jacobi sampled if dim is big)
    trip = list(itertools.product(range(alg.dim), repeat=3))
    if len(trip) > 4096:
        import random
        random.seed(7)
        trip = random.sample(trip, 4096)
    for s, t, r in trip:
        f, g, h = 1 << s, 1 << t, 1 << r
        if poisson(alg, C, f, alg.mul(g, h)) != \
                alg.mul(poisson(alg, C, f, g), h) ^ \
                alg.mul(g, poisson(alg, C, f, h)):
            bad_leib += 1
        if poisson(alg, C, f, poisson(alg, C, g, h)) ^ \
                poisson(alg, C, g, poisson(alg, C, h, f)) ^ \
                poisson(alg, C, h, poisson(alg, C, f, g)):
            bad_jac += 1
    print("Leibniz on %d triples          : %d failures (must be 0)"
          % (len(trip), bad_leib))
    print("Jacobi  on %d triples          : %d failures (must be 0)"
          % (len(trip), bad_jac))
    return C


def act_field(alg, D, f):
    out = 0
    for i in range(alg.m):
        if D[i]:
            out ^= alg.mul(D[i], alg.partial(i, f))
    return out


def omega_pair(alg, G, D, E):
    out = 0
    for i in range(alg.m):
        for j in range(alg.m):
            if G[i][j] and D[i] and E[j]:
                out ^= alg.mul(G[i][j], alg.mul(D[i], E[j]))
    return out


# ------------------------------------------------ the two structural sweeps

def sweep(m, limit=None, seed=11):
    """A), B), C) on ALL closed forms in m indeterminates (or a sample)."""
    alg = Alg(m, (1,) * m)
    Z = closed_space(alg)
    print("\n=== sweep, m = %d : dim Z^2 = %d" % (m, len(Z)))
    vecs = list(range(2 ** len(Z))) if len(Z) <= 20 else None
    if vecs is None or (limit and len(vecs) > limit):
        import random
        random.seed(seed)
        n = limit or 2000
        vecs = [random.getrandbits(len(Z)) for _ in range(n)]
        how = "%d random" % n
    else:
        how = "all %d" % len(vecs)

    bad_det = bad_diag = bad_alt = 0
    tested = nd = 0
    for v in vecs:
        om = {}
        for k in range(len(Z)):
            if (v >> k) & 1:
                for key, c in Z[k].items():
                    om[key] = om.get(key, 0) ^ c
        om = {k: c for k, c in om.items() if c}
        tested += 1
        G = gram(alg, om)
        D = det(alg, G)
        if not is_const(alg, D):                      # A) for EVERY closed form
            bad_det += 1
        if alg.const_term(D) == 0:
            continue                                  # degenerate
        nd += 1
        C = inv_gram(alg, G)
        if not all(is_const(alg, C[i][i]) for i in range(m)):
            bad_diag += 1                             # B)
        if alternating(om) != all(C[i][i] == 0 for i in range(m)):
            bad_alt += 1                              # C)
    print("%s closed forms, %d of them non-degenerate" % (how, nd))
    print("A) det G is a constant          : %d failures (must be 0)" % bad_det)
    print("B) diag(G^{-1}) is constant     : %d failures (must be 0)" % bad_diag)
    print("C) G alternating <=> G^{-1} is  : %d failures (must be 0)" % bad_alt)


def diagonals(mmax=7):
    """the diagonal of the dual bivector of the standard shape, for each m."""
    print("\n=== diag C for the standard shape ===")
    for m in range(1, mmax + 1):
        alg = Alg(m, (1,) * m)
        om = standard(alg)
        C = dual_bivector(alg, om)
        print("  m = %d : diag G = %s, diag C = %s"
              % (m,
                 tuple(alg.const_term(gram(alg, om)[i][i]) for i in range(m)),
                 tuple(alg.const_term(C[i][i]) for i in range(m))))


def simultaneous(m):
    """can a LINEAR change put both diagonals at (1,0,...,0) at once?"""
    alg = Alg(m, (1,) * m)
    om = standard(alg)
    G0 = [[alg.const_term(x) for x in row] for row in gram(alg, om)]
    best = []
    for J in itertools.product([0, 1], repeat=m * m):
        J = [list(J[i * m:(i + 1) * m]) for i in range(m)]
        G = matmul(transpose(J), matmul(G0, J))
        if not invertible(G):
            continue
        C = matinv(G)
        dG = tuple(G[i][i] for i in range(m))
        dC = tuple(C[i][i] for i in range(m))
        if dG == dC:
            best.append((dG, dC))
    print("  m = %d : linear changes with diag G = diag C : %d"
          % (m, len(best)), "e.g." + str(best[0]) if best else "")


# ---- tiny F_2 matrix helpers, for `simultaneous` only
def transpose(A):
    return [list(r) for r in zip(*A)]


def matmul(A, B):
    n, k, p = len(A), len(B), len(B[0])
    return [[sum(A[i][t] * B[t][j] for t in range(k)) % 2 for j in range(p)]
            for i in range(n)]


def invertible(A):
    return matinv(A) is not None


def matinv(A):
    n = len(A)
    M = [row[:] + [1 if i == j else 0 for j in range(n)]
         for i, row in enumerate(A)]
    for c in range(n):
        piv = next((r for r in range(c, n) if M[r][c]), None)
        if piv is None:
            return None
        M[c], M[piv] = M[piv], M[c]
        for r in range(n):
            if r != c and M[r][c]:
                M[r] = [(x + y) % 2 for x, y in zip(M[r], M[c])]
    return [row[n:] for row in M]


def pf(alg, A):
    """Pfaffian of an alternating matrix over O, by expansion along row 0."""
    n = len(A)
    if n == 0:
        return alg.one
    if n % 2:
        return 0
    idx = list(range(n))
    out = 0
    for t in range(1, n):
        if not A[0][t]:
            continue
        rest = [i for i in idx if i not in (0, t)]
        sub = [[A[i][j] for j in rest] for i in rest]
        out ^= alg.mul(A[0][t], pf(alg, sub))       # char 2: no signs
    return out


def pfcheck(m, trials=200, seed=5):
    """The diagonal of the dual bivector, in closed form.

    For omega = (du_1)^(2) + omega' with Gram G = A + E_11:
        m odd : C_11 = 1,  C_ii = ( Pf(A_ii^) / Pf(A_11^) )^2   for i != 1;
        m even: C_11 = 0,  C_ii = ( Pf(A_1^i^) / Pf(A) )^2      for i != 1.
    """
    import random
    random.seed(seed)
    alg = Alg(m, (1,) * m)
    Z = closed_space(alg)
    bad = tested = 0
    for _ in range(trials):
        v = random.getrandbits(len(Z))
        om = {}
        for k in range(len(Z)):
            if (v >> k) & 1:
                for key, c in Z[k].items():
                    om[key] = om.get(key, 0) ^ c
        om = {k: c for k, c in om.items() if c}
        om.pop(('b', 0), None)
        for i in range(1, m):
            om.pop(('b', i), None)               # diagonal := (1,0,...,0)
        om[('b', 0)] = alg.one
        if not nondeg(alg, om):
            continue
        tested += 1
        G = gram(alg, om)
        A = [[0 if i == j else G[i][j] for j in range(m)] for i in range(m)]
        C = inv_gram(alg, G)
        want = []
        for i in range(m):
            if m % 2:
                keep = [k for k in range(m) if k != i]
                num = pf(alg, submat(A, keep))
                den = pf(alg, submat(A, [k for k in range(m) if k != 0]))
            else:
                keep = [k for k in range(m) if k not in (0, i)]
                num = pf(alg, submat(A, keep)) if i else 0
                den = pf(alg, A)
            if i == 0:
                want.append(alg.one if m % 2 else 0)
            else:
                # both are constants; over F_2 the denominator is 1
                want.append(alg.one if (alg.const_term(num) and
                                        alg.const_term(den)) else 0)
        if [C[i][i] for i in range(m)] != want:
            bad += 1
    print("  m = %d : Pfaffian formula for diag C on %d closed non-degenerate "
          "forms : %d failures (must be 0)" % (m, tested, bad))


def submat(A, keep):
    return [[A[i][j] for j in keep] for i in keep]


def selfbracket(mmax=5):
    """{f,f} = sum_i c_i lambda_i(f)^2, lambda_i(f) = the coefficient of u_i.

    So the self-bracket is a semilinear (Frobenius) form in the LINEAR part of
    f alone -- the exact algebraic shadow of the diagonal of the bivector.
    """
    print("\n=== the self-bracket ===")
    for m in range(1, mmax + 1):
        alg = Alg(m, (1,) * m)
        for name, om in (("standard", standard(alg)),
                         ("second", second(alg)) if m >= 2 else
                         ("standard", standard(alg))):
            C = dual_bivector(alg, om)
            c = [alg.const_term(C[i][i]) for i in range(m)]
            bad = 0
            for f in range(2 ** alg.dim if alg.dim <= 8 else 0):
                lam = [(f >> alg.idx[tuple(1 if k == i else 0
                                           for k in range(m))]) & 1
                       for i in range(m)]
                want = alg.one if sum(ci * li for ci, li in zip(c, lam)) % 2 \
                    else 0
                if poisson(alg, C, f, f) != want:
                    bad += 1
            print("  m = %d %-9s: c = %s, {f,f} = sum c_i lambda_i(f)^2 on all "
                  "%d f : %d failures" % (m, name, tuple(c),
                                          2 ** alg.dim if alg.dim <= 8 else 0,
                                          bad))
            if m == 1:
                break


def shearing():
    """the same shapes over O(m;un) with un != One -- no orbits needed here."""
    print("\n=== an arbitrary shearing vector ===")
    for m, un in ((2, (2, 1)), (2, (2, 2)), (3, (2, 1, 1)), (3, (1, 2, 2)),
                  (4, (2, 1, 1, 2))):
        alg = Alg(m, un)
        for name, om in (("standard", standard(alg)), ("second", second(alg))):
            G = gram(alg, om)
            D = det(alg, G)
            C = inv_gram(alg, G)
            ok_det = is_const(alg, D)
            ok_diag = all(is_const(alg, C[i][i]) for i in range(m))
            bad = 0
            for s in range(alg.dim):
                Hs = hamiltonian(alg, C, 1 << s)
                if lie(alg, Hs, om):
                    bad += 1
                for t in range(alg.dim):
                    if poisson(alg, C, 1 << s, 1 << t) != \
                            act_field(alg, Hs, 1 << t):
                        bad += 1
            print("  m=%d un=%s %-9s: det G const %s, diag C const %s, "
                  "L_{H_f}omega=0 and {f,g}=H_f(g) : %d failures"
                  % (m, un, name, ok_det, ok_diag, bad))
            print("      " + show_bracket(alg, C))


def main():
    mmax = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    for m in range(1, mmax + 1):
        alg = Alg(m, (1,) * m)
        check_shape(alg, standard(alg), "standard shape")
        if m >= 2:
            check_shape(alg, second(alg), "second shape")
    diagonals(7)
    print("\n=== can one linear change normalize both diagonals? ===")
    for m in (2, 3, 4):
        simultaneous(m)
    print("\n=== the Pfaffian formula for diag C ===")
    for m in (2, 3, 4, 5):
        pfcheck(m)
    selfbracket(3)
    shearing()
    sweep(2)
    sweep(3, limit=3000)


if __name__ == "__main__":
    main()
