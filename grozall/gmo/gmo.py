"""The symmetry algebra of a module of differential operators on the line.

The object of [GMO] = Gargoubi--Mathonet--Ovsienko, arXiv:math-ph/0506044: for
D^k_{lam,mu} the space of differential operators of order <= k from the lam-densities
to the mu-densities on S^1 (or on the line), the commutant

    I^k_{lam,mu} = End_{vect(1)} D^k_{lam,mu}

of the action.  Here we compute its LOCAL part, i.e. the endomorphisms that are
differential operators in the coefficients; [GMO]'s one non-local generator, the
Adler-type trace of their eq. (4.4), lives on (lam,mu)=(0,1) alone and is invisible
to this engine by construction -- which is exactly the discrepancy the calibration
below predicts and finds.

Model.  An operator A: F_lam -> F_mu is a sum  A = sum_i a_i D^i,  D=d/dx, with
coefficients a_i arbitrary functions.  The action of X = f d/dx is

    L_X(A) = (f D + mu f') o A  -  A o (f D + lam f'),

the composition taken in the Weyl algebra.  A local endomorphism is

    T(A)_j = sum_{i,r} c[j][i][r] * a_i^{(r)},   c constant,

and constancy of the c is not an assumption: invariance under f=1 forces it.
Invariance under f = x, x^2, x^3 then generates the rest of vect(1)_{>=-1}, which
for a local map is invariance.

Everything is exact: the entries are Fractions, the nullspace is taken by
lib/exactlin.

Self-tests: python3 gmo.py --test
Census:     python3 gmo.py --census
"""

import sys
import os
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "lib"))
from exactlin import nullspace  # noqa: E402

Q = Fraction


# ---------------------------------------------------------------------------
# Linear-in-jets expressions.
#
# An expression is a dict  {(i, r, p): coefficient}  standing for
#      sum  coefficient * a_i^{(r)} * x^p .
# Everything in sight is linear in the a's, which is what keeps this cheap.
# ---------------------------------------------------------------------------

def e_add(u, v, s=1):
    w = dict(u)
    for key, c in v.items():
        w[key] = w.get(key, 0) + s * c
        if w[key] == 0:
            del w[key]
    return w


def e_scale(u, s):
    if s == 0:
        return {}
    return {k: s * c for k, c in u.items()}


def e_mul_x(u, p):
    """Multiply by x^p."""
    if p == 0:
        return dict(u)
    return {(i, r, px + p): c for (i, r, px), c in u.items()}


def e_diff(u):
    """d/dx of an expression: the jet index goes up, the x-power comes down."""
    w = {}
    for (i, r, p), c in u.items():
        key = (i, r + 1, p)
        w[key] = w.get(key, 0) + c
        if w[key] == 0:
            del w[key]
        if p > 0:
            key = (i, r, p - 1)
            w[key] = w.get(key, 0) + p * c
            if w[key] == 0:
                del w[key]
    return w


def e_diff_n(u, n):
    for _ in range(n):
        u = e_diff(u)
    return u


# ---------------------------------------------------------------------------
# Differential operators with such coefficients:  a list P with P[j] the
# coefficient of D^j, each P[j] an expression as above.
# ---------------------------------------------------------------------------

def binom(n, k):
    if k < 0 or k > n:
        return 0
    num, den = 1, 1
    for t in range(k):
        num *= n - t
        den *= t + 1
    return num // den


def op_zero(k):
    return [{} for _ in range(k + 1)]


def op_add(P, Q_, s=1):
    n = max(len(P), len(Q_))
    out = []
    for j in range(n):
        u = P[j] if j < len(P) else {}
        v = Q_[j] if j < len(Q_) else {}
        out.append(e_add(u, v, s))
    return out


def op_compose_left_XD(P, f_pow, weight):
    """(x^{f_pow} D + weight * (x^{f_pow})') o P .

    With f = x^n one has f' = n x^{n-1}.
    """
    n = f_pow
    out = [{} for _ in range(len(P) + 1)]
    for j, Pj in enumerate(P):
        if not Pj:
            continue
        # f * D o (Pj D^j) = f * (Pj' D^j + Pj D^{j+1})
        out[j] = e_add(out[j], e_mul_x(e_diff(Pj), n))
        out[j + 1] = e_add(out[j + 1], e_mul_x(Pj, n))
        # weight * f' * Pj D^j
        if n != 0 and weight != 0:
            out[j] = e_add(out[j], e_scale(e_mul_x(Pj, n - 1), weight * n))
    return out


def op_compose_right_XD(P, f_pow, weight):
    """P o (x^{f_pow} D + weight * (x^{f_pow})').

    Uses  D^j o g = sum_s binom(j,s) g^{(s)} D^{j-s}  for a scalar function g;
    here g = x^n or weight*n*x^{n-1}, whose derivatives are exact.
    """
    n = f_pow
    out = [{} for _ in range(len(P) + 1)]
    for j, Pj in enumerate(P):
        if not Pj:
            continue
        # term  Pj D^j o (x^n D)
        for s in range(0, min(j, n) + 1):
            # (x^n)^{(s)} = n(n-1)...(n-s+1) x^{n-s}
            fac = 1
            for t in range(s):
                fac *= n - t
            if fac == 0:
                continue
            coef = binom(j, s) * fac
            out[j - s + 1] = e_add(out[j - s + 1],
                                   e_scale(e_mul_x(Pj, n - s), coef))
        # term  Pj D^j o (weight * n * x^{n-1})
        if n != 0 and weight != 0:
            m = n - 1
            for s in range(0, min(j, max(m, 0)) + 1):
                fac = 1
                for t in range(s):
                    fac *= m - t
                if fac == 0:
                    continue
                coef = binom(j, s) * fac * weight * n
                out[j - s] = e_add(out[j - s], e_scale(e_mul_x(Pj, m - s), coef))
    return out


def lie_action(P, f_pow, lam, mu):
    """L_X(A) for X = x^{f_pow} d/dx, A the operator P: F_lam -> F_mu."""
    left = op_compose_left_XD(P, f_pow, mu)
    right = op_compose_right_XD(P, f_pow, lam)
    return op_add(left, right, s=-1)


# ---------------------------------------------------------------------------
# The invariance system.
# ---------------------------------------------------------------------------

def symmetry_space(k, lam, mu, rmax=None, fpows=(1, 2, 3), verbose=False):
    """Basis of the space of local invariant endomorphisms of D^k_{lam,mu}.

    Unknowns c[j][i][r], 0 <= i,j <= k, 0 <= r <= rmax.  Returns the nullspace
    basis (rows), each row a vector of the c's in the order given by `unknowns`.
    """
    if rmax is None:
        rmax = k
    unknowns = [(j, i, r) for j in range(k + 1) for i in range(k + 1)
                for r in range(rmax + 1)]
    index = {u: n for n, u in enumerate(unknowns)}
    N = len(unknowns)

    rows = {}                     # monomial -> row of coefficients

    def add(monomial, col, coef):
        if coef == 0:
            return
        row = rows.setdefault(monomial, [0] * N)
        row[col] += coef

    for f_pow in fpows:
        # For each generator a_i of the source, the image of the basis operator
        # a_i D^i under L_X, and the two sides of T o L_X = L_X o T.
        for i0 in range(k + 1):
            P = op_zero(k)
            P[i0] = {(i0, 0, 0): 1}
            LP = lie_action(P, f_pow, lam, mu)
            # LP may have a D^{k+1} term; on D^k it is zero -- and it is, since
            # the action preserves the order.  Guard it.
            if len(LP) > k + 1 and LP[k + 1]:
                raise AssertionError("the action raised the order")
            LP = LP[:k + 1]

            #  side 1:  T(L_X A)_j = sum_{i,r} c[j][i][r] * (LP_i)^{(r)}
            for j in range(k + 1):
                for i in range(k + 1):
                    for r in range(rmax + 1):
                        col = index[(j, i, r)]
                        for (ii, rr, p), c in e_diff_n(LP[i], r).items():
                            add((f_pow, i0, j, ii, rr, p), col, c)

            #  side 2:  L_X(T A).  T A has coefficients (TA)_j = c[j][i0][r] a_i0^{(r)}
            #  -- only i = i0 contributes, A being the basis element.
            #  Its Lie action is linear in the c, so compute it per unknown.
            for m in range(k + 1):
                for r in range(rmax + 1):
                    col = index[(m, i0, r)]
                    Pm = op_zero(k)
                    Pm[m] = {(i0, r, 0): 1}
                    LPm = lie_action(Pm, f_pow, lam, mu)[:k + 1]
                    for j in range(k + 1):
                        for (ii, rr, p), c in LPm[j].items():
                            add((f_pow, i0, j, ii, rr, p), col, -c)

    mat = [r for r in rows.values() if any(r)]
    if verbose:
        print(f"    system: {len(mat)} x {N}")
    if not mat:
        return [[Q(1) if n == m else Q(0) for n in range(N)] for m in range(N)], unknowns
    basis = nullspace([[Q(c) for c in row] for row in mat])
    return basis, unknowns


def dim_symmetry(k, lam, mu, rmax=None, fpows=(1, 2, 3)):
    basis, _ = symmetry_space(k, lam, mu, rmax=rmax, fpows=fpows)
    return len(basis)


# ---------------------------------------------------------------------------
# Self-tests.
# ---------------------------------------------------------------------------

def _op_to_pairs(P):
    return [(j, sorted(Pj.items())) for j, Pj in enumerate(P) if Pj]


def test():
    ok = 0

    def chk(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            print(f"FAIL: {name}")
            raise SystemExit(1)

    # 1. The action is a Lie algebra action: [L_X, L_Y] = L_{[X,Y]} on D^k.
    #    [x^m D, x^n D] = (n-m) x^{m+n-1} D, which for our f-powers we check
    #    on a generic basis operator.
    lam, mu, k = Q(2, 3), Q(-1, 5), 3
    for m in (1, 2):
        for n in (2, 3):
            for i0 in range(k + 1):
                P = op_zero(k)
                P[i0] = {(i0, 0, 0): 1}
                A = lie_action(lie_action(P, n, lam, mu)[:k + 1], m, lam, mu)[:k + 1]
                B = lie_action(lie_action(P, m, lam, mu)[:k + 1], n, lam, mu)[:k + 1]
                C = op_add(A, B, s=-1)
                # [x^m D, x^n D] = (n-m) x^{m+n-1} D
                Pc = op_zero(k)
                Pc[i0] = {(i0, 0, 0): 1}
                R = lie_action(Pc, m + n - 1, lam, mu)[:k + 1]
                R = [e_scale(u, n - m) for u in R]
                chk(f"Lie action m={m} n={n} i0={i0}",
                    _op_to_pairs(C) == _op_to_pairs(R))

    # 2. The order-0 module D^0_{lam,mu} = F_{mu-lam} is irreducible, so its
    #    symmetry algebra is the scalars.
    for lam, mu in ((Q(0), Q(1)), (Q(1, 2), Q(3, 2)), (Q(-2, 3), Q(5, 3))):
        chk(f"D^0 trivial ({lam},{mu})", dim_symmetry(0, lam, mu) == 1)

    # 3. The identity is always a symmetry, and the conjugation is one when
    #    lam + mu = 1: check that the dimension is at least 2 there.
    chk("conjugation k=2", dim_symmetry(2, Q(1, 3), Q(2, 3)) >= 2)

    # 4. The known splitting D^1_{lam,mu} = F_{mu-lam-1} + F_{mu-lam}: the two
    #    summands are non-isomorphic unless the weights collide, so the algebra
    #    is R^2 for generic weights -- [GMO] Theorem 3.6.
    chk("D^1 generic", dim_symmetry(1, Q(1, 7), Q(5, 11)) == 2)

    print(f"gmo.py: {ok} checks pass")


# ---------------------------------------------------------------------------
# The census.
# ---------------------------------------------------------------------------

GMO_SPECIAL = {
    # (k, lam, mu) -> (name of the algebra in [GMO], its dimension)
    2: {"generic": 2},
    3: {"generic": 1},
    4: {"generic": 1},
    5: {"generic": 1},
}


def census():
    print("Local symmetries of D^k_{lam,mu} over vect(1): dim End")
    print()
    # generic weights, chosen off every exceptional locus of [GMO]
    gen = (Q(1, 7), Q(5, 11))
    for k in range(0, 6):
        print(f"  k={k}  generic (lam,mu)={gen[0]},{gen[1]}: "
              f"dim={dim_symmetry(k, gen[0], gen[1])}")
    print()
    print("  the loci [GMO] single out:")
    cases = [
        (2, Q(0), Q(1), "b        (dim 4), [GMO] Thm 3.5 5"),
        (2, Q(-1, 2), Q(3, 2), "t2       (dim 3), [GMO] Thm 3.5 3"),
        (2, Q(0), Q(2), "a + R    (dim 3), [GMO] Thm 3.5 4"),
        (2, Q(-1), Q(1), "a + R    (dim 3), [GMO] Thm 3.5 4"),
        (2, Q(1, 3), Q(2, 3), "R^2      (dim 2), lam+mu=1"),
        (3, Q(0), Q(1), "b + R^2  (dim 6), [GMO] Thm 3.4 7"),
        (3, Q(-2, 3), Q(5, 3), "R^3      (dim 3), [GMO] Thm 3.4 6  <-- Grozman"),
        (3, Q(-1, 2), Q(3, 2), "t2       (dim 3), [GMO] Thm 3.4 5"),
        (3, Q(0), Q(3), "a + R    (dim 3), [GMO] Thm 3.4 4"),
        (3, Q(0), Q(2), "a + R    (dim 3), [GMO] Thm 3.4 4"),
        (3, Q(-1), Q(1), "a + R    (dim 3), [GMO] Thm 3.4 4"),
        (3, Q(-2), Q(1), "a + R    (dim 3), [GMO] Thm 3.4 4"),
        (3, Q(1, 3), Q(2, 3), "R^2      (dim 2), lam+mu=1"),
        (3, Q(1), Q(3), "a        (dim 2), mu-lam=2"),
        (4, Q(-2, 3), Q(5, 3), "R^3      (dim 3), [GMO] Thm 3.2 2  <-- Grozman"),
        (4, Q(0), Q(3), "a + R    (dim 3), [GMO] Thm 3.2 3"),
        (4, Q(-2), Q(1), "a + R    (dim 3), [GMO] Thm 3.2 3"),
        (4, Q(0), Q(4, 5), "R^3      (dim 3), [GMO] Thm 3.2 2"),
        (4, Q(-1, 4), Q(1), "R^3      (dim 3), [GMO] Thm 3.2 2"),
        (4, Q(0), Q(1), "b + R^2  (dim 6), [GMO] Thm 3.2 4"),
        (5, Q(0), Q(1), "b + R^2  (dim 6), [GMO] Thm 3.1 3"),
        (5, Q(0), Q(0), "R^3      (dim 3), [GMO] Thm 3.1 2"),
        (5, Q(1), Q(1), "R^3      (dim 3), [GMO] Thm 3.1 2"),
        (5, Q(1, 3), Q(2, 3), "R^2      (dim 2), lam+mu=1"),
        (5, Q(-2, 3), Q(5, 3), "R^2      (dim 2), lam+mu=1, k>=5 stable"),
    ]
    for k, lam, mu in [(c[0], c[1], c[2]) for c in cases]:
        pass
    for k, lam, mu, note in cases:
        d = dim_symmetry(k, lam, mu)
        print(f"  k={k}  (lam,mu)=({lam},{mu}): dim={d}   [GMO]: {note}")


if __name__ == "__main__":
    if "--test" in sys.argv:
        test()
    elif "--census" in sys.argv:
        census()
    else:
        test()
        print()
        census()


def scan(k, fix, value, others, rmax=None):
    """Scan one line of the (lam,mu)-plane and report where the dimension jumps.

    fix='lam' fixes lam=value and runs mu over `others`, and conversely.
    """
    out = []
    for w in others:
        lam, mu = (value, w) if fix == "lam" else (w, value)
        out.append(((lam, mu), dim_symmetry(k, lam, mu, rmax=rmax)))
    return out
