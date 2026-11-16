#!/usr/bin/env python3
"""Cross-validation against the group's own p = 2 Hamiltonian algebra.

In the Section on Eick's algebras of the text, the non-alternating Hamiltonian algebra
in THREE indeterminates at p = 2 is written with the bracket

    {f,g} = d_1 f d_3 g + d_3 f d_1 g + d_2 f d_2 g,      H_f as printed there,

i.e. it is the algebra preserving the form with Gram matrix Pi_3 (antidiagonal
1's, so with a 1 in the middle: non-alternating).  In our normalization that
form is  omega = dx_1^{(2)} + dx_2 dx_3  after renaming the variables.

Checked here:
  1. H_f, defined by  iota_{H_f} omega = df, reproduces that formula;
  2. every H_f preserves omega, and the H_f span O/K (dim 2^m - 1);
  3. the bracket of the H_f is the Poisson bracket above;
  4. h(omega) is BIGGER than {H_f}, and the quotient is H^1 of the divided
     power de Rham complex -- so the two must not be confused.
"""
from nalt2 import *
from run1 import show_form


def inv_gram(alg, G):
    """inverse of the Gram matrix over O, by Gauss-Jordan (char 2, m small)."""
    m = len(G)
    A = [row[:] + [alg.one if i == j else 0 for j in range(m)]
         for i, row in enumerate(G)]
    for c in range(m):
        piv = next((r for r in range(c, m) if alg.const_term(A[r][c])), None)
        assert piv is not None, "Gram matrix is not invertible"
        A[c], A[piv] = A[piv], A[c]
        inv = invert(alg, A[c][c])
        A[c] = [alg.mul(inv, v) for v in A[c]]
        for r in range(m):
            if r != c and A[r][c]:
                f = A[r][c]
                A[r] = [A[r][k] ^ alg.mul(f, A[c][k]) for k in range(2 * m)]
    return [row[m:] for row in A]


def invert(alg, f):
    """inverse of a unit of O: (1+n)^{-1} = 1 + n + n^2 + ... , n nilpotent."""
    assert alg.const_term(f) == 1
    n = f ^ alg.one
    out, term = alg.one, alg.one
    for _ in range(alg.dim):
        term = alg.mul(term, n)
        if not term:
            break
        out ^= term
    return out


def hamiltonian(alg, Ginv, f):
    """H_f with iota_{H_f} omega = df."""
    grad = [alg.partial(k, f) for k in range(alg.m)]
    return [
        # the contraction of D with omega has i-th coefficient sum_j G_ij f_j,
        # so f = Ginv * grad
        eval_row(alg, Ginv[i], grad) for i in range(alg.m)]


def eval_row(alg, row, vec):
    out = 0
    for a, b in zip(row, vec):
        out ^= alg.mul(a, b)
    return out


def poisson(alg, f, g):
    """the bracket printed in the text, rewritten in our variables.

    There it is  {f,g} = d_{y_1}f d_{y_3}g + d_{y_3}f d_{y_1}g + d_{y_2}f d_{y_2}g
    with y_2 the variable carrying the diagonal; ours is x_1, so
    y_1 = x_2, y_2 = x_1, y_3 = x_3.
    """
    return alg.mul(alg.partial(1, f), alg.partial(2, g)) ^ \
           alg.mul(alg.partial(2, f), alg.partial(1, g)) ^ \
           alg.mul(alg.partial(0, f), alg.partial(0, g))


def de_rham_h1(alg):
    """dim H^1 = dim ker(d: Omega^1 -> Omega^2) - dim d(O)."""
    b2 = g2_basis(alg.m)
    rows = []
    for i in range(alg.m):
        for t in range(alg.dim):
            of = [0] * alg.m
            of[i] = 1 << t
            r = []
            for k in b2:
                c = d1(alg, of).get(k, 0)
                r.extend((c >> s) & 1 for s in range(alg.dim))
            rows.append(r)
    Z1 = alg.m * alg.dim - rank(rows, p=2)
    dO = [[(alg.partial(k, 1 << t) >> s) & 1
           for k in range(alg.m) for s in range(alg.dim)]
          for t in range(alg.dim)]
    return Z1 - rank(dO, p=2)


def run(m=3):
    alg = Alg(m, (1,) * m)
    # omega = dx_1^{(2)} + dx_2 dx_3   (the y-variables of eick130625 are
    # y_1 = x_2, y_2 = x_1, y_3 = x_3)
    om = {('b', 0): alg.one, ('a', 1, 2): alg.one}
    print("omega =", show_form(alg, om), " closed:", not d2(alg, om),
          " non-degenerate:", nondeg(alg, om))
    G = gram(alg, om)
    Gi = inv_gram(alg, G)

    Hs, bad_pres, bad_formula = [], 0, 0
    for t in range(alg.dim):
        f = 1 << t
        H = hamiltonian(alg, Gi, f)
        Hs.append(H)
        if lie(alg, H, om):
            bad_pres += 1
        # the formula printed in eick130625: H_f = d_1f d_1 + d_3f d_2 + d_2f d_3
        want = [alg.partial(0, f), alg.partial(2, f), alg.partial(1, f)]
        if H != want:
            bad_formula += 1
    print("H_f preserves omega : %d failures (must be 0)" % bad_pres)
    print("H_f agrees with the formula printed in the text : %d "
          "failures (must be 0)" % bad_formula)
    print("dim span{H_f} = %d   (dim O - 1 = %d)"
          % (span_dim(alg, Hs), alg.dim - 1))

    bad = 0
    for s in range(alg.dim):
        for t in range(alg.dim):
            f, g = 1 << s, 1 << t
            if bracket(alg, hamiltonian(alg, Gi, f), hamiltonian(alg, Gi, g)) \
                    != hamiltonian(alg, Gi, poisson(alg, f, g)):
                bad += 1
    print("[H_f, H_g] = H_{f,g} for the printed Poisson bracket : %d failures "
          "(must be 0)" % bad)

    h = preserving_algebra(alg, om)
    print("dim h(omega) = %d, dim span{H_f} = %d, difference = %d"
          % (len(h), span_dim(alg, Hs), len(h) - span_dim(alg, Hs)))
    print("dim H^1 of the divided power de Rham complex = %d" % de_rham_h1(alg))
    print("derived series of h(omega) :", derived_series(alg, h))


if __name__ == "__main__":
    run(3)
