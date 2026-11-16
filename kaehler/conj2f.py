r"""Theorems 1.8.2 and 3.2.1, identified by the action of the o(2)-generator.

The closures are simple of sdim (4|4) with g_ev = sl(2) (+) C, resp. of sdim (11|8)
with g_ev = sp(4) (+) C.  For the identification with osp(2|2), resp. osp(2|4), the
central element z of g_ev must act on g_od with eigenvalues +-lambda, each eigenspace
being the tautological module of sl(2), resp. sp(4).  This script computes the
characteristic polynomial of ad z on g_od exactly, and the dimensions of the
eigenspaces.

Run (from this folder):  ../venv/bin/python conj2f.py
"""
import sys
from fractions import Fraction

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import nullspace, rank                       # noqa: E402

from flatmodel import Model, closure, d_operator, \
    form_to_multiplication, form_to_convolution            # noqa: E402
from conj2 import sd, centre, reduce_basis                 # noqa: E402
from conj2c import is_simple                               # noqa: E402
from conj2d import build                                   # noqa: E402
from conj2e import partner_forms                           # noqa: E402

half = Fraction(1, 2)


def coords(vecs):
    """monomial keys -> column index, as a stable ordered list."""
    keys = []
    seen = set()
    for v in vecs:
        for k in v:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    return keys


def matrix_of(M, op, basis):
    """matrix of ad(op) on the span of `basis` (assumed ad(op)-stable)."""
    keys = coords(basis + [M.br(op, b) for b in basis])
    idx = {k: i for i, k in enumerate(keys)}
    # columns = basis vectors in monomial coordinates
    B = [[Fraction(0)] * len(basis) for _ in keys]
    for j, b in enumerate(basis):
        for k, v in b.items():
            B[idx[k]][j] = Fraction(v)
    cols = []
    for b in basis:
        img = M.br(op, b)
        col = [Fraction(0)] * len(keys)
        for k, v in img.items():
            col[idx[k]] = Fraction(v)
        cols.append(col)
    # solve B * X = cols  (each column separately), B has full column rank
    import sympy
    Bm = sympy.Matrix([[sympy.Rational(x.numerator, x.denominator) for x in row]
                       for row in B])
    Cm = sympy.Matrix([[sympy.Rational(c[i].numerator, c[i].denominator)
                        for c in cols] for i in range(len(keys))])
    X = Bm.solve_least_squares(Cm) if Bm.rows != Bm.cols else Bm.solve(Cm)
    return sympy.simplify(X)


def centre_vectors(M, vecs):
    """actual central elements of the span of `vecs` (not just their number)."""
    keys = coords([M.br(a, b) for a in vecs for b in vecs] or [{}])
    idx = {k: i for i, k in enumerate(keys)}
    rows = []                       # one row per (monomial, v) condition
    for v in vecs:
        block = [[Fraction(0)] * len(vecs) for _ in keys]
        for j, u in enumerate(vecs):
            for k, val in M.br(u, v).items():
                block[idx[k]][j] = Fraction(val)
        rows.extend(block)
    if not rows:
        return []
    out = []
    for sol in nullspace(rows):
        z = {}
        for j, co in enumerate(sol):
            if co:
                z = M.add(z, M.smul(Fraction(co), vecs[j]))
        if z:
            out.append(z)
    return out


def report(M, red, name, ev_expect):
    import sympy
    ev = [e for e in red if M.parity(e) == 0]
    od = [e for e in red if M.parity(e) == 1]
    zs = centre_vectors(M, ev)
    print('  %s: sdim %s, simple %s, centre of g_ev %s'
          % (name, sd(M, red), is_simple(M, red)[0], centre(M, ev)))
    if not zs:
        print('     no central element in g_ev -- identification inconclusive')
        return
    z = zs[0]
    Z = matrix_of(M, z, od)
    p = Z.charpoly()
    print('     char. poly of ad z on g_od :', sympy.factor(p.as_expr()))
    eigs = Z.eigenvals()
    print('     eigenvalues (with multiplicity):',
          {sympy.nsimplify(k): v for k, v in eigs.items()})
    print('     dim g_od = %d, expected two eigenspaces of dim %d each'
          % (len(od), ev_expect))


print('=' * 72)
print('2a)  Kaehler: the twisted Dolbeault algebra, expected osp(2|2) = sl(1|2)')
print('=' * 72)
for nn in (1, 2):
    M, n, q, p, c, x, t, d, dc, Xp, Xm, H = build(nn)
    alpha, alpha_c = partner_forms(M, n, q, p, c, x)
    Dp = M.add(d, M.smul(-half, alpha))
    Dcp = M.add(dc, M.smul(-half, alpha_c))
    basis, sdm = closure(M, [Dp, Dcp, Xp, Xm], limit=600)
    red = reduce_basis(M, basis)
    report(M, red, 'C^{%d|0}' % (2 * n), 2)
    print()

print('=' * 72)
print('2b)  hyper-Kaehler: expected osp(2|4)')
print('=' * 72)
# rebuild the hyper-Kaehler flat model exactly as in conj2.py / conj2c.py
for nn in (1,):
    N = 4 * nn
    M = Model([0] * N)
    idx = list(range(N))
    c = lambda a: M.gen('c', a)
    x = lambda a: M.gen('x', a)
    d = d_operator(M)
    # the three Kaehler forms of the flat hyper-Kaehler C^4 (quaternionic units)
    pairs_i = [(0, 1), (2, 3)]
    pairs_j = [(0, 2), (3, 1)]
    pairs_k = [(0, 3), (1, 2)]
    E, F = {}, {}
    for lab, prs in (('i', pairs_i), ('j', pairs_j), ('k', pairs_k)):
        E[lab] = form_to_multiplication(M, [(1, pr) for pr in prs])
        F[lab] = form_to_convolution(M, [(1, pr) for pr in prs])
    # the connection 1-form for the twist along omega_i
    alpha1 = M.add(M.mul(c(0), x(1)), M.smul(-1, M.mul(c(1), x(0))),
                   M.mul(c(2), x(3)), M.smul(-1, M.mul(c(3), x(2))))
    D1 = M.add(d, M.smul(-half, alpha1))
    gens = [D1] + [E[l] for l in 'ijk'] + [F[l] for l in 'ijk']
    basis, sdm = closure(M, gens, limit=800)
    red = reduce_basis(M, basis)
    report(M, red, 'C^{%d|0}' % N, 4)
