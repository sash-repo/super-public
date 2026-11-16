"""svect^L_lambda(1|2): the one-parameter family, and its cocycle.

    svect^L_lambda(1|2) = Ker div_lambda,    div_lambda(X) = div(X) + lambda f/t
                                             for X = f d_t + sum g_i d_{theta_i},

a family of simple stringy Lie superalgebras that has NO polynomial counterpart:
t^lambda is not a Laurent polynomial unless lambda is an integer, and the
volume element t^lambda vol is what the definition needs.

This module
-----------
1) builds a basis of each homogeneous component over the field Q(lambda)
   (sympy; the components have dimension <= 6, so this is cheap);
2) reads off the structure constants c^k_{ij}(lambda) in that basis, so that
   the whole family becomes ONE graded vector space carrying a curve of
   brackets mu_lambda -- Jacobi is verified identically in lambda;
3) differentiates: cdot := d mu_lambda / d lambda at lambda = lambda_0 is a
   2-cocycle of mu_{lambda_0}, whose class in H^2 does not depend on the frame;
4) tests whether cdot is a coboundary.

Why (4) is a PROOF and not evidence.  For a weight-0 (degree-preserving)
cochain, delta phi (e_i,e_j) = [phi e_i, e_j] + [e_i, phi e_j] - phi([e_i,e_j])
involves phi in degrees deg e_i, deg e_j and deg e_i + deg e_j only.  So if the
core window is closed under those, the space of coboundaries RESTRICTED to the
core is already complete: no phi living further out can help.  Hence
"cdot|core is not in B|core" settles that cdot is not a coboundary, i.e.
H^2(g;g) != 0.  (The opposite verdict, h = 0, stays evidence: the COCYCLE side
is the truncated one.)
"""

from fractions import Fraction as Q
import sys, os

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]

import sympy as sp
import yb
from stringy import add, mul, d_t, d_odd
from defco import _solve

LAM = sp.Symbol("lam")
V = yb.Vect(2)                      # the ambient vect^L(1|2)


# ---------------------------------------------------------------- divergence

def div(X, lam):
    """div_lambda(f d_t + sum g_i d_i) = d_t f + lambda f / t
                                         + sum_i (-1)^{p(g_i)} d_i g_i.

    Written out here rather than taken from yb.SVect because the coefficients
    are polynomials in lambda and yb's smul() coerces to Fraction.
    """
    out = {}
    for (mu, a, S), c in X.items():
        f = {(a, S): c}
        if mu == 't':
            out = add(out, d_t(f), {(a - 1, S): lam * c})
        else:
            sgn = (-1) ** (len(S) % 2)
            out = add(out, {k: sgn * v for k, v in d_odd(mu, f).items()})
    return {k: sp.expand(v) for k, v in out.items() if sp.expand(v) != 0}


# ------------------------------------------------------------------- a basis

_BAS = {}


def basis(d, p, lam=LAM):
    """A basis of the degree-d, parity-p part of Ker div_lambda over Q(lam)."""
    if (d, p) in _BAS:
        return _BAS[(d, p)]
    keys = V.homog_keys(d, (p,))
    if not keys:
        _BAS[(d, p)] = []
        return []
    rows = {}
    for j, k in enumerate(keys):
        for t, c in div({k: sp.Integer(1)}, lam).items():
            rows.setdefault(t, {})[j] = c
    if rows:
        M = sp.Matrix([[r.get(j, 0) for j in range(len(keys))]
                       for r in rows.values()])
        ker = M.nullspace()
    else:
        ker = [sp.Matrix([1 if i == j else 0 for i in range(len(keys))])
               for j in range(len(keys))]
    out = []
    for v in ker:
        v = sp.simplify(v * sp.lcm([sp.denom(sp.together(x)) for x in v] or [1]))
        out.append({keys[i]: sp.expand(v[i]) for i in range(len(keys))
                    if sp.expand(v[i]) != 0})
    _BAS[(d, p)] = out
    return out


def vectors(N):
    """[(element, parity, degree, index)] over a window of N steps."""
    out = []
    for d in range(-N, N + 1):
        for p in (0, 1):
            for b in basis(d, p):
                out.append((b, p, d, len(out)))
    return out


def coords(d, p, elt):
    """Coordinates of a homogeneous element in the basis of (d,p)."""
    bas = basis(d, p)
    if not elt:
        return [sp.Integer(0)] * len(bas)
    keys = sorted({k for b in bas for k in b} | set(elt), key=repr)
    M = sp.Matrix([[b.get(k, 0) for b in bas] for k in keys])
    rhs = sp.Matrix([elt.get(k, 0) for k in keys])
    sol = M.solve_least_squares(rhs) if M.rows > M.cols else None
    sol, params = M.gauss_jordan_solve(rhs)
    sol = sol.subs({q: 0 for q in params})
    return [sp.simplify(x) for x in sol]


# ----------------------------------------------------- structure constants

def struct(N):
    """{(i,j): {k: c^k_ij(lambda)}} on a window of N steps."""
    vec = vectors(N)
    idx = {}
    for (b, p, d, i) in vec:
        idx.setdefault((d, p), []).append(i)
    out = {}
    for (bi, pi, di, i) in vec:
        for (bj, pj, dj, j) in vec:
            if j < i:
                continue
            br = V.brk(bi, bj)
            if not br:
                continue
            d, p = di + dj, (pi + pj) % 2
            if not basis(d, p):
                raise ValueError("bracket leaves the kernel at (%d,%d)" % (d, p))
            co = coords(d, p, br)
            cell = {}
            for a, k in enumerate(idx.get((d, p), [])):
                if co[a] != 0:
                    cell[k] = co[a]
            if cell:
                out[(i, j)] = cell
    return vec, out


# ---------------------------------------------------------- the computation

def report(N=2, lam0=Q(1, 3), core=2):
    print("svect^L_lambda(1|2): the family and its cocycle   (window N=%d)" % N)
    vec, mu = struct(N)
    print("  %d basis vectors in degrees [-%d,%d]" % (len(vec), N, N))
    dims = {}
    for (b, p, d, i) in vec:
        dims[d] = dims.get(d, 0) + 1
    print("  dims by degree: %s" % [dims.get(d, 0) for d in range(-N, N + 1)])

    # ---- 1. the structure constants really do depend on lambda
    moving = [(ij, k, c) for ij, cell in mu.items() for k, c in cell.items()
              if c.free_symbols]
    print("  structure constants depending on lambda: %d of %d"
          % (len(moving), sum(len(c) for c in mu.values())))
    if moving:
        (i, j), k, c = moving[0]
        print("    e.g.  c^%d_{%d,%d} = %s" % (k, i, j, sp.simplify(c)))

    # ---- 2. super Jacobi, identically in lambda
    bad, tested = jacobi_check(vec, mu, N)
    print("  super Jacobi identically in lambda: %s  (%d triples tested)"
          % ("OK" if bad == 0 else "FAILS in %d" % bad, tested))

    # ---- 3. the derivative is a cocycle at lambda_0
    cdot = {ij: {k: sp.diff(c, LAM).subs(LAM, sp.Rational(lam0))
                 for k, c in cell.items()}
            for ij, cell in mu.items()}
    cdot = {ij: {k: c for k, c in cell.items() if c != 0}
            for ij, cell in cdot.items()}
    cdot = {ij: cell for ij, cell in cdot.items() if cell}
    print("  nonzero entries of d mu / d lambda at lambda = %s: %d"
          % (lam0, sum(len(c) for c in cdot.values())))
    return vec, mu, cdot


def jacobi_check(vec, mu, N=None):
    """Super Jacobi for mu_lambda, as an identity in lambda.

    Only the triples all of whose pairwise sums AND whose total degree stay
    inside the window are tested: outside it the structure constants of
    `struct` are truncated, not wrong, and a truncated sum would report a
    failure that is an artefact of the window.
    """
    par = {i: p for (b, p, d, i) in vec}
    deg = {i: d for (b, p, d, i) in vec}
    if N is None:
        N = max(abs(d) for d in deg.values())
    win = set(range(-N, N + 1))

    def br(i, j):
        if (i, j) in mu:
            return mu[(i, j)]
        if (j, i) in mu:
            s = -(-1) ** (par[i] * par[j])
            return {k: s * c for k, c in mu[(j, i)].items()}
        return {}

    bad = 0
    n = len(vec)
    tested = 0
    for i in range(n):
        for j in range(i, n):
            if deg[i] + deg[j] not in win:
                continue
            for k in range(j, n):
                if (deg[j] + deg[k] not in win or deg[i] + deg[k] not in win
                        or deg[i] + deg[j] + deg[k] not in win):
                    continue
                tested += 1
                acc = {}
                for (x, y, z) in ((i, j, k), (j, k, i), (k, i, j)):
                    s = (-1) ** (par[x] * par[z])
                    for m, c in br(x, y).items():
                        for r, e in br(m, z).items():
                            acc[r] = acc.get(r, 0) + s * c * e
                if any(sp.simplify(v) != 0 for v in acc.values()):
                    bad += 1
    return bad, tested


# --------------------------------------------- is the derivative trivial?

def numeric(mu, lam0):
    out = {}
    for ij, cell in mu.items():
        c = {k: Q(str(sp.Rational(sp.simplify(v.subs(LAM, sp.Rational(lam0))))))
             for k, v in cell.items()}
        c = {k: v for k, v in c.items() if v}
        if c:
            out[ij] = c
    return out


def bracket(mu, par, i, j):
    if (i, j) in mu:
        return mu[(i, j)]
    if (j, i) in mu:
        s = -(-1) ** (par[i] * par[j])
        return {k: s * c for k, c in mu[(j, i)].items()}
    return {}


def is_coboundary(vec, mu0, cdot, core):
    """Solve delta(phi) = cdot on the core pairs.  Returns (solvable, sizes).

    On a core closed under (d_i, d_j) |-> d_i + d_j this is EXACT: a
    degree-preserving phi can only act in the degrees the core already holds,
    so no phi outside can contribute.  Hence an insoluble system proves that
    cdot is not a coboundary.
    """
    par = {i: p for (b, p, d, i) in vec}
    deg = {i: d for (b, p, d, i) in vec}
    idx = {}
    for (b, p, d, i) in vec:
        idx.setdefault((d, p), []).append(i)

    # unknowns: phi(e_i) = sum_k x_{i,k} e_k, same degree and parity
    unk = {}
    for (b, p, d, i) in vec:
        for k in idx.get((d, p), []):
            unk[(i, k)] = len(unk)

    rows, rhs = [], []
    npair = 0
    for (b, p, d, i) in vec:
        for (b2, p2, d2, j) in vec:
            if j < i:
                continue
            if abs(deg[i]) > core or abs(deg[j]) > core:
                continue
            if abs(deg[i] + deg[j]) > core:
                continue
            npair += 1
            # delta(phi)(e_i,e_j) as a linear form in the unknowns
            lin = {}
            for k in idx.get((deg[i], par[i]), []):
                for m, c in bracket(mu0, par, k, j).items():
                    lin.setdefault(m, {})[unk[(i, k)]] = \
                        lin.setdefault(m, {}).get(unk[(i, k)], Q(0)) + c
            for k in idx.get((deg[j], par[j]), []):
                for m, c in bracket(mu0, par, i, k).items():
                    lin.setdefault(m, {})[unk[(j, k)]] = \
                        lin.setdefault(m, {}).get(unk[(j, k)], Q(0)) + c
            for m, c in bracket(mu0, par, i, j).items():
                for k in idx.get((deg[m], par[m]), []):
                    lin.setdefault(k, {})[unk[(m, k)]] = \
                        lin.setdefault(k, {}).get(unk[(m, k)], Q(0)) - c
            target = bracket(cdot, par, i, j)
            for m in set(lin) | set(target):
                rows.append(lin.get(m, {}))
                rhs.append(target.get(m, Q(0)))

    n = len(unk)
    mat = [[r.get(c, Q(0)) for c in range(n)] for r in rows]
    sol = _solve(mat, rhs)
    return sol is not None, {"pairs": npair, "unknowns": n, "equations": len(rows)}


def cocycle_check(vec, mu0, cdot, N):
    """cdot must satisfy the deformation (cocycle) identity at lambda_0."""
    par = {i: p for (b, p, d, i) in vec}
    deg = {i: d for (b, p, d, i) in vec}
    win = set(range(-N, N + 1))
    bad = tested = 0
    n = len(vec)
    for i in range(n):
        for j in range(i, n):
            if deg[i] + deg[j] not in win:
                continue
            for k in range(j, n):
                if (deg[j] + deg[k] not in win or deg[i] + deg[k] not in win
                        or deg[i] + deg[j] + deg[k] not in win):
                    continue
                tested += 1
                acc = {}
                for (x, y, z) in ((i, j, k), (j, k, i), (k, i, j)):
                    s = (-1) ** (par[x] * par[z])
                    for m, c in bracket(cdot, par, x, y).items():
                        for r, e in bracket(mu0, par, m, z).items():
                            acc[r] = acc.get(r, Q(0)) + s * c * e
                    for m, c in bracket(mu0, par, x, y).items():
                        for r, e in bracket(cdot, par, m, z).items():
                            acc[r] = acc.get(r, Q(0)) + s * c * e
                if any(v for v in acc.values()):
                    bad += 1
    return bad, tested


def controls(vec, mu0, core):
    """A 'NO' from is_coboundary is worth nothing unless the test can say YES.

    Two cochains that ARE coboundaries by construction:
      1) the bracket itself, which is delta(id);
      2) delta(phi) for an explicitly chosen, lopsided phi.
    """
    par = {i: p for (b, p, d, i) in vec}
    deg = {i: d for (b, p, d, i) in vec}
    idx = {}
    for (b, p, d, i) in vec:
        idx.setdefault((d, p), []).append(i)
    c1 = is_coboundary(vec, mu0, mu0, core)[0]

    phi = {}
    for n, (b, p, d, i) in enumerate(vec):
        for m, k in enumerate(idx[(d, p)]):
            phi[(i, k)] = Q((n * 7 + m * 3) % 5 - 2)
    dphi = {}
    for (b, p, d, i) in vec:
        for (b2, p2, d2, j) in vec:
            if j < i:
                continue
            acc = {}
            for k in idx[(deg[i], par[i])]:
                for m, c in bracket(mu0, par, k, j).items():
                    acc[m] = acc.get(m, Q(0)) + phi[(i, k)] * c
            for k in idx[(deg[j], par[j])]:
                for m, c in bracket(mu0, par, i, k).items():
                    acc[m] = acc.get(m, Q(0)) + phi[(j, k)] * c
            for m, c in bracket(mu0, par, i, j).items():
                for k in idx[(deg[m], par[m])]:
                    acc[k] = acc.get(k, Q(0)) - c * phi[(m, k)]
            acc = {k: v for k, v in acc.items() if v}
            if acc:
                dphi[(i, j)] = acc
    c2 = is_coboundary(vec, mu0, dphi, core)[0]
    return c1, c2


def verdict(N=3, lam0=Q(1, 3), core=1):
    vec, mu, cdot = report(N, lam0)
    mu0 = numeric(mu, lam0)
    cd = numeric(cdot, 0) if False else {ij: {k: Q(str(sp.Rational(v)))
                                              for k, v in cell.items()}
                                         for ij, cell in cdot.items()}
    bad, tested = cocycle_check(vec, mu0, cd, N)
    print("  d mu / d lambda is a 2-cocycle of mu_{%s}: %s  (%d triples)"
          % (lam0, "OK" if bad == 0 else "FAILS in %d" % bad, tested))
    c1, c2 = controls(vec, mu0, core)
    print("  controls: delta(id) recognised as a coboundary: %s; "
          "delta(phi): %s" % ("YES" if c1 else "NO -- TEST BROKEN",
                              "YES" if c2 else "NO -- TEST BROKEN"))
    if not (c1 and c2):
        raise SystemExit("the coboundary test cannot say YES; verdict void")
    ok, sizes = is_coboundary(vec, mu0, cd, core)
    print("  d mu / d lambda is a coboundary on the core |deg| <= %d: %s"
          % (core, "YES" if ok else "NO"))
    print("    (%d core pairs, %d unknowns phi, %d equations)"
          % (sizes["pairs"], sizes["unknowns"], sizes["equations"]))
    if not ok:
        print("\n  ==> H^2(svect^L_%s(1|2); svect^L_%s(1|2)) != 0:"
              % (lam0, lam0))
        print("      the lambda-family is a nontrivial deformation.")
    return ok


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    core = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    verdict(N=N, core=core)
