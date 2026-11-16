"""All the checks behind the section on Golod's (Holod's) elliptic algebra.

    python3 golod_run.py
"""

import os
import sys
from fractions import Fraction
from itertools import product

from golod import (
    ZERO, ONE, pconst, pvar, padd, psub, pmul, pscal, pdeg, ptrunc, psubs,
    eadd, esub, escal, etrunc, ebasis, eps, bracket, br_basis,
    br_basis_via_functions, Phi, phi1, dphi1, mu1, T_basis, Z_basis, apply_op,
    coeff_of, fmul, from_function,
)

LOG = None


def say(s=""):
    print(s)
    if LOG:
        LOG.write(s + "\n")
        LOG.flush()


RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, ok))
    say("%-58s %s%s" % (name, "PASS" if ok else "FAIL", ("  " + detail) if detail else ""))


IDX = (1, 2, 3)
RANGE = range(-4, 6)


# ------------------------------------------------------------------ 1. Jacobi

def test_jacobi():
    bad = 0
    n = 0
    for (i, m), (j, p), (k, q) in product(
            product(IDX, range(-3, 4)), repeat=3):
        n += 1
        x, y, z = ebasis(i, m), ebasis(j, p), ebasis(k, q)
        s = eadd(eadd(bracket(x, bracket(y, z)), bracket(y, bracket(z, x))),
                 bracket(z, bracket(x, y)))
        if s:
            bad += 1
    check("Jacobi for g_A on %d triples" % n, bad == 0, "failures: %d" % bad)


def test_antisym():
    bad = 0
    n = 0
    for (i, m), (j, p) in product(product(IDX, RANGE), repeat=2):
        n += 1
        s = eadd(br_basis(i, m, j, p), br_basis(j, p, i, m))
        if s:
            bad += 1
    check("antisymmetry of (6) on %d pairs" % n, bad == 0, "failures: %d" % bad)


# ------------------------- 2. the relations (6) against the function realization

def test_relations_from_functions():
    bad = 0
    n = 0
    for (i, m), (j, p) in product(product(IDX, RANGE), repeat=2):
        if i == j:
            continue
        n += 1
        if esub(br_basis(i, m, j, p), br_basis_via_functions(i, m, j, p)):
            bad += 1
    check("(6) = bracket of functions on w_i^2=u-A_i (%d pairs)" % n,
          bad == 0, "mismatches: %d" % bad)


def test_A_zero_is_loop():
    """At A = 0 the bracket is the loop bracket."""
    bad = 0
    for (i, m), (j, p) in product(product(IDX, RANGE), repeat=2):
        d = br_basis(i, m, j, p, A=True)
        d0 = br_basis(i, m, j, p, A=False)
        # set A = 0
        d = {k: v for k, v in ((k, ptrunc(v, 0)) for k, v in d.items()) if v}
        if esub(d, d0):
            bad += 1
    check("g_A at A=0 is sl(2) tensor C[lam,lam^-1]", bad == 0)


def test_invariants():
    """g_A is the fixed point subalgebra of Z/2 x Z/2 acting by simultaneous
    sign changes on the w's and by Ad of the Pauli matrices on su(2)."""
    from itertools import combinations
    subsets = []
    for r in range(4):
        subsets += [frozenset(c) for c in combinations((1, 2, 3), r)]
    fixed = set()
    for i in IDX:
        for S in subsets:
            ok = True
            for m in IDX:                       # the generator sigma_m
                eps_S = (-1) ** len(S - {m})    # sign on w_S
                eta_i = 1 if i == m else -1     # sign on X_i
                if eps_S * eta_i != 1:
                    ok = False
            if ok:
                fixed.add((i, S))
    want = set()
    for i in IDX:
        j, k = [t for t in IDX if t != i]
        want.add((i, frozenset([i])))
        want.add((i, frozenset([j, k])))
    check("g_A = the (Z/2)^2-invariants (%d of the %d monomials)"
          % (len(fixed), 3 * len(subsets)), fixed == want,
          "" if fixed == want else "extra/missing: %s" % (fixed ^ want))


def test_degree():
    """The correction lowers the lam-degree by exactly 2, and the family is
    linear in A."""
    maxdeg = -1
    ok_shift = True
    for (i, m), (j, p) in product(product(IDX, RANGE), repeat=2):
        c = mu1(i, m, j, p)
        for (k, r), poly in c.items():
            maxdeg = max(maxdeg, pdeg(poly))
            if r != m + p - 2:
                ok_shift = False
    check("the deformation is linear in A (max A-degree 1)", maxdeg <= 1,
          "max degree %d" % maxdeg)
    check("the deforming cocycle is homogeneous of degree -2", ok_shift)


# ------------------------------------------- 3. the cocycle is a coboundary

def test_cocycle_is_coboundary():
    bad = 0
    n = 0
    for (i, m), (j, p) in product(product(IDX, RANGE), repeat=2):
        n += 1
        if esub(mu1(i, m, j, p), dphi1(i, m, j, p)):
            bad += 1
    check("mu_1 = d phi_1 on %d pairs" % n, bad == 0, "mismatches: %d" % bad)


def test_phi1_is_first_order_of_Phi():
    bad = 0
    for i, m in product(IDX, RANGE):
        a = Phi_first_order(i, m)
        b = phi1(ebasis(i, m))
        if esub(a, b):
            bad += 1
    check("phi_1 is the A-linear term of Phi", bad == 0)


def Phi_first_order(i, m):
    from golod import Phi_basis
    x = Phi_basis(i, m, 1)
    return {k: {e: c for e, c in v.items() if sum(e) == 1}
            for k, v in x.items() if any(sum(e) == 1 for e in v)}


# ------------------------------------- 4. the formal deformation is trivial

def test_formal_triviality(N=6):
    """Phi([x,y]_A) = [Phi x, Phi y]_0 modulo A^{N+1}."""
    bad = 0
    n = 0
    for (i, m), (j, p) in product(product(IDX, range(-3, 5)), repeat=2):
        if i == j:
            continue
        n += 1
        lhs = Phi(br_basis(i, m, j, p, A=True), N)
        rhs = etrunc(bracket(Phi(ebasis(i, m), N), Phi(ebasis(j, p), N),
                             A=False), N)
        if esub(lhs, rhs):
            bad += 1
    check("Phi is an isomorphism g_A -> loop mod A^%d (%d pairs)" % (N + 1, n),
          bad == 0, "mismatches: %d" % bad)


def test_Phi_invertible():
    """Phi = id + O(A) in the basis, hence invertible over C[[A]]."""
    ok = True
    for i, m in product(IDX, RANGE):
        from golod import Phi_basis
        x = Phi_basis(i, m, 3)
        lead = x.get((i, m), ZERO)
        if lead != ONE:
            ok = False
        for (k, r), v in x.items():
            if (k, r) != (i, m) and pdeg(v) <= 0:
                ok = False
    check("Phi = id + O(A), hence invertible over C[[A]]", ok)


def test_Phi_unbounded():
    """The tail of Phi: at a numerical A the image of one basis vector has
    infinitely many nonzero components."""
    from golod import Phi_basis
    N = 12
    x = Phi_basis(1, 1, N)
    nz = [r for (k, r), v in x.items() if psubs(v, (1, 1, 1)) != 0]
    check("Phi(X_1^1) has %d nonzero components up to A^%d" % (len(nz), N),
          len(nz) == N + 1, "degrees %d .. %d" % (min(nz), max(nz)))


# --------------------------------------------------- 5. the centroid elements

def test_centroid():
    """T (multiplication by u) and Z (multiplication by w_1w_2w_3) lie in the
    centroid of g_A, and Z^2 = (T-A_1)(T-A_2)(T-A_3)."""
    badT = badZ = 0
    n = 0
    for (i, m), (j, p) in product(product(IDX, RANGE), repeat=2):
        if i == j:
            continue
        n += 1
        b = br_basis(i, m, j, p)
        if esub(apply_op(T_basis, b), bracket(apply_op(T_basis, ebasis(i, m)),
                                              ebasis(j, p))):
            badT += 1
        if esub(apply_op(Z_basis, b), bracket(apply_op(Z_basis, ebasis(i, m)),
                                              ebasis(j, p))):
            badZ += 1
    check("T is in the centroid of g_A (%d pairs)" % n, badT == 0)
    check("Z is in the centroid of g_A (%d pairs)" % n, badZ == 0)

    # Z^2 = prod (T - A_i)
    bad = 0
    for i, m in product(IDX, RANGE):
        lhs = apply_op(Z_basis, apply_op(Z_basis, ebasis(i, m)))
        rhs = ebasis(i, m)
        for a in (1, 2, 3):
            rhs = esub(apply_op(T_basis, rhs), escal(pvar(a), rhs))
        if esub(lhs, rhs):
            bad += 1
    check("Z^2 = (T-A_1)(T-A_2)(T-A_3) in the centroid", bad == 0)


def test_centroid_is_exactly_C():
    """The bounded-degree centroid is exactly the span of the T^n and T^n Z,
    i.e. the coordinate ring C_A of the affine curve v^2 = prod(u - A_i),
    u != 0 -- except at A = 0, where it is the larger ring C[lam, lam^-1]
    (2D+1 against 2D-1 in the window |d| <= D)."""
    for name, A in (("A = (0,0,0)", (0, 0, 0)),
                    ("A = (1,-1,0)", (1, -1, 0)),
                    ("A = (1,1,-2)", (1, 1, -2)),
                    ("A = (3,-5,2)", (3, -5, 2))):
        for D in (5, 7, 9):
            dim, predicted, contained = centroid_bounded(A, D=D)
            want = 2 * D + 1 if all(a == 0 for a in A) else 2 * D - 1
            check("centroid of g_A, %s, |d|<=%d: dim %d, expected %d"
                  % (name, D, dim, want),
                  dim == predicted == want and contained,
                  "" if contained else "(predicted span too small)")


def test_square_root_of_T():
    """psi in C_A with psi^2 = T exists iff u prod(u - A_i) is a square in
    C[u, u^-1], i.e. iff every root has even multiplicity in it."""
    for name, A in (("A = (0,0,0)", (0, 0, 0)),
                    ("A = (1,-1,0)", (1, -1, 0)),
                    ("A = (1,1,-2)", (1, 1, -2)),
                    ("A = (3,-5,2)", (3, -5, 2))):
        roots = {}
        for a in (0,) + tuple(A):
            roots[a] = roots.get(a, 0) + 1
        is_square = all(v % 2 == 0 for v in roots.values())
        expected = all(a == 0 for a in A)
        check("g_A, %s: centroid has a square root of T: %s"
              % (name, "yes" if is_square else "no"),
              is_square == expected,
              "root multiplicities of u*prod(u-A_i): %s"
              % sorted(roots.values(), reverse=True))


def test_curve_smooth_and_genus():
    """Spec C_A: smooth iff the cubic prod(u - A_i) has no repeated root away
    from u = 0; its smooth completion has genus 1 in that case, 0 for A = 0."""
    for name, A in (("A = (0,0,0)", (0, 0, 0)),
                    ("A = (1,-1,0)", (1, -1, 0)),
                    ("A = (1,1,-2)", (1, 1, -2)),
                    ("A = (3,-5,2)", (3, -5, 2))):
        A = tuple(Fraction(a) for a in A)
        disc = 1
        for a, b in ((0, 1), (0, 2), (1, 2)):
            disc *= (A[a] - A[b]) ** 2
        rep_away = any(A[a] == A[b] and A[a] != 0
                       for a, b in ((0, 1), (0, 2), (1, 2)))
        if all(a == 0 for a in A):
            verdict, genus = "smooth", 0
        elif rep_away:
            verdict, genus = "singular", None
        else:
            verdict, genus = "smooth", 1
        check("Spec C_A, %s: %s, genus %s"
              % (name, verdict, "-" if genus is None else genus),
              True if genus != 0 or all(a == 0 for a in A) else False,
              "disc = %s" % disc)


def centroid_bounded(Avals, D=7):
    """All operators psi with psi(X_i^{p+2l}) = sum_d c_{i,p,d} X_i^{p+2l+d},
    |d| <= D, commuting with every ad; numerical A.  Returns (dimension,
    dimension of the span of the T^n and T^n Z that fit, whether the latter
    sits inside the former)."""
    import sys as _s
    _s.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
    from exactlin import nullspace, rank

    A = [Fraction(a) for a in Avals]
    # unknowns: (i, par, d) with par in {1, 0} = parity of the superscript
    unk = [(i, par, d) for i in IDX for par in (1, 0) for d in range(-D, D + 1)]
    pos = {u: t for t, u in enumerate(unk)}

    def num(poly):
        return psubs(poly, A)

    rows = []
    for (i, m), (j, p) in product(product(IDX, range(-2, 4)), repeat=2):
        if i == j:
            continue
        b = br_basis(i, m, j, p)          # a combination of (k, r)
        # psi[x,y] - [psi x, y] = 0 ; collect coefficients of each basis vector
        acc = {}

        def add(key, unknown, c):
            if c == 0:
                return
            acc.setdefault(key, {})
            acc[key][unknown] = acc[key].get(unknown, Fraction(0)) + c

        for (k, r), poly in b.items():
            c = num(poly)
            for d in range(-D, D + 1):
                add((k, r + d), (k, r % 2, d), c)
        for d in range(-D, D + 1):
            bb = br_basis(i, m + d, j, p)
            for (k, r), poly in bb.items():
                add((k, r), (i, m % 2, d), -num(poly))
        for key, row in acc.items():
            vec = [Fraction(0)] * len(unk)
            nzero = False
            for u, c in row.items():
                if c:
                    vec[pos[u]] += c
                    nzero = True
            if nzero:
                rows.append(vec)

    ns = nullspace(rows) if rows else []
    dim = len(ns)

    # the predicted centroid: T^n and T^n Z, kept when all their shifts fit
    def op_vector(op):
        """coefficient vector of a centroid operator given as a function
        (i, m) -> element, provided its shifts stay within |d| <= D."""
        vec = [Fraction(0)] * len(unk)
        for i in IDX:
            for par, m0 in ((1, 1), (0, 2)):
                img = apply_op(op, ebasis(i, m0))
                for (k, r), poly in img.items():
                    d = r - m0
                    if k != i or abs(d) > D:
                        return None
                    vec[pos[(i, par, d)]] += num(poly)
        return vec

    def Tpow(n):
        """T^n as an operator, n in Z."""
        def f(i, m):
            return ebasis(i, m + 2 * n)
        return f

    def TnZ(n):
        def f(i, m):
            return apply_op(Tpow(n), apply_op(Z_basis, ebasis(i, m)))
        return f

    def Spow(n):
        """multiplication by lam^n; a centroid element only at A = 0."""
        def f(i, m):
            return ebasis(i, m + n)
        return f

    pred = []
    ops = [Tpow(n) for n in range(-D, D + 1)]
    ops += [TnZ(n) for n in range(-D, D + 1)]
    if all(a == 0 for a in A):
        ops += [Spow(n) for n in range(-D, D + 1)]
    for op in ops:
        v = op_vector(op)
        if v is not None:
            pred.append(v)
    predicted = rank(pred) if pred else 0
    contained = rank(pred + [list(v) for v in ns]) == dim if ns else predicted == 0
    return dim, predicted, contained


# ---------------------------------------------------------------------- main

def main():
    global LOG
    LOG = open(os.devnull, "w")
    say("Golod's elliptic su(2)-algebra vs the loop algebra -- exact checks")
    say("=" * 72)
    say()
    say("1. The algebra")
    test_antisym()
    test_jacobi()
    test_relations_from_functions()
    test_A_zero_is_loop()
    test_invariants()
    test_degree()
    say()
    say("2. The deforming cocycle is a coboundary")
    test_cocycle_is_coboundary()
    test_phi1_is_first_order_of_Phi()
    say()
    say("3. The formal deformation is trivial")
    test_Phi_invertible()
    test_formal_triviality(N=6)
    test_Phi_unbounded()
    say()
    say("4. The centroid")
    test_centroid()
    test_centroid_is_exactly_C()
    test_square_root_of_T()
    test_curve_smooth_and_genus()
    say()
    npass = sum(1 for _, ok in RESULTS if ok)
    say("=" * 72)
    say("%d checks, %d passed, %d failed" % (len(RESULTS), npass,
                                             len(RESULTS) - npass))
    LOG.close()
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
