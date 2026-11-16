#!/usr/bin/env python3
"""
run_prolong.py -- Cartan prolongs inside vect(m|n), exactly.

Purpose: in the *standard* grading of vect(n|n) the pair (id, pe^a(n)) prolongs to
le(n), which is infinite-dimensional; le(n) is therefore a graded subalgebra of
vect(n|n) of the case 3^o.  It is not maximal, since pe^a(n) sits inside
c(pe^a(n)) = pe^a(n) + C.E.  What is the prolong of the latter -- finite, like the
conformal cases c(o) and c(pe^{sy}) of table (60), or infinite?

Output: printed to stdout.  Usage: run_prolong.py [n ...]  (default 2 3)
"""

import sys
from fractions import Fraction as F

import os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import nullspace                                    # noqa: E402

from wgrad import (VF, bracket, monomials, le_D, divergence, Span,
                   to_vec, svect_basis)                           # noqa: E402

LOG = None


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    if LOG:
        LOG.write(s + "\n")
        LOG.flush()


def deg_k_basis(ne, no, k):
    """basis of vect_k = { monomial of degree k+1 } x { d_j }, standard grading."""
    out = []
    for m in monomials(ne, no, k + 1):
        if sum(m[0]) + len(m[1]) != k + 1:
            continue
        for j in range(ne + no):
            out.append(VF({(m, j): F(1)}, ne, no))
    return out


def coords(fields, keys):
    idx = {k: i for i, k in enumerate(keys)}
    rows = []
    for X in fields:
        v = [F(0)] * len(keys)
        for key, c in X.t.items():
            v[idx[key]] = c
        rows.append(v)
    return rows


def annihilator(rows, dim):
    """rows spanning U <= K^dim; return rows spanning U^perp (as functionals)."""
    if not rows:
        return [[F(1) if i == j else F(0) for i in range(dim)] for j in range(dim)]
    # f is a functional on K^dim killing every row: that is the nullspace of the
    # (|rows| x dim) matrix itself, not of its transpose.
    return nullspace(rows)


def prolong(u0, ne, no, kmax):
    """the Cartan prolong (g_{-1}, u0)_* degree by degree; u0 a list of VFs."""
    dims = {0: len(coords_rank(u0, ne, no, 0))}
    layers = {0: list(u0)}
    d_i = [VF({(((0,) * ne, ()), j): F(1)}, ne, no) for j in range(ne + no)]
    for k in range(1, kmax + 1):
        prev = layers[k - 1]
        keys_prev = [(m, j) for m in monomials(ne, no, k)
                     if sum(m[0]) + len(m[1]) == k for j in range(ne + no)]
        idx_prev = {key: i for i, key in enumerate(keys_prev)}
        Uprev = coords(prev, keys_prev)
        N = annihilator(Uprev, len(keys_prev))          # functionals killing U_{k-1}
        basis = deg_k_basis(ne, no, k)
        M = []                                          # rows = conditions
        for i, D in enumerate(d_i):
            brs = [bracket(X, D) for X in basis]
            cols = []
            for B in brs:
                v = [F(0)] * len(keys_prev)
                for key, c in B.t.items():
                    v[idx_prev[key]] = c
                cols.append(v)
            for f in N:
                M.append([sum(f[t] * cols[b][t] for t in range(len(keys_prev)))
                          for b in range(len(basis))])
        if not M:
            layers[k] = basis
        else:
            ns = nullspace(M)
            lay = []
            for vec in ns:
                X = VF({}, ne, no)
                for j, c in enumerate(vec):
                    if c:
                        X = X.add(basis[j], F(c))
                if X:
                    lay.append(X)
            layers[k] = lay
        dims[k] = len(layers[k])
    return layers, dims


def coords_rank(fields, ne, no, k):
    keys = sorted({key for X in fields for key in X.t}, key=str)
    S = Span({key: i for i, key in enumerate(keys)})
    out = []
    for X in fields:
        v = to_vec(X, S.index)
        if S.add(v):
            out.append(X)
    return out


def le_layers(n, kmax):
    """le(n)_k = span of D_f with deg f = k+2."""
    ne = no = n
    out = {}
    for k in range(-1, kmax + 1):
        fields = []
        for m in monomials(ne, no, k + 2):
            if sum(m[0]) + len(m[1]) != k + 2:
                continue
            X = le_D(m, ne, no, n)
            if X:
                fields.append(X)
        out[k] = coords_rank(fields, ne, no, k) if fields else []
    return out


def sdim(fields, ne):
    ev = od = 0
    for X in fields:
        (m, j) = next(iter(X.t))
        p = (len(m[1]) + (1 if j >= ne else 0)) % 2
        if p:
            od += 1
        else:
            ev += 1
    return ev, od


def main(ns=(2, 3)):
    for n in ns:
        ne = no = n
        say("=" * 74)
        say(f"vect({n}|{n}) in the standard grading")
        say("=" * 74)
        L = le_layers(n, 3)
        say(f"  dim le({n})_k  for k = 0,1,2,3: "
            + ", ".join(str(len(L[k])) for k in range(0, 4)))

        E = VF({(((1 if i == j else 0,) if False else None), None): None}
               for i, j in []) if False else None
        # the Euler operator E = sum u_i d_i
        t = {}
        for j in range(ne + no):
            a = [0] * ne
            S = ()
            if j < ne:
                a[j] = 1
            else:
                S = (j - ne,)
            t[((tuple(a), S), j)] = F(1)
        E = VF(t, ne, no)

        u0_pe = L[0]
        u0_cpe = L[0] + [E]
        say(f"  dim pe^a({n}) = le({n})_0 = {len(u0_pe)}, "
            f"dim c(pe^a({n})) = {len(u0_cpe)}")

        for name, u0 in [("pe^a", u0_pe), ("c(pe^a)", u0_cpe)]:
            layers, dims = prolong(u0, ne, no, 3)
            say(f"  prolong (id, {name}({n}))_* : dims k=0..3 = "
                + ", ".join(str(dims[k]) for k in range(0, 4)))
            diff = [dims[k] - len(L[k]) for k in range(0, 4)]
            say(f"      minus dim le({n})_k          = "
                + ", ".join(str(d) for d in diff))

        # is le(n) inside svect(n|n)?
        bad = [X for k in range(0, 3) for X in L[k] if divergence(X)]
        say(f"  fields of le({n})_0..le({n})_2 with non-zero divergence: "
            f"{len(bad)} of {sum(len(L[k]) for k in range(0,3))}"
            "   (so le(n) is not inside svect(n|n))")
    say("\nDone.")



# ---------------------------------------------------------------------------
# The two periplectic algebras, and the maximality step for le(n) + C.E
# ---------------------------------------------------------------------------

def linear_fields_preserving_odd_form(n, eps):
    """{ X in gl(n|n) : om(Xu,v) + (-1)^{p(X)p(u)} om(u,Xv) = 0 },  as vector fields.

    Basis of V: e_1..e_n even (= the x's), f_1..f_n odd (= the xi's);
    om(e_i,f_j) = delta_ij,  om(f_j,e_i) = eps*delta_ij,  om(e,e) = om(f,f) = 0.
    eps = +1: odd symmetric form;  eps = -1: odd antisymmetric one.
    A matrix M is realized as the linear vector field sum_{k,l} M_{kl} u_l d_k.
    """
    ne = no = n
    N = 2 * n
    par = [0] * n + [1] * n

    def om(a, b):
        if a < n and b >= n:
            return F(1) if b - n == a else F(0)
        if a >= n and b < n:
            return F(eps) if a - n == b else F(0)
        return F(0)

    cols = []          # one column per matrix unit E_{kl}
    units = [(k, l) for k in range(N) for l in range(N)]
    rows = []
    for (a, b) in [(a, b) for a in range(N) for b in range(N)]:
        row = []
        for (k, l) in units:
            pX = (par[k] + par[l]) % 2
            # om(E_{kl} a, b) = delta_{la} om(k,b)
            v1 = om(k, b) if l == a else F(0)
            v2 = om(a, k) if l == b else F(0)
            sign = F(-1) ** (pX * par[a])
            row.append(v1 + sign * v2)
        rows.append(row)
    ns = nullspace(rows)
    out = []
    for vec in ns:
        t = {}
        for idx, c in enumerate(vec):
            if not c:
                continue
            k, l = units[idx]
            a = [0] * ne
            S = ()
            if l < ne:
                a[l] = 1
            else:
                S = (l - ne,)
            key = ((tuple(a), S), k)
            t[key] = t.get(key, F(0)) + F(c)
        X = VF(t, ne, no)
        if X:
            out.append(X)
    return out


def module_generated(gens, acting, ne, no, ambient):
    """dim of the `acting`-submodule generated by `gens`, inside `ambient`.

    The coordinate space must be that of the whole ambient layer: taking only the
    keys occurring in the generators silently truncates the module being built.
    """
    keys = sorted({k for X in ambient for k in X.t}, key=str)
    S = Span({k: i for i, k in enumerate(keys)})
    frontier = []
    for X in gens:
        v = to_vec(X, S.index)
        if v is not None and S.add(v):
            frontier.append(X)
    while frontier:
        nxt = []
        for X in frontier:
            for A in acting:
                C = bracket(A, X)
                if not C:
                    continue
                v = to_vec(C, S.index)
                if v is None:
                    continue
                if S.add(v):
                    nxt.append(C)
        frontier = nxt
    return S.dim(), len(keys)


def extra(ns=(2, 3)):
    say("\n" + "=" * 74)
    say("The two periplectic algebras in gl(n|n) and their prolongs")
    say("=" * 74)
    for n in ns:
        ne = no = n
        for eps, name in ((1, "odd symmetric"), (-1, "odd antisymmetric")):
            u0 = linear_fields_preserving_odd_form(n, eps)
            E = None
            t = {}
            for j in range(2 * n):
                a = [0] * ne
                S = ()
                if j < ne:
                    a[j] = 1
                else:
                    S = (j - ne,)
                t[((tuple(a), S), j)] = F(1)
            E = VF(t, ne, no)
            for tag, uu in (("pe", u0), ("c(pe)", u0 + [E])):
                layers, dims = prolong(uu, ne, no, 3)
                say(f"  n={n}, {name:>17} form, {tag:>6}: dim = "
                    + ", ".join(f"{dims[k]}" for k in range(0, 4)))
        L = le_layers(n, 2)
        say(f"      for comparison, dim le({n})_k = "
            + ", ".join(str(len(L[k])) for k in range(0, 3)))

    say("\n" + "=" * 74)
    say("Maximality step: does gl(n|n) = vect(n|n)_0 generate all of vect_1 from le_1?")
    say("=" * 74)
    for n in ns:
        ne = no = n
        L = le_layers(n, 1)
        g0 = deg_k_basis(ne, no, 0)
        g1 = deg_k_basis(ne, no, 1)
        got, full = module_generated(L[1], g0, ne, no, g1)
        say(f"  n={n}: the gl({n}|{n})-module generated by le({n})_1 has dim {got}; "
            f"dim vect({n}|{n})_1 = {full}  ==> "
            + ("all of it" if got == full else "PROPER submodule"))


if __name__ == "__main__":
    _ns = tuple(int(x) for x in sys.argv[1:]) or (2, 3)
    main(_ns)
    extra(_ns)
