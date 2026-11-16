#!/usr/bin/env python3
"""
run_cartan.py -- the super Cartan question, by machine, for small V:
which pairs (id, u_0) with u_0 irreducible in gl(V) have an INFINITE prolong?

For the Lie case the answer is Cartan's: gl, sl, sp, csp -- four of them, giving
vect, svect, h, and the conformal-symplectic prolong.  In the super case no such
theorem is on record; what is on record is the classification of the simple
W-graded vectorial Lie superalgebras [LS1], whose depth-1 part is the same list
read backwards.  Here we test the small cases directly.

A prolong is declared infinite when dim g_k > 0 for every k up to KMAX and the
dimensions grow; finite when a layer vanishes (then all the higher ones do).

Output: printed to stdout.
"""

import sys
from fractions import Fraction as F

import os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import nullspace                              # noqa: E402

from wgrad import VF, monomials                             # noqa: E402
from run_prolong import prolong, deg_k_basis, le_layers     # noqa: E402

LOG = None
KMAX = 3


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    if LOG:
        LOG.write(s + "\n")
        LOG.flush()


def euler_field(ne, no):
    t = {}
    for j in range(ne + no):
        a = [0] * ne
        S = ()
        if j < ne:
            a[j] = 1
        else:
            S = (j - ne,)
        t[((tuple(a), S), j)] = F(1)
    return VF(t, ne, no)


def preserving(ne, no, om):
    """{ X in gl(ne|no) : om(Xu,v) + (-1)^{p(X)p(u)} om(u,Xv) = 0 } as vector fields."""
    N = ne + no
    par = [0] * ne + [1] * no
    units = [(k, l) for k in range(N) for l in range(N)]
    rows = []
    for a in range(N):
        for b in range(N):
            row = []
            for (k, l) in units:
                pX = (par[k] + par[l]) % 2
                v1 = om(k, b) if l == a else F(0)
                v2 = om(a, k) if l == b else F(0)
                row.append(v1 + F(-1) ** (pX * par[a]) * v2)
            rows.append(row)
    out = []
    for vec in nullspace(rows):
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


def form_even(ne, no, sym_even=+1):
    """even non-degenerate form: symmetric on the even part (o), antisymmetric
    on the odd part (sp).  sym_even = -1 gives sp on the even part instead."""
    assert no % 2 == 0

    def om(a, b):
        if a < ne and b < ne:
            return F(1) if a == b and sym_even == 1 else (
                F(0) if sym_even == 1 else
                (F(1) if (b == a + ne // 2) else F(-1) if (a == b + ne // 2) else F(0)))
        if a >= ne and b >= ne:
            i, j = a - ne, b - ne
            k = no // 2
            if j == i + k:
                return F(1)
            if i == j + k:
                return F(-1)
            return F(0)
        return F(0)
    return om


def form_odd(ne, no, eps):
    assert ne == no

    def om(a, b):
        if a < ne and b >= ne:
            return F(1) if b - ne == a else F(0)
        if a >= ne and b < ne:
            return F(eps) if a - ne == b else F(0)
        return F(0)
    return om


def sl_fields(ne, no):
    """sl(ne|no) as linear vector fields: supertrace zero."""
    N = ne + no
    out = []
    for k in range(N):
        for l in range(N):
            if k == l:
                continue
            a = [0] * ne
            S = ()
            if l < ne:
                a[l] = 1
            else:
                S = (l - ne,)
            out.append(VF({((tuple(a), S), k): F(1)}, ne, no))
    # the diagonal part: sum c_k u_k d_k with sum (-1)^{p(k)} c_k = 0
    diag = []
    for k in range(1, N):
        t = {}
        for j, c in ((0, F(1) if k < ne or True else F(1)), (k, None)):
            pass
        s0 = F(1)
        sk = F(-1) * (F(-1) ** ((0 if 0 < ne else 1) + (0 if k < ne else 1)))
        for j, c in ((0, s0), (k, sk)):
            a = [0] * ne
            S = ()
            if j < ne:
                a[j] = 1
            else:
                S = (j - ne,)
            t[((tuple(a), S), j)] = c
        diag.append(VF(t, ne, no))
    return out + diag


def q_fields(n):
    """q(n) < gl(n|n): matrices (A B / B A), as linear vector fields."""
    ne = no = n
    out = []
    for i in range(n):
        for j in range(n):
            # A-part: e_ij on the even block and on the odd block
            t = {}
            a = [0] * ne
            a[j] = 1
            t[((tuple(a), ()), i)] = F(1)
            t[(((0,) * ne, (j,)), ne + i)] = F(1)
            out.append(VF(t, ne, no))
            # B-part: the odd one
            t = {}
            a = [0] * ne
            a[j] = 1
            t[((tuple(a), ()), ne + i)] = F(1)
            t[(((0,) * ne, (j,)), i)] = F(1)
            out.append(VF(t, ne, no))
    return out


def verdict(dims):
    if any(dims[k] == 0 for k in range(1, KMAX + 1)):
        first = min(k for k in range(1, KMAX + 1) if dims[k] == 0)
        return f"FINITE (g_{first} = 0)"
    return "infinite (still growing)"


def run(label, ne, no, u0, withE=True):
    for tag, uu in ((label, u0),
                    ("c(" + label + ")", u0 + [euler_field(ne, no)]) if withE else (None, None)):
        if tag is None:
            continue
        layers, dims = prolong(uu, ne, no, KMAX)
        say(f"  {tag:<16} dim g_0..g_{KMAX} = "
            + ", ".join(str(dims[k]) for k in range(KMAX + 1))
            + f"   -> {verdict(dims)}")


def main():
    say("=" * 74)
    say("V = C^4 (the Lie case: Cartan's theorem says gl, sl, sp, csp)")
    say("=" * 74)
    ne, no = 4, 0
    run("gl(4)", ne, no, deg_k_basis(ne, no, 0), withE=False)
    run("sl(4)", ne, no, sl_fields(ne, no))
    run("o(4)", ne, no, preserving(ne, no, form_even(ne, no, +1)))
    run("sp(4)", ne, no, preserving(ne, no, form_even(ne, no, -1)))

    say("\n" + "=" * 74)
    say("V = C^{2|2}")
    say("=" * 74)
    ne, no = 2, 2
    run("gl(2|2)", ne, no, deg_k_basis(ne, no, 0), withE=False)
    run("sl(2|2)", ne, no, sl_fields(ne, no))
    run("osp(2|2)", ne, no, preserving(ne, no, form_even(ne, no, +1)))
    run("pe^{sy}(2)", ne, no, preserving(ne, no, form_odd(ne, no, +1)))
    run("pe^{a}(2)", ne, no, preserving(ne, no, form_odd(ne, no, -1)))
    run("q(2)", ne, no, q_fields(2))

    say("\n" + "=" * 74)
    say("V = C^{3|3}")
    say("=" * 74)
    ne, no = 3, 3
    run("pe^{sy}(3)", ne, no, preserving(ne, no, form_odd(ne, no, +1)))
    run("pe^{a}(3)", ne, no, preserving(ne, no, form_odd(ne, no, -1)))
    run("q(3)", ne, no, q_fields(3))

    say("\nDone.")


if __name__ == "__main__":
    main()
