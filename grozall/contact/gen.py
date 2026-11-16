"""Bilinear k(3)-invariant differential operators  T(k1,l1) x T(k2,l2) -> T(k3,l3).

An operator invariant under k_{-2}+k_{-1} has constant coefficients in the invariant
frame e_1,e_2,R (checked in test_contact.py and again here), so it is a combination of

    u_1^c u_2^(k3-c) . (D_1 F^{(m)}) . (D_2 G^{(n)}),   D = e_1^i e_2^j R^s,

F^{(m)} being the coefficient of u_1^m u_2^(k1-m) in F.  Invariance under the grading
element z = K_{2t-pq} makes the WEIGHTED order

    W := (i_1+j_1) + (i_2+j_2) + 2(s_1+s_2)

constant on an operator, and ties the target weight to the source ones:

    lam3 = lam1 + lam2 + (W + k1 + k2 - k3)/2.

So for fixed types and fixed W the ansatz is finite-dimensional, and W (not the classical
order) is the invariant notion of order here.  What is left to impose is k_0 and k_1;
k_{<=1} generates k(3) (checked), so that is the whole condition.
"""

import os, sys
from fractions import Fraction as Fr
from itertools import product

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import nullspace, rank  # noqa: E402

from contact import (  # noqa: E402
    Module, madd, mscal, component, from_components, pmul, padd, pscal,
    pmono, pconst, e1, e2, reeb, k_piece,
)

K0 = [pmono(1, 0, 0), pmono(0, 2, 0), pmono(0, 1, 1), pmono(0, 0, 2)]
K1 = k_piece(1)
K2 = k_piece(2)
K3 = k_piece(3)
KMINUS = [pconst(1), pmono(0, 1, 0), pmono(0, 0, 1)]


def apply_D(op, poly):
    """op = (i,j,s):  e_1^i e_2^j R^s applied to a polynomial (R first, then e_2, e_1)."""
    i, j, s = op
    out = poly
    for _ in range(s):
        out = reeb(out)
        if not out:
            return {}
    for _ in range(j):
        out = e2(out)
        if not out:
            return {}
    for _ in range(i):
        out = e1(out)
        if not out:
            return {}
    return out


def d_ops(W):
    """All pairs (D_1,D_2) of weighted order summing to W."""
    out = []
    for s1 in range(W // 2 + 1):
        for s2 in range(W // 2 + 1 - s1):
            E = W - 2 * (s1 + s2)
            for i1 in range(E + 1):
                for j1 in range(E + 1 - i1):
                    for i2 in range(E + 1 - i1 - j1):
                        j2 = E - i1 - j1 - i2
                        out.append(((i1, j1, s1), (i2, j2, s2)))
    return out


def term_basis(k1, k2, k3, W):
    out = []
    for D1, D2 in d_ops(W):
        for m in range(k1 + 1):
            for n in range(k2 + 1):
                for c in range(k3 + 1):
                    out.append((m, n, D1, D2, c))
    return out


def apply_term(term, F, G):
    m, n, D1, D2, c = term
    a = apply_D(D1, component(F, m))
    if not a:
        return {}
    b = apply_D(D2, component(G, n))
    if not b:
        return {}
    return from_components({c: pmul(a, b)})


def target_weight(k1, l1, k2, l2, k3, W):
    num = W + k1 + k2 - k3
    return Fr(l1) + Fr(l2) + Fr(num, 2)


def defect(term, f, F, G, M1, M2, M3, LfF=None, LfG=None):
    """L_f(T(F,G)) - T(L_f F, G) - T(F, L_f G) for one basis term."""
    if LfF is None:
        LfF = M1.act(f, F)
    if LfG is None:
        LfG = M2.act(f, G)
    out = M3.act(f, apply_term(term, F, G))
    out = madd(out, mscal(-1, apply_term(term, LfF, G)))
    out = madd(out, mscal(-1, apply_term(term, F, LfG)))
    return out


def rows_for(terms, f, F, G, M1, M2, M3):
    """The linear conditions on the coefficient vector coming from one (f,F,G)."""
    LfF = M1.act(f, F)
    LfG = M2.act(f, G)
    cols = []
    keys = set()
    for term in terms:
        d = defect(term, f, F, G, M1, M2, M3, LfF, LfG)
        cols.append(d)
        keys |= set(d)
    if not keys:
        return []
    keys = sorted(keys)
    idx = {k: i for i, k in enumerate(keys)}
    mat = [[Fr(0)] * len(terms) for _ in keys]
    for jj, d in enumerate(cols):
        for k, v in d.items():
            mat[idx[k]][jj] = v
    return mat


def monomial_tests(k, maxdeg):
    out = []
    for a in range(maxdeg + 1):
        for b in range(maxdeg + 1 - a):
            for c in range(maxdeg + 1 - a - b):
                for i in range(k + 1):
                    out.append({(a, b, c, i): Fr(1)})
    return out


def rand_field(k, maxdeg, seed=0, nmon=30):
    """A random element of T(k,*) of (t,p,q)-degree at most maxdeg, nmon monomials."""
    import random
    rnd = random.Random(seed)
    keys = [(a, b, c, i)
            for a in range(maxdeg + 1)
            for b in range(maxdeg + 1 - a)
            for c in range(maxdeg + 1 - a - b)
            for i in range(k + 1)]
    if len(keys) > nmon:
        keys = rnd.sample(keys, nmon)
    return {key: Fr(rnd.randint(1, 9)) for key in keys}


def solve(k1, l1, k2, l2, k3, W, deg=None, gens=None, verify_deg=None,
          verify_gens=None, nverify=2, nsolve=2, quiet=True):
    """Return (lam3, list of coefficient vectors spanning the invariant operators)."""
    l3 = target_weight(k1, l1, k2, l2, k3, W)
    M1, M2, M3 = Module(k1, l1), Module(k2, l2), Module(k3, l3)
    terms = term_basis(k1, k2, k3, W)
    if deg is None:
        deg = W + 1
    if gens is None:
        gens = K0 + K1
    if verify_deg is None:
        verify_deg = deg + 2
    if verify_gens is None:
        verify_gens = K0 + K1 + K2 + K3

    def kernel(mat):
        if not mat:
            return [[Fr(1) if i == j else Fr(0) for i in range(len(terms))]
                    for j in range(len(terms))]
        return nullspace(mat)

    tests1 = [rand_field(k1, deg, seed=17 + i) for i in range(nsolve)]
    tests2 = [rand_field(k2, deg, seed=93 + i) for i in range(nsolve)]
    mat = []
    for f in gens:
        for F in tests1:
            for G in tests2:
                mat.extend(rows_for(terms, f, F, G, M1, M2, M3))
    ns = kernel(mat)

    # verification on independent, denser and higher-degree fields; if it fails, the
    # solve was under-determined -- the failing conditions are added and it is redone.
    for _ in range(5):
        bad = []
        v1 = [rand_field(k1, verify_deg, seed=1000 + i) for i in range(nverify)]
        v2 = [rand_field(k2, verify_deg, seed=2000 + i) for i in range(nverify)]
        for f in verify_gens:
            for F in v1:
                for G in v2:
                    LfF, LfG = M1.act(f, F), M2.act(f, G)
                    for vec in ns:
                        acc = {}
                        for cf, term in zip(vec, terms):
                            if cf:
                                acc = madd(acc, mscal(
                                    cf, defect(term, f, F, G, M1, M2, M3, LfF, LfG)))
                        if acc:
                            bad.append((f, F, G))
                            break
        if not bad:
            break
        if not quiet:
            print("    verification added %d conditions" % len(bad))
        for f, F, G in bad:
            mat.extend(rows_for(terms, f, F, G, M1, M2, M3))
        ns = kernel(mat)
    else:
        raise RuntimeError("verification did not converge")
    return l3, terms, ns


def nonzero_operators(terms, ns, k1, k2, k3, deg=4):
    """How many of the found operators are non-zero AS OPERATORS (not just as vectors)."""
    tests1 = monomial_tests(k1, deg)
    tests2 = monomial_tests(k2, deg)
    keep = []
    for vec in ns:
        alive = False
        for F in tests1:
            for G in tests2:
                acc = {}
                for cf, term in zip(vec, terms):
                    if cf:
                        acc = madd(acc, mscal(cf, apply_term(term, F, G)))
                if acc:
                    alive = True
                    break
            if alive:
                break
        if alive:
            keep.append(vec)
    return keep


def term_latex(term, k1, k2, k3):
    m, n, D1, D2, c = term

    def dl(D):
        i, j, s = D
        out = ""
        if i:
            out += "e_1^{%d}" % i if i > 1 else "e_1"
        if j:
            out += "e_2^{%d}" % j if j > 1 else "e_2"
        if s:
            out += "R^{%d}" % s if s > 1 else "R"
        return out or ""

    def uu(c, k):
        if k == 0:
            return ""
        parts = []
        if c:
            parts.append("u_1^{%d}" % c if c > 1 else "u_1")
        if k - c:
            parts.append("u_2^{%d}" % (k - c) if k - c > 1 else "u_2")
        return "".join(parts)

    f = "F" + ("_{%s}" % uu(m, k1) if k1 else "")
    g = "G" + ("_{%s}" % uu(n, k2) if k2 else "")
    return "%s(%s %s)(%s %s)" % (uu(c, k3), dl(D1), f, dl(D2), g)
