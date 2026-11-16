"""
The rank-2 case: bilinear operators between weighted densities on S_c^{1|2},
invariant under lf(S_c^{1|2}) = osp(2|2) = sl(1|2).

This module is the exact-arithmetic engine for the rank-2 case sl(1|2).

Setting.  Coordinates (t | xi, eta), xi and eta odd and isotropic,
contact form  alpha = dt + xi d(eta) + eta d(xi)   (the isotropic basis of
pair2.py: xi = (th_1 + i th_2)/sqrt2, eta = (th_1 - i th_2)/sqrt2, in which
th_1 d th_1 + th_2 d th_2 = xi d eta + eta d xi).

Contact fields (eq. (6) of the text, transported to the isotropic basis):

  K_f = (2-E)(f) d_t
        - (-1)^{p(f)} ( d_xi(f) d_eta + d_eta(f) d_xi )
        + d_t(f) ( xi d_xi + eta d_eta ),          E := xi d_xi + eta d_eta.

Contact bracket:

  {f,g} = (2-E)(f) d_t(g) - d_t(f) (2-E)(g)
          - (-1)^{p(f)} ( d_xi(f) d_eta(g) + d_eta(f) d_xi(g) ).

Rank-1 modules  F_{l1,l2} = F alpha^{l1/2} [d xi]^{l2}  (eq. (35)):

  rho_{l1,l2}(f) phi = K_f(phi) + l1 d_t(f) phi + l2 c(f) phi,
  c(f) := d_t(f) - d_xi d_eta (f)          (the [d xi]-multiplier).

Everything below is verified, not assumed: the bracket homomorphism
[K_f,K_g] = K_{{f,g}} and the module homomorphism
[rho(f),rho(g)] = rho({f,g}) with SYMBOLIC l1, l2 are checked in main().

A monomial is the key (m, e, g) meaning  t^m xi^e eta^g, e,g in {0,1};
an element of the module is a dict {key: sympy coefficient}.
"""

import itertools
from fractions import Fraction

import sympy as sp

# Coefficient backend.  With symbolic weights every coefficient has to be
# expanded (sympy); with numeric ones Fractions are ~100 times faster, and
# expansion is a no-op.  set_numeric() switches.
_EX = sp.expand
_SY = sp.sympify


def set_numeric(flag=True):
    global _EX, _SY
    if flag:
        _EX = (lambda x: x)
        _SY = lambda c: c if isinstance(c, Fraction) else Fraction(c)
    else:
        _EX = sp.expand
        _SY = sp.sympify

# ------------------------------------------------------------------ algebra


def add(*fs):
    out = {}
    for f in fs:
        for k, v in f.items():
            out[k] = out.get(k, 0) + v
    return {k: _EX(v) for k, v in out.items() if _EX(v) != 0}


def smul(c, f):
    c = _SY(c)
    return {k: _EX(c * v) for k, v in f.items() if _EX(c * v) != 0}


def mul(f, g):
    """Product in C[t] (x) Lambda(xi,eta), monomials written t^m xi^e eta^g."""
    out = {}
    for (m1, e1, g1), c1 in f.items():
        for (m2, e2, g2), c2 in g.items():
            if e1 + e2 > 1 or g1 + g2 > 1:
                continue
            sgn = -1 if (g1 and e2) else 1           # move xi^e2 past eta^g1
            key = (m1 + m2, e1 + e2, g1 + g2)
            out[key] = out.get(key, 0) + sgn * c1 * c2
    return {k: _EX(v) for k, v in out.items() if _EX(v) != 0}


def d_t(f):
    out = {}
    for (m, e, g), c in f.items():
        if m:
            out[(m - 1, e, g)] = out.get((m - 1, e, g), 0) + m * c
    return {k: _EX(v) for k, v in out.items() if _EX(v) != 0}


def d_xi(f):
    """Left derivative: d_xi(xi eta^g) = eta^g."""
    out = {}
    for (m, e, g), c in f.items():
        if e:
            out[(m, 0, g)] = out.get((m, 0, g), 0) + c
    return {k: _EX(v) for k, v in out.items() if _EX(v) != 0}


def d_eta(f):
    """Left derivative: d_eta(xi^e eta) = (-1)^e xi^e."""
    out = {}
    for (m, e, g), c in f.items():
        if g:
            s = -1 if e else 1
            out[(m, e, 0)] = out.get((m, e, 0), 0) + s * c
    return {k: _EX(v) for k, v in out.items() if _EX(v) != 0}


def E_op(f):
    """The odd Euler operator xi d_xi + eta d_eta."""
    return {k: _EX((k[1] + k[2]) * v) for k, v in f.items()
            if _EX((k[1] + k[2]) * v) != 0}


def parity(f):
    ps = {(e + g) % 2 for (m, e, g) in f}
    if len(ps) > 1:
        raise ValueError("inhomogeneous parity")
    return ps.pop() if ps else 0


def deg(f):
    """Degree for deg t = 2, deg xi = deg eta = 1."""
    ds = {2 * m + e + g for (m, e, g) in f}
    if len(ds) > 1:
        raise ValueError("inhomogeneous degree")
    return ds.pop() if ds else 0


def charge(f):
    cs = {e - g for (m, e, g) in f}
    if len(cs) > 1:
        raise ValueError("inhomogeneous charge")
    return cs.pop() if cs else 0


def mono(m=0, e=0, g=0, c=1):
    return {(m, e, g): _SY(c)} if c else {}


# ------------------------------------------------------- contact structure


def two_minus_E(f):
    return add(smul(2, f), smul(-1, E_op(f)))


def K(f, phi):
    """The contact vector field K_f applied to the function phi."""
    s = 1 if parity(f) == 0 else -1
    out = mul(two_minus_E(f), d_t(phi))
    out = add(out, smul(-s, mul(d_xi(f), d_eta(phi))))
    out = add(out, smul(-s, mul(d_eta(f), d_xi(phi))))
    out = add(out, mul(d_t(f), E_op(phi)))
    return out


def kb(f, g):
    """The contact bracket {f,g}."""
    s = 1 if parity(f) == 0 else -1
    out = mul(two_minus_E(f), d_t(g))
    out = add(out, smul(-1, mul(d_t(f), two_minus_E(g))))
    out = add(out, smul(-s, mul(d_xi(f), d_eta(g))))
    out = add(out, smul(-s, mul(d_eta(f), d_xi(g))))
    return out


def c2(f):
    """The second 1-cocycle of k(1|2) with values in F (the one that is not
    the contact multiplier).  Found by solving the cocycle condition inside
    the 6-dimensional space of operators of degree -2 and charge 0
    (computed separately): the solution space is 2-dimensional, spanned by d_t and

        c2(f) = d_eta d_xi (f) + eta d_eta d_t (f) - xi d_xi d_t (f)
                + xi eta d_t^2 (f).

    Normalization: c2(t) = 0 and c2(xi eta) = 1."""
    out = d_eta(d_xi(f))
    out = add(out, mul(mono(0, 0, 1), d_eta(d_t(f))))
    out = add(out, smul(-1, mul(mono(0, 1, 0), d_xi(d_t(f)))))
    out = add(out, mul(mono(0, 1, 1), d_t(d_t(f))))
    return out


def c_xi(f):
    """The [d xi]-multiplier:  L_{K_f}[d xi] = c_xi(f) [d xi].

    Fixed by c_xi(t) = 1 (the degree of [d xi] is 1) and c_xi(xi eta) = 1
    (its charge is +1); then c_eta = d_t - c2 and c_xi + c_eta = 2 d_t,
    which is eq. (124): mu_G nu_G = m_G."""
    return add(d_t(f), c2(f))


def rho(l1, l2, f, phi):
    """The action of K_f on F_{l1,l2}."""
    out = K(f, phi)
    out = add(out, smul(l1, mul(d_t(f), phi)))
    out = add(out, smul(l2, mul(c_xi(f), phi)))
    return out


# --------------------------------------------------------- the algebra lf


def lf_basis():
    """The 8 generating functions of lf(S_c^{1|2}) = osp(2|2), with names."""
    return [
        ("1",     mono(0, 0, 0)),
        ("xi",    mono(0, 1, 0)),
        ("eta",   mono(0, 0, 1)),
        ("t",     mono(1, 0, 0)),
        ("xieta", mono(0, 1, 1)),
        ("t xi",  mono(1, 1, 0)),
        ("t eta", mono(1, 0, 1)),
        ("t^2",   mono(2, 0, 0)),
    ]


ODD_GENS = ["xi", "eta", "t xi", "t eta"]


# ------------------------------------------------------------ the operators


def ansatz_terms(N, q, maxord=None):
    """All terms  xi^s eta^u d_t^m d_xi^p1 d_eta^r1 (x) d_t^m' d_xi^p2 d_eta^r2
    of total degree drop N and charge shift q.

    Returned as tuples (s, u, m1, p1, r1, m2, p2, r2).
    """
    out = []
    for s, u in itertools.product((0, 1), repeat=2):
        for p1, r1, p2, r2 in itertools.product((0, 1), repeat=4):
            if s - u - p1 - p2 + r1 + r2 != q:
                continue
            rest = N + s + u - (p1 + r1 + p2 + r2)
            if rest < 0 or rest % 2:
                continue
            half = rest // 2
            for m1 in range(half + 1):
                m2 = half - m1
                if maxord is not None and (m1 > maxord or m2 > maxord):
                    continue
                out.append((s, u, m1, p1, r1, m2, p2, r2))
    return out


def apply_term(term, phi, psi):
    """Evaluate one ansatz term on the monomials phi, psi."""
    s, u, m1, p1, r1, m2, p2, r2 = term
    a = phi
    for _ in range(m1):
        a = d_t(a)
    if p1:
        a = d_xi(a)
    if r1:
        a = d_eta(a)
    b = psi
    for _ in range(m2):
        b = d_t(b)
    if p2:
        b = d_xi(b)
    if r2:
        b = d_eta(b)
    out = mul(a, b)
    if u:
        out = mul(mono(0, 0, 1), out)
    if s:
        out = mul(mono(0, 1, 0), out)
    return out


def apply_B(terms, coeffs, phi, psi):
    out = {}
    for term, c in zip(terms, coeffs):
        if c == 0:
            continue
        out = add(out, smul(c, apply_term(term, phi, psi)))
    return out


# --------------------------------------------------- the invariance system


def basis_monomials(dmax):
    """All monomials of degree <= dmax."""
    out = []
    for m in range(dmax // 2 + 1):
        for e, g in itertools.product((0, 1), repeat=2):
            if 2 * m + e + g <= dmax:
                out.append(mono(m, e, g))
    return out


def invariance_matrix(N, q, wa, wb, dmax, gens=None, maxord=None):
    """Rows of the linear system on the coefficients of B, columns = terms.

    wa = (a1, a2), wb = (b1, b2) are the weights of the two arguments;
    the target weight is forced:  c1 = a1 + b1 + N + q,  c2 = a2 + b2 - q.
    """
    terms = ansatz_terms(N, q, maxord=maxord)
    if not terms:
        return terms, []
    a1, a2 = wa
    b1, b2 = wb
    c1, c2 = a1 + b1 + N + q, a2 + b2 - q
    gens = gens or ODD_GENS
    lf = dict(lf_basis())
    rows = {}
    for gname in gens:
        f = lf[gname]
        pf = parity(f)
        for phi in basis_monomials(dmax):
            pphi = parity(phi)
            for psi in basis_monomials(dmax):
                for j, term in enumerate(terms):
                    lhs = rho(c1, c2, f, apply_term(term, phi, psi))
                    r1_ = apply_term(term, rho(a1, a2, f, phi), psi)
                    sgn = -1 if (pf and pphi) else 1
                    r2_ = apply_term(term, phi, rho(b1, b2, f, psi))
                    res = add(lhs, smul(-1, r1_), smul(-sgn, r2_))
                    for key, val in res.items():
                        rows.setdefault((gname, tuple(sorted(phi)),
                                         tuple(sorted(psi)), key),
                                        [0] * len(terms))[j] += val
    mat = [[_EX(x) for x in row] for row in rows.values()]
    mat = [row for row in mat if any(x != 0 for x in row)]
    return terms, mat


def nullspace(mat, ncols):
    if not mat:
        return sp.eye(ncols).columnspace()
    M = sp.Matrix(mat)
    return M.nullspace()


def numeric_dim(mat, ncols):
    """Nullity over Q of an exact rational matrix, via ../lib/exactlin.py."""
    if not mat:
        return ncols
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "lib"))
    from exactlin import rank
    rows = [[Fraction(x) for x in row] for row in mat]
    return ncols - rank(rows)


def hom_dim(N, q, wa, wb, dmax=6, maxord=None, want_vectors=False,
            numeric=False):
    terms, mat = invariance_matrix(N, q, wa, wb, dmax, maxord=maxord)
    if not terms:
        return (0, terms, []) if want_vectors else 0
    if numeric and not want_vectors:
        return numeric_dim(mat, len(terms))
    ns = nullspace(mat, len(terms))
    return (len(ns), terms, ns) if want_vectors else len(ns)


# ------------------------------------------------------------------ checks


def check_bracket(verbose=True):
    """[K_f, K_g] = K_{{f,g}} on a spanning set."""
    fs = [mono(m, e, g) for m in range(3)
          for e in (0, 1) for g in (0, 1)]
    tests = [mono(m, e, g) for m in range(3) for e in (0, 1) for g in (0, 1)]
    bad = 0
    for f in fs:
        for g in fs:
            sgn = -1 if (parity(f) and parity(g)) else 1
            for phi in tests:
                lhs = add(K(f, K(g, phi)), smul(-sgn, K(g, K(f, phi))))
                rhs = K(kb(f, g), phi)
                if add(lhs, smul(-1, rhs)):
                    bad += 1
    if verbose:
        print("check_bracket: %s" % ("OK" if bad == 0 else "FAILED %d" % bad))
    return bad == 0


def check_module(verbose=True):
    """[rho(f),rho(g)] = rho({f,g}) with symbolic weights."""
    l1, l2 = sp.symbols('l1 l2')
    fs = [mono(m, e, g) for m in range(3)
          for e in (0, 1) for g in (0, 1)]
    tests = [mono(m, e, g) for m in range(3) for e in (0, 1) for g in (0, 1)]
    bad = 0
    for f in fs:
        for g in fs:
            sgn = -1 if (parity(f) and parity(g)) else 1
            for phi in tests:
                lhs = add(rho(l1, l2, f, rho(l1, l2, g, phi)),
                          smul(-sgn, rho(l1, l2, g, rho(l1, l2, f, phi))))
                rhs = rho(l1, l2, kb(f, g), phi)
                if add(lhs, smul(-1, rhs)):
                    bad += 1
    if verbose:
        print("check_module: %s" % ("OK" if bad == 0 else "FAILED %d" % bad))
    return bad == 0


def check_cartan(verbose=True):
    """The two Cartan elements act by  deg + l1 + l2  and  charge + l2."""
    l1, l2 = sp.symbols('l1 l2')
    bad = 0
    for m in range(3):
        for e, g in itertools.product((0, 1), repeat=2):
            phi = mono(m, e, g)
            d, ch = deg(phi), charge(phi)
            r1 = rho(l1, l2, mono(1, 0, 0), phi)
            if add(r1, smul(-(d + l1 + l2), phi)):
                bad += 1
            r2 = rho(l1, l2, mono(0, 1, 1), phi)
            if add(r2, smul(-(ch + l2), phi)):
                bad += 1
    if verbose:
        print("check_cartan: %s" % ("OK" if bad == 0 else "FAILED %d" % bad))
    return bad == 0


def main():
    ok = True
    ok &= check_bracket()
    ok &= check_module()
    ok &= check_cartan()
    print("all structure checks:", "OK" if ok else "FAILED")


if __name__ == "__main__":
    main()
