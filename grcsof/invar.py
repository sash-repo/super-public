#!/usr/bin/env python3
"""Classification of i(osp(2|2))-invariant unary and binary differential
operators between spaces of weighted densities over k(1|2), in the xi-eta
realisation.  Conventions: super12.py / k12.py (= those of [BLS1]).

Normalisation of the weights (fixed once and for all here):

    F_{lam,nu} := F . alpha_1^{lam/2} beta^{nu},
    L_{K_f}(phi) = K_f(phi) + ( lam d_t(f) + (-1)^{p(f)} nu div_2(f) ) phi.

This is the normalisation of [BLS1, 2.2.1a] and the one used in the paper
(with it, an operator of degree s shifts lam by -s; e.g. the order-k
Gordan-Rankin-Cohen bracket in sdim 1|1 has s=-2k and maps
F_{lam_1} x F_{lam_2} -> F_{lam_1+lam_2+2k}, as in the paper).

Grading: deg t = 2, deg xi = deg eta = 1; deg d_t = -2, deg D_xi = deg D_eta = -1.
K_xieta-weight ("charge"): wt(xi) = -1, wt(eta) = +1, wt(D_xi) = +1, wt(D_eta) = -1.
For a homogeneous operator of degree s and charge w the target weights are
    lam = lam_1 + lam_2 - s,   nu = nu_1 + nu_2 - w.
"""
from sympy import symbols, S, Matrix, expand, simplify, factor, Rational, zeros
from super12 import add, smul, mul, mono, parity, par_mon
from k12 import K, kb, D_xi, D_eta, div2, L, OSP, d_t

# ---------------------------------------------------------------- symbol table
DOPS = {                        # name -> (operator, #D's, charge)
    '1':    (lambda u: u, 0, 0),
    'x':    (D_xi, 1, +1),
    'e':    (D_eta, 1, -1),
    'xe':   (lambda u: D_xi(D_eta(u)), 2, 0),
}


def dtn(n):
    def r(u):
        for _ in range(n):
            u = d_t(u)
        return u
    return r


def opterm(name, n):
    """d_t^n D_name as a single operator, with its degree and charge."""
    op, nD, w = DOPS[name]
    f = dtn(n)
    return (lambda u: op(f(u))), -2 * n - nD, w, nD


def gen_unary(s, w):
    out = []
    for nm in DOPS:
        _, nD, wt = DOPS[nm]
        if wt != w:
            continue
        r = -s - nD
        if r >= 0 and r % 2 == 0:
            o, dg, ww, nd = opterm(nm, r // 2)
            out.append((f"d^{r//2}D{nm}", o, nd))
    return out


def gen_binary(s, w):
    """all pairs (P,Q) of total degree s and total charge w."""
    out = []
    for n1 in DOPS:
        for n2 in DOPS:
            _, nD1, w1 = DOPS[n1]
            _, nD2, w2 = DOPS[n2]
            if w1 + w2 != w:
                continue
            rest = -s - nD1 - nD2
            if rest < 0 or rest % 2:
                continue
            for a in range(rest // 2 + 1):
                b = rest // 2 - a
                o1, _, _, _ = opterm(n1, a)
                o2, _, _, _ = opterm(n2, b)
                out.append((f"(d^{a}D{n1})x(d^{b}D{n2})", o1, o2,
                            (nD1 + nD2) % 2, nD2 % 2))
    return out


def monos(nmax):
    return [(a, e, f) for a in range(nmax + 1) for e in (0, 1) for f in (0, 1)]


# ------------------------------------------------------------------- equations
def eqs_unary(terms, lam, nu, lam2, nu2, pB, nmax=6, names=None, fields=None):
    cs = symbols(f'c0:{len(terms)}')
    names = names or [t[0] for t in terms]
    eqs = set()
    for nm in (fields or OSP):
        f = OSP[nm]
        pf = parity(f)
        s = -1 if (pf and pB) else 1
        Lsrc, Ltgt = L(f, lam, nu), L(f, lam2, nu2)
        for m in monos(nmax):
            u = mono(*m)
            Bu = add(*[smul(cs[i], terms[i][1](u)) for i in range(len(terms))])
            BLu = add(*[smul(cs[i], terms[i][1](Lsrc(u))) for i in range(len(terms))])
            res = add(Ltgt(Bu), smul(-s, BLu))
            for v in res.values():
                e = expand(v)
                if e:
                    eqs.add(e)
    return cs, sorted(eqs, key=lambda e: str(e))


def eqs_binary(terms, l1, n1, l2, n2, lt, nt, pB, nmax=5, fields=None):
    """terms: list of (name, P, Q, parity_of_term, parity_of_Q); all of parity pB.

    The term is the honest tensor product P (x) Q, i.e. it carries the Koszul
    sign:   (P (x) Q)(phi (x) psi) = (-1)^{p(Q) p(phi)} P(phi) . Q(psi).
    (These are the factors (-1)^{p(varphi)} written out in [GO], [Om] and in
    the paper.)"""
    cs = symbols(f'c0:{len(terms)}')
    eqs = set()
    MS = monos(nmax)
    for nm in (fields or OSP):
        f = OSP[nm]
        pf = parity(f)
        sB = -1 if (pf and pB) else 1
        L1, L2, Lt = L(f, l1, n1), L(f, l2, n2), L(f, lt, nt)
        for m1 in MS:
            u = mono(*m1)
            pu = (m1[1] + m1[2]) % 2
            Lu = L1(u)
            for m2 in MS:
                v = mono(*m2)
                Lv = L2(v)
                def B(x, y, px):
                    return add(*[smul(cs[i] * (-1 if (terms[i][4] and px) else 1),
                                      mul(terms[i][1](x), terms[i][2](y)))
                                 for i in range(len(terms))])
                s2 = -1 if (pf and pu) else 1
                res = add(Lt(B(u, v, pu)),
                          smul(-sB, add(B(Lu, v, (pu + pf) % 2),
                                        smul(s2, B(u, Lv, pu)))))
                for val in res.values():
                    e = expand(val)
                    if e:
                        eqs.add(e)
    return cs, sorted(eqs, key=lambda e: str(e))


def matrix_of(eqs, cs):
    return Matrix([[expand(e).coeff(c) for c in cs] for e in eqs])
