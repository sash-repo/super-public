#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
antilin.py -- real forms of svect^L_lambda(1|2):  for which lambda does the
algebra admit an anti-linear involutive automorphism?

This concerns item 2) of Theorem 5.5 (p. 315) of Berezin's book,
2nd ed. (Appendix D5, "Automorphisms and real forms", after V. Serganova):

    "2) The Lie superalgebras of the series svect^L_lambda, svect^{L'}_lambda
        with lambda not in R have no real forms."

This script checks that statement in exact arithmetic.  Everything is built on
sv2.py (the algebra) and re-uses nothing but its calculus.

What is computed.

 1. The transformation law of the substitution
        psi_b:  t |-> t^{-1},   theta_i |-> t^{b_i} theta_i     (b_i in Z)
    on the family: psi_b is an involution of the Laurent superalgebra R(1|2)
    and psi_b(svect^L_mu) = svect^L_{-mu-2-B}, where B = b_1 + b_2.  (For
    b = 0 this is the inversion t |-> 1/t; a pure twist, i.e. the identity
    on t, gives mu |-> mu - B, also re-checked here.)

 2. Complex conjugation C of the coefficients is an anti-linear isomorphism
    svect^L_lambda -> svect^L_{lambdabar}.  Hence

        Phi := psi_b o C  is an anti-linear automorphism of svect^L_lambda
        exactly when  -lambdabar - 2 - B = lambda,  i.e.  2 Re lambda = -2-B,

    and Phi^2 = psi_b^2 o C^2 = id because psi_b is an involution defined over
    Q.  So EVERY lambda with 2 Re lambda in Z -- in particular the non-real
    ones -- carries a real structure.  The script verifies, for numerical
    lambda with Im lambda = 1, that Phi
        (a) maps svect^L_lambda to itself,
        (b) is an automorphism of the bracket,
        (c) is anti-linear,
        (d) is involutive,
        (e) has a fixed-point set of the right real dimension, closed under
            the bracket -- i.e. a genuine real form.

 3. The converse: an anti-linear automorphism forces svect^L_lambda to be
    isomorphic to its conjugate svect^L_{lambdabar}, and the isomorphism
    classification of the family (lambda ~ lambda+1, lambda ~ -lambda-2; see
    item 8.8 of Appendix D3 and Subsection 4.2.12 of the paper) then forces
    Im lambda = 0 or 2 Re lambda in Z.  Machine illustration: for
    lambda = 1/3 + i no monomial substitution of either type works.

Conclusion:  svect^L_lambda(1|2) has a real form  <=>  Im lambda = 0 or
2 Re lambda in Z.  Item 2) as printed is too strong.

Run:  ../venv/bin/python antilin.py   (plain python3 works too)
"""

import io
import contextlib
import sys

from sympy import (I, Rational, Symbol, expand, simplify, conjugate, re, im,
                   nsimplify, Matrix, solve)

import sv2
from sv2 import (VF, mono, fadd, fscal, fclean, bracket, div_lam, lam,
                 TH, COMPS, L, J, G, Gbar)

FAIL = []
CHECKS = [0]


def check(cond, msg):
    CHECKS[0] += 1
    if not cond:
        FAIL.append(msg)
        print("  FAIL: " + msg)


# --------------------------------------------------------------- utilities

def subs_vf(X, val):
    """substitute a numerical value for lambda in every coefficient."""
    return VF({a: fclean({k: expand(c.subs(lam, val)) if hasattr(c, 'subs')
                          else c for k, c in X.c[a].items()})
               for a in COMPS}, X.p)


def conj_vf(X):
    """C: complex conjugation of the coefficients (anti-linear)."""
    return VF({a: fclean({k: expand(conjugate(c)) for k, c in X.c[a].items()})
               for a in COMPS}, X.p)


def scal_vf(c, X):
    return VF({a: fclean({k: expand(c * v) for k, v in X.c[a].items()})
               for a in COMPS}, X.p)


def sub_vf(X, Y):
    return VF({a: fadd(X.c[a], fscal(-1, Y.c[a])) for a in COMPS}, X.p)


def is_zero_vf(X):
    return all(not fclean(X.c[a]) for a in COMPS)


# ------------------------------------------------- the substitution psi_b

def psi_f(b, f):
    """psi_b on functions:  t^m theta_S |-> t^{-m + sum_{i in S} b_i} theta_S."""
    out = {}
    for (m, S), c in f.items():
        k = (expand(-m + sum(b[i] for i in S)), S)
        out[k] = out.get(k, 0) + c
    return fclean(out)


def tw_f(b, f, inv=False):
    """the pure twist  theta_i |-> t^{b_i} theta_i  (inv: its inverse)."""
    out = {}
    for (m, S), c in f.items():
        sh = sum(b[i] for i in S)
        k = (expand(m + (-sh if inv else sh)), S)
        out[k] = out.get(k, 0) + c
    return fclean(out)


def psi_push(b, X):
    """(psi_b)_* X = psi_b o X o psi_b^{-1};  psi_b^{-1} = psi_b."""
    comps = {'t': psi_f(b, X.act(mono(-1)))}
    for i in TH:
        comps[i] = psi_f(b, X.act(mono(b[i], (i,))))
    return VF(comps, X.p)


def tw_push(b, X):
    comps = {'t': tw_f(b, X.act(mono(1)))}
    for i in TH:
        comps[i] = tw_f(b, X.act(tw_f(b, mono(0, (i,)), inv=True)))
    return VF(comps, X.p)


def mu_of(X, sign):
    """the unique mu with div_mu(X) = 0, or None (or 'any')."""
    mu = Symbol('mu_')
    eqs = [expand(v) for v in div_lam(X, sign, l=mu).values()]
    eqs = [e for e in eqs if e != 0]
    if not eqs:
        return 'any'
    sol = solve(eqs, mu, dict=True)
    if not sol:
        return None
    return simplify(sol[0][mu])


def basis(rng=(-2, -1, 0, 1, 2)):
    """a window of the basis of svect^L_lambda(1|2), lambda formal."""
    out = []
    for n in rng:
        out.append(('L_%d' % n, L(n)))
        for x in ('+', '-', '0'):
            out.append(('J%s_%d' % (x, n), J(x, n)))
    for k in rng:
        r = Rational(2 * k + 1, 2)
        for al in TH:
            out.append(('G%d^%s' % (al, r), G(al, r)))
            out.append(('Gb^%s%d' % (r, al), Gbar(al, r)))
    return out


# ============================================ 1. the law of the substitution

def sec1(sign):
    print("\n1. psi_b: t |-> 1/t, theta_i |-> t^{b_i} theta_i")
    B0 = basis((-1, 0, 1))

    # psi_b is an involution of R(1|2)
    bad = 0
    for b in ({1: 0, 2: 0}, {1: -1, 2: -1}, {1: -2, 2: -1}, {1: 3, 2: -5}):
        for m in (-2, 0, 3):
            for S in ((), (1,), (2,), (1, 2)):
                f = mono(m, S)
                if psi_f(b, psi_f(b, f)) != f:
                    bad += 1
    check(bad == 0, "psi_b is not an involution on functions (%d)" % bad)
    print("   psi_b o psi_b = id on R(1|2): %d failures" % bad)

    # the law lambda |-> -lambda - 2 - B
    for b in ({1: 0, 2: 0}, {1: -1, 2: -1}, {1: -2, 2: -1}, {1: -3, 2: 0},
              {1: 2, 2: 1}):
        B = b[1] + b[2]
        mus = set()
        for nm, X in B0:
            Y = psi_push(b, X)
            m = mu_of(Y, sign)
            if m == 'any':
                continue
            mus.add(simplify(m))
        check(len(mus) == 1, "psi_%s: inconsistent mu %s" % (b, mus))
        mu = mus.pop()
        want = -lam - 2 - B
        check(simplify(mu - want) == 0,
              "psi_%s gives mu = %s, expected %s" % (b, mu, want))
        print("   b = (%2d,%2d), B = %2d:  svect_lambda -> svect_{%s}"
              % (b[1], b[2], B, mu))

    # the pure twist, for the record
    for b in ({1: 1, 2: 0}, {1: 1, 2: 1}, {1: 2, 2: -1}):
        B = b[1] + b[2]
        mus = {simplify(mu_of(tw_push(b, X), sign)) for nm, X in B0
               if mu_of(tw_push(b, X), sign) != 'any'}
        check(len(mus) == 1, "twist %s: inconsistent mu %s" % (b, mus))
        mu = mus.pop()
        check(simplify(mu - (lam - B)) == 0,
              "twist %s gives mu = %s" % (b, mu))
        print("   twist b = (%2d,%2d):    svect_lambda -> svect_{%s}"
              % (b[1], b[2], mu))

    # psi_b preserves the bracket (it is a substitution, but check anyway)
    bad = 0
    b = {1: -2, 2: -1}
    for na, X in B0:
        for nb, Y in B0:
            Z = psi_push(b, bracket(X, Y))
            W = bracket(psi_push(b, X), psi_push(b, Y))
            if not is_zero_vf(sub_vf(Z, W)):
                bad += 1
    check(bad == 0, "psi_b is not a Lie automorphism (%d)" % bad)
    print("   psi_b[X,Y] = [psi_b X, psi_b Y] on %d pairs: %d failures"
          % (len(B0) ** 2, bad))


# ================================================ 2. the real structure Phi

def real_structure(val, b, sign, rng=(-2, -1, 0, 1, 2), verbose=True):
    """Phi = psi_b o C at lambda = val; returns the list of checks."""
    B0 = [(nm, subs_vf(X, val)) for nm, X in basis(rng)]

    def Phi(X):
        return psi_push(b, conj_vf(X))

    # (a) Phi preserves svect^L_lambda
    bad = 0
    for nm, X in B0:
        Y = Phi(X)
        d = div_lam(Y, sign, l=val)
        if any(expand(v) != 0 for v in d.values()):
            bad += 1
    check(bad == 0, "lambda = %s: Phi leaves svect (%d)" % (val, bad))

    # (b) automorphism of the bracket
    badb = 0
    for na, X in B0:
        for nb, Y in B0:
            if not is_zero_vf(sub_vf(Phi(bracket(X, Y)),
                                     bracket(Phi(X), Phi(Y)))):
                badb += 1
    check(badb == 0, "lambda = %s: Phi is not a Lie automorphism (%d)"
          % (val, badb))

    # (c) anti-linearity
    badc = 0
    for nm, X in B0:
        if not is_zero_vf(sub_vf(Phi(scal_vf(I, X)), scal_vf(-I, Phi(X)))):
            badc += 1
    check(badc == 0, "lambda = %s: Phi is not anti-linear (%d)" % (val, badc))

    # (d) involutive
    badd = 0
    for nm, X in B0:
        if not is_zero_vf(sub_vf(Phi(Phi(X)), X)):
            badd += 1
    check(badd == 0, "lambda = %s: Phi^2 != id (%d)" % (val, badd))

    if verbose:
        print("   lambda = %-10s b = (%d,%d): preserved %s, bracket %s, "
              "anti-linear %s, Phi^2 = id %s"
              % (val, b[1], b[2], bad == 0, badb == 0, badc == 0, badd == 0))
    return Phi, B0


def coords(X, keys):
    return [X.c[a].get(k, 0) for (a, k) in keys]


def all_keys(fields):
    ks = []
    for X in fields:
        for a in COMPS:
            for k in X.c[a]:
                if (a, k) not in ks:
                    ks.append((a, k))
    return ks


def fixed_dimension(Phi, fields):
    """dim_R of the Phi-fixed points of the SMALLEST Phi-stable space
    containing the given fields, and its dim_C.

    The span is made Phi-stable by adjoining Phi(fields) (Phi^2 = id, so no
    further step is needed); then Phi is written as a matrix M with
    Phi(sum c_k e_k) = sum (M conj(c))_j e_j in a C-basis (e_k), M conj(M) = 1
    is verified, and the fixed points are the real kernel of

        [[A - 1, B], [B, -A - 1]],   M = A + iB,   c = x + iy.
    """
    fam = list(fields) + [Phi(X) for X in fields]
    keys = all_keys(fam)
    cols = [Matrix([expand(c) for c in coords(X, keys)]) for X in fam]
    # a C-basis of the span, greedily
    bas, basc = [], []
    M0 = Matrix.zeros(len(keys), 0)
    for X, col in zip(fam, cols):
        M1 = M0.row_join(col)
        if M1.rank() > M0.rank():
            M0, bas, basc = M1, bas + [X], basc + [col]
    n = len(bas)
    # the matrix of Phi
    Mcols = []
    for X in bas:
        rhs = Matrix([expand(c) for c in coords(Phi(X), keys)])
        sol = M0.solve_least_squares(rhs) if M0.rows != M0.cols else M0.solve(rhs)
        Mcols.append([simplify(s) for s in sol])
    M = Matrix(Mcols).T
    check(simplify(M * M.conjugate() - Matrix.eye(n)) == Matrix.zeros(n, n),
          "M conj(M) != 1 (%s)" % (M * M.conjugate()))
    A = Matrix(n, n, lambda i, j: expand(re(M[i, j])))
    B = Matrix(n, n, lambda i, j: expand(im(M[i, j])))
    E = Matrix.eye(n)
    R = (A - E).row_join(B).col_join(B.row_join(-A - E))
    return n, 2 * n - R.rank()


def sec2(sign):
    print("\n2. Phi = psi_b o C is a real structure whenever 2 Re lambda in Z")
    cases = [(I, {1: -1, 2: -1}),                       # Re = 0,   B = -2
             (Rational(1, 2) + I, {1: -2, 2: -1}),      # Re = 1/2, B = -3
             (-1 + 2 * I, {1: 0, 2: 0}),                # Re = -1,  B = 0
             (Rational(3, 2) + I / 3, {1: -3, 2: -2}),  # Re = 3/2, B = -5
             (Rational(1, 3), {1: -1, 2: -1}) ]         # real lambda: B = -2-2/3?
    for val, b in cases[:-1]:
        Phi, B0 = real_structure(val, b, sign)
    # the real case is covered by b = 0 with the id-substitution, see sec3

    print("\n   the fixed points form a real form (mode window -1..1):")
    for val, b in [(I, {1: -1, 2: -1}), (Rational(1, 2) + I, {1: -2, 2: -1})]:
        Phi, B0 = real_structure(val, b, sign, rng=(-1, 0, 1), verbose=False)
        ev = [X for nm, X in B0 if X.p == 0]
        od = [X for nm, X in B0 if X.p == 1]
        for nmp, fam in (('even', ev), ('odd', od)):
            dimC, dimR = fixed_dimension(Phi, fam)
            check(dimR == dimC,
                  "lambda = %s, %s: dim_R fixed = %s, dim_C = %s"
                  % (val, nmp, dimR, dimC))
            print("     lambda = %-8s %-4s part: dim_C = %d, dim_R (fixed) = %d"
                  % (val, nmp, dimC, dimR))

    print("\n   and the real form is closed under the bracket:")
    for val, b in [(I, {1: -1, 2: -1})]:
        Phi, B0 = real_structure(val, b, sign, rng=(-1, 0, 1), verbose=False)
        fam = [X for nm, X in B0]
        real_elts = ([fadd_vf(X, Phi(X)) for X in fam]
                     + [scal_vf(I, sub_vf(X, Phi(X))) for X in fam])
        bad = 0
        for X in real_elts:
            if not is_zero_vf(sub_vf(Phi(X), X)):
                bad += 1
        check(bad == 0, "the listed elements are not Phi-fixed (%d)" % bad)
        badbr = 0
        for X in real_elts[:12]:
            for Y in real_elts[:12]:
                Z = bracket(X, Y)
                if not is_zero_vf(sub_vf(Phi(Z), Z)):
                    badbr += 1
        check(badbr == 0, "the fixed set is not closed (%d)" % badbr)
        print("     lambda = %s: %d fixed elements, %d bracket failures"
              % (val, len(real_elts) - bad, badbr))


def fadd_vf(X, Y):
    return VF({a: fadd(X.c[a], Y.c[a]) for a in COMPS}, X.p)


# ====================================== 3. the converse, and a control case

def sec3(sign):
    print("\n3. What the isomorphism classification forbids")
    print("   Ant(g) nonempty  =>  g_lambda = conj(g_lambda) = g_{lambdabar}")
    print("   =>  lambdabar in {lambda + n, -lambda + n | n in Z}")
    print("   =>  Im lambda = 0  or  2 Re lambda in Z.")

    print("\n   control: lambda = 1/3 + i (2 Re lambda = 2/3 not in Z)")
    val = Rational(1, 3) + I
    B0 = [(nm, subs_vf(X, val)) for nm, X in basis((0, 1))]
    hits = 0
    for b1 in range(-6, 7):
        for b2 in range(-6, 7):
            b = {1: b1, 2: b2}
            for kind, push in (('psi', psi_push), ('twist', tw_push)):
                ok = True
                for nm, X in B0:
                    Y = push(b, conj_vf(X))
                    if any(expand(v) != 0
                           for v in div_lam(Y, sign, l=val).values()):
                        ok = False
                        break
                if ok:
                    hits += 1
                    print("     UNEXPECTED: %s with b = %s works" % (kind, b))
    check(hits == 0, "a monomial anti-linear map exists at lambda = 1/3+i")
    print("     338 monomial candidates (13 x 13 x 2): %d work" % hits)

    print("\n   and the real case, lambda real: C itself is a real structure")
    for val in (Rational(0), Rational(1, 3), Rational(-2)):
        B0 = [(nm, subs_vf(X, val)) for nm, X in basis((-1, 0, 1))]
        bad = 0
        for nm, X in B0:
            Y = conj_vf(X)
            if any(expand(v) != 0 for v in div_lam(Y, sign, l=val).values()):
                bad += 1
            if not is_zero_vf(sub_vf(conj_vf(conj_vf(X)), X)):
                bad += 1
        check(bad == 0, "C is not a real structure at lambda = %s (%d)"
              % (val, bad))
        print("     lambda = %-5s: C preserves svect and C^2 = id: %s"
              % (val, bad == 0))


def main():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sign = sv2.sec0()
    sv2.GOOD_SIGN[0] = sign
    print("antilin.py -- real forms of svect^L_lambda(1|2)   (sign = %s)"
          % sign)

    sec1(sign)
    sec2(sign)
    sec3(sign)

    print("\n%d checks, %d failures" % (CHECKS[0], len(FAIL)))
    for m in FAIL:
        print("  " + m)
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
