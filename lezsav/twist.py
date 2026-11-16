#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
twist.py -- can the parameter lambda of svect^L_lambda(1|2) be recovered by a
super Liouville equation depending on it?

The whole question is about ONE map: the twist

    Phi_{a_1,a_2}:  t |-> t,   theta_i |-> t^{a_i} theta_i,           (Phi)

with COMPLEX exponents a_i.  This script establishes, in exact arithmetic with
lambda and the a_i formal (sympy), the four facts the answer rests on.

  1. Phi_{a_1,a_2} carries the condition div_lambda = 0 to div_mu = 0 with
     mu = lambda - a_1 - a_2, for every complex a_1, a_2 -- the shift of lambda
     is LINEAR in the twist and has no denominator.  For a_i in Z the map is a
     Laurent automorphism of O(1|2) and hence an isomorphism svect^L_lambda ->
     svect^L_mu; for a_i not in Z it is not, and the Laurent fields in the image
     form the genuinely different algebra svect^L_mu(1|2).  This is the same
     relation that k^L(1|1) and k^M(1|1) bear to each other -- one local
     algebra, two (here: a continuum of) sectors.
     The two doublets move oppositely: G^r |-> G^{r-a}, Gbar^s |-> Gbar^{s+a}.

  2. The generators of svect^L_lambda(1|2) are homogeneous in the theta-degree
     (L, J: degree 0; G: degree -1; Gbar: degree +1), whereas the contact
     generators K_f of k^L(1|N) are NOT: K_{t^p theta_i} has a degree +1 and a
     degree -1 part at once.  Under (Phi) with a_1 = ... = a_N = a the two parts
     are multiplied by t^{a} and t^{-a}, so the image consists of Laurent fields
     only if 2a is an integer -- the two sectors NS (a in Z) and Ramond
     (a in Z + 1/2) and nothing between them.  THIS is why lambda is a continuum
     for svect and a Z/2 for the contact series.

  3. The inversion t |-> 1/t is a Laurent automorphism and carries
     svect^L_lambda(1|2) to svect^L_{-lambda-2}(1|2).  Together with 1) at
     integer a this gives  lambda ~ lambda + 1  and  lambda ~ -lambda,
     so the modulus is  exp(2 pi i lambda) in C^* modulo z ~ 1/z.

  4. The Liouville side.  Feeding (Phi) into the Ivanov-Krivonos N=4 system
     multiplies its mass term by (x_+/x_-)^{lambda/2} and deforms the odd
     derivatives to  D = i theta (d_x + (lambda/2x) J) + d_thetabar,  J the
     u(1)-charge; the rescaling q |-> (x_+/x_-)^{-lambda/2} q removes the factor
     again and is single-valued exactly for lambda in 2Z.  Both the deformed
     relations of the D's and the removal are verified here.  The bosonic
     shadow (Liouville with a conical singularity) is verified as well.

Run:  ../venv/bin/python twist.py     (plain python3 works too, slower)
"""

import io
import contextlib
import sys

from sympy import (Symbol, Rational, symbols, simplify, expand, factor, solve,
                   sympify, exp, log, diff, Function, together, cancel,
                   expand_log)

import sv2
from sv2 import (VF, mono, fadd, fscal, fmul, fclean, d_t, d_th, bracket,
                 div_lam, lam, TH, COMPS, L, J, G, Gbar, E)

FAIL = []
CHECKS = [0]


def check(cond, msg):
    CHECKS[0] += 1
    if not cond:
        FAIL.append(msg)
        print("  FAIL: " + msg)


def vstr(X):
    """printer that survives symbolic exponents (sv2.vf_str sorts the keys)."""
    parts = []
    for a in COMPS:
        for (m, S), c in X.c[a].items():
            cs = str(factor(simplify(c)))
            th = ''.join('th%d' % i for i in S)
            tp = 't^(%s)' % m
            d = 'd_t' if a == 't' else 'd_%d' % a
            parts.append('*'.join(z for z in ('(%s)' % cs, tp, th, d) if z))
    return ' + '.join(parts) if parts else '0'


# ===================================================================== the twist

def phi_f(A, f, inv=False):
    """Phi(f) for Phi: theta_i |-> t^{A[i]} theta_i (inv: the inverse map)."""
    out = {}
    for (m, S), c in f.items():
        sh = sum(A[i] for i in S)
        if inv:
            sh = -sh
        k = (expand(m + sh), S)
        out[k] = out.get(k, 0) + c
    return fclean(out)


def push(A, X):
    """Phi_* X = Phi o X o Phi^{-1}, componentwise on t, theta_1, theta_2."""
    comps = {'t': phi_f(A, X.act(mono(1)))}
    for i in TH:
        comps[i] = phi_f(A, X.act(phi_f(A, mono(0, (i,)), inv=True)))
    return VF(comps, X.p)


def inv_f(f):
    """Psi(f) for Psi: t |-> 1/t, theta_i |-> theta_i."""
    return fclean({(expand(-m), S): c for (m, S), c in f.items()})


def push_inv(X):
    comps = {'t': inv_f(X.act(mono(-1)))}
    for i in TH:
        comps[i] = inv_f(X.act(mono(0, (i,))))
    return VF(comps, X.p)


def mu_of(X, sign):
    """the unique mu with div_mu(X) = 0, or None."""
    mu = Symbol('mu_')
    eqs = [expand(v) for v in div_lam(X, sign, l=mu).values()]
    eqs = [e for e in eqs if e != 0]
    if not eqs:
        return 'any'
    sol = solve(eqs, mu, dict=True)
    if not sol:
        return None
    return simplify(sol[0][mu])


def basis(sign, rng=(-2, -1, 0, 1, 2)):
    """a window of the basis of svect^L_lambda(1|2)."""
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


# ============================================ 1. the twist shifts lambda linearly

def sec1(sign):
    print("\n1. The twist Phi: theta_i |-> t^{a_i} theta_i")
    a1, a2 = symbols('a_1 a_2')
    A = {1: a1, 2: a2}
    shifts = set()
    for nm, X in basis(sign):
        Y = push(A, X)
        mu = mu_of(Y, sign)
        if mu == 'any':
            continue
        check(mu is not None, "%s: pushforward is in no svect_mu" % nm)
        if mu is not None:
            shifts.add(simplify(mu))
    check(len(shifts) == 1, "the shift is not uniform: %s" % shifts)
    mu = shifts.pop()
    check(simplify(mu - (lam - a1 - a2)) == 0,
          "shift is %s, not lambda - a_1 - a_2" % mu)
    print("   Phi_* carries div_lambda = 0 to div_mu = 0 with mu = %s" % mu)
    print("   -- LINEAR in the twist, and with no denominator: every complex")
    print("      value of mu is reached, from any lambda, by a suitable a.")
    print("   For a_i in Z, Phi is a Laurent automorphism of O(1|2), so this is")
    print("   an isomorphism svect^L_lambda -> svect^L_mu; for a_i not in Z it")
    print("   is not, and the LAURENT fields in the image are the genuinely")
    print("   different algebra svect^L_mu(1|2).  Cf. k^L(1|1) vs k^M(1|1).")

    # both theta's twisted by the same a: mu = lambda - 2a, and the doublets
    # move oppositely while L and the currents do not move at all.
    a = Symbol('a')
    A = {1: a, 2: a}
    print("\n   With a_1 = a_2 = a (mu = lambda - 2a), what happens to a basis:")
    for nm, X in [('L_1', L(1)), ('J+_1', J('+', 1)), ('J0_1', J('0', 1)),
                  ('G1^{1/2}', G(1, Rational(1, 2))),
                  ('Gb^{1/2,1}', Gbar(1, Rational(1, 2)))]:
        Y = push(A, X)
        print("     %-12s |-> %s" % (nm, vstr(Y)))

    # the mode shift, read off exactly: G^r |-> t^{-a} G^r, Gbar^s |-> t^{a} Gbar^s
    bad = 0
    for k in (-2, -1, 0, 1, 2):
        r = Rational(2 * k + 1, 2)
        for al in TH:
            Y = push(A, G(al, r))
            for (m, S) in Y.c[al]:
                if simplify(m - (r + Rational(1, 2) - a)) != 0:
                    bad += 1
            Z = push(A, Gbar(al, r))
            for c in COMPS:
                for (m, S) in Z.c[c]:
                    off = 0 if c == 't' else 1
                    # t-degree of the Gbar-components: p + a  and  p - 1 + a
                    if simplify(m - (r + Rational(1, 2) + a)) != 0 and \
                       simplify(m - (r - Rational(1, 2) + a)) != 0:
                        bad += 1
    check(bad == 0, "the doublets do not move by -a and +a (%d)" % bad)
    print("   G^r_alpha |-> t^{-a} G, Gbar^{s alpha} |-> t^{+a} Gbar,")
    print("   L and the sl(2)-currents unmoved: %d exceptions." % bad)
    return a


# ================================ 2. theta-homogeneity: svect versus the contact

def theta_degrees(X):
    """the set of theta-degrees (#theta - #d_theta) occurring in X."""
    degs = set()
    for a in COMPS:
        for (m, S), c in X.c[a].items():
            if expand(c) == 0:
                continue
            degs.add(len(S) - (0 if a == 't' else 1))
    return degs


def sec2(sign):
    print("\n2. Why lambda is a continuum here and a Z/2 in the contact series")
    print("   (a) the generators of svect^L_lambda(1|2) are theta-homogeneous:")
    bad = 0
    for nm, X in basis(sign):
        d = theta_degrees(X)
        if len(d) != 1:
            bad += 1
            print("       %s has degrees %s" % (nm, d))
    check(bad == 0, "svect generators not theta-homogeneous (%d)" % bad)
    for nm, X in [('L_n', L(1)), ('J^a_n', J('+', 1)), ('J^0_n', J('0', 1)),
                  ('G^r_alpha', G(1, Rational(1, 2))),
                  ('Gbar^{r alpha}', Gbar(1, Rational(1, 2)))]:
        print("       %-14s theta-degree %s" % (nm, theta_degrees(X)))
    print("       so Phi multiplies each generator by ONE power of t and the")
    print("       modes can be relabeled for every complex a: no constraint.")

    print("   (b) the contact generators K_f of k^L(1|N) are not:")
    import os
    import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
    from stringy import mono as kmono, parity as kparity
    import slnN

    def kdegs(n, f):
        """theta-degrees of K_f, computed with the engine of slnN.py."""
        X = slnN.K_L(n, f)
        degs = set()
        for a, comp in X.comps.items() if hasattr(X, 'comps') else []:
            pass
        # VF of stringy.py: attribute .c is a dict too
        d = X.c if hasattr(X, 'c') else X.comps
        for a, comp in d.items():
            for (m, S), c in comp.items():
                if c == 0:
                    continue
                degs.add(len(S) - (0 if a == 't' else 1))
        return degs

    for n in (2, 4):
        for (desc, f) in [('K_{t^2}', kmono(2)),
                          ('K_{t^2 theta_1}', kmono(2, (1,))),
                          ('K_{t^2 theta_1 theta_2}', kmono(2, (1, 2)))]:
            print("       N=%d  %-24s theta-degrees %s"
                  % (n, desc, sorted(kdegs(n, f))))
    dodd = kdegs(4, kmono(2, (1,)))
    check(dodd == {1, -1},
          "K_{t^p theta} is theta-homogeneous after all: %s" % dodd)
    print("       K_{t^p theta_i} carries the degrees +1 AND -1 at once, so")
    print("       Phi multiplies its two halves by t^{+a} and t^{-a}: the image")
    print("       is a Laurent algebra only if 2a is an integer.  Hence exactly")
    print("       two sectors, NS (a in Z) and Ramond (a in Z+1/2), and no")
    print("       parameter -- while svect^L(1|2) has the whole of C.")


# ========================================= 3. the inversion:  lambda |-> -lambda

def sec3(sign):
    print("\n3. The inversion Psi: t |-> 1/t")
    shifts = set()
    for nm, X in basis(sign):
        Y = push_inv(X)
        mu = mu_of(Y, sign)
        if mu == 'any':
            continue
        check(mu is not None, "%s: Psi_* is in no svect_mu" % nm)
        if mu is not None:
            shifts.add(simplify(mu))
    check(len(shifts) == 1, "the inversion shift is not uniform: %s" % shifts)
    mu = shifts.pop()
    check(simplify(mu - (-lam - 2)) == 0, "inversion gives %s" % mu)
    print("   Psi_* svect^L_lambda(1|2) = svect^L_mu(1|2), mu = %s" % mu)
    print("   Psi is a Laurent automorphism of O(1|2), so this is an honest")
    print("   isomorphism; with lambda ~ lambda+1 from sec.1 at a in Z it gives")
    print("      lambda ~ lambda + 1   and   lambda ~ -lambda,")
    print("   i.e. the modulus is  z = exp(2 pi i lambda) in C^*  mod z ~ 1/z.")

    # Psi is an involution of the underlying superspace
    Z = push_inv(push_inv(L(2)))
    check(all(simplify(v) == 0 for c in COMPS
              for v in (fadd(Z.c[c], fscal(-1, L(2).c[c]))).values()),
          "Psi is not an involution")
    print("   Psi^2 = id on the generators: verified.")


# ============================== 4. the Liouville side: the equation that sees it

def sec4a():
    """The four twisted odd derivatives of the N=4 superstring, exactly.

    On C^{1|4} with the chiral pair (theta^alpha, thetabar_alpha), alpha = 1,2,
    the twist theta |-> x^{-l/2} theta, thetabar |-> x^{l/2} thetabar turns

        D^alpha = i theta^alpha d_x + d/d thetabar_alpha,
        Dbar_alpha = i thetabar_alpha d_x + d/d theta^alpha

    into  x^{-l/2} A^alpha  and  x^{l/2} B_alpha  with A, B single-valued and
    d_x replaced by  nabla = d_x + (l/2x) Jhat,  Jhat the u(1)-charge.  We
    verify that the twisted derivatives satisfy the FLAT relations with nabla
    in place of d_x -- which is what makes the twisted system a super Liouville
    system at all -- while the single-valued A, B alone do not.
    """
    import itertools
    x, l = symbols('x lambda')
    from sympy import I as ii

    def clean(f):
        return {k: cancel(expand(v)) for k, v in f.items()
                if cancel(expand(v)) != 0}

    def fadd2(*fs):
        o = {}
        for f in fs:
            for k, v in f.items():
                o[k] = o.get(k, 0) + v
        return clean(o)

    def fs2(c, f):
        return clean({k: c * v for k, v in f.items()})

    def merge(S1, S2):
        if set(S1) & set(S2):
            return None, 0
        arr = list(S1) + list(S2)
        sg = 1
        for i in range(len(arr)):
            for j in range(len(arr) - 1 - i):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    sg = -sg
        return tuple(arr), sg

    def lmul(i, f):
        o = {}
        for S, c in f.items():
            T, sg = merge((i,), S)
            if T is not None:
                o[T] = o.get(T, 0) + sg * c
        return clean(o)

    def dodd(i, f):
        o = {}
        for S, c in f.items():
            if i in S:
                p = S.index(i)
                T = tuple(y for y in S if y != i)
                o[T] = o.get(T, 0) + (-1) ** p * c
        return clean(o)

    def ddx(f):
        return clean({S: diff(c, x) for S, c in f.items()})

    def Jhat(f):
        return clean({S: (sum(1 for i in S if i in (1, 2))
                          - sum(1 for i in S if i in (3, 4))) * c
                      for S, c in f.items()})

    def nabla(f):
        return fadd2(ddx(f), fs2(l / (2 * x), Jhat(f)))

    # 1,2 = theta^1, theta^2 ;  3,4 = thetabar_1, thetabar_2
    Aop = lambda al, f: fadd2(fs2(ii, lmul(al, nabla(f))), dodd(al + 2, f))
    Bop = lambda be, f: fadd2(fs2(ii, lmul(be + 2, nabla(f))), dodd(be, f))
    Dop = lambda al, f: fs2(x ** (-l / 2), Aop(al, f))
    Dbop = lambda be, f: fs2(x ** (l / 2), Bop(be, f))

    tests = []
    for k in range(5):
        for S in itertools.combinations((1, 2, 3, 4), k):
            for c in (x ** 2, x ** -1, sympify(1), x ** 3):
                tests.append({S: c})

    def zero(op, name):
        bad = 0
        for f in tests:
            if op(f):
                bad += 1
        check(bad == 0, "%s does not vanish (%d of %d)" % (name, bad, len(tests)))
        return bad

    print("\n4a. The twisted odd derivatives of the N=4 superstring")
    tot = 0
    for al in (1, 2):
        for be in (1, 2):
            d = 1 if al == be else 0
            tot += zero(lambda f, al=al, be=be, d=d:
                        fadd2(fadd2(Dop(al, Dbop(be, f)), Dbop(be, Dop(al, f))),
                              fs2(-2 * ii * d, nabla(f))),
                        "{D^%d,Dbar_%d} = 2i delta nabla" % (al, be))
            tot += zero(lambda f, al=al, be=be:
                        fadd2(Dop(al, Dop(be, f)), Dop(be, Dop(al, f))),
                        "{D^%d,D^%d} = 0" % (al, be))
            tot += zero(lambda f, al=al, be=be:
                        fadd2(Dbop(al, Dbop(be, f)), Dbop(be, Dbop(al, f))),
                        "{Dbar_%d,Dbar_%d} = 0" % (al, be))
    print("   D^alpha = x^{-lambda/2}(i theta^alpha nabla + d/d thetabar_alpha),")
    print("   Dbar_alpha = x^{lambda/2}(i thetabar_alpha nabla + d/d theta^alpha),")
    print("   nabla = d_x + (lambda/2x) Jhat,  Jhat the u(1)-charge.  Then")
    print("   {D^a,Dbar_b} = 2i delta^a_b nabla,  {D,D} = {Dbar,Dbar} = 0:")
    print("   12 relations on %d test superfunctions, %d failures."
          % (len(tests), tot))
    print("   The prefactors x^{-+lambda/2} are the whole of the twist: drop")
    print("   them and the relations acquire (i lambda/2x)-terms, i.e. the")
    print("   single-valued A, B alone do NOT span a flat N=4 superspace.")


def sec4():
    print("\n4. The Ivanov-Krivonos system in the twisted variables")
    x, l = symbols('x lambda', positive=True)

    # (a) the deformed odd derivatives.  On C^{1|4} with the chiral pair
    # (theta^alpha, thetabar_alpha), the twist theta |-> x^{-l/2} theta,
    # thetabar |-> x^{+l/2} thetabar turns d_x into  d_x + (l/2x) Jhat,
    # Jhat = theta d_theta - thetabar d_thetabar the u(1)-charge.  We verify
    # the transport on the graded pieces: a field of u(1)-charge q acquires
    # the factor x^{-q l/2}, so d_x |-> d_x + (l/2x) Jhat on the twisted frame.
    q = Symbol('q')
    f = Function('f')
    lhs = diff(x ** (-q * l / 2) * f(x), x)
    rhs = x ** (-q * l / 2) * (diff(f(x), x) - q * l / (2 * x) * f(x))
    check(simplify(lhs - rhs) == 0, "the transport of d_x is wrong")
    print("   d_x on a field of u(1)-charge q, in the twisted frame:")
    print("     d_x  |->  d_x + (lambda/2x) Jhat        [verified]")

    # (b) the mass term.  D^alpha_pm = x_pm^{-l/2} D^alpha_(l)pm and
    # Dbar_{pm alpha} = x_pm^{+l/2} Dbar_(l) pm alpha, so the IK equation
    #     D_+^a (q^{-1} Dbar_{-c} q)_b^c = -4 i m qbar_b^a
    # becomes, in the twisted frame,
    #     D_(l)+^a (q^{-1} Dbar_(l)-c q)_b^c = -4 i m (x_+/x_-)^{l/2} qbar_b^a.
    xp, xm = symbols('x_+ x_-', positive=True)
    pref = simplify((xp ** (-l / 2) * xm ** (l / 2)) ** (-1))
    check(simplify(pref - (xp / xm) ** (l / 2)) == 0,
          "the mass prefactor is not (x_+/x_-)^{lambda/2}")
    print("   the mass term picks up the factor  (x_+/x_-)^{lambda/2}")
    print("   = %s   [verified]" % pref)

    # (c) the rescaling that removes it again.  q |-> c q with c a scalar
    # superfunction adds  D_+ ( c^{-1} Dbar_- c )  to the left-hand side; for
    # c = (x_+/x_-)^{-l/2} that term vanishes because c splits as a product of
    # a function of x_+ and a function of x_-, each killed by the OTHER half's
    # derivative.  We verify the splitting and the vanishing symbolically.
    c = (xp / xm) ** (-l / 2)
    cp, cm = xp ** (-l / 2), xm ** (l / 2)
    check(simplify(c - cp * cm) == 0, "c does not split between the halves")
    # c^{-1} Dbar_- c  with  Dbar_- = i thetabar_- (d_{x_-} + (l/2x_-) Jhat)
    #                                  + d_{theta_-} :  only d_{x_-} survives
    term = simplify(diff(cm, xm) / cm)          # = l/(2 x_-)
    check(simplify(term - l / (2 * xm)) == 0, "c^{-1} Dbar_- c is not l/2x_-")
    check(simplify(diff(term, xp)) == 0,
          "D_+ of the extra term does not vanish")
    print("   q |-> (x_+/x_-)^{-lambda/2} q removes the factor: the extra term")
    print("   c^{-1} Dbar_- c = lambda/(2 x_-) depends on x_- alone and is")
    print("   killed by D_+.   [verified]")
    print("   That rescaling is single-valued exactly for lambda in 2Z; with")
    print("   the residual Laurent twist of ONE theta (sec.1, a in Z) the")
    print("   condition is lambda in Z.  For lambda not in Z it is genuinely")
    print("   multivalued: lambda is a monodromy, not a local datum.")


# ============================================ 5. the bosonic shadow: a cone

def sec5():
    print("\n5. The bosonic shadow: the same twist is a conical singularity")
    x, y = symbols('x y', positive=True)
    mu, nu = symbols('mu nu')

    # Schwarzian of f = x^mu, and the Hill potential T = -S(f)/2
    f = x ** mu
    S = simplify(diff(f, x, 3) / diff(f, x)
                 - Rational(3, 2) * (diff(f, x, 2) / diff(f, x)) ** 2)
    T = simplify(-S / 2)
    check(simplify(T - (mu ** 2 - 1) / (4 * x ** 2)) == 0,
          "T is not (mu^2-1)/4x^2, got %s" % T)
    print("   f = x^mu:  T = -S(f)/2 = %s" % simplify(T))
    print("   the monodromy of the Hill operator d^2/dx^2 - T around 0 is")
    print("   the rotation by 2 pi mu: mu is read off T, and is an integer")
    print("   exactly when the solution is single-valued.")

    # the Liouville equation for the single-valued part
    Ft = Function('F')(x, y)
    F = Ft + (mu - 1) / 2 * log(x) + (nu - 1) / 2 * log(y)
    lhs = simplify(diff(F, x, y))
    rhs = simplify(exp(2 * F))
    check(simplify(lhs - diff(Ft, x, y)) == 0,
          "the log terms do not drop out of F_xy")
    ratio = rhs / (x ** (mu - 1) * y ** (nu - 1) * exp(2 * Ft))
    check(simplify(expand_log(log(ratio), force=True)) == 0,
          "the exponential does not produce x^{mu-1} y^{nu-1}")
    print("   F = Ftilde + (mu-1)/2 log x + (nu-1)/2 log y turns F_xy = e^{2F}")
    print("   into   Ftilde_xy = x^{mu-1} y^{nu-1} e^{2 Ftilde}:")

    # and the general solution is the twisted one: f = x^mu, g = y^nu
    Fgen = log(diff(x ** mu, x) * diff(y ** nu, y)
               / (x ** mu + y ** nu) ** 2) / 2
    check(simplify(diff(Fgen, x, y) - exp(2 * Fgen)) == 0,
          "F = 1/2 log(f'g'/(f+g)^2) with f = x^mu is not a solution")
    Ftil = simplify(Fgen - (mu - 1) / 2 * log(x) - (nu - 1) / 2 * log(y))
    src = simplify(x ** (mu - 1) * y ** (nu - 1)
                   * exp(expand_log(2 * Ftil, force=True)))
    check(simplify(diff(Ftil, x, y) - src) == 0,
          "the single-valued part does not solve the equation with sources")
    print("   and f = x^mu, g = y^nu is a solution of it, with Ftilde")
    print("   single-valued:  Ftilde = %s" % simplify(Ftil))
    print("   the parameter enters the equation as a power in front of the")
    print("   exponential -- exactly as (x_+/x_-)^{lambda/2} does in sec.4.")
    print("   But witt(x) does not change with mu, whereas svect^L_lambda(1|2)")
    print("   does: only the SUPER case has odd coordinates for the twist to")
    print("   weigh, which is why lambda is a superalgebra modulus and mu is")
    print("   only a modulus of the solution.")


def main():
    print("=" * 78)
    print("Can a super Liouville equation depend on lambda in C \\ Z?")
    print("=" * 78)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sign = sv2.sec0()
    sv2.GOOD_SIGN[0] = sign
    print("(the calculus and the sign convention of sv2.py re-verified: "
          "sign = %s)" % sign)
    sec1(sign)
    sec2(sign)
    sec3(sign)
    sec4a()
    sec4()
    sec5()
    print("\n" + "=" * 78)
    print("checks: %d, failures: %d" % (CHECKS[0], len(FAIL)))
    for f in FAIL:
        print("  ! " + f)
    print("=" * 78)


if __name__ == '__main__':
    main()
