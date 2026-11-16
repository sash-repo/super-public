#!/usr/bin/env python3
"""Exact check of the two facts behind eq. (27) of the text, i.e. of the relation
   between the modules Vol^lambda of lambda-densities and the modules cF_lambda
   defined through the powers of the contact form alpha_1, see eq. (26).

   For k^L(1|n) realised in generating functions, see eq. (3):
       K_f = (2 - E)(f) d_t - (-1)^{p(f)} sum_i (d_{theta_i} f - theta_i d_t f) d_{theta_i},
   we check, for all monomials f = t^a * (product of thetas) in a window,

     (A)  Div K_f = (2 - n) d_t(f)                                    [eq. (35)]
     (B)  L_{K_f}(alpha_1) = 2 d_t(f) . alpha_1                       [the multiplier]

   (A) says that the multiplier of K_f on Vol is (2-n) d_t f; (B) says that on
   cF_lambda = cF alpha_1^{lambda/2} (n > 0) it is (lambda/2) . 2 d_t f =
   lambda d_t f, and on cF_lambda = cF alpha_1^{lambda} (n = 0) it is
   2 lambda d_t f.  Comparing:

       n > 0:  lambda = 2 - n     ==>  Vol^lambda = cF_{lambda(2-n)},
       n = 0:  2 lambda = 2       ==>  Vol       = cF_1,  Vol^lambda = cF_lambda,

   which is exactly the two-line eq. (27) as corrected, and it shows that the
   one-line version Vol^lambda = cF_{lambda(2-n)} inherited by the head is wrong
   at n = 0 (it would give cF_{2 lambda}).

   Conventions.  Functions and vector fields are those of stringy.py (coefficients
   of the vector fields on the left).  A 1-form is written with its coefficients
   on the right, omega = sum_b dx^b omega_b, so that d g = sum_b dx^b (d_b g) with
   the left partial derivatives, and p(dx^b) = p(b) + 1.  With that convention

       (L_X omega)_b = sum_a (d_b X^a) omega_a + (-1)^{p(b) p(X)} X(omega_b),

   as one gets from L_X (d g) = d (X g); and, for a function c,

       (c . omega)_b = (-1)^{p(c)(p(b)+1)} c omega_b.
"""

from fractions import Fraction
from itertools import combinations

import stringy as S


# ------------------------------------------------------------------ helpers

def p_coord(b):
    return 0 if b == 't' else 1


def d_b(b, f):
    return S.d_t(f) if b == 't' else S.d_odd(b, f)


def K(f, n):
    """The contact field K_f of eq. (3) for k^L(1|n); odd labels 1..n."""
    pf = S.parity(f)
    # (2 - E)(f), where E = sum theta_i d_{theta_i}
    Ef = {}
    for i in range(1, n + 1):
        Ef = S.add(Ef, S.mul(S.mono(0, (i,)), S.d_odd(i, f)))
    comps = {'t': S.add(S.smul(2, f), S.smul(-1, Ef))}
    for i in range(1, n + 1):
        term = S.add(S.d_odd(i, f),
                     S.smul(-1, S.mul(S.mono(0, (i,)), S.d_t(f))))
        comps[i] = S.smul((-1) ** pf * -1, term)
    return S.VF(comps, tuple(range(1, n + 1)))


def div(X):
    """Super divergence sum_a (-1)^{p(a) p(X^a)} d_a X^a."""
    out = {}
    for a, Xa in X.c.items():
        sgn = (-1) ** (p_coord(a) * S.parity(Xa))
        out = S.add(out, S.smul(sgn, d_b(a, Xa)))
    return out


def lie_form(X, omega, coords):
    """(L_X omega)_b = sum_a (d_b X^a) omega_a + (-1)^{p(b) p(X)} X(omega_b)."""
    pX = X.parity()
    out = {}
    for b in coords:
        acc = {}
        for a, Xa in X.c.items():
            if a in omega:
                acc = S.add(acc, S.mul(d_b(b, Xa), omega[a]))
        if b in omega:
            acc = S.add(acc, S.smul((-1) ** (p_coord(b) * pX), X.apply(omega[b])))
        if acc:
            out[b] = acc
    return out


def scal_form(c, omega, coords, side="right"):
    """(omega . c)_b = omega_b c   resp.  (c . omega)_b = (-1)^{p(c)(p(b)+1)} c omega_b.

    In the convention omega = sum_b dx^b omega_b the multiplier of a contact
    field sits on the RIGHT: L_{K_f} alpha_1 = alpha_1 . (2 d_t f).  For even f
    the two sides agree, and it is the even f that fix the scaling of the module
    cF_lambda; the check below is run both ways.
    """
    pc = S.parity(c)
    out = {}
    for b in coords:
        if b in omega:
            if side == "right":
                v = S.mul(omega[b], c)
            else:
                v = S.smul((-1) ** (pc * (p_coord(b) + 1)), S.mul(c, omega[b]))
            if v:
                out[b] = v
    return out


def form_eq(w1, w2, coords):
    return all(S.eq(w1.get(b, {}), w2.get(b, {})) for b in coords)


# ------------------------------------------------------------------- the run

def monomials(n, amin=-2, amax=3):
    """t^a * theta_S for all S, |S| any."""
    out = []
    for a in range(amin, amax + 1):
        for k in range(n + 1):
            for Sset in combinations(range(1, n + 1), k):
                out.append((a, Sset))
    return out


def run():
    bad_div = bad_mult = bad_mult_ev = tot = tot_ev = 0
    for n in range(0, 5):
        coords = ['t'] + list(range(1, n + 1))
        alpha = {'t': S.const(1)}
        for i in range(1, n + 1):
            alpha[i] = S.mono(0, (i,))          # theta_i dtheta_i = dtheta_i . theta_i
        for (a, Sset) in monomials(n):
            f = S.mono(a, Sset)
            Kf = K(f, n)
            tot += 1

            # (A) divergence
            lhs = div(Kf)
            rhs = S.smul(2 - n, S.d_t(f))
            if not S.eq(lhs, rhs):
                bad_div += 1
                print(f"   DIV  n={n} f=t^{a}{Sset}: {S.show(lhs)} != {S.show(rhs)}")

            # (B) contact multiplier
            L = lie_form(Kf, alpha, coords)
            R = scal_form(S.smul(2, S.d_t(f)), alpha, coords)
            if not form_eq(L, R, coords):
                bad_mult += 1
                print(f"   MULT n={n} f=t^{a}{Sset}: "
                      f"{ {b: S.show(v) for b, v in L.items()} } != "
                      f"{ {b: S.show(v) for b, v in R.items()} }")
            if S.parity(f) == 0:
                tot_ev += 1
                Rl = scal_form(S.smul(2, S.d_t(f)), alpha, coords, side="left")
                if not form_eq(L, Rl, coords):
                    bad_mult_ev += 1

    print(f"\n{tot} generating functions tested, n = 0..4")
    print(f"  (A) Div K_f = (2-n) d_t f                  : {bad_div} failures")
    print(f"  (B) L_{{K_f}} alpha_1 = alpha_1 . (2 d_t f)   : {bad_mult} failures")
    print(f"  (B') the same with the multiplier on the left,")
    print(f"       f even ({tot_ev} functions)              : {bad_mult_ev} failures")

    # sanity: the engine must be able to see a wrong statement
    wrong = 0
    for n in (2, 3):
        for (a, Sset) in monomials(n):
            f = S.mono(a, Sset)
            if not S.eq(div(K(f, n)), S.smul(1 - n, S.d_t(f))):
                wrong += 1
    print(f"  control (2-n replaced by 1-n)        : {wrong} failures (must be > 0)")

    # what eq. (27) then says
    print("\n  multiplier of K_f on Vol^lambda  : lambda (2-n) d_t f")
    print("  multiplier of K_f on cF_mu       : mu d_t f      (n > 0)")
    print("                                     2 mu d_t f    (n = 0)")
    print("  ==> Vol^lambda = cF_{lambda(2-n)} for n > 0, and = cF_lambda for n = 0.")
    return bad_div + bad_mult


if __name__ == "__main__":
    raise SystemExit(1 if run() else 0)
