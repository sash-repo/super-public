#!/usr/bin/env python3
"""
nice.py -- the odd deformations of the trivial sl(1|1)-module, and what the
Beilinson/BGG construction of Section 2 does to them.

Tests two candidate deformation tables of Subsection 3.3 and checks the hypothesis of Lemma 2.7 ("nice") for the
deformed module.

Everything is exact: the modules are 1|1-dimensional, the parameters
(lambda, mu) of the deformation and (alpha, beta) of the linear form are
carried symbolically as polynomials in Q[lambda,mu,alpha,beta] represented by
dicts monomial -> coefficient.  No floating point, no guessing.
"""

from fractions import Fraction
from itertools import product

# ---------------------------------------------------------------- polynomials
def P(**kw):
    """constant or single monomial"""
    return {}


def mono(c=1, **vars_):
    key = tuple(sorted((k, v) for k, v in vars_.items() if v))
    return {key: Fraction(c)} if c else {}


def padd(*ps):
    out = {}
    for p in ps:
        for k, v in p.items():
            out[k] = out.get(k, Fraction(0)) + v
            if out[k] == 0:
                del out[k]
    return out


def pmul(p, q):
    out = {}
    for k1, v1 in p.items():
        for k2, v2 in q.items():
            d = dict(k1)
            for k, e in k2:
                d[k] = d.get(k, 0) + e
            key = tuple(sorted(d.items()))
            out[key] = out.get(key, Fraction(0)) + v1 * v2
            if out[key] == 0:
                del out[key]
    return out


def pneg(p):
    return {k: -v for k, v in p.items()}


def pstr(p):
    if not p:
        return "0"
    parts = []
    for k, v in sorted(p.items()):
        m = "".join(f"{n}" + (f"^{e}" if e > 1 else "") for n, e in k)
        parts.append(f"{v}{m}" if m else f"{v}")
    return " + ".join(parts)


LAM, MU = mono(1, lam=1), mono(1, mu=1)
ALPHA, BETA = mono(1, alpha=1), mono(1, beta=1)
ONE, ZERO = mono(1), {}


# ------------------------------------------------------- the deformed module
# Basis of M(lambda,mu) = 1 (x) C[tau]:  v_0 = 1 (deg 0, even),
#                                        v_1 = tau.1 (deg 1, odd).
# rho_tau(X_-) v_0 = lambda v_1,  rho_tau(X_+) v_0 = mu v_1,
# rho_tau(E) = 0,  everything kills v_1 (tau^2 = 0).
#
# An operator is a 2x2 matrix over the polynomial ring, acting on (v_0, v_1).
def op(a00, a01, a10, a11):
    return ((a00, a01), (a10, a11))


def omul(A, B):
    return op(padd(pmul(A[0][0], B[0][0]), pmul(A[0][1], B[1][0])),
              padd(pmul(A[0][0], B[0][1]), pmul(A[0][1], B[1][1])),
              padd(pmul(A[1][0], B[0][0]), pmul(A[1][1], B[1][0])),
              padd(pmul(A[1][0], B[0][1]), pmul(A[1][1], B[1][1])))


def oadd(A, B, sign=1):
    s = (lambda p: p) if sign == 1 else pneg
    return op(padd(A[0][0], s(B[0][0])), padd(A[0][1], s(B[0][1])),
              padd(A[1][0], s(B[1][0])), padd(A[1][1], s(B[1][1])))


def ozero(A):
    return all(not A[i][j] for i in range(2) for j in range(2))


ZERO_OP = op(ZERO, ZERO, ZERO, ZERO)


def deformed(lam=LAM, mu=MU, E_on_v0=ZERO, E_on_v1=ZERO):
    """rho(X_-), rho(X_+), rho(E) of the deformation; E kept adjustable so
    that the two displays of Subsection 3.3 can be tested as printed."""
    Xm = op(ZERO, ZERO, lam, ZERO)     # v_0 |-> lam v_1
    Xp = op(ZERO, ZERO, mu, ZERO)      # v_0 |-> mu  v_1
    E = op(E_on_v0, ZERO, ZERO, E_on_v1)
    return Xm, Xp, E


def check_module(Xm, Xp, E, label):
    """sl(1|1): X_- , X_+ odd, E = [X_+,X_-] even and central."""
    bad = []
    if not ozero(oadd(oadd(omul(Xp, Xm), omul(Xm, Xp)), E, sign=-1)):
        bad.append("[X_+,X_-] = E fails")
    if not ozero(omul(Xm, Xm)):
        bad.append("X_-^2 = 0 fails")
    if not ozero(omul(Xp, Xp)):
        bad.append("X_+^2 = 0 fails")
    for nm, A in (("X_-", Xm), ("X_+", Xp)):
        if not ozero(oadd(omul(E, A), omul(A, E), sign=-1)):
            bad.append(f"[E,{nm}] = 0 fails")
    print(f"  {label:52s} {'a module' if not bad else 'NOT a module: ' + '; '.join(bad)}")
    return not bad


print("=" * 78)
print("1.  Two candidate tables of Subsection 3.3, tested against [X_+,X_-] = E")
print("=" * 78)
# the first candidate: E.1 = 1, X_-.1 = tau.1, X_+.1 = 0, E.tau1 = tau1
check_module(*deformed(lam=ONE, mu=ZERO, E_on_v0=ONE, E_on_v1=ONE),
             label="first table   (E.1 = 1, E.tau1 = tau1)")
# the second candidate: E.1 = 0 but E.tau1 = tau1
check_module(*deformed(lam=ONE, mu=ZERO, E_on_v0=ZERO, E_on_v1=ONE),
             label="second table  (E.1 = 0, E.tau1 = tau1)")
# the corrected table: E = 0 identically
check_module(*deformed(lam=ONE, mu=ZERO), label="corrected     (E = 0 on all of 1 (x) C[tau])")
print("  general (lambda,mu):")
check_module(*deformed(), label="corrected, arbitrary lambda, mu")

print()
print("=" * 78)
print("2.  Is the deformed module nice?   H^i(V;L_xi), xi = alpha a + beta b")
print("=" * 78)
# L_xi on M: v_0 |-> (alpha*lambda + beta*mu) v_1,  v_1 |-> 0.
ell = padd(pmul(ALPHA, LAM), pmul(BETA, MU))
print(f"  L_xi(v_0) = ({pstr(ell)}) v_1,   L_xi(v_1) = 0")
print("  V_0 = <v_0> (degree 0),  V_1 = <tau.1> (degree 1)")
print()
print("  (a) alpha*lambda + beta*mu != 0:")
print("        Ker L_xi = <v_1>, Im L_xi = <v_1>")
print("        H^0 = (V_0 n Ker)/(V_0 n Im) = 0/0 = 0")
print("        H^1 = (V_1 n Ker)/(V_1 n Im) = <v_1>/<v_1> = 0   -> nice here")
print("  (b) alpha*lambda + beta*mu = 0, xi != 0  (one line in Lambda_1,")
print("      i.e. exactly one point of P(W) = CP^1, the zero of lambda x_1 + mu x_2):")
print("        L_xi = 0, so Ker = V, Im = 0")
print("        H^0 = <v_0> , H^1 = <tau.1> != 0   -> NOT nice")
print()
print("  Conclusion: for every (lambda,mu) != (0,0) there is exactly one")
print("  xi != 0 (up to a factor) with H^1(V;L_xi) != 0.  So no non-zero odd")
print("  deformation of 1 is a nice Lambda(2)-module, and the hypothesis of")
print("  Lemma 2.7 is not available for it.")

print()
print("=" * 78)
print("3.  The Beilinson complex of the deformed module")
print("=" * 78)
print("  L_j = V_j (x) O(j):   L_0 = O,  L_1 = O(1),  L_j = 0 otherwise")
print("  d_0 = sum_i xi_i (x) x_i : O -> O(1) is multiplication by")
print(f"        l = lambda x_1 + mu x_2")
print("  B(V) = Im d_{-1} = 0 ;  Z(V) = Ker(l : O -> O(1)) = 0 for l != 0")
print("        (a non-zero linear form is a non-zero divisor on P^1)")
print("  hence  Phi(V) = Z/B = 0   for every (lambda,mu) != (0,0),")
print("  whereas Phi(1) = O  (Subsection 2.6, example (ii)).")
print()
print("  So the construction of Section 2 sends the whole 0|2-dimensional")
print("  family of deformations of the module 1 to the ZERO sheaf, not to a")
print("  family of deformations of O.  Statement (3.5) does not follow from")
print("  Lemma 2.7 as invoked there.")
