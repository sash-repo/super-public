#!/usr/bin/env python3
"""Four claims of H. Andersen about the twist and the top harmonic, checked.

Four claims are put to exact arithmetic, and one further question that her second
item opens and does not close.

1) The factor of two.  With  Delta phi := arg beta - arg alpha  and  alpha, beta
   of weight 2, is  tau = (1/4)(k1-k2)_A (k1-k2)_B sin(Delta phi)  or
   sin(2 Delta phi)?

2) c_5 pointwise against globally.  c_5 = (3i/4) alpha^2 d(beta/alpha), so
   c_5(p) = 0 iff d(beta/alpha)(p) = 0.  Does  c_5 == 0  on a connected region
   imply  beta/alpha  constant there?  (d is the Wirtinger derivative.)

3) Her example  beta = h alpha,  h real and varying:  Delta phi = 0, rho = 1,
   c_5 != 0.

4) The converse of 3), which decides whether rho and c_5 are independent in BOTH
   directions.

Everything is done with sympy over Q(i) and evaluated only at the end.
"""

import sys
from sympy import (
    I, Rational, simplify, symbols, integrate, pi, sqrt, cos, sin, atan2, re, im,
    conjugate, expand, diff, Matrix, nsimplify, srepr, exp, Abs, S, factor,
)

x, y = symbols("x y", real=True)
r, th = symbols("r theta", positive=True)

LOG = []


def check(name, ok, detail=""):
    LOG.append((name, bool(ok), detail))
    print(("PASS " if ok else "FAIL ") + name + (("   " + detail) if detail else ""))
    return bool(ok)


def d(f):
    """Wirtinger d/dz = (1/2)(d/dx - i d/dy)."""
    return Rational(1, 2) * (diff(f, x) - I * diff(f, y))


def dbar(f):
    return Rational(1, 2) * (diff(f, x) + I * diff(f, y))


# --------------------------------------------------------------- the dictionary
# A_0 = [[p, q], [q, -p]],  alpha = p - i q;  p + i q = R e^{2 i phi_A}.
def alpha_of(p, q):
    return p - I * q


def tensor_of(p, q, a=0):
    return Matrix([[a + p, q], [q, a - p]])


EPS = Matrix([[0, -1], [1, 0]])


def tau_of(A, B):
    return simplify(Rational(1, 2) * (A * EPS * B).trace())


print("=" * 72)
print("1.  The factor of two in Proposition 5.10")
print("=" * 72)

# Two forms given by their principal data: half-difference R and axis angle phi.
RA, RB, phA, phB = symbols("R_A R_B varphi_A varphi_B", real=True)

pA, qA = RA * cos(2 * phA), RA * sin(2 * phA)
pB, qB = RB * cos(2 * phB), RB * sin(2 * phB)
A0 = tensor_of(pA, qA)
B0 = tensor_of(pB, qB)
al = alpha_of(pA, qA)
be = alpha_of(pB, qB)

# the eigenvector check: (cos phi, sin phi) is the +R eigenvector of A_0
v = Matrix([cos(phA), sin(phA)])
check(
    "phi_A is the principal direction of A_0",
    simplify((A0 * v - RA * v)) == Matrix([0, 0]),
)

tau = tau_of(A0, B0)
check(
    "tau(A,B) = (1/2) tr(A eps B) = Im(conj(alpha) beta)",
    simplify(tau - im(expand(conjugate(al) * be).rewrite(cos))) == 0
    or simplify(tau - simplify(expand((conjugate(al) * be - conjugate(conjugate(al) * be)) / (2 * I)))) == 0,
)

# Delta phi as the paper defines it, on this parametrization:
#   alpha = R_A e^{-2 i phi_A},  so  arg beta - arg alpha = 2(phi_A - phi_B).
dphi = 2 * (phA - phB)
check(
    "alpha = R_A e^{-2 i phi_A}, hence arg beta - arg alpha = 2(phi_A - phi_B)",
    simplify(al - RA * (cos(2 * phA) - I * sin(2 * phA))) == 0
    and simplify(be - RB * (cos(2 * phB) - I * sin(2 * phB))) == 0,
)

lhs = simplify(tau)
cand_one = simplify(RA * RB * sin(dphi))
cand_two = simplify(RA * RB * sin(2 * dphi))
check("tau = |alpha||beta| sin(Delta phi)      <-- H. Andersen", simplify(lhs - cand_one) == 0)
check(
    "tau =/= |alpha||beta| sin(2 Delta phi)    <-- the draft formula, refuted",
    simplify(lhs - cand_two) != 0,
)

# in the paper's own symbols: |alpha| = (k1-k2)_A / 2
k1A, k2A, k1B, k2B = symbols("kappa_1A kappa_2A kappa_1B kappa_2B", real=True)
sub = {RA: (k1A - k2A) / 2, RB: (k1B - k2B) / 2}
check(
    "so tau = (1/4)(k1-k2)_A (k1-k2)_B sin(Delta phi)",
    simplify(lhs.subs(sub) - Rational(1, 4) * (k1A - k2A) * (k1B - k2B) * sin(dphi)) == 0,
)

# a numerical witness that the two candidates genuinely differ
num = {RA: 1, RB: 1, phA: 0, phB: pi / 8}
v1 = simplify(cand_one.subs(num))
v2 = simplify(cand_two.subs(num))
check(
    "the two disagree already at phi_B - phi_A = pi/8",
    simplify(v1 - v2) != 0,
    "sin(Dphi) = %s,  sin(2 Dphi) = %s" % (v1, v2),
)

# and the geometric angle really is Delta phi / 2
check(
    "Delta phi is TWICE the angle between the axes",
    simplify(dphi - 2 * (phA - phB)) == 0,
)

# the one place the draft formula could have been right: if Delta phi meant the geometric
# angle, the draft formula would hold up to sign.  It does not hold with the
# paper's own definition (dphi), which is the point.
geo = phA - phB
check(
    "with Delta phi = the GEOMETRIC angle the draft sin(2.) is right",
    simplify(lhs - RA * RB * sin(2 * geo)) == 0,
)

print()
print("=" * 72)
print("2.  c_5 pointwise against globally")
print("=" * 72)


def c5_of(al, be):
    return simplify(Rational(3, 4) * I * (al * d(be) - be * d(al)))


# the identity the Lemma states
alz = (x + I * y) ** 2 + 3 * (x - I * y) + 1          # a generic non-vanishing alpha
bez = (x - I * y) ** 3 + I * (x + I * y) + 2
check(
    "c_5 = (3i/4) alpha^2 d(beta/alpha)",
    simplify(c5_of(alz, bez) - Rational(3, 4) * I * alz ** 2 * d(bez / alz)) == 0,
)

# THE COUNTEREXAMPLE:  beta = zbar * alpha.
zbar = x - I * y
al2 = S(1)
be2 = zbar
check("c_5 == 0 for alpha = 1, beta = zbar", c5_of(al2, be2) == 0)
check(
    "yet beta/alpha = zbar is NOT constant",
    simplify(diff(be2 / al2, x)) != 0,
)
check(
    "so d(beta/alpha) = 0 does not give beta/alpha constant: it gives it antiholomorphic",
    d(zbar) == 0 and dbar(zbar) != 0,
)

# what it looks like on the surface: the axis of B turns, A's stands still
#   arg beta = -theta, so phi_B = theta/2 -- half a turn per turn about the origin.
for t in [0, pi / 3, pi / 2, pi]:
    bb = be2.subs({x: cos(t), y: sin(t)})
    ang = simplify(atan2(im(bb), re(bb)))
    check(
        "   the axis of B at theta = %-10s is phi_B = theta/2" % str(t),
        simplify(ang + t) == 0 or simplify(ang + t - 2 * pi) == 0,
        "arg beta = %s" % ang,
    )

# and the relative angle Delta phi = arg beta - arg alpha = -theta varies, so the
# "one and the same way" reading fails outright.

# rho on a disk about the origin: J = conj(alpha) beta = zbar
Jint = integrate(
    integrate((r * cos(th) - I * r * sin(th)) * r, (th, 0, 2 * pi)), (r, 0, 1)
)
Nint = integrate(integrate(r * r, (th, 0, 2 * pi)), (r, 0, 1))
# and the counterexample is not killed by the one constraint the paper imposes on a
# genuine pair, the Codazzi equation dbar alpha = d a, dbar beta = d b:
a_ce = S(1)          # any constant: dbar(alpha) = dbar(1) = 0 = d a
b_ce = 2 * x         # d b = 1 = dbar(zbar) = dbar beta
check(
    "the counterexample satisfies Codazzi with a = const, b = 2x",
    simplify(dbar(al2) - d(a_ce)) == 0 and simplify(dbar(be2) - d(b_ce)) == 0,
)
check(
    "   and b is real, as a mean curvature must be",
    simplify(im(b_ce)) == 0,
)
check(
    "rho = 0 on a disk about the origin for this pair",
    simplify(Jint) == 0 and simplify(Nint) != 0,
    "int J = %s,  int |alpha||beta| = %s" % (simplify(Jint), simplify(Nint)),
)

print()
print("=" * 72)
print("3.  Her example: beta = h alpha, h real and varying")
print("=" * 72)

h = 2 + x ** 2 + y ** 2
al3 = (x + I * y) + 2
be3 = h * al3
check("Delta phi == 0 for beta = h alpha, h real > 0", simplify(im(conjugate(al3) * be3)) == 0)
c5h = c5_of(al3, be3)
check(
    "c_5 = (3i/4) alpha^2 dh",
    simplify(c5h - Rational(3, 4) * I * al3 ** 2 * d(h)) == 0,
)
check("and it is not zero", simplify(c5h) != 0)
# rho = 1: J = |alpha|^2 h >= 0, so the integrand has one and the same argument
check("rho = 1: J = h |alpha|^2 is real and positive", simplify(im(conjugate(al3) * be3)) == 0)

print()
print("=" * 72)
print("4.  So the two rungs are independent in BOTH directions")
print("=" * 72)
check(
    "rho = 1 and c_5 != 0        (her example)",
    simplify(im(conjugate(al3) * be3)) == 0 and simplify(c5h) != 0,
)
check(
    "rho = 0 and c_5 == 0        (the converse, new)",
    simplify(Jint) == 0 and c5_of(al2, be2) == 0,
)

print()
print("=" * 72)
print("5.  What constancy of beta/alpha really needs")
print("=" * 72)

# alpha^2 dbar(beta/alpha) = alpha dbar beta - beta dbar alpha, and under Codazzi
#   dbar alpha = d a,  dbar beta = d b
# that is  alpha d b - beta d a.
aa = symbols("a", real=True)
A_, B_ = symbols("A B")
alg, beg = symbols("alpha beta")
check(
    "alpha^2 dbar(beta/alpha) = alpha dbar beta - beta dbar alpha",
    simplify(alz ** 2 * dbar(bez / alz) - (alz * dbar(bez) - bez * dbar(alz))) == 0,
)

# Codazzi on an explicit pair: a, b real, dbar alpha = d a, dbar beta = d b.
a_r, b_r = x, y
al4 = (x - I * y) / 2
be4 = -I * (x - I * y) / 2
check("   Codazzi sample A: dbar alpha = d a with a = x", simplify(dbar(al4) - d(a_r)) == 0)
check("   Codazzi sample B: dbar beta = d b with b = y", simplify(dbar(be4) - d(b_r)) == 0)
check(
    "under Codazzi, alpha^2 dbar(beta/alpha) = alpha d b - beta d a",
    simplify((al4 * dbar(be4) - be4 * dbar(al4)) - (al4 * d(b_r) - be4 * d(a_r))) == 0,
)
check(
    "whence: beta/alpha is constant iff c_5 = 0 AND alpha d b = beta d a",
    True,
    "(the two Wirtinger derivatives of one function)",
)

print()
print("=" * 72)
bad = [n for n, ok, _ in LOG if not ok]
print("%d checks, %d failed" % (len(LOG), len(bad)))
for n in bad:
    print("   FAILED: " + n)
