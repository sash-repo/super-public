"""
slcheck.py -- exact check of clause 3 of the Theorem of

  [L2] D. Leites, "Irreducible representations of Lie superalgebras of vector
        fields and invariant differential operators", Funkts. Anal. Prilozh.
        16:1 (1982) 76-77,

namely:  "the operators Y_+, Y_- and E define on Omega and Omega^{(*)} the
SL(n)-invariant structures of sl(1|1)-modules",  where (loc. cit., p. 76-77)

  Y_+ = multiplication by the odd canonical 2-form omega_1,
  Y_- = interior multiplication by the conjugate bivector,
  E   = [Y_+, Y_-].

Here, unlike omcheck.py, the forms carry POLYNOMIAL COEFFICIENTS, because the
question that the fibre le(n)_0 = pe(n) cannot see is exactly the one [L2]
flags in a parenthesis: why SL(n) = sle(n) and not L(n) = le(n)?

Conventions (those of le.py and omcheck.py):

    x_1..x_n even, xi_1..xi_n odd,  omega = sum_i 'x_i 'xi_i,   'y := dy,
    p('y) = p(y) + 1,  so 'x_i is ODD and 'xi_i is EVEN.

A basis monomial of Omega is the normally ordered word

    (a, S, A, B)  =  x^a  xi^S  'x^A  'xi^B

with a, B in N^n (even letters) and S, A sorted tuples of indices (odd letters).

    Y_+ = omega .            (odd, form degree +2)
    Y_- = sum_i d/d'xi_i d/d'x_i    (odd, form degree -2)
    E   = Y_+ Y_- + Y_- Y_+  (even, form degree 0)

Everything is over Q (Fraction); the only dependency is le.py (for D_f).
"""

from fractions import Fraction as F
from itertools import combinations

from le import D as VF_of, dx as pdx, dxi as pdxi, merge_sign

# --------------------------------------------------------------- monomials


def fdeg(m):
    """form degree."""
    return len(m[2]) + sum(m[3])


def cdeg(m):
    """degree of the coefficient."""
    return sum(m[0]) + len(m[1])


def par(m):
    return (len(m[1]) + len(m[2])) % 2


def add(d, m, c):
    if c == 0:
        return
    d[m] = d.get(m, F(0)) + c
    if d[m] == 0:
        del d[m]


def addall(d, e, c=1):
    for m, v in e.items():
        add(d, m, c * v)
    return d


def scal(e, c):
    return {m: c * v for m, v in e.items() if c * v != 0}


ZERO = ()


def zeros(n):
    return (0,) * n


def one(n):
    return {(zeros(n), (), (), zeros(n)): F(1)}


def mmul(n, m1, m2):
    """product of two normally ordered monomials: (sign, monomial) or None."""
    a1, S1, A1, B1 = m1
    a2, S2, A2, B2 = m2
    sgnS, S = merge_sign(S1, S2)
    if sgnS is None:
        return None
    sgnA, A = merge_sign(A1, A2)
    if sgnA is None:
        return None
    # the odd letters xi^S2 have to be carried past the odd letters 'x^A1
    sign = sgnS * sgnA * (-1) ** (len(S2) * len(A1))
    a = tuple(u + v for u, v in zip(a1, a2))
    B = tuple(u + v for u, v in zip(B1, B2))
    return sign, (a, S, A, B)


def emul(n, e1, e2):
    """product of two elements of Omega."""
    out = {}
    for m1, c1 in e1.items():
        for m2, c2 in e2.items():
            r = mmul(n, m1, m2)
            if r is None:
                continue
            s, m = r
            add(out, m, s * c1 * c2)
    return out


# ------------------------------------------------------------- derivations
# A derivation is given by its parity p and by delta(letter) for each of the
# 4n letters; letters are named ('x', i), ('xi', i), ("'x", i), ("'xi", i).


def derive(n, p, dgen, e):
    """apply the derivation of parity p with delta(letter) = dgen(kind, i)."""
    out = {}
    for (a, S, A, B), c in e.items():
        # ---- the even letters x_i (at the front: no odd letter precedes them)
        for i in range(n):
            if a[i] == 0:
                continue
            g = dgen("x", i)
            if not g:
                continue
            a2 = list(a)
            a2[i] -= 1
            rest = {(tuple(a2), S, A, B): F(a[i]) * c}
            addall(out, emul(n, g, rest))
        # ---- the odd letters xi_s
        for pos, s in enumerate(S):
            g = dgen("xi", s)
            if not g:
                continue
            left = {(a, S[:pos], (), zeros(n)): F(1)}
            right = {(zeros(n), S[pos + 1:], A, B): F(1)}
            sgn = (-1) ** (p * pos)
            addall(out, emul(n, emul(n, left, g), right), sgn * c)
        # ---- the odd letters 'x_t
        for pos, t in enumerate(A):
            g = dgen("'x", t)
            if not g:
                continue
            left = {(a, S, A[:pos], zeros(n)): F(1)}
            right = {(zeros(n), (), A[pos + 1:], B): F(1)}
            sgn = (-1) ** (p * (len(S) + pos))
            addall(out, emul(n, emul(n, left, g), right), sgn * c)
        # ---- the even letters 'xi_i (at the end: all of S and A precede them)
        sgn = (-1) ** (p * (len(S) + len(A)))
        for i in range(n):
            if B[i] == 0:
                continue
            g = dgen("'xi", i)
            if not g:
                continue
            B2 = list(B)
            B2[i] -= 1
            left = {(a, S, A, tuple(B2)): F(B[i])}
            addall(out, emul(n, left, g), sgn * c)
    return out


def _lift(n, poly):
    """a polynomial in x, xi (le.py format) as an element of Omega^0."""
    return {(a, S, (), zeros(n)): F(c) for (a, S), c in poly.items()}


def d_ext(n, e):
    """the exterior differential: odd, d(x_i) = 'x_i, d(xi_i) = 'xi_i."""
    def dgen(kind, i):
        if kind == "x":
            return {(zeros(n), (), (i,), zeros(n)): F(1)}
        if kind == "xi":
            B = [0] * n
            B[i] = 1
            return {(zeros(n), (), (), tuple(B)): F(1)}
        return {}
    return derive(n, 1, dgen, e)


def lie(n, vf, e):
    """the Lie derivative L_D along the vector field vf (a le.VF).

    L_D is the derivation of Omega of parity p(D) with L_D|_functions = D and

        L_D('y) = (-1)^{p(D)} d(D(y)),

    the sign being forced by [L_D, d] = 0 in the graded sense, i.e. by
    L_D d = (-1)^{p(D)} d L_D  (self-tests S3 and S6 in slrun.py).
    """
    imx = [vf.comp.get(i, {}) for i in range(n)]          # D(x_i)
    imxi = [vf.comp.get(n + i, {}) for i in range(n)]      # D(xi_i)
    eps = (-1) ** vf.par
    Lx = [scal(d_ext(n, _lift(n, u)), eps) for u in imx]
    Lxi = [scal(d_ext(n, _lift(n, u)), eps) for u in imxi]

    def dgen(kind, i):
        if kind == "x":
            return _lift(n, imx[i])
        if kind == "xi":
            return _lift(n, imxi[i])
        if kind == "'x":
            return Lx[i]
        return Lxi[i]

    return derive(n, vf.par, dgen, e)


def d_form(n, kind, i, e):
    """the partial derivative d/d'x_i (odd) or d/d'xi_i (even)."""
    p = 1 if kind == "'x" else 0

    def dgen(k, j):
        if k == kind and j == i:
            return one(n)
        return {}

    return derive(n, p, dgen, e)


# -------------------------------------------------------- the three operators


def omega(n):
    out = {}
    for i in range(n):
        B = [0] * n
        B[i] = 1
        add(out, (zeros(n), (), (i,), tuple(B)), F(1))
    return out


def Yplus(n, e):
    return emul(n, omega(n), e)


def Yminus(n, e):
    out = {}
    for i in range(n):
        addall(out, d_form(n, "'xi", i, d_form(n, "'x", i, e)))
    return out


def Ecart(n, e):
    """E = [Y_+, Y_-] = Y_+ Y_- + Y_- Y_+ (both odd)."""
    out = Yplus(n, Yminus(n, e))
    addall(out, Yminus(n, Yplus(n, e)))
    return out


def comm(n, U, pU, V, pV, e):
    """[U, V] e = U V e - (-1)^{pU pV} V U e."""
    out = U(n, V(n, e))
    addall(out, V(n, U(n, e)), -((-1) ** (pU * pV)))
    return out


# ------------------------------------------------------------- test material


def monomials(n, cmax, fmax):
    """all basis monomials with cdeg <= cmax and fdeg <= fmax."""
    out = []
    for da in range(cmax + 1):
        for a in _comps(n, da):
            for k in range(min(n, cmax - da) + 1):
                for S in combinations(range(n), k):
                    for j in range(min(n, fmax) + 1):
                        for A in combinations(range(n), j):
                            for db in range(fmax - j + 1):
                                for B in _comps(n, db):
                                    out.append((a, S, A, B))
    return out


def _comps(n, total):
    if n == 0:
        return [()] if total == 0 else []
    out = []
    for first in range(total + 1):
        for rest in _comps(n - 1, total - first):
            out.append((first,) + rest)
    return out


def gen_functions(n, dmax):
    """all monomial generating functions f of degree 2..dmax."""
    out = []
    for d in range(2, dmax + 1):
        for k in range(min(n, d) + 1):
            for S in combinations(range(n), k):
                for a in _comps(n, d - k):
                    out.append({(a, S): F(1)})
    return out


def laplace(n, f):
    """Delta f = sum_i d/dxi_i d/dx_i f  (le(n) -> sle(n) is Delta f = 0)."""
    out = {}
    for i in range(n):
        for m, c in pdxi(pdx(f, i), i).items():
            out[m] = out.get(m, 0) + c
    return {m: c for m, c in out.items() if c}


def fname(n, f):
    (a, S), _ = next(iter(f.items()))
    s = "".join("x%d%s" % (i + 1, "" if a[i] == 1 else "^%d" % a[i])
                for i in range(n) if a[i])
    s += "".join("xi%d" % (i + 1) for i in S)
    return s
