"""Exact arithmetic in the algebra of Laurent polynomials in z = e^{ia/2}, over Q(i).

x = z^2 = e^{ia} is the coordinate on the supercircle S; a is the angle.
Even powers of z are the periodic functions (elements of R(0) = C[x,x^{-1}]);
odd powers are the anti-periodic ones (sections of the Moebius bundle),
which is what the sqrt(x) of the k^M-realization needs.

A Laurent polynomial is a dict {int exponent of z: sympy number}.
The one operation that is NOT a ring operation is `star`: it conjugates the
COEFFICIENTS and leaves z alone.  Composed with a substitution z -> sigma(z)
it is the anti-linear automorphism phi of the paper.

No floating point anywhere: coefficients are sympy Rationals + I.
"""

from sympy import Rational, I, simplify, expand, conjugate, nsimplify, S

# ---------------------------------------------------------------- Laurent ring


def L(*pairs):
    """L((exp, coeff), ...) -> Laurent polynomial."""
    d = {}
    for e, c in pairs:
        c = S(c)
        if c != 0:
            d[e] = d.get(e, 0) + c
    return {e: expand(c) for e, c in d.items() if expand(c) != 0}


ZERO = {}
ONE = {0: S(1)}


def const(c):
    c = S(c)
    return {} if c == 0 else {0: c}


def zpow(n, c=1):
    c = S(c)
    return {} if c == 0 else {n: c}


def add(*fs):
    d = {}
    for f in fs:
        for e, c in f.items():
            d[e] = d.get(e, 0) + c
    return {e: expand(c) for e, c in d.items() if expand(c) != 0}


def neg(f):
    return {e: -c for e, c in f.items()}


def sub(f, g):
    return add(f, neg(g))


def mul(*fs):
    r = ONE
    for f in fs:
        d = {}
        for e1, c1 in r.items():
            for e2, c2 in f.items():
                d[e1 + e2] = d.get(e1 + e2, 0) + c1 * c2
        r = {e: expand(c) for e, c in d.items() if expand(c) != 0}
    return r


def smul(c, f):
    c = S(c)
    return {} if c == 0 else {e: expand(c * cc) for e, cc in f.items()}


def is_zero(f):
    return all(expand(c) == 0 for c in f.values())


def eq(f, g):
    return is_zero(sub(f, g))


def inv(f):
    """Inverse of a monomial c*z^e (the only invertibles we need)."""
    items = [(e, c) for e, c in f.items() if expand(c) != 0]
    if len(items) != 1:
        raise ValueError("not a monomial: %s" % pretty(f))
    e, c = items[0]
    return {-e: expand(1 / c)}


def star(f):
    """Conjugate the coefficients; z is left alone."""
    return {e: expand(conjugate(c)) for e, c in f.items()}


def subst(f, lam, s):
    """z -> lam * z^s  (s = +1 or -1)."""
    lam = S(lam)
    return {e * s: expand(c * lam ** e) for e, c in f.items()}


def ddx(f):
    """d/dx, where x = z^2, i.e. (1/(2z)) d/dz."""
    d = {}
    for e, c in f.items():
        if e != 0:
            d[e - 2] = d.get(e - 2, 0) + Rational(e, 2) * c
    return {e: expand(c) for e, c in d.items() if expand(c) != 0}


def pretty(f):
    if not f:
        return "0"
    out = []
    for e in sorted(f):
        c = f[e]
        out.append("(%s)z^%d" % (c, e))
    return " + ".join(out)


def periodic(f):
    """True if f is a function on the circle (even powers of z only)."""
    return all(e % 2 == 0 for e in f)


# ------------------------------------------------------------------ matrices


def mzero(n, m=None):
    m = n if m is None else m
    return [[ZERO for _ in range(m)] for _ in range(n)]


def mid(n, f=None):
    f = ONE if f is None else f
    return [[f if i == j else ZERO for j in range(n)] for i in range(n)]


def mdiag(*fs):
    n = len(fs)
    return [[fs[i] if i == j else ZERO for j in range(n)] for i in range(n)]


def mblock(*blocks):
    n = sum(len(b) for b in blocks)
    M = mzero(n)
    o = 0
    for b in blocks:
        k = len(b)
        for i in range(k):
            for j in range(k):
                M[o + i][o + j] = b[i][j]
        o += k
    return M


def mmul(*Ms):
    R = Ms[0]
    for M in Ms[1:]:
        n, k, m = len(R), len(M), len(M[0])
        R = [[add(*[mul(R[i][t], M[t][j]) for t in range(k)]) for j in range(m)]
             for i in range(n)]
    return R


def mt(M):
    return [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]


def mstar(M):
    return [[star(c) for c in row] for row in M]


def msubst(M, lam, s):
    return [[subst(c, lam, s) for c in row] for row in M]


def msmul(c, M):
    return [[smul(c, e) for e in row] for row in M]


def mmulf(f, M):
    return [[mul(f, e) for e in row] for row in M]


def madd(*Ms):
    n, m = len(Ms[0]), len(Ms[0][0])
    return [[add(*[M[i][j] for M in Ms]) for j in range(m)] for i in range(n)]


def msub(A, B):
    return madd(A, msmul(-1, B))


def meq(A, B):
    return all(eq(A[i][j], B[i][j]) for i in range(len(A)) for j in range(len(A[0])))


def mdet(M):
    n = len(M)
    if n == 1:
        return M[0][0]
    tot = ZERO
    for j in range(n):
        minor = [[M[i][k] for k in range(n) if k != j] for i in range(1, n)]
        term = mul(M[0][j], mdet(minor))
        tot = add(tot, smul((-1) ** j, term))
    return tot


def minv(M):
    """Inverse via adjugate; det must be a monomial."""
    n = len(M)
    d = mdet(M)
    di = inv(d)
    if n == 1:
        return [[di]]
    C = mzero(n)
    for i in range(n):
        for j in range(n):
            minor = [[M[r][c] for c in range(n) if c != j] for r in range(n) if r != i]
            C[j][i] = smul((-1) ** (i + j), mul(mdet(minor), di))
    return C


def mpretty(M):
    return "[" + ",\n ".join("[" + ", ".join(pretty(c) for c in row) + "]" for row in M) + "]"


# ------------------------------------------------------- the standard objects

x = zpow(2)          # x = z^2
xinv = zpow(-2)
z = zpow(1)


def cosn(n):
    """cos(n a / 2) as a Laurent polynomial in z."""
    return L((n, Rational(1, 2)), (-n, Rational(1, 2)))


def sinn(n):
    """sin(n a / 2)."""
    return L((n, -I / 2), (-n, I / 2))


def tau(n=2):
    """tau(a) = [[cos a, sin a], [-sin a, cos a]] for n = 2 (rotation)."""
    c, s = cosn(n), sinn(n)
    return [[c, s], [neg(s), c]]


def tauM(n=2):
    """tau^M(a) = [[cos a, sin a], [sin a, -cos a]] (reflection)."""
    c, s = cosn(n), sinn(n)
    return [[c, s], [s, neg(c)]]


J2 = [[ZERO, ONE], [smul(-1, ONE), ZERO]]


def Jn(m):
    """J_{2m}: block-diagonal with m copies of J2."""
    return mblock(*[J2] * m)
