"""Exact arithmetic in Q(zeta_24) and Z-lattices in C^n with a transcendental
modulus tau, for the additional-lattices computation of Sec. 4.7.7 of [P].

Field: Q(z), z = zeta_24, minimal polynomial Phi_24 = x^8 - x^4 + 1.
It contains i = z^6, omega = z^8, sqrt2 = z^3 + z^21, sqrt3 = z^2 + z^22.

Lattice elements are polynomials in tau (degree <= DEGTAU) with coefficients in
C^n over Q(z); tau is treated as transcendental, so "coefficient of tau^k" is
exact bookkeeping and never a numerical approximation.
"""
from fractions import Fraction as Fr

DEG = 8                      # degree of Q(zeta_24) over Q


# ------------------------------------------------------------------ the field
def F(*c):
    c = list(c) + [0] * (DEG - len(c))
    return tuple(Fr(x) for x in c[:DEG])


ZERO = F(0)
ONE = F(1)


def fadd(a, b):
    return tuple(x + y for x, y in zip(a, b))


def fsub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def fneg(a):
    return tuple(-x for x in a)


def fscal(q, a):
    q = Fr(q)
    return tuple(q * x for x in a)


def fmul(a, b):
    r = [Fr(0)] * (2 * DEG - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                if y:
                    r[i + j] += x * y
    # x^8 = x^4 - 1
    for k in range(2 * DEG - 2, DEG - 1, -1):
        if r[k]:
            c = r[k]
            r[k] = Fr(0)
            r[k - DEG + 4] += c
            r[k - DEG] -= c
    return tuple(r[:DEG])


def fis0(a):
    return all(x == 0 for x in a)


def feq(a, b):
    return fis0(fsub(a, b))


# powers of z
def zpow(k):
    k %= 24
    r = ONE
    x = F(0, 1)
    for _ in range(k):
        r = fmul(r, x)
    return r


ZP = [zpow(k) for k in range(24)]

I = ZP[6]
OMEGA = ZP[8]                       # e^{2 pi i/3}
SQ2 = fadd(ZP[3], ZP[21])
SQ3 = fadd(ZP[2], ZP[22])


def fconj(a):
    """complex conjugation: z -> z^{-1}."""
    r = ZERO
    for k, x in enumerate(a):
        if x:
            r = fadd(r, fscal(x, ZP[(-k) % 24]))
    return r


def _solve(mat, rhs):
    """exact rational solve; mat is list of rows, returns solution or None."""
    m = [row[:] + [rhs[k]] for k, row in enumerate(mat)]
    rows, cols = len(m), len(m[0]) - 1
    piv = []
    r = 0
    for c in range(cols):
        p = None
        for k in range(r, rows):
            if m[k][c] != 0:
                p = k
                break
        if p is None:
            continue
        m[r], m[p] = m[p], m[r]
        inv = Fr(1) / m[r][c]
        m[r] = [x * inv for x in m[r]]
        for k in range(rows):
            if k != r and m[k][c] != 0:
                f = m[k][c]
                m[k] = [x - f * y for x, y in zip(m[k], m[r])]
        piv.append(c)
        r += 1
        if r == rows:
            break
    for k in range(r, rows):
        if m[k][cols] != 0:
            return None
    sol = [Fr(0)] * cols
    for k, c in enumerate(piv):
        sol[c] = m[k][cols]
    return sol


def finv(a):
    """inverse in Q(z): solve a*x = 1 by linear algebra on the basis."""
    cols = []
    for k in range(DEG):
        e = [Fr(0)] * DEG
        e[k] = Fr(1)
        cols.append(fmul(a, tuple(e)))
    mat = [[cols[j][i] for j in range(DEG)] for i in range(DEG)]
    rhs = [Fr(1)] + [Fr(0)] * (DEG - 1)
    sol = _solve(mat, rhs)
    if sol is None:
        raise ZeroDivisionError("not invertible")
    return tuple(sol)


def fdiv(a, b):
    return fmul(a, finv(b))


def fnum(a):
    import cmath
    z = cmath.exp(2j * cmath.pi / 24)
    return sum(complex(x) * z ** k for k, x in enumerate(a))


def fstr(a):
    return "%.6f%+.6fi" % (fnum(a).real, fnum(a).imag)


# ------------------------------------------------------- vectors and matrices
def vadd(v, w):
    return tuple(fadd(a, b) for a, b in zip(v, w))


def vsub(v, w):
    return tuple(fsub(a, b) for a, b in zip(v, w))


def vscal(c, v):
    return tuple(fmul(c, a) for a in v)


def herm(v, w):
    """<v|w> = sum v_j conj(w_j), linear in the first argument."""
    r = ZERO
    for a, b in zip(v, w):
        r = fadd(r, fmul(a, fconj(b)))
    return r


def mat_apply(M, v):
    return tuple(
        tuple_sum(fmul(M[i][j], v[j]) for j in range(len(v))) for i in range(len(M)))


def tuple_sum(it):
    r = ZERO
    for x in it:
        r = fadd(r, x)
    return r


def mat_mul(A, B):
    n = len(A)
    return tuple(tuple(tuple_sum(fmul(A[i][k], B[k][j]) for k in range(n))
                       for j in range(n)) for i in range(n))


def mat_id(n):
    return tuple(tuple(ONE if i == j else ZERO for j in range(n)) for i in range(n))


def mat_eq(A, B):
    return all(feq(a, b) for ra, rb in zip(A, B) for a, b in zip(ra, rb))


def mat_inv(A):
    """inverse of a matrix over Q(z), by solving n^2 rational equations."""
    n = len(A)
    N = n * DEG
    # build the Q-linear map x -> A x on C^n viewed as Q^{n*DEG}
    cols = []
    for j in range(n):
        for d in range(DEG):
            e = [Fr(0)] * DEG
            e[d] = Fr(1)
            v = tuple(tuple(e) if k == j else ZERO for k in range(n))
            Av = mat_apply(A, v)
            cols.append([x for comp in Av for x in comp])
    mat = [[cols[c][r] for c in range(N)] for r in range(N)]
    inv_cols = []
    for j in range(n):
        for d in range(DEG):
            rhs = [Fr(0)] * N
            rhs[j * DEG + d] = Fr(1)
            sol = _solve([row[:] for row in mat], rhs)
            if sol is None:
                raise ZeroDivisionError("singular matrix")
            inv_cols.append(sol)
    # column (j,d) of the inverse gives the image of the basis vector
    B = [[ZERO] * n for _ in range(n)]
    for jj, col in enumerate(inv_cols):
        j, d = divmod(jj, DEG)
        if d != 0:
            continue
        for i in range(n):
            B[i][j] = tuple(col[i * DEG:(i + 1) * DEG])
    return tuple(tuple(row) for row in B)


def mat_trace(A):
    return tuple_sum(A[i][i] for i in range(len(A)))


def refl(e, mu, n):
    """R v = v - (1-mu) <v|e> e, in matrix form."""
    one_minus = fsub(ONE, mu)
    rows = []
    for i in range(n):
        row = []
        for j in range(n):
            # image of the j-th basis vector
            c = ONE if i == j else ZERO
            c = fsub(c, fmul(fmul(one_minus, fconj(e[j])), e[i]))
            row.append(c)
        rows.append(tuple(row))
    return tuple(rows)


def group_gen(gens, limit=200000):
    n = len(gens[0])
    seen = {mat_id(n)}
    frontier = [mat_id(n)]
    while frontier:
        new = []
        for g in frontier:
            for s in gens:
                h = mat_mul(s, g)
                if h not in seen:
                    seen.add(h)
                    new.append(h)
                    if len(seen) > limit:
                        raise RuntimeError("group too big")
        frontier = new
    return seen
