"""Exact arithmetic in Q(zeta_8) and Z-lattices in C^n, for the invariant-lattice
computation of Popov's Sections 4.7-4.8.

Everything is exact: field elements are 4-tuples of Fractions in the basis
1, z, z^2, z^3 with z = zeta_8 (z^4 = -1); i = z^2, sqrt2 = z - z^3.
"""
from fractions import Fraction as Fr
from itertools import product

# ---------------------------------------------------------------- field Q(z8)
DEG = 4


def F(*c):
    c = list(c) + [0] * (DEG - len(c))
    return tuple(Fr(x) for x in c[:DEG])


ZERO = F(0)
ONE = F(1)
Z8 = F(0, 1)
I = F(0, 0, 1)          # zeta_8^2
SQ2 = F(0, 1, 0, -1)     # z - z^3 = sqrt 2


def fadd(a, b):
    return tuple(x + y for x, y in zip(a, b))


def fsub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def fneg(a):
    return tuple(-x for x in a)


def fmul(a, b):
    r = [Fr(0)] * (2 * DEG - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                if y:
                    r[i + j] += x * y
    for k in range(2 * DEG - 2, DEG - 1, -1):
        if r[k]:
            r[k - DEG] -= r[k]      # z^4 = -1
            r[k] = Fr(0)
    return tuple(r[:DEG])


def fconj(a):
    # z -> z^{-1}: 1->1, z->-z^3, z^2->-z^2, z^3->-z
    return (a[0], -a[3], -a[2], -a[1])


def fscal(q, a):
    q = Fr(q)
    return tuple(q * x for x in a)


def fis0(a):
    return all(x == 0 for x in a)


def fmatmulvec(M, v):
    # M list of rows (field elts), v vector
    return tuple(_dot(row, v) for row in M)


def _dot(u, v):
    s = ZERO
    for a, b in zip(u, v):
        s = fadd(s, fmul(a, b))
    return s


def finv(a):
    # solve a*x = 1 by 4x4 rational system
    cols = []
    for k in range(DEG):
        ek = [Fr(0)] * DEG
        ek[k] = Fr(1)
        cols.append(fmul(a, tuple(ek)))
    # matrix M with M[r][k] = coefficient r of a*z^k ; solve M x = e_0
    M = [[cols[k][r] for k in range(DEG)] for r in range(DEG)]
    rhs = [Fr(1)] + [Fr(0)] * (DEG - 1)
    x = solve_rat(M, rhs)
    return tuple(x)


def fdiv(a, b):
    return fmul(a, finv(b))


def herm(u, v):
    """<u|v> = sum u_k conj(v_k), linear in u."""
    s = ZERO
    for a, b in zip(u, v):
        s = fadd(s, fmul(a, fconj(b)))
    return s


def fnum(a):
    """complex float value (for sanity checks)"""
    import cmath
    z = cmath.exp(1j * cmath.pi / 4)
    return sum(complex(x) * z ** k for k, x in enumerate(a))


def fstr(a):
    names = ['1', 'z', 'i', 'z^3']
    parts = [f"{x}*{n}" for x, n in zip(a, names) if x]
    return " + ".join(parts) if parts else "0"


# ------------------------------------------------------------ rational solver
def solve_rat(M, rhs):
    n = len(M)
    m = len(M[0])
    A = [row[:] + [rhs[r]] for r, row in enumerate(M)]
    piv = []
    r = 0
    for c in range(m):
        p = None
        for rr in range(r, n):
            if A[rr][c] != 0:
                p = rr
                break
        if p is None:
            continue
        A[r], A[p] = A[p], A[r]
        pv = A[r][c]
        A[r] = [x / pv for x in A[r]]
        for rr in range(n):
            if rr != r and A[rr][c] != 0:
                f = A[rr][c]
                A[rr] = [x - f * y for x, y in zip(A[rr], A[r])]
        piv.append(c)
        r += 1
        if r == n:
            break
    for rr in range(r, n):
        if A[rr][m] != 0:
            raise ValueError("inconsistent system")
    x = [Fr(0)] * m
    for k, c in enumerate(piv):
        x[c] = A[k][m]
    return x


# --------------------------------------------------- vectors in C^n as Q^{4n}
def vec_to_coords(v):
    out = []
    for a in v:
        out.extend(a)
    return out


def coords_to_vec(c, n):
    return tuple(tuple(c[DEG * k:DEG * (k + 1)]) for k in range(n))


# --------------------------------------------------------------- HNF over Z
def hnf_rows(rows):
    """Row-style HNF of a list of integer rows; returns list of nonzero rows,
    upper triangular w.r.t. pivot columns, positive pivots, entries above pivot
    reduced."""
    A = [list(r) for r in rows if any(r)]
    if not A:
        return []
    m = len(A[0])
    res = []
    r = 0
    for c in range(m):
        # find rows (from r on) with nonzero in column c, reduce by gcd
        while True:
            nz = [k for k in range(r, len(A)) if A[k][c] != 0]
            if len(nz) <= 1:
                break
            nz.sort(key=lambda k: abs(A[k][c]))
            k0 = nz[0]
            for k in nz[1:]:
                q = A[k][c] // A[k0][c]
                A[k] = [x - q * y for x, y in zip(A[k], A[k0])]
        nz = [k for k in range(r, len(A)) if A[k][c] != 0]
        if not nz:
            continue
        k0 = nz[0]
        A[r], A[k0] = A[k0], A[r]
        if A[r][c] < 0:
            A[r] = [-x for x in A[r]]
        for k in range(r):
            q = A[k][c] // A[r][c]
            if q:
                A[k] = [x - q * y for x, y in zip(A[k], A[r])]
        r += 1
        if r == len(A):
            break
    return [row for row in A[:r] if any(row)]


def hnf_with_transform(rows):
    """returns (H, U) with U*A = H (H in row HNF, U unimodular)."""
    n = len(rows)
    m = len(rows[0]) if n else 0
    A = [list(r) for r in rows]
    U = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    r = 0
    for c in range(m):
        while True:
            nz = [k for k in range(r, n) if A[k][c] != 0]
            if len(nz) <= 1:
                break
            nz.sort(key=lambda k: abs(A[k][c]))
            k0 = nz[0]
            for k in nz[1:]:
                q = A[k][c] // A[k0][c]
                A[k] = [x - q * y for x, y in zip(A[k], A[k0])]
                U[k] = [x - q * y for x, y in zip(U[k], U[k0])]
        nz = [k for k in range(r, n) if A[k][c] != 0]
        if not nz:
            continue
        k0 = nz[0]
        A[r], A[k0] = A[k0], A[r]
        U[r], U[k0] = U[k0], U[r]
        if A[r][c] < 0:
            A[r] = [-x for x in A[r]]
            U[r] = [-x for x in U[r]]
        r += 1
        if r == n:
            break
    return A, U


# ------------------------------------------------------------------- lattices
class Lat:
    """Z-lattice given by rational generators in Q^m (rows)."""

    def __init__(self, gens):
        gens = [[Fr(x) for x in g] for g in gens]
        den = 1
        for g in gens:
            for x in g:
                den = den * x.denominator // _gcd(den, x.denominator)
        ints = [[int(x * den) for x in g] for g in gens]
        H = hnf_rows(ints)
        # reduce den
        if H:
            g = 0
            for row in H:
                for x in row:
                    g = _gcd(g, abs(x))
            d = _gcd(g, den)
            if d > 1:
                H = [[x // d for x in row] for row in H]
                den //= d
        self.den = den
        self.H = H

    @property
    def rank(self):
        return len(self.H)

    def basis(self):
        return [[Fr(x, self.den) for x in row] for row in self.H]

    def __eq__(self, o):
        return self.den == o.den and self.H == o.H

    def __hash__(self):
        return hash((self.den, tuple(map(tuple, self.H))))

    def contains_vec(self, v):
        v = [Fr(x) for x in v]
        try:
            x = coeffs_in(self, v)
        except ValueError:
            return False
        return all(t.denominator == 1 for t in x)

    def contains(self, o):
        return all(self.contains_vec(b) for b in o.basis())

    def index_in(self, o):
        """[o : self] for self <= o, both of the same rank."""
        assert o.contains(self)
        A = [coeffs_in(o, b) for b in self.basis()]
        return abs(det_rat(A))


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)


def coeffs_in(L, v):
    """coordinates of v in the basis of L (rationals); raises if v not in span."""
    B = L.basis()
    # solve x B = v  ->  B^T x^T = v^T
    m = len(v)
    M = [[B[k][j] for k in range(len(B))] for j in range(m)]
    return solve_rat(M, [Fr(x) for x in v])


def det_rat(A):
    n = len(A)
    A = [row[:] for row in A]
    det = Fr(1)
    for c in range(n):
        p = None
        for r in range(c, n):
            if A[r][c] != 0:
                p = r
                break
        if p is None:
            return Fr(0)
        if p != c:
            A[c], A[p] = A[p], A[c]
            det = -det
        det *= A[c][c]
        pv = A[c][c]
        A[c] = [x / pv for x in A[c]]
        for r in range(c + 1, n):
            if A[r][c]:
                f = A[r][c]
                A[r] = [x - f * y for x, y in zip(A[r], A[c])]
    return det


def lat_sum(*Ls):
    gens = []
    for L in Ls:
        gens.extend(L.basis())
    return Lat(gens)


def lat_from_vecs(vs):
    return Lat([vec_to_coords(v) for v in vs])


def apply_mat(M, L, n):
    """image of lattice L (in C^n coords) under the C-linear map M."""
    out = []
    for b in L.basis():
        v = coords_to_vec(b, n)
        out.append(vec_to_coords(fmatmulvec(M, v)))
    return Lat(out)


def scale_lat(alpha, L, n):
    out = []
    for b in L.basis():
        v = coords_to_vec(b, n)
        out.append(vec_to_coords(tuple(fmul(alpha, x) for x in v)))
    return Lat(out)


def line_part(L, e, n):
    """L cap C e  (e a vector of C^n) as a lattice, plus the coefficient lattice."""
    # conditions: v_k * e_l - v_l * e_k = 0 for all k<l  -> Q-linear in coords of v
    B = L.basis()
    rows = []
    conds = []
    for k in range(n):
        for l in range(k + 1, n):
            conds.append((k, l))
    if not conds:
        return L
    Mrows = []
    for b in B:
        v = coords_to_vec(b, n)
        r = []
        for (k, l) in conds:
            val = fsub(fmul(v[k], e[l]), fmul(v[l], e[k]))
            r.extend(val)
        Mrows.append(r)
    den = 1
    for r in Mrows:
        for x in r:
            den = den * x.denominator // _gcd(den, x.denominator)
    Mint = [[int(x * den) for x in r] for r in Mrows]
    H, U = hnf_with_transform(Mint)
    ker = [U[k] for k in range(len(H)) if not any(H[k])]
    gens = []
    for x in ker:
        g = [Fr(0)] * len(B[0])
        for c, b in zip(x, B):
            if c:
                g = [gg + c * bb for gg, bb in zip(g, b)]
        gens.append(g)
    return Lat(gens) if gens else Lat([[Fr(0)] * len(B[0])])
