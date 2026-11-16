"""
omcheck.py -- exact check of the graph of [L3, Subsec. 2.4 b)]:
the structure of the superspace Omega of (integro)differential forms with CONSTANT
coefficients in n even and n odd variables as a module over L_0 = pe(n), together
with the action of the odd canonical form omega = sum_i dx_i dxi_i.

Conventions (same as le.py):

    x_1..x_n even, xi_1..xi_n odd,   omega = sum_i dx_i dxi_i,
    'y := dy  with  p('y) = p(y) + 1.

So 'x_i is ODD and 'xi_i is EVEN, and

    Omega(const) = Lambda('x_1..'x_n)  (x)  C['xi_1..'xi_n],
    Omega^i      = its degree-i part   (deg 'x = deg 'xi = 1).

Multiplication by omega is an odd operator raising the degree by 2; omega^2 = 0
because omega is odd in a supercommutative superalgebra, so (Omega, omega.) is a
complex, and [L3] computes its homology.

L_0 = pe(n) acts by the Lie derivative.  For D = D_f with f quadratic,

    D_f(x_k) = (-1)^{p(f)} df/dxi_k,      D_f(xi_k) = df/dx_k,
    L_D('y)  = d(D_f(y)),                 L_D a derivation of parity p(f)+1.

Everything is over Q (Fraction); no dependencies.
"""

from fractions import Fraction as F
from itertools import combinations

# --------------------------------------------------------------- monomials
# A basis monomial of Omega(const) is (A, B):
#   A = sorted tuple of indices of the odd generators 'x_a  (exterior)
#   B = tuple of length n of exponents of the even generators 'xi_i (symmetric)
# written as  'x_{a1} ... 'x_{ak} * ('xi)^B.


def deg(m):
    A, B = m
    return len(A) + sum(B)


def par(m):
    return len(m[0]) % 2


def basis(n, i):
    """all monomials of Omega^i."""
    out = []
    for k in range(0, min(n, i) + 1):
        for A in combinations(range(n), k):
            for B in compositions(n, i - k):
                out.append((A, B))
    return out


def compositions(n, total):
    """all tuples of n nonnegative ints summing to total."""
    if n == 0:
        return [()] if total == 0 else []
    out = []
    for first in range(total + 1):
        for rest in compositions(n - 1, total - first):
            out.append((first,) + rest)
    return out


def add(d, m, c):
    if c == 0:
        return
    d[m] = d.get(m, F(0)) + c
    if d[m] == 0:
        del d[m]


def mul_x(m, i):
    """'x_i * m  -- returns (sign, monomial) or None."""
    A, B = m
    if i in A:
        return None
    sign = (-1) ** sum(1 for a in A if a < i)
    return sign, (tuple(sorted(A + (i,))), B)


# ------------------------------------------------------------- the form omega


def omega_mul(n, v):
    """multiply the element v (dict monomial->coeff) by omega = sum_i 'x_i 'xi_i."""
    out = {}
    for (A, B), c in v.items():
        for i in range(n):
            r = mul_x((A, B), i)
            if r is None:
                continue
            sign, (A2, B2) = r
            B3 = list(B2)
            B3[i] += 1
            add(out, (A2, tuple(B3)), sign * c)
    return out


# ------------------------------------------------------- pe(n) by Lie derivative
# A generating function f is a dict {(a, S): coeff} exactly as in le.py, but we
# only ever need the quadratic ones, so we hard-code the three families.


def d_of_linear(n, lin):
    """d of a linear function given as (cx, cxi): coefficient lists.
    Returns an element of Omega^1."""
    cx, cxi = lin
    out = {}
    for j in range(n):
        if cx[j]:
            add(out, ((j,), (0,) * n), F(cx[j]))
        if cxi[j]:
            B = [0] * n
            B[j] = 1
            add(out, ((), tuple(B)), F(cxi[j]))
    return out


class Gen:
    """one quadratic generating function f of le(n), i.e. one element of pe(n).

    kind = 'gl'   : f = x_i xi_j          (p(f) = 1, so p(D) = 0)
    kind = 'lam'  : f = xi_i xi_j, i<j    (p(f) = 0, so p(D) = 1)
    kind = 'sym'  : f = x_i x_j,   i<=j   (p(f) = 0, so p(D) = 1)
    """

    def __init__(self, n, kind, i, j):
        self.n, self.kind, self.i, self.j = n, kind, i, j
        self.pf = 1 if kind == "gl" else 0
        self.pD = (self.pf + 1) % 2
        # images of the coordinates: D(x_k) and D(xi_k) as linear functions
        self.dx = [self._Dx(k) for k in range(n)]
        self.dxi = [self._Dxi(k) for k in range(n)]
        # images of the generators of Omega: L_D('x_k) and L_D('xi_k)
        self.Lx = [d_of_linear(n, self.dx[k]) for k in range(n)]
        self.Lxi = [d_of_linear(n, self.dxi[k]) for k in range(n)]

    def _zero(self):
        return ([0] * self.n, [0] * self.n)

    def _Dx(self, k):
        """D_f(x_k) = (-1)^{p(f)} df/dxi_k  as a linear function."""
        n, i, j = self.n, self.i, self.j
        cx, cxi = self._zero()
        s = -1 if self.pf else 1
        if self.kind == "gl":
            # f = x_i xi_j ; df/dxi_k (left derivative) = delta_{jk} x_i
            if k == j:
                cx[i] += s
        elif self.kind == "lam":
            # f = xi_i xi_j, i<j ; df/dxi_i = xi_j, df/dxi_j = -xi_i
            if k == i:
                cxi[j] += s
            if k == j:
                cxi[i] += -s
        # kind == 'sym': f = x_i x_j has no xi, derivative is 0
        return (cx, cxi)

    def _Dxi(self, k):
        """D_f(xi_k) = df/dx_k."""
        n, i, j = self.n, self.i, self.j
        cx, cxi = self._zero()
        if self.kind == "gl":
            # f = x_i xi_j ; df/dx_i = xi_j
            if k == i:
                cxi[j] += 1
        elif self.kind == "sym":
            # f = x_i x_j
            if k == i:
                cx[j] += 1
            if k == j:
                cx[i] += 1
        return (cx, cxi)

    def act(self, v):
        """the Lie derivative L_D on an element of Omega(const)."""
        n = self.n
        out = {}
        for (A, B), c in v.items():
            # ---- differentiate the exterior part
            for pos, a in enumerate(A):
                rest = A[:pos] + A[pos + 1:]
                # Koszul sign: L_D moves past 'x_{a1}..'x_{a_{pos-1}} (all odd)
                sgn = (-1) ** (self.pD * pos)
                for m2, c2 in self.Lx[a].items():
                    # m2 is 'x_b or 'xi_b ; put it back in position `pos`
                    r = _insert(m2, rest, pos)
                    if r is None:
                        continue
                    s2, A3, B3 = r
                    add(out, (A3, _addB(B, B3)), sgn * s2 * c * c2)
            # ---- differentiate the symmetric part (all 'xi are even; L_D moves
            #      past the whole exterior part A)
            sgn = (-1) ** (self.pD * len(A))
            for i in range(n):
                if B[i] == 0:
                    continue
                B2 = list(B)
                B2[i] -= 1
                for m2, c2 in self.Lxi[i].items():
                    r = _insert(m2, A, len(A))
                    if r is None:
                        continue
                    s2, A3, B3 = r
                    add(out, (A3, _addB(tuple(B2), B3)), sgn * s2 * B[i] * c * c2)
        return out


def _addB(B1, B2):
    return tuple(a + b for a, b in zip(B1, B2))


def _insert(m2, A, pos):
    """multiply the degree-1 monomial m2 into the exterior word A, inserting it
    at position `pos` (i.e. m2 sits right after A[:pos]).  Returns
    (sign, sorted exterior word, extra 'xi-exponents), or None if it vanishes."""
    A2, B2 = m2
    if not A2:
        # m2 = 'xi_b, even: no sign, no exterior part
        return 1, A, B2
    b = A2[0]
    if b in A:
        return None
    # sign of sorting the word  A[:pos] + (b,) + A[pos:]  (all letters odd)
    w = list(A[:pos]) + [b] + list(A[pos:])
    sign = 1
    for p in range(len(w)):
        for q in range(len(w) - 1 - p):
            if w[q] > w[q + 1]:
                w[q], w[q + 1] = w[q + 1], w[q]
                sign = -sign
    return sign, tuple(w), B2


# ------------------------------------------------------------ linear algebra


def to_vec(v, index):
    out = [F(0)] * len(index)
    for m, c in v.items():
        out[index[m]] = c
    return out


def rank(rows):
    """rank of a list of rows over Q."""
    rows = [r[:] for r in rows]
    m = len(rows)
    if m == 0:
        return 0
    ncol = len(rows[0])
    r = 0
    for col in range(ncol):
        piv = None
        for i in range(r, m):
            if rows[i][col] != 0:
                piv = i
                break
        if piv is None:
            continue
        rows[r], rows[piv] = rows[piv], rows[r]
        pv = rows[r][col]
        rows[r] = [t / pv for t in rows[r]]
        for i in range(m):
            if i != r and rows[i][col] != 0:
                f = rows[i][col]
                rows[i] = [a - f * b for a, b in zip(rows[i], rows[r])]
        r += 1
        if r == m:
            break
    return r


def nullspace(rows, ncol):
    """basis of {v : rows . v = 0} as lists of Fractions."""
    rows = [r[:] for r in rows]
    m = len(rows)
    pivots = []
    r = 0
    for col in range(ncol):
        piv = None
        for i in range(r, m):
            if rows[i][col] != 0:
                piv = i
                break
        if piv is None:
            continue
        rows[r], rows[piv] = rows[piv], rows[r]
        pv = rows[r][col]
        rows[r] = [t / pv for t in rows[r]]
        for i in range(m):
            if i != r and rows[i][col] != 0:
                f = rows[i][col]
                rows[i] = [a - f * b for a, b in zip(rows[i], rows[r])]
        pivots.append(col)
        r += 1
        if r == m:
            break
    free = [c for c in range(ncol) if c not in pivots]
    out = []
    for fc in free:
        v = [F(0)] * ncol
        v[fc] = F(1)
        for i, pc in enumerate(pivots):
            v[pc] = -rows[i][fc]
        out.append(v)
    return out


# ------------------------------------------------------------------ weights


def weight(m):
    """the h-weight of a monomial: wt('x_j) = -eps_j, wt('xi_j) = +eps_j."""
    A, B = m
    n = len(B)
    w = [0] * n
    for a in A:
        w[a] -= 1
    for j in range(n):
        w[j] += B[j]
    return tuple(w)


def gens_pe(n):
    """all of pe(n); returns (n_plus, all)."""
    npl, allg = [], []
    for i in range(n):
        for j in range(n):
            g = Gen(n, "gl", i, j)
            allg.append(g)
            if j < i:
                npl.append(g)
    for i in range(n):
        for j in range(i + 1, n):
            g = Gen(n, "lam", i, j)
            allg.append(g)
            npl.append(g)          # pe_{+1} = Lambda^2 is in n_+
    for i in range(n):
        for j in range(i, n):
            allg.append(Gen(n, "sym", i, j))
    return npl, allg
