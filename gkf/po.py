"""The Poisson superalgebra po(0|n) = Lambda(xi_1..xi_n) and its Hamiltonian quotients.

Basis: Grassmann monomials, encoded as sorted tuples of indices 0..n-1.

The Poisson bracket is taken with respect to a symmetric Gram matrix B on the odd
space:   {f, g} = (-1)^{p(f)} sum_{i,j} B_{ij} (d_i f)(d_j g),
where d_i is the LEFT derivative.  We use the hyperbolic (split) shape of B, so
that the maximal torus of g_0 = o(n) is diagonal in this basis: for n = 2m the
pairs are (i, i+m), for n = 2m+1 there is one extra self-paired index 2m.

h(0|n)      = po(0|n) / K.1               (Grassmann degrees 1..n)
h^{(1)}(0|n)= span of Grassmann degrees 1..n-1   (dim 2^n - 2), simple for n >= 4.

Grading: deg (a monomial of Grassmann degree d) = d - 2, so that the bracket has
degree 0 and h^{(1)}(0|n) = g_{-1} + g_0 + ... + g_{n-3}.
Parity: p = d mod 2.
"""

from fractions import Fraction
from itertools import combinations


# ---------------------------------------------------------------- Grassmann


def gram(n):
    """Hyperbolic Gram matrix of the symmetric form on the odd space."""
    B = [[0] * n for _ in range(n)]
    m = n // 2
    for i in range(m):
        B[i][i + m] = 1
        B[i + m][i] = 1
    if n % 2:
        B[n - 1][n - 1] = 1
    return B


def lderiv(mon, i):
    """Left derivative d_i of a Grassmann monomial: (sign, monomial) or None."""
    if i not in mon:
        return None
    pos = mon.index(i)
    return (-1) ** pos, mon[:pos] + mon[pos + 1:]


def gmul(a, b):
    """Product of two Grassmann monomials: (sign, monomial) or None."""
    if set(a) & set(b):
        return None
    merged = list(a) + list(b)
    # count inversions to sort
    sign = 1
    arr = merged[:]
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] > arr[j]:
                sign = -sign
    return sign, tuple(sorted(merged))


class Po:
    """h^{(1)}(0|n) (default) or h(0|n) or po(0|n), with its bracket."""

    def __init__(self, n, kind="h1"):
        self.n = n
        self.kind = kind
        self.B = gram(n)
        if kind == "po":
            degs = range(0, n + 1)
        elif kind == "h":
            degs = range(1, n + 1)
        elif kind == "h1":
            degs = range(1, n)
        else:
            raise ValueError(kind)
        self.basis = []
        for d in degs:
            for c in combinations(range(n), d):
                self.basis.append(c)
        self.index = {m: k for k, m in enumerate(self.basis)}
        self.dim = len(self.basis)

    # ---- gradings -------------------------------------------------------
    def deg(self, k):
        return len(self.basis[k]) - 2

    def par(self, k):
        return len(self.basis[k]) % 2

    def weight(self, k):
        """Weight with respect to the diagonal torus of o(n): eta_i - zeta_i count."""
        n, m = self.n, self.n // 2
        w = [0] * m
        for i in self.basis[k]:
            if i < m:
                w[i] += 1
            elif i < 2 * m:
                w[i - m] -= 1
        return tuple(w)

    # ---- bracket --------------------------------------------------------
    def bracket(self, a, b):
        """[x_a, x_b] as a dict {basis index: coefficient}."""
        ma, mb = self.basis[a], self.basis[b]
        sgn0 = (-1) ** (len(ma) % 2)
        out = {}
        for i in range(self.n):
            da = lderiv(ma, i)
            if da is None:
                continue
            for j in range(self.n):
                bij = self.B[i][j]
                if not bij:
                    continue
                db = lderiv(mb, j)
                if db is None:
                    continue
                pr = gmul(da[1], db[1])
                if pr is None:
                    continue
                c = sgn0 * bij * da[0] * db[0] * pr[0]
                mon = pr[1]
                if mon in self.index:          # else it fell out of the quotient
                    k = self.index[mon]
                    out[k] = out.get(k, 0) + c
                elif len(mon) == 0 and self.kind == "po":
                    pass
        return {k: v for k, v in out.items() if v}

    def table(self):
        """Full structure-constant table, cached."""
        if not hasattr(self, "_tab"):
            self._tab = [[self.bracket(a, b) for b in range(self.dim)]
                         for a in range(self.dim)]
        return self._tab


# ---------------------------------------------------------------- checks


def check_algebra(g, verbose=True):
    """Graded antisymmetry, Jacobi, and the compatibility of the gradings."""
    tab = g.table()
    d = g.dim
    bad = []
    # graded antisymmetry: [a,b] = -(-1)^{p(a)p(b)} [b,a]
    for a in range(d):
        for b in range(d):
            lhs = tab[a][b]
            s = -(-1) ** (g.par(a) * g.par(b))
            rhs = {k: s * v for k, v in tab[b][a].items()}
            if lhs != rhs:
                bad.append(("antisym", a, b))
    # degree and parity additivity
    for a in range(d):
        for b in range(d):
            for k in tab[a][b]:
                if g.deg(k) != g.deg(a) + g.deg(b):
                    bad.append(("deg", a, b, k))
                if g.par(k) != (g.par(a) + g.par(b)) % 2:
                    bad.append(("par", a, b, k))
    # Jacobi: [a,[b,c]] = [[a,b],c] + (-1)^{p(a)p(b)} [b,[a,c]]
    for a in range(d):
        for b in range(d):
            for c in range(d):
                lhs = {}
                for k, v in tab[b][c].items():
                    for l, w in tab[a][k].items():
                        lhs[l] = lhs.get(l, 0) + v * w
                rhs = {}
                for k, v in tab[a][b].items():
                    for l, w in tab[k][c].items():
                        rhs[l] = rhs.get(l, 0) + v * w
                s = (-1) ** (g.par(a) * g.par(b))
                for k, v in tab[a][c].items():
                    for l, w in tab[b][k].items():
                        rhs[l] = rhs.get(l, 0) + s * v * w
                lhs = {k: v for k, v in lhs.items() if v}
                rhs = {k: v for k, v in rhs.items() if v}
                if lhs != rhs:
                    bad.append(("jacobi", a, b, c))
    if verbose:
        print(f"  {g.kind}(0|{g.n}): dim = {g.dim}, defects = {len(bad)}")
        if bad:
            print("   ", bad[:5])
    return not bad


def sdim(g):
    ev = sum(1 for k in range(g.dim) if g.par(k) == 0)
    return ev, g.dim - ev


if __name__ == "__main__":
    for n in range(3, 8):
        g = Po(n, "h1")
        ok = check_algebra(g)
        e, o = sdim(g)
        by = {}
        for k in range(g.dim):
            by[g.deg(k)] = by.get(g.deg(k), 0) + 1
        print(f"    sdim = ({e}|{o}), degrees {dict(sorted(by.items()))}, ok={ok}")
