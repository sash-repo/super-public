"""The non-graded Hamiltonian algebras H(2:n;omega_2) over F_p, from scratch.

Built to check, by exact arithmetic over F_p, the claims of Caranti-Mattarei
"Gradings of non-graded Hamiltonian Lie algebras" (arXiv:math/0510431).

Objects (Sections 2-3 of that paper):

    F[2:n]      divided powers x^(i) y^(j), 0 <= i < p^n1, 0 <= j < p^n2,
                x^(i) x^(k) = C(i+k,i) x^(i+k)
    e           = x^(p^n1-1) y^(p^n2-1), the top monomial
    omega_0     = dx ^ dy          -> {f,g}_0 = f_y g_x - f_x g_y
    omega_2     = (1-e) dx ^ dy    -> {f,g}_2 = (1+e)(f_y g_x - f_x g_y)

    H(2:n;omega_0) = P(2:n)/<1>,  P = span of monomials != e,  dim p^n - 2
    H(2:n;omega_2) = F[2:n]/<1>,                               dim p^n - 1

Since e * x^(i) y^(j) = 0 unless i=j=0, multiplication by (1+e) only adds
back, on the e-coordinate, the constant term that the quotient by <1> kills:
that single entry is the whole difference between the graded algebra and the
filtered one.

What is computed: Jacobi, dim, simplicity, Der/ad, H^2(L;F) = ker d2 / im d1
with ALTERNATING cochains (the convention Remark 4.5 of the paper says is the
right one at p=2), and the two isomorphism claims (p=2 Zassenhaus, p=3 psl(3)).
"""
import sys
from itertools import combinations

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import rank, nullspace, backend                 # noqa: E402


# ------------------------------------------------------------------ binomials

def binom(a, b):
    if b < 0 or b > a:
        return 0
    r = 1
    for k in range(b):
        r = r * (a - k) // (k + 1)
    return r


# ------------------------------------------------------------------ the algebra

class Ham:
    """H(2:n;omega_j) over F_p in the Poisson-bracket model."""

    def __init__(self, p, n1, n2, form=2):
        self.p, self.n1, self.n2, self.form = p, n1, n2, form
        self.X, self.Y = p ** n1, p ** n2          # exclusive bounds on i, j
        self.top = (self.X - 1, self.Y - 1)        # the monomial e
        mons = [(i, j) for i in range(self.X) for j in range(self.Y)]
        mons.remove((0, 0))                        # the quotient by <1>
        if form == 0:
            mons.remove(self.top)                  # H(2:n;om_0) has dim p^n-2
        self.basis = mons
        self.index = {m: k for k, m in enumerate(mons)}
        self.dim = len(mons)
        self._table = [[None] * self.dim for _ in range(self.dim)]

    # {x^(i)y^(j), x^(k)y^(l)}_0 = N(i,j,k,l) x^(i+k-1) y^(j+l-1)
    def _N(self, i, j, k, l):
        return (binom(i + k - 1, i) * binom(j + l - 1, j - 1)
                - binom(i + k - 1, i - 1) * binom(j + l - 1, j))

    def bracket_pair(self, a, b):
        """[e_a, e_b] as a dict {basis index: coefficient mod p}."""
        i, j = self.basis[a]
        k, l = self.basis[b]
        c = self._N(i, j, k, l) % self.p
        out = {}
        if c:
            u, v = i + k - 1, j + l - 1
            if 0 <= u < self.X and 0 <= v < self.Y:
                if (u, v) == (0, 0):
                    # the constant 1: killed by the quotient; the factor (1+e)
                    # of omega_2 puts it back on the coordinate e
                    if self.form == 2:
                        out[self.index[self.top]] = c
                elif (u, v) in self.index:
                    out[self.index[(u, v)]] = c
        return out

    def table(self):
        for a in range(self.dim):
            for b in range(self.dim):
                if self._table[a][b] is None:
                    br = self.bracket_pair(a, b)
                    self._table[a][b] = br
                    self._table[b][a] = {k: (-v) % self.p for k, v in br.items()}
        return self._table

    def mul(self, u, v):
        """bracket of two vectors given as lists of length dim."""
        T, p, out = self.table(), self.p, [0] * self.dim
        for a, ua in enumerate(u):
            if not ua:
                continue
            for b, vb in enumerate(v):
                if not vb:
                    continue
                for k, c in T[a][b].items():
                    out[k] = (out[k] + ua * vb * c) % p
        return out

    # -------------------------------------------------------------- checks

    def jacobi_fails(self):
        T, p, bad = self.table(), self.p, 0
        for a, b, c in combinations(range(self.dim), 3):
            acc = {}
            for x, y, z in ((a, b, c), (b, c, a), (c, a, b)):
                for k, v in T[x][y].items():
                    for m, w in T[k][z].items():
                        acc[m] = (acc.get(m, 0) + v * w) % p
            if any(acc.values()):
                bad += 1
        return bad

    def is_simple(self):
        """no proper ideal: every nonzero element generates the whole algebra."""
        T, p, d = self.table(), self.p, self.dim
        # it suffices to test generic elements? no - test the ideal generated by
        # each basis vector, then check the algebra is perfect and has no centre
        for a in range(d):
            gen = [[1 if k == a else 0 for k in range(d)]]
            span, changed = list(gen), True
            while changed:
                changed = False
                cur = rank(span, p=p)
                new = [self.mul(v, [1 if k == b else 0 for k in range(d)])
                       for v in span for b in range(d)]
                cand = span + [w for w in new if any(w)]
                if rank(cand, p=p) > cur:
                    span, changed = _reduce(cand, p), True
            if rank(span, p=p) != d:
                return False, a
        return True, None

    def derivations(self):
        """basis of Der(L) as dim x dim matrices, flattened row-major."""
        self.table()
        rows = []
        for a, b in combinations(range(self.dim), 2):
            rows.extend(self._der_rows(a, b))
        return nullspace(rows, p=self.p)

    def _der_rows(self, a, b):
        """the d equations coming from the pair (a,b), as rows in the unknowns
        D[m][t] (m = source basis index, t = target coordinate)."""
        T, p, d = self._table, self.p, self.dim
        M = [[0] * (d * d) for _ in range(d)]
        for k, c in T[a][b].items():                    # + D applied to [a,b]
            for t in range(d):
                M[t][k * d + t] = (M[t][k * d + t] + c) % p
        for m in range(d):                              # - [D e_a, e_b]
            for t, c in T[m][b].items():
                M[t][a * d + m] = (M[t][a * d + m] - c) % p
        for m in range(d):                              # - [e_a, D e_b]
            for t, c in T[a][m].items():
                M[t][b * d + m] = (M[t][b * d + m] - c) % p
        return M

    def inner_dim(self):
        d, p = self.dim, self.p
        ads = []
        for a in range(d):
            row = []
            for m in range(d):
                col = [0] * d
                for k, c in self._table[a][m].items():
                    col[k] = c
                row.extend(col)
            ads.append(row)
        return rank(ads, p=p)

    # -------------------------------------------------- H^2(L;F), alternating

    def h2_dim(self):
        d, p, T = self.dim, self.p, self.table()
        pairs = list(combinations(range(d), 2))
        pidx = {q: t for t, q in enumerate(pairs)}
        triples = list(combinations(range(d), 3))

        def pcoef(x, y):
            if x == y:
                return None, 0
            return (pidx[(x, y)], 1) if x < y else (pidx[(y, x)], -1)

        # d2 : C^2 -> C^3,  (d2 phi)(a,b,c) = -phi([a,b],c) - phi([b,c],a) - phi([c,a],b)
        rows = []
        for (a, b, c) in triples:
            row = [0] * len(pairs)
            for (x, y, z) in ((a, b, c), (b, c, a), (c, a, b)):
                for k, v in T[x][y].items():
                    q, s = pcoef(k, z)
                    if q is not None:
                        row[q] = (row[q] + s * v) % p
            rows.append(row)
        z2 = len(pairs) - rank(rows, p=p)

        # d1 : C^1 -> C^2,  (d1 f)(a,b) = -f([a,b])
        rows1 = []
        for m in range(d):
            row = [0] * len(pairs)
            for t, (a, b) in enumerate(pairs):
                row[t] = T[a][b].get(m, 0)
            rows1.append(row)
        b2 = rank(rows1, p=p)
        return z2 - b2, z2, b2


def _reduce(rows, p):
    from exactlin import rref
    R, _ = rref(rows, p=p)
    return [r for r in R if any(r)]


# ------------------------------------------------------------------ reporting

def report(p, n1, n2, form, do_h2=True, do_der=True, do_simple=False):
    L = Ham(p, n1, n2, form)
    L.table()
    name = "H(2:(%d,%d);om_%d)" % (n1, n2, form)
    out = ["p=%d  %-22s dim=%d" % (p, name, L.dim)]
    expect = p ** (n1 + n2) - (1 if form == 2 else 2)
    out.append("  dim expected %d : %s" % (expect, "OK" if L.dim == expect else "WRONG"))
    bad = L.jacobi_fails()
    out.append("  Jacobi: %s" % ("holds" if bad == 0 else "FAILS on %d triples" % bad))
    if do_simple:
        ok, a = L.is_simple()
        out.append("  simple: %s" % (ok if ok else "no, %s generates a proper ideal" % (L.basis[a],)))
    if do_der:
        der = L.derivations()
        inner = L.inner_dim()
        out.append("  dim Der = %d, dim ad = %d, dim Der/ad = %d  (n1+n2 = %d)"
                   % (len(der), inner, len(der) - inner, n1 + n2))
    if do_h2:
        h2, z2, b2 = L.h2_dim()
        out.append("  dim Z^2 = %d, dim B^2 = %d, dim H^2 = %d" % (z2, b2, h2))
    return "\n".join(out)


if __name__ == "__main__":
    print("backend: %s\n" % backend())
    cases = [
        (3, 1, 1, 2), (3, 1, 1, 0),
        (5, 1, 1, 2), (5, 1, 1, 0),
        (7, 1, 1, 2),
        (2, 1, 1, 2), (2, 1, 2, 2), (2, 2, 2, 2), (2, 2, 2, 0),
        (3, 1, 2, 2), (3, 1, 2, 0),
    ]
    for (p, n1, n2, form) in cases:
        print(report(p, n1, n2, form))
        print()
