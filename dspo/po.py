#!/usr/bin/env python3
"""
Exact-arithmetic engine for the Poisson Lie superalgebra po(0|m) = Lambda(m)
with the Poisson bracket of formula (20) of the text, and for its quotients
h(0|m) = po(0|m)/c and h^{(1)}(0|m) = [h,h].

Everything is done over Q with Fraction arithmetic (structure constants are
integers), so every number printed here is exact.
Running it checks the Jacobi identity and super anti-commutativity and prints,
for m = 5, 6 (default), the rank of ad_x and sdim g_x for every homologic
monomial x (even ones included), grouped by degree; the odd rows are those of
the tables of Subsecs. 3.1.3, 3.1.4, which porun.py prints in full.

Conventions (formula (20) of the text):
  generators  xi_1..xi_k, eta_1..eta_k  (+ theta if m is odd)
  {f,g} = (-1)^p(f) ( df/dtheta dg/dtheta
                      + sum_j ( df/dxi_j dg/deta_j + df/deta_j dg/dxi_j ) )
with LEFT derivatives.  deg_L(f) = deg(f) - 2 is the standard Z-grading.

Usage:  python3 po.py [m ...]
"""
import sys
from fractions import Fraction
from itertools import combinations

# ---------------------------------------------------------------- Grassmann

def popcount(x):
    return bin(x).count("1")

def deriv(i, mask):
    """left derivative d/dz_i of the monomial `mask`; returns (sign, mask) or None."""
    bit = 1 << i
    if not (mask & bit):
        return None
    sign = -1 if (popcount(mask & (bit - 1)) & 1) else 1
    return sign, mask ^ bit

def mono_prod(a, b):
    """product of two Grassmann monomials; returns (sign, mask) or None."""
    if a & b:
        return None
    # move the generators of b past those of a
    sign = 1
    bb = b
    while bb:
        low = bb & -bb
        i = low.bit_length() - 1
        # number of generators of a above i that must be passed
        if popcount(a & ~((1 << (i + 1)) - 1)) & 1:
            sign = -sign
        bb ^= low
    return sign, a | b


class Po:
    """po(0|m): Lambda on m odd generators with the Poisson bracket."""

    def __init__(self, m):
        self.m = m
        self.k = m // 2
        self.odd_m = (m % 2 == 1)
        # generator indices: xi_j = j-1, eta_j = k+j-1, theta = 2k
        self.pairs = []
        for j in range(self.k):
            self.pairs.append((j, self.k + j))
            self.pairs.append((self.k + j, j))
        if self.odd_m:
            self.pairs.append((2 * self.k, 2 * self.k))
        self.dim = 1 << m
        self.names = ([f"xi{j+1}" for j in range(self.k)]
                      + [f"eta{j+1}" for j in range(self.k)]
                      + (["theta"] if self.odd_m else []))

    def parity(self, mask):
        return popcount(mask) & 1

    def degL(self, mask):
        return popcount(mask) - 2

    def name(self, mask):
        if mask == 0:
            return "1"
        return "".join(self.names[i] for i in range(self.m) if mask & (1 << i))

    def bracket_mono(self, a, b):
        """{a,b} for monomials a,b -> dict mask->int coefficient."""
        out = {}
        sgn_f = -1 if (popcount(a) & 1) else 1
        for (i, j) in self.pairs:
            da = deriv(i, a)
            if da is None:
                continue
            db = deriv(j, b)
            if db is None:
                continue
            pr = mono_prod(da[1], db[1])
            if pr is None:
                continue
            c = sgn_f * da[0] * db[0] * pr[0]
            out[pr[1]] = out.get(pr[1], 0) + c
        return {k: v for k, v in out.items() if v}

    def bracket(self, f, g):
        """f,g: dict mask->coeff."""
        out = {}
        for a, ca in f.items():
            for b, cb in g.items():
                for msk, c in self.bracket_mono(a, b).items():
                    out[msk] = out.get(msk, 0) + ca * cb * c
        return {k: v for k, v in out.items() if v}

    def ad_matrix(self, x):
        """matrix of ad_x in the monomial basis (column j = ad_x(e_j))."""
        n = self.dim
        M = [[0] * n for _ in range(n)]
        for j in range(n):
            for msk, c in self.bracket({b: co for b, co in x.items()}, {j: 1}).items():
                M[msk][j] = c
        return M

    # -------------------------------------------------------------- checks
    def check_identities(self, ntest=400):
        """super-anticommutativity and the super Jacobi identity on monomials."""
        import random
        rnd = random.Random(7)
        n = self.dim
        for _ in range(ntest):
            a, b, c = (rnd.randrange(n) for _ in range(3))
            pa, pb = self.parity(a), self.parity(b)
            l = self.bracket({a: 1}, {b: 1})
            r = self.bracket({b: 1}, {a: 1})
            s = -1 if (pa * pb) % 2 == 0 else 1     # {a,b} = -(-1)^{pa pb}{b,a}
            for msk in set(l) | set(r):
                if l.get(msk, 0) != s * r.get(msk, 0):
                    return f"anticommutativity fails on {self.name(a)},{self.name(b)}"
            # Jacobi: {a,{b,c}} = {{a,b},c} + (-1)^{pa pb}{b,{a,c}}
            lhs = self.bracket({a: 1}, self.bracket({b: 1}, {c: 1}))
            r1 = self.bracket(self.bracket({a: 1}, {b: 1}), {c: 1})
            r2 = self.bracket({b: 1}, self.bracket({a: 1}, {c: 1}))
            sg = 1 if (pa * pb) % 2 == 0 else -1
            for msk in set(lhs) | set(r1) | set(r2):
                if lhs.get(msk, 0) != r1.get(msk, 0) + sg * r2.get(msk, 0):
                    return f"Jacobi fails on {self.name(a)},{self.name(b)},{self.name(c)}"
        return None


# ------------------------------------------------------- exact linear algebra

def rref(rows, ncols):
    """exact reduced row echelon form over Q; returns (rows, pivots)."""
    M = [[Fraction(v) for v in r] for r in rows]
    piv = []
    r = 0
    for c in range(ncols):
        p = None
        for i in range(r, len(M)):
            if M[i][c] != 0:
                p = i
                break
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        pv = M[r][c]
        M[r] = [v / pv for v in M[r]]
        for i in range(len(M)):
            if i != r and M[i][c] != 0:
                f = M[i][c]
                M[i] = [a - f * b for a, b in zip(M[i], M[r])]
        piv.append(c)
        r += 1
        if r == len(M):
            break
    return M[:r], piv

def col_space(M):
    """basis of the column space of M (list of columns, exact)."""
    cols = list(zip(*M)) if M else []
    rows, piv = rref([list(c) for c in cols], len(M))
    # pivots of the row-reduced set of columns: take the independent original columns
    ind = independent_subset([list(c) for c in cols])
    return [list(cols[i]) for i in ind]

def independent_subset(vectors):
    """indices of a maximal independent subset, exact."""
    basis = []      # (pivot, vector)
    keep = []
    for idx, v in enumerate(vectors):
        w = [Fraction(x) for x in v]
        for (p, b) in basis:
            if w[p] != 0:
                f = w[p]
                w = [a - f * c for a, c in zip(w, b)]
        p = next((i for i, x in enumerate(w) if x != 0), None)
        if p is not None:
            w = [x / w[p] for x in w]
            basis.append((p, w))
            keep.append(idx)
    return keep

def rank(M):
    if not M:
        return 0
    cols = [list(c) for c in zip(*M)]
    return len(independent_subset(cols))

def null_space(M, ncols):
    """exact basis of {v : Mv = 0}."""
    R, piv = rref(M, ncols)
    free = [c for c in range(ncols) if c not in piv]
    out = []
    for f in free:
        v = [Fraction(0)] * ncols
        v[f] = Fraction(1)
        for i, c in enumerate(piv):
            v[c] = -R[i][f]
        out.append(v)
    return out

def solve_in_span(basis, v):
    """coordinates of v in the span of `basis` (list of vectors), or None."""
    n = len(v)
    A = [list(b) for b in basis]
    rows = [[Fraction(A[j][i]) for j in range(len(A))] + [Fraction(v[i])] for i in range(n)]
    R, piv = rref(rows, len(A) + 1)
    if len(A) in piv:
        return None
    sol = [Fraction(0)] * len(A)
    for i, c in enumerate(piv):
        sol[c] = R[i][len(A)]
    return sol


# --------------------------------------------------------- quotient algebra

class Quot:
    """
    g_x = Ker ad_x / Im ad_x with its induced bracket, as an abstract
    superalgebra given by a structure-constant table.
    """
    def __init__(self, alg, x, verbose=False):
        self.alg = alg
        n = alg.dim
        M = alg.ad_matrix(x)
        self.M = M
        self.rank = rank(M)
        ker = null_space(M, n)
        cols = [list(c) for c in zip(*M)]
        im_idx = independent_subset(cols)
        im = [cols[i] for i in im_idx]
        self.dim_ker = len(ker)
        self.dim_im = len(im)
        # complement of Im inside Ker
        allv = im + ker
        keep = independent_subset(allv)
        rep_idx = [i - len(im) for i in keep if i >= len(im)]
        self.reps = [ker[i] for i in rep_idx]        # representatives of g_x
        self.im = im
        self.dim = len(self.reps)
        # parity and degree of the representatives (they are homogeneous when
        # x is homogeneous, which is the only case we use)
        self.par = []
        self.deg = []
        for v in self.reps:
            supp = [i for i, c in enumerate(v) if c != 0]
            ps = {alg.parity(i) for i in supp}
            ds = {alg.degL(i) for i in supp}
            self.par.append(ps.pop() if len(ps) == 1 else None)
            self.deg.append(ds.pop() if len(ds) == 1 else None)
        self.basis_all = self.im + self.reps      # spans Ker

    def sdim(self):
        ev = sum(1 for p in self.par if p == 0)
        od = sum(1 for p in self.par if p == 1)
        return (ev, od)

    def _build_projector(self):
        """
        Precompute the projection Ker -> coordinates w.r.t. self.reps.
        B is the n x K matrix whose columns are basis_all = im + reps; pick K
        independent rows and invert that K x K block once.
        """
        B = self.basis_all
        K = len(B)
        n = self.alg.dim
        rows = [[Fraction(B[j][i]) for j in range(K)] for i in range(n)]
        keep = independent_subset(rows)          # K independent row indices
        assert len(keep) == K
        S = [rows[i][:] for i in keep]
        # invert S by Gauss--Jordan
        A = [S[i] + [Fraction(1) if i == j else Fraction(0) for j in range(K)]
             for i in range(K)]
        for c in range(K):
            p = next(i for i in range(c, K) if A[i][c] != 0)
            A[c], A[p] = A[p], A[c]
            pv = A[c][c]
            A[c] = [v / pv for v in A[c]]
            for i in range(K):
                if i != c and A[i][c] != 0:
                    f = A[i][c]
                    A[i] = [a - f * b for a, b in zip(A[i], A[c])]
        self._rows_keep = keep
        self._inv = [r[K:] for r in A]
        self._nim = len(self.im)

    def project(self, v):
        """coordinates of v (in Ker) modulo Im, in terms of self.reps."""
        if not hasattr(self, "_inv"):
            self._build_projector()
        sub = [Fraction(v[i]) for i in self._rows_keep]
        K = len(sub)
        out = []
        for r in range(self._nim, K):
            row = self._inv[r]
            s = Fraction(0)
            for j in range(K):
                if sub[j]:
                    s += row[j] * sub[j]
            out.append(s)
        return out

    def table(self):
        """structure constants c[i][j] = vector of coordinates of [e_i,e_j]."""
        if hasattr(self, "_tab"):
            return self._tab
        alg = self.alg
        d = self.dim
        tab = [[None] * d for _ in range(d)]
        vecs = []
        for v in self.reps:
            vecs.append({i: c for i, c in enumerate(v) if c != 0})
        for i in range(d):
            for j in range(d):
                br = alg.bracket(vecs[i], vecs[j])
                w = [Fraction(0)] * alg.dim
                for msk, c in br.items():
                    w[msk] = Fraction(c)
                tab[i][j] = self.project(w)
        self._tab = tab
        return tab

    # ---- structure of the quotient
    def span_bracket(self, A, B):
        """span of [A,B] for subspaces A,B given by coordinate vectors."""
        tab = self.table()
        d = self.dim
        out = []
        for a in A:
            for b in B:
                w = [Fraction(0)] * d
                for i in range(d):
                    if a[i] == 0:
                        continue
                    for j in range(d):
                        if b[j] == 0:
                            continue
                        f = a[i] * b[j]
                        for t in range(d):
                            w[t] += f * tab[i][j][t]
                out.append(w)
        keep = independent_subset(out)
        return [out[i] for i in keep]

    def derived_series(self):
        d = self.dim
        cur = [[Fraction(1) if i == j else Fraction(0) for j in range(d)] for i in range(d)]
        seq = [len(cur)]
        for _ in range(12):
            nxt = self.span_bracket(cur, cur)
            seq.append(len(nxt))
            if len(nxt) == len(cur) or len(nxt) == 0:
                break
            cur = nxt
        return seq, cur

    def lower_central(self):
        d = self.dim
        g = [[Fraction(1) if i == j else Fraction(0) for j in range(d)] for i in range(d)]
        cur = g
        seq = [len(cur)]
        for _ in range(12):
            nxt = self.span_bracket(g, cur)
            seq.append(len(nxt))
            if len(nxt) == len(cur) or len(nxt) == 0:
                break
            cur = nxt
        return seq


# ------------------------------------------------------------------ drivers

def homologic(alg, x):
    return not alg.bracket(x, x)

def monomial_reps(alg, deg):
    """all monomials of a given degree."""
    return [sum(1 << i for i in c) for c in combinations(range(alg.m), deg)]

def scan(m, max_deg=None, verbose=True):
    alg = Po(m)
    err = alg.check_identities()
    print(f"=== po(0|{m}) : dim = {alg.dim} = {alg.dim//2}|{alg.dim//2}, "
          f"identities: {'OK' if err is None else err}")
    rows = {}
    for d in range(1, (max_deg or m) + 1):
        for msk in monomial_reps(alg, d):
            x = {msk: 1}
            if not homologic(alg, x):
                continue
            q = Quot(alg, x)
            ev, od = q.sdim()
            key = (d, q.rank)
            rows.setdefault(key, []).append((alg.name(msk), q, ev, od))
    for key in sorted(rows):
        d, r = key
        names = [t[0] for t in rows[key]]
        q = rows[key][0][1]
        ev, od = rows[key][0][2], rows[key][0][3]
        print(f"  deg {d}  rank ad_x = {r:3d}   dim ker = {q.dim_ker}  dim Im = {q.dim_im}"
              f"   sdim g_x = {ev}|{od}   ({len(names)} monomials, e.g. {names[0]})")
    return alg, rows


if __name__ == "__main__":
    ms = [int(a) for a in sys.argv[1:]] or [5, 6]
    for m in ms:
        scan(m)
        print()
