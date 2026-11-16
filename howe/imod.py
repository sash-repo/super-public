"""
imod.py -- the induced le(n)-module I(V) = C[partial_1..partial_n, delta_1..delta_n] (x) V
and the search for singular vectors.

Basis of L_{-1}: index i < n is partial_i = -D_{xi_i} (even, weight eps_i),
                 index n+i    is delta_i   =  D_{x_i}  (odd,  weight -eps_i).
A monomial of U(L_{-1}) is (a, S): a = exponents of the partial's, S = sorted delta indices.

For g in L_j (j >= 0) the action is the derivation recursion

    g . (a u' (x) v) = [g,a] . (u' (x) v) + (-1)^{p(g)p(a)} a . (g . (u' (x) v)),

with [g,a] in L_{j-1}; the recursion bottoms out at L_0 acting on V and L_{>0} killing V.
"""

from fractions import Fraction as F
from itertools import combinations
from le import abr, par_le, pwt, nullspace, pmul, padd
from pemod import Pe, Irrep, mon


class Le:
    def __init__(self, n, odd_up=True):
        self.n = n
        self.odd_up = odd_up
        self.pe = Pe(n, odd_up)
        # L_{-1}
        self.neg = ([{((0,) * n, (i,)): F(-1)} for i in range(n)]      # partial_i = -D_{xi_i}
                    + [mon(n, [1 if k == i else 0 for k in range(n)], []) for i in range(n)])
        self.neg_par = [par_le(f) for f in self.neg]
        self.neg_wt = [pwt(f, n) for f in self.neg]
        self.negname = ['d%d' % (i + 1) for i in range(n)] + ['t%d' % (i + 1) for i in range(n)]
        # L_1 : generating functions of degree 3
        self.pos1 = self._monos(3)
        self.pos1_par = [par_le(f) for f in self.pos1]
        self.pos1_wt = [pwt(f, n) for f in self.pos1]

    def _monos(self, d):
        n = self.n
        out = []

        def rec(i, a, rem):
            if i == n:
                for k in range(min(rem, n) + 1):
                    if k == rem:
                        for S in combinations(range(n), k):
                            out.append(mon(n, a, S))
                return
            for e in range(rem + 1):
                rec(i + 1, a + [e], rem - e)

        rec(0, [], d)
        return out

    # ---- decompose a generating function of degree 1 in the L_{-1} basis
    def as_neg(self, f):
        n = self.n
        out = {}
        for (a, S), c in f.items():
            if S:
                out[S[0]] = out.get(S[0], 0) - c        # xi_i = -(partial_i)
            else:
                i = a.index(1)
                out[n + i] = out.get(n + i, 0) + c
        return {k: v for k, v in out.items() if v}


class Induced:
    """I(V) with V = V(lambda) irreducible over L_0 = pe(n)."""

    def __init__(self, le, lam):
        self.le = le
        self.n = le.n
        self.pe = le.pe
        self.V = Irrep(le.pe, lam)
        self.lam = self.V.lam

    # ---- monomials of U(L_{-1}) of degree m
    def umonos(self, m):
        n = self.n
        out = []
        for k in range(min(m, n) + 1):
            for S in combinations(range(n), k):
                r = m - k
                for a in self._comps(r, n):
                    out.append((a, S))
        return out

    def _comps(self, r, n):
        if n == 1:
            return [(r,)]
        out = []
        for e in range(r + 1):
            for rest in self._comps(r - e, n - 1):
                out.append((e,) + rest)
        return out

    def uwt(self, u):
        a, S = u
        return tuple(a[i] - (1 if i in S else 0) for i in range(self.n))

    # ---- left multiplication by a basis element of L_{-1}
    def lmul(self, k, elt):
        n = self.n
        out = {}
        for (u, mu, i), c in elt.items():
            a, S = u
            if k < n:
                b = list(a)
                b[k] += 1
                key = ((tuple(b), S), mu, i)
                out[key] = out.get(key, 0) + c
            else:
                j = k - n
                if j in S:
                    continue
                sg = (-1) ** sum(1 for t in S if t < j)
                key = ((a, tuple(sorted(S + (j,)))), mu, i)
                out[key] = out.get(key, 0) + sg * c
        return {k2: v for k2, v in out.items() if v}

    # ---- action of D_f (f a generating function of degree >= 2) on I(V)
    def act(self, f, elt):
        """elt: dict (u, mu, i) -> coeff.  f homogeneous of degree d >= 2 (so D_f in L_{d-2})."""
        out = {}
        for key, c in elt.items():
            for k2, c2 in self._act_one(f, key).items():
                out[k2] = out.get(k2, 0) + c * c2
        return {k: v for k, v in out.items() if v}

    def _act_one(self, f, key):
        n = self.n
        u, mu, i = key
        a, S = u
        deg = sum(a) + len(S)
        d = sum(next(iter(f))[0]) + len(next(iter(f))[1])   # degree of f
        if deg == 0:
            if d > 2:
                return {}
            vec = self.pe.as_vector(f)
            out = {}
            for b, cb in vec.items():
                nu = tuple(x + y for x, y in zip(mu, self.pe.bwt[b]))
                for j, cj in self.V.act(b, mu, {i: F(1)}).items():
                    out[((a, S), nu, j)] = out.get(((a, S), nu, j), 0) + cb * cj
            return {k: v for k, v in out.items() if v}
        # split off the leftmost factor of u
        if any(a):
            k = next(t for t in range(n) if a[t])
            b = list(a)
            b[k] -= 1
            rest = ((tuple(b), S), mu, i)
        else:
            k = n + S[0]
            rest = ((a, S[1:]), mu, i)
        pg = par_le(f)                        # parity of D_f = p(f) + 1
        pa = self.le.neg_par[k]
        br = abr(f, self.le.neg[k], n)        # in L_{d-3}
        out = {}
        if br:
            if d == 2:                        # [g,a] in L_{-1}: multiply
                for kk, ck in self.le.as_neg(br).items():
                    for k2, c2 in self.lmul(kk, {rest: F(1)}).items():
                        out[k2] = out.get(k2, 0) + ck * c2
            else:
                for k2, c2 in self._act_one(br, rest).items():
                    out[k2] = out.get(k2, 0) + c2
        sg = (-1) ** (pg * pa)
        for k2, c2 in self._act_one(f, rest).items():
            for k3, c3 in self.lmul(k, {k2: F(1)}).items():
                out[k3] = out.get(k3, 0) + sg * c2 * c3
        return {k: v for k, v in out.items() if v}

    # ---- singular vectors of degree m and weight nu
    def singular(self, m, nu):
        cols = []
        for u in self.umonos(m):
            w = self.uwt(u)
            mu = tuple(a - b for a, b in zip(nu, w))
            for i in range(len(self.V.wtbasis(mu))):
                cols.append((u, mu, i))
        if not cols:
            return []
        eqs = []
        ops = list(self.le.pos1) + [self.pe.basis[len(self.pe.nneg) + self.n + t]
                                    for t in range(len(self.pe.npos))]
        for f in ops:
            img = {}
            for key in cols:
                for k2, c2 in self._act_one(f, key).items():
                    row = img.setdefault(k2, {})
                    row[key] = row.get(key, 0) + c2
            eqs.extend(img.values())
        return nullspace(eqs, cols)
