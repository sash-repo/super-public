"""
mimod.py -- the induced modules I(V) over M(n) and over M_{a,b}(n), and the search
for singular vectors.

M(n)_{<0} = C.D + <'partial_i, 'delta_i>  with  ['partial_i,'delta_j] = 2 delta_{ij} D,
D odd and central, D^2 = 0.  Hence

    U(M(n)_{<0}) = C['partial] (x) Lambda('delta) (x) Lambda(D),

with PBW monomials u = 'partial^a 'delta^S D^t, t in {0,1}; deg u = |a| + |S| + 2t = m
is Kochetkov's degree of the singular vector.

M(n)_0 = pe(n) + C z (z = 'tau central in M(n)_0), so V = V(lambda) (x) C_c with
V(lambda) the irreducible pe(n)-module and c the eigenvalue of z; the weight of V in
Kochetkov's basis {'(x_i xi_i); z} is the (n+1)-tuple (lambda, c).

M_{a,b}(n) is the Cartan prolong (ba(n), spe(n) + C(a z + b h))_*, computed here degree
by degree inside M(n); see Mab.
"""

from fractions import Fraction as F
from itertools import combinations

from le import nullspace, padd, pscal
from pemod import Pe, Irrep, mon
from mn import (mono, one, tau, mbr, mdeg, mwt, par_m, layer, monos, Delta, mdiv)


# --------------------------------------------------------------------- the algebra
class Mn:
    """M(n) with Kochetkov's Borel (the S^2-up one of pe(n), as for le(n)/sle(n))."""

    def __init__(self, n, odd_up=False):
        self.n = n
        self.pe = Pe(n, odd_up)
        # L_{-1} and L_{-2}: 'partial_i = '(-xi_i), 'delta_i = 'x_i, D = '(1/2)
        self.neg = ([pscal(mono(n, [0] * n, [i]), -1) for i in range(n)]
                    + [mono(n, [1 if k == i else 0 for k in range(n)], []) for i in range(n)]
                    + [pscal(one(n), F(1, 2))])
        self.neg_par = [par_m(f) for f in self.neg]          # 0,..,0, 1,..,1, 1
        self.neg_wt = [mwt(f, n) for f in self.neg]
        self.negname = (["'d%d" % (i + 1) for i in range(n)]
                        + ["'t%d" % (i + 1) for i in range(n)] + ['D'])
        self.pos1 = layer(n, 1)

    def as_neg(self, f):
        """decompose a generating function of M-degree -1 or -2 in the basis self.neg"""
        n = self.n
        out = {}
        for (a, S), c in f.items():
            if not S and not any(a):                 # the constant -> D = '(1/2)
                out[2 * n] = out.get(2 * n, 0) + 2 * c
            elif S:
                out[S[0]] = out.get(S[0], 0) - c     # xi_i = -'partial_i
            else:
                out[n + a.index(1)] = out.get(n + a.index(1), 0) + c
        return {k: v for k, v in out.items() if v}

    # ---- the two coordinates cutting out M_{a,b}(n)_0 inside M(n)_0
    def zcoef(self, f):
        return f.get(((0,) * self.n, (self.n,)), 0)

    def hcoef(self, f):
        n = self.n
        s = sum(f.get((tuple(1 if k == i else 0 for k in range(n)), (i,)), 0)
                for i in range(n))
        return F(s, n)


class Mab:
    """M_{a,b}(n) = (ba(n), spe(n) + C(az+bh))_*, computed degree by degree."""

    def __init__(self, mn, a, b):
        self.mn = mn
        self.n = mn.n
        self.a = F(a)
        self.b = F(b)
        self._lay = {}

    def phi(self, f):
        """the functional cutting M_{a,b}(n)_0 out of M(n)_0"""
        return self.b * self.mn.zcoef(f) - self.a * self.mn.hcoef(f)

    def layer(self, j):
        """a basis of M_{a,b}(n)_j as a list of generating functions"""
        if j in self._lay:
            return self._lay[j]
        n = self.n
        if j < 0:
            out = layer(n, j)
        elif j == 0:
            bas = layer(n, 0)
            rows = [{k: self.phi(f) for k, f in enumerate(bas) if self.phi(f)}]
            out = [self._combine(bas, s) for s in nullspace([r for r in rows if r],
                                                            list(range(len(bas))))]
        else:
            prev = self.layer(j - 1)
            bas = layer(n, j)
            # coordinates of the previous layer inside M(n)_{j-1}
            amb = layer(n, j - 1)
            idx = {next(iter(f)): i for i, f in enumerate(amb)}
            pcols = []
            for f in prev:
                pcols.append({idx[m]: c for m, c in f.items()})
            piv, red = _rref(pcols, len(amb))
            eqs = []
            for k in range(2 * n):                     # bracket with L_{-1}
                col = {}
                for ci, f in enumerate(bas):
                    br = mbr(f, self.mn.neg[k], n)
                    v = {idx[m]: c for m, c in br.items()}
                    for p, r in zip(piv, red):         # reduce modulo M_{a,b}(n)_{j-1}
                        if p in v:
                            c = v[p]
                            for kk, vv in r.items():
                                v[kk] = v.get(kk, 0) - c * vv
                                if v[kk] == 0:
                                    del v[kk]
                    for kk, vv in v.items():
                        col.setdefault(kk, {})[ci] = vv
                eqs.extend(col.values())
            out = [self._combine(bas, s)
                   for s in nullspace(eqs, list(range(len(bas))))]
        self._lay[j] = out
        return out

    def _combine(self, bas, sol):
        out = {}
        for k, c in sol.items():
            out = padd(out, bas[k], c)
        return out


def _rref(rows, ncols):
    from le import rref
    return rref(rows, ncols)


# --------------------------------------------------------------------- the module
class MInduced:
    """I(V) with V = V(lambda) (x) C_c irreducible over M(n)_0 = pe(n) + C z.

    ops = the list of generating functions that must kill a singular vector, on top of
    n_+ subset pe(n): M(n)_1 for M(n), M_{a,b}(n)_1 for M_{a,b}(n).
    """

    def __init__(self, mn, lam, zc, ops=None):
        self.mn = mn
        self.n = mn.n
        self.pe = mn.pe
        self.V = Irrep(mn.pe, lam)
        self.lam = self.V.lam
        self.zc = F(zc)
        self.ops = list(mn.pos1 if ops is None else ops)

    # ---- PBW monomials of U(M(n)_{<0}) of degree m
    def umonos(self, m):
        n = self.n
        out = []
        for t in (0, 1):
            r0 = m - 2 * t
            if r0 < 0:
                continue
            for k in range(min(r0, n) + 1):
                for S in combinations(range(n), k):
                    for a in _comps(r0 - k, n):
                        out.append((a, S, t))
        return out

    def uwt(self, u):
        """the pe-part of the weight (the z-part is -deg u)"""
        a, S, t = u
        return tuple(a[i] - (1 if i in S else 0) for i in range(self.n))

    @staticmethod
    def udeg(u):
        a, S, t = u
        return sum(a) + len(S) + 2 * t

    # ---- left multiplication by neg[k]
    def lmul(self, k, elt):
        n = self.n
        out = {}
        for (u, mu, i), c in elt.items():
            a, S, t = u
            if k < n:                                   # 'partial_k, even
                b = list(a)
                b[k] += 1
                key = ((tuple(b), S, t), mu, i)
                out[key] = out.get(key, 0) + c
            elif k < 2 * n:                             # 'delta_j, odd
                j = k - n
                if j not in S:
                    sg = (-1) ** sum(1 for s in S if s < j)
                    key = ((a, tuple(sorted(S + (j,))), t), mu, i)
                    out[key] = out.get(key, 0) + sg * c
                if a[j] and t == 0:                     # [ 'delta_j, 'partial_j ] = -2D
                    b = list(a)
                    b[j] -= 1
                    sg = -2 * a[j] * (-1) ** len(S)
                    key = ((tuple(b), S, 1), mu, i)
                    out[key] = out.get(key, 0) + sg * c
            else:                                       # D, odd, central, D^2 = 0
                if t == 0:
                    key = ((a, S, 1), mu, i)
                    out[key] = out.get(key, 0) + (-1) ** len(S) * c
        return {k2: v for k2, v in out.items() if v}

    # ---- action of a generating function of M-degree >= 0
    def act(self, f, elt):
        out = {}
        for key, c in elt.items():
            for k2, c2 in self._act_one(f, key).items():
                out[k2] = out.get(k2, 0) + c * c2
        return {k: v for k, v in out.items() if v}

    def _act_one(self, f, key):
        n = self.n
        u, mu, i = key
        a, S, t = u
        d = mdeg(f, n)
        if self.udeg(u) == 0:
            if d > 0:
                return {}
            # L_0 = pe(n) + C z
            zc = self.mn.zcoef(f)
            f2 = {m: c for m, c in f.items() if n not in m[1]}
            out = {}
            if zc:
                out[((a, S, t), mu, i)] = zc * self.zc
            if f2:
                vec = self.pe.as_vector(f2)
                for b, cb in vec.items():
                    nu = tuple(x + y for x, y in zip(mu, self.pe.bwt[b]))
                    for j, cj in self.V.act(b, mu, {i: F(1)}).items():
                        kk = ((a, S, t), nu, j)
                        out[kk] = out.get(kk, 0) + cb * cj
            return {k: v for k, v in out.items() if v}
        # split off the leftmost factor
        if any(a):
            k = next(q for q in range(n) if a[q])
            b = list(a)
            b[k] -= 1
            rest = ((tuple(b), S, t), mu, i)
        elif S:
            k = n + S[0]
            rest = (((0,) * n, S[1:], t), mu, i)
        else:
            k = 2 * n
            rest = (((0,) * n, (), 0), mu, i)
        pf = par_m(f)
        pa = self.mn.neg_par[k]
        br = mbr(f, self.mn.neg[k], n)
        out = {}
        if br:
            db = mdeg(br, n)
            if db >= 0:
                for k2, c2 in self._act_one(br, rest).items():
                    out[k2] = out.get(k2, 0) + c2
            else:
                for kk, ck in self.mn.as_neg(br).items():
                    for k2, c2 in self.lmul(kk, {rest: F(1)}).items():
                        out[k2] = out.get(k2, 0) + ck * c2
        sg = (-1) ** (pf * pa)
        for k2, c2 in self._act_one(f, rest).items():
            for k3, c3 in self.lmul(k, {k2: F(1)}).items():
                out[k3] = out.get(k3, 0) + sg * c2 * c3
        return {k: v for k, v in out.items() if v}

    # ---- singular vectors of degree m and weight nu = (pe-weight, z-weight)
    def singular(self, m, nu):
        cols = []
        for u in self.umonos(m):
            w = self.uwt(u)
            mu = tuple(x - y for x, y in zip(nu[:self.n], w))
            for i in range(len(self.V.wtbasis(mu))):
                cols.append((u, mu, i))
        if not cols:
            return []
        eqs = []
        ops = list(self.ops) + [self.pe.basis[len(self.pe.nneg) + self.n + q]
                                for q in range(len(self.pe.npos))]
        for f in ops:
            img = {}
            for key in cols:
                for k2, c2 in self._act_one(f, key).items():
                    row = img.setdefault(k2, {})
                    row[key] = row.get(key, 0) + c2
            eqs.extend(img.values())
        return nullspace(eqs, cols)

    # ---- the z-weight is forced: it is -m + c
    def zwt(self, m):
        return self.zc - m


def _comps(r, n):
    if n == 0:
        return [()] if r == 0 else []
    if n == 1:
        return [(r,)]
    out = []
    for e in range(r + 1):
        for rest in _comps(r - e, n - 1):
            out.append((e,) + rest)
    return out
