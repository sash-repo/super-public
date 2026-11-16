#!/usr/bin/env python3
r"""
blattner.py --- realization of an abstract Z-graded Lie algebra g_{<=0} over F_2
by vector fields on O(m;1), m = dim g_-, all heights 1 (i.e. the restricted case).

Construction: the coinduced module  M = Hom_{u(g_0)}(u(g_{<=0}), K)
~ u(g_-)^*  with g_{<=0} acting by (X f)(u) = f(u X).  Here u(g_-) is the
restricted enveloping algebra; its p-map is FORCED, e^[2] being the unique
y in g_{2 deg e} with ad y = (ad e)^2 on g_{<=0}.  The PBW monomials e^S
(S a set of basis indices, increasing order) have coproduct
sum_{A u B = S} e^A (x) e^B, so the dual basis x^S multiplies as the
monomials of O(m;1): x^A x^B = x^{A u B} if A, B disjoint, 0 otherwise.
Hence X acts by the derivation  D_X = sum_i (X.x_i) d_i,
(X.x_i) = sum_T c_{T,i} x^T,  c_{T,i} = coefficient of the one-letter word e_i
(with the g_0-factor on the left killed) in e^T X.

Everything is checked: D_{[X,Y]} = [D_X, D_Y] on the whole basis.
"""
import itertools
from functools import lru_cache

from f2vf import Space


class Abstract:
    """basis 0..n-1; deg, par lists; br(i,j) -> int bitmask over basis (F_2)."""

    def __init__(self, deg, par, br, names=None):
        self.n = len(deg)
        self.deg = list(deg)
        self.par = list(par)
        self._br = br
        self.names = names or [str(i) for i in range(self.n)]
        self._cache = {}

    def br(self, i, j):
        k = (i, j)
        v = self._cache.get(k)
        if v is None:
            v = self._cache[k] = self._br(i, j)
        return v

    def br_vec(self, a, b):
        """bracket of two bitmask vectors"""
        out = 0
        ia = a
        i = 0
        while ia:
            if ia & 1:
                jb = b
                j = 0
                while jb:
                    if jb & 1:
                        out ^= self.br(i, j)
                    jb >>= 1
                    j += 1
            ia >>= 1
            i += 1
        return out


def bits(v):
    i = 0
    while v:
        if v & 1:
            yield i
        v >>= 1
        i += 1


def solve_f2(rows, rhs, nvar):
    """solve sum_v y_v rows[v] = rhs over F_2 (rows[v], rhs ints). Return y (int) or None."""
    piv = {}
    for v in range(nvar):
        r = rows[v]
        c = 1 << v
        while r:
            p = r.bit_length() - 1
            if p in piv:
                r ^= piv[p][0]
                c ^= piv[p][1]
            else:
                piv[p] = (r, c)
                break
    r = rhs
    c = 0
    while r:
        p = r.bit_length() - 1
        if p not in piv:
            return None
        r ^= piv[p][0]
        c ^= piv[p][1]
    return c


class Realization:
    def __init__(self, A, weights=None, verbose=True):
        self.A = A
        neg = [i for i in range(A.n) if A.deg[i] < 0]
        # order: by degree (-1 first), then index
        neg.sort(key=lambda i: (-A.deg[i], i))
        self.neg = neg
        self.pos = {b: p for p, b in enumerate(neg)}      # basis index -> coordinate
        self.m = len(neg)
        self.pmap = {}
        for b in neg:
            self.pmap[b] = self._pmap(b)
        degs = [-A.deg[b] for b in neg]
        pars = [A.par[b] for b in neg]
        self.sp = Space([1] * self.m, degs, pars, weights)
        self._mult = {}

    # --- the forced p-map on g_-
    def _pmap(self, t):
        A = self.A
        d2 = 2 * A.deg[t]
        cands = [c for c in range(A.n) if A.deg[c] == d2]
        # rows: ad(c) on the whole basis, flattened
        n = A.n

        def adflat(vecfun):
            out = 0
            for b in range(n):
                out |= vecfun(b) << (b * n)
            return out
        target = adflat(lambda b: A.br_vec(1 << t, A.br(t, b)))
        if not cands:
            if target:
                raise ValueError('no p-map for basis element %s: (ad e)^2 != 0 but g_{2deg} = 0'
                                 % A.names[t])
            return 0
        rows = [adflat(lambda b, c=c: A.br(c, b)) for c in cands]
        y = solve_f2(rows, target, len(cands))
        if y is None:
            raise ValueError('no p-map for %s' % A.names[t])
        out = 0
        for v in bits(y):
            out |= 1 << cands[v]
        return out

    # --- u(g_-): words are tuples of coordinate positions, strictly increasing
    def mult_letter(self, S, a):
        """e^S * e_a  (a a coordinate position) -> set of words"""
        key = (S, a)
        r = self._mult.get(key)
        if r is not None:
            return r
        if not S or a > S[-1]:
            r = {S + (a,)}
        else:
            t = S[-1]
            Sp = S[:-1]
            r = set()
            if a == t:
                for c in bits(self.pmap[self.neg[t]]):
                    r ^= self.mult_letter(Sp, self.pos[c])
            else:
                # e^{S'} e_t e_a = e^{S'} e_a e_t + e^{S'} [e_t, e_a]
                for W in self.mult_letter(Sp, a):
                    r ^= self.mult_letter(W, t)
                for c in bits(self.A.br(self.neg[t], self.neg[a])):
                    r ^= self.mult_letter(Sp, self.pos[c])
        r = frozenset(r)
        self._mult[key] = r
        return r

    def mult_elem(self, words, v):
        """(sum of words) * (element v of g_-, bitmask over basis)"""
        out = set()
        for W in words:
            for c in bits(v):
                assert self.A.deg[c] < 0
                out ^= self.mult_letter(W, self.pos[c])
        return out

    def words_of_degree(self, dg):
        """all increasing words T with sum of degrees = dg (dg < 0), T nonempty"""
        out = []
        degs = [self.A.deg[b] for b in self.neg]

        def rec(start, rem, cur):
            if rem == 0 and cur:
                out.append(tuple(cur))
                return
            for p in range(start, self.m):
                if degs[p] >= rem:          # degs negative; need degs[p] >= rem
                    cur.append(p)
                    rec(p + 1, rem - degs[p], cur)
                    cur.pop()
        rec(0, dg, [])
        return out

    def field(self, X):
        """D_X for a basis element X of g_{<=0}"""
        A = self.A
        sp = self.sp
        res = set()
        dX = A.deg[X]
        for i in range(self.m):
            dT = A.deg[self.neg[i]] - dX
            if dT > 0:
                continue
            Ts = [()] if dT == 0 else self.words_of_degree(dT)
            for T in Ts:
                if dX < 0:
                    prod = self.mult_elem({T}, 1 << X)
                else:
                    prod = set()
                    for k in range(len(T)):
                        words = {()}
                        for j, t in enumerate(T):
                            v = A.br(self.neg[t], X) if j == k else (1 << self.neg[t])
                            if v == 0:
                                words = set()
                                break
                            words = self.mult_elem(words, v)
                        prod ^= words
                    if not T:
                        prod = set()
                if (i,) in prod:
                    Apack = 0
                    for t in T:
                        Apack |= sp.one[t]
                    res ^= {(Apack << 5) | i}
        return res

    def realize(self, check=True):
        A = self.A
        D = {X: self.field(X) for X in range(A.n)}
        if check:
            sp = self.sp
            bad = 0
            for X in range(A.n):
                for Y in range(X + 1, A.n):
                    lhs = sp.bracket(D[X], D[Y])
                    rhs = set()
                    for c in bits(A.br(X, Y)):
                        rhs ^= D[c]
                    if lhs != rhs:
                        bad += 1
            if bad:
                raise AssertionError('realization is not a homomorphism: %d bad pairs' % bad)
            # faithful on g_0? (it is injective on g_- automatically)
        self.D = D
        return D


# ------------------------------------------------------------------ N~=1 core
def neg_pmap_choices(A):
    """for each basis element e of g_-: the solutions y in g_{2deg e} of
    ad y = (ad e)^2 ON g_- ONLY: (particular, [basis of the ambiguity])"""
    neg = [i for i in range(A.n) if A.deg[i] < 0]
    n = A.n
    out = {}
    for t in neg:
        d2 = 2 * A.deg[t]
        cands = [c for c in range(n) if A.deg[c] == d2]

        def flat(fun):
            v = 0
            for j, b in enumerate(neg):
                v |= fun(b) << (j * n)
            return v
        target = flat(lambda b: A.br_vec(1 << t, A.br(t, b)))
        rows = [flat(lambda b, c=c: A.br(c, b)) for c in cands]
        y = solve_f2(rows, target, len(cands)) if cands else (0 if target == 0 else None)
        if y is None:
            raise ValueError('no p-map on g_- itself for %s' % A.names[t])
        part = 0
        for v in bits(y or 0):
            part |= 1 << cands[v]
        # ambiguity: kernel of c -> ad c on g_-
        from f2vf import kernel_bits
        ker = kernel_bits(rows) if cands else []
        amb = []
        for kv in ker:
            z = 0
            for v in bits(kv):
                z |= 1 << cands[v]
            amb.append(z)
        out[t] = (part, amb)
    return out


def g0_prime(A, pm):
    """{x in g_0 : (ad e)^2 x = [e^[2], x] for all e in g_-}, for the p-map pm"""
    zero = [i for i in range(A.n) if A.deg[i] == 0]
    rows = []
    for x in zero:
        v = 0
        sh = 0
        for t, y in pm.items():
            lhs = A.br_vec(1 << t, A.br(t, x))
            rhs = A.br_vec(y, 1 << x)
            v |= (lhs ^ rhs) << sh
            sh += A.n
        rows.append(v)
    from f2vf import kernel_bits
    ker = kernel_bits(rows)
    out = []
    for kv in ker:
        z = 0
        for v in bits(kv):
            z |= 1 << zero[v]
        out.append(z)
    return out


def best_core(A, maxtries=1 << 16):
    """Search the p-maps on g_- (particular + ambiguity) for those making the
    largest part g_0' of g_0 compatible.  Returns (dim, list of (pm, g0' basis))."""
    ch = neg_pmap_choices(A)
    items = list(ch.items())
    namb = [len(a) for _, (p, a) in items]
    total = 1 << sum(namb)
    if total > maxtries:
        raise ValueError('too many p-map choices: 2^%d' % sum(namb))
    best = -1
    found = []
    for code in range(total):
        pm = {}
        c = code
        for t, (p, amb) in items:
            y = p
            for z in amb:
                if c & 1:
                    y ^= z
                c >>= 1
            pm[t] = y
        g0p = g0_prime(A, pm)
        if len(g0p) > best:
            best = len(g0p)
            found = [(pm, g0p)]
        elif len(g0p) == best:
            found.append((pm, g0p))
    return best, found, total


def sub_abstract(A, vecs_neg, vecs0):
    """the abstract algebra spanned by basis elements of g_- and vectors (bitmasks
    over A's basis) spanning a subalgebra g_0' of g_0."""
    from absutil import Coords
    # the span of vecs0 must be parity-graded: project and re-echelonize
    from f2vf import BitEchelon
    oddmask = 0
    for i in range(A.n):
        if A.par[i]:
            oddmask |= 1 << i
    Espan = BitEchelon()
    for v in vecs0:
        Espan.add(v)
    E0, E1 = BitEchelon(), BitEchelon()
    for v in vecs0:
        v1 = v & oddmask
        v0 = v ^ v1
        if Espan.reduce(v0) or Espan.reduce(v1):
            raise ValueError("g_0' is not parity-graded")
        if v0:
            E0.add(v0)
        if v1:
            E1.add(v1)
    vecs0 = list(E0.piv.values()) + list(E1.piv.values())
    basis = [1 << i for i in range(A.n) if A.deg[i] < 0] + list(vecs0)
    degs = [A.deg[i] for i in range(A.n) if A.deg[i] < 0] + [0] * len(vecs0)
    pars = [A.par[i] for i in range(A.n) if A.deg[i] < 0]
    pars += [0] * len(E0) + [1] * len(E1)
    co = Coords(basis, as_int=True)
    return Abstract(degs, pars, lambda i, j: co(A.br_vec(basis[i], basis[j])))
