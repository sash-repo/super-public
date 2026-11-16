"""The Lie algebra K_lambda of Sect. 11 (anti-symmetric elements of U_lambda=U(sl2)/(C-lam^2+1))
in the conjugate picture sigma'(e)=e, sigma'(f)=f, sigma'(h)=-h, its triangular decomposition,
the anti-involution tau (e<->f, h->h) and the Shapovalov form of its Verma modules with
generalized highest weight chi in K_0^* (K_0 = odd polynomials in h)."""
import sympy as sp
from functools import lru_cache

h = sp.Symbol('h')

class U:
    """U_lambda: elements of weight 2d are X_d p(h), X_d=e^d (d>=0), f^{-d} (d<0)."""
    def __init__(self, lam, dom):
        self.lam = lam; self.dom = dom
        self.P = lambda e: sp.Poly(e, h, domain=dom)
        self.u = self.P((lam**2 - (h-1)**2)/4)          # e f = u(h), f e = u(h+2)
    def sh(self, p, s):                                # p(h) -> p(h+s)
        return p.compose(self.P(h + s))
    def Pp(self, b):  # e^b f^b
        r = self.P(1)
        for k in range(b): r = r * self.sh(self.u, -2*k)
        return r
    def Qq(self, b):  # f^b e^b
        r = self.P(1)
        for k in range(1, b+1): r = r * self.sh(self.u, 2*k)
        return r
    def XX(self, d1, d2):  # X_{d1} X_{d2} = X_{d1+d2} r(h)
        if d1 >= 0 and d2 >= 0 or d1 <= 0 and d2 <= 0: return self.P(1)
        if d1 > 0:  # e^a f^b
            a, b = d1, -d2
            if a >= b: return self.Pp(b)
            return self.sh(self.Pp(a), -2*(b-a))
        b, a = -d1, d2  # f^b e^a
        if b >= a: return self.Qq(a)
        return self.sh(self.Qq(b), 2*(a-b))
    def mul(self, x, y):
        out = {}
        for d1, p in x.items():
            for d2, q in y.items():
                r = self.XX(d1, d2) * self.sh(p, 2*d2) * q
                d = d1 + d2
                out[d] = out.get(d, self.P(0)) + r
        return {d: p for d, p in out.items() if not p.is_zero}
    def br(self, x, y):
        a = self.mul(x, y); b = self.mul(y, x)
        out = dict(a)
        for d, p in b.items(): out[d] = out.get(d, self.P(0)) - p
        return {d: p for d, p in out.items() if not p.is_zero}
    def sigp(self, x):   # sigma'(X_d p(h)) = p(-h) X_d = X_d p(-h-2d)
        return {d: p.compose(self.P(-h - 2*d)) for d, p in x.items()}
    def tau(self, x):    # tau(e^d p) = f^d p(h-2d); tau(f^m p) = e^m p(h+2m)
        return {-d: self.sh(p, -2*d) for d, p in x.items()}

class K:
    def __init__(self, Ua):
        self.U = Ua
    def basis(self, d, j):
        P = self.U.P
        if d > 0: return {d: P((h + d)**(2*j+1))}
        if d < 0: return {d: P((h + d)**(2*j+1))}   # f^m (h-m)^{2j+1}, m=-d
        return {0: P(h**(2*j+1))}
    def coords(self, x):
        """expand a K-element into basis keys (d,j); assert anti-symmetry."""
        out = {}
        for d, p in x.items():
            q = self.U.sh(p, -d)            # p(h)=q(h+d)
            for (k,), c in q.terms():
                assert k % 2 == 1, ("not antisymmetric", d, q)
                out[(d, (k-1)//2)] = c
        return out
    @lru_cache(maxsize=None)
    def brk(self, a, b):
        return tuple(sorted(self.coords(self.U.br(self.basis(*a), self.basis(*b))).items()))

class Verma:
    """M = U(K_-) v, keys of K_- are (d,j) with d<0, PBW monomials = sorted tuples."""
    def __init__(self, Kalg, chi):
        self.K = Kalg; self.chi = chi   # chi(j) = value on h^{2j+1}
        self.cache = {}
    def act(self, x, mono):
        """basis element x acting on monomial (tuple) -> dict mono->coef"""
        key = (x, mono)
        if key in self.cache: return self.cache[key]
        d = x[0]
        res = {}
        def add(dct, c=1):
            for m, v in dct.items():
                res[m] = res.get(m, 0) + c*v
        if d < 0:
            if not mono or x <= mono[0]:
                res = {(x,) + mono: 1}
            else:
                y = mono[0]; rest = mono[1:]
                for m, v in self.act(x, rest).items():
                    add(self.act(y, m), v)
                for z, c in self.K.brk(x, y):
                    add(self.act(z, rest), c)
        else:
            if not mono:
                if d == 0: res = {(): self.chi(x[1])}
            else:
                y = mono[0]; rest = mono[1:]
                for z, c in self.K.brk(x, y):
                    add(self.act(z, rest), c)
                for m, v in self.act(x, rest).items():
                    add(self.act(y, m), v)
        res = {m: sp.expand(v) for m, v in res.items() if sp.expand(v) != 0}
        self.cache[key] = res
        return res
    def form(self, y, z):
        """(y v | z v), y,z monomials; (g x|w)=(x|tau(g) w)"""
        if not y: return 1 if not z else 0
        g = y[0]; t = (-g[0], g[1])   # tau(B(-m,j)) = B(m,j)
        tot = 0
        for m, c in self.act(t, z).items():
            tot += c * self.form(y[1:], m)
        return sp.expand(tot)

def monomials(deg, N):
    """PBW monomials of total degree -deg with j<N"""
    keys = [(-d, j) for d in range(1, deg+1) for j in range(N)]
    keys.sort()
    out = []
    def rec(start, rem, cur):
        if rem == 0: out.append(tuple(cur)); return
        for i in range(start, len(keys)):
            k = keys[i]
            if -k[0] <= rem: rec(i, rem + k[0], cur + [k])
    rec(0, deg, [])
    return out

def gram(V, deg, N):
    ms = monomials(deg, N)
    return ms, sp.Matrix(len(ms), len(ms), lambda i, k: V.form(ms[i], ms[k]))
