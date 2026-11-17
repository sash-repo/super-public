"""Generation of A_c by A_c^{<=4}, over Q[c] (coefficients fmpq_poly), and the classical fiber.
Elements at mode degree Delta are state vectors; reduce them to quasi-primary-complement
coordinates (sum over n of K_n-coordinates) by the c-independent relations
d u + (k - Delta) u = 0, u in V_k."""
import sys
import os; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"), os.path.dirname(os.path.abspath(__file__))]
from flint import fmpq_poly, fmpq
from fractions import Fraction as Fr
from va import VA, parts, binom
from exactlin import rank, solve
import sympy as sp

def conv(x):
    x = Fr(x); return fmpq_poly([fmpq(x.numerator, x.denominator)])
C = fmpq_poly([0, 1])

def lin(*ps):
    out = {}
    for a, st in ps:
        for k, v in st.items(): out[k] = out.get(k, 0) + a * v
    return {k: v for k, v in out.items() if v}
def wt(m): return sum(m)

class Red:
    """reduction at mode degree Delta, weights <= N"""
    def __init__(self, Delta, N, V):
        self.D = Delta; self.N = N; self.V = V   # V: a VA (any c) used for L_{-1} only
        self.K = {}; self.dec = {}
        for n in range(2, N + 1):
            Vn = parts(n); Vm = parts(n - 1) if n > 2 else []
            dimg = [self._dvec(u, Vn) for u in Vm]       # images of d, rows
            # choose K_n: monomials completing span(dimg)
            sel = []
            cur = list(dimg)
            for p in Vn:
                row = [Fr(1) if q == p else Fr(0) for q in Vn]
                if rank(cur + [row]) > rank(cur) if cur else True:
                    cur.append(row); sel.append(p)
            self.K[n] = sel
            self.dec[n] = (Vn, Vm, dimg, sel)
    def _dvec(self, u, Vn):
        d = VA(0).L(-1, {u: Fr(1)})
        return [d.get(q, Fr(0)) for q in Vn]
    def reduce(self, st):
        """st: dict mono -> fmpq_poly. returns dict n -> list of fmpq_poly (K_n coords)"""
        st = dict(st)
        out = {}
        for n in range(self.N, 1, -1):
            Vn, Vm, dimg, sel = self.dec[n]
            comp = {k: v for k, v in st.items() if wt(k) == n}
            for k in comp: st.pop(k)
            if any(wt(k) > self.N for k in st): raise ValueError
            # solve comp = sum a_p e_p (p in sel) + sum b_u d(u): basis matrix rows
            rows = [[Fr(1) if q == p else Fr(0) for q in Vn] for p in sel] + dimg
            # invert once: express each monomial q of Vn
            if not hasattr(self, '_inv'): self._inv = {}
            if n not in self._inv:
                inv = {}
                M = [list(col) for col in zip(*rows)]   # columns = basis vectors
                for q in Vn:
                    rhs = [Fr(1) if q2 == q else Fr(0) for q2 in Vn]
                    inv[q] = solve(M, rhs)
                self._inv[n] = inv
            inv = self._inv[n]
            a = [0] * len(sel); b = [0] * len(Vm)
            for q, v in comp.items():
                x = inv[q]
                for i in range(len(sel)): a[i] = a[i] + v * conv(x[i]) if x[i] else a[i]
                for i in range(len(Vm)): b[i] = b[i] + v * conv(x[len(sel) + i]) if x[len(sel) + i] else b[i]
            out[n] = a
            # d u == -(k - Delta) u with k = n-1
            for i, u in enumerate(Vm):
                if b[i] != 0:
                    st[u] = st.get(u, 0) - (n - 1 - self.D) * b[i]
                    if not st[u]: st.pop(u)
        return out

class Alg:
    def __init__(self, N):
        self.V = VA(C, conv); self.N = N
    def br(self, e1, e2):
        D1, s1 = e1; D2, s2 = e2
        out = {}
        for a, x in s1.items():
            p = wt(a) - 1 - D1
            for b, y in s2.items():
                for j in range(0, wt(a) + wt(b)):
                    r = self.V.mode({a: conv(1)}, j, {b: conv(1)})
                    if r:
                        out = lin((1, out), (x * y * conv(binom(p, j)), r))
        out = {k: v for k, v in out.items() if k != ()}
        return (D1 + D2, out)

LAM = {(2, 2): conv(1), (4,): conv(Fr(-3, 5))}
TT = {(2,): conv(1)}
def Lm(a): return (3 - a, LAM)
def Tm(a): return (1 - a, TT)

c = sp.Symbol('c')
def topoly(p):
    return sum(sp.Rational(int(q.p), int(q.q)) * c**i for i, q in enumerate(p.coeffs())) if p != 0 else sp.Integer(0)
