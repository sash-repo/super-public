"""N-extended superfields over jets of component fields (exact, over Q).
Superfield = sum_I theta^I c_I, theta's on the LEFT; I = sorted tuple of
indices 1..N.  D_i = d/dtheta_i + theta_i d/dx (D_i^2 = d/dx),
Q_i = d/dtheta_i - theta_i d/dx (anticommutes with every D_j)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sdkp'))
from spdo import P, PAR
from fractions import Fraction
from itertools import combinations

def tmul(I, J):
    """theta^I theta^J = sign * theta^K, or (0, None)."""
    if set(I) & set(J): return 0, None
    seq = list(I) + list(J); s = 1
    for a in range(len(seq)):
        for b in range(len(seq) - 1 - a):
            if seq[b] > seq[b + 1]:
                seq[b], seq[b + 1] = seq[b + 1], seq[b]; s = -s
    return s, tuple(seq)

class SF:
    def __init__(self, N, c=None, p=None):
        self.N = N; self.c = c or {}; self.p = p   # p = parity of the superfield
    @staticmethod
    def field(N, names, p):
        """names: dict I -> component name; parity of c_I = p + |I| mod 2"""
        c = {}
        for I, nm in names.items():
            PAR[nm] = (p + len(I)) % 2
            c[I] = P.var(nm)
        return SF(N, c, p)
    def comp_par(self, I): return (self.p + len(I)) % 2
    def __add__(self, o):
        c = dict(self.c)
        for I, v in o.c.items():
            w = c[I] + v if I in c else v
            if w.iszero(): c.pop(I, None)
            else: c[I] = w
        return SF(self.N, c, self.p if self.p is not None else o.p)
    def __neg__(self): return SF(self.N, {I: -v for I, v in self.c.items()}, self.p)
    def __sub__(self, o): return self + (-o)
    def scale(self, k): return SF(self.N, {I: v * k for I, v in self.c.items()}, self.p)
    def __mul__(self, o):
        c = {}
        for I, a in self.c.items():
            pa = self.comp_par(I)
            for J, b in o.c.items():
                s, K = tmul(I, J)
                if not s: continue
                if pa and len(J) % 2: s = -s
                v = (a * b) * s
                c[K] = c[K] + v if K in c else v
        c = {K: v for K, v in c.items() if not v.iszero()}
        return SF(self.N, c, (self.p + o.p) % 2)
    def dx(self):
        return SF(self.N, {I: v.dx() for I, v in self.c.items() if not v.dx().iszero()}, self.p)
    def dth(self, i):
        c = {}
        for I, v in self.c.items():
            if i in I:
                k = I.index(i); s = -1 if k % 2 else 1
                J = I[:k] + I[k + 1:]
                c[J] = c[J] + v * s if J in c else v * s
        return SF(self.N, c, 1 - self.p)
    def thmul(self, i):
        c = {}
        for I, v in self.c.items():
            s, K = tmul((i,), I)
            if s: c[K] = c[K] + v * s if K in c else v * s
        return SF(self.N, c, 1 - self.p)
    def D(self, i): return self.dth(i) + self.dx().thmul(i)
    def Q(self, i): return self.dth(i) - self.dx().thmul(i)
    def iszero(self): return all(v.iszero() for v in self.c.values())

# ---- derivations of the jet algebra ----
def derive(p, act, xpar):
    """apply the derivation of parity xpar sending jet var v -> act(v) (a P)."""
    r = P()
    for (ev, od), c in p.t.items():
        for idx, (v, a) in enumerate(ev):
            ev2 = dict(ev); ev2[v] -= 1
            if not ev2[v]: del ev2[v]
            rest = P({(tuple(sorted(ev2.items())), od): Fraction(1)})
            r = r + act(v) * rest * (c * a)
        evp = P({(ev, ()): Fraction(1)})
        for j, v in enumerate(od):
            pre = P({((), od[:j]): Fraction(1)}); post = P({((), od[j + 1:]): Fraction(1)})
            s = -1 if (xpar and j % 2) else 1
            r = r + evp * pre * act(v) * post * (c * s)
    return r

def dxk(p, k):
    for _ in range(k): p = p.dx()
    return p

def oddness(p):
    """set of fermion degrees occurring in p"""
    return sorted({len(od) for (ev, od) in p.t})

def split_by_odd(p):
    out = {}
    for (ev, od), c in p.t.items():
        out.setdefault(len(od), P())
        out[len(od)] = out[len(od)] + P({(ev, od): c})
    return out
