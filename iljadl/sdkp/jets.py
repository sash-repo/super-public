"""Supercommutative polynomials in (x,t)-jets of component fields and the
coordinates x, y, t; total derivatives; the super dKP system in Kovalevskaya
form with respect to y."""
import sys; sys.path.insert(0, '.')
from fractions import Fraction as F
from spdo import mono_mul
import spdo

PAR = {}         # component name -> parity
def isodd(v): return len(v) == 3 and PAR[v[0]] == 1
COORD = {'x', 'y', 't'}

class Q:
    __slots__ = ('t',)
    def __init__(self, t=None): self.t = t or {}
    @staticmethod
    def var(v):
        if isodd(v): return Q({((), (v,)): F(1)})
        return Q({(((v, 1),), ()): F(1)})
    @staticmethod
    def const(c): return Q({((), ()): F(c)}) if c else Q()
    def __add__(self, o):
        if not isinstance(o, Q): o = Q.const(o)
        t = dict(self.t)
        for m, c in o.t.items():
            c2 = t.get(m, 0) + c
            if c2: t[m] = c2
            else: t.pop(m, None)
        return Q(t)
    __radd__ = __add__
    def __neg__(self): return Q({m: -c for m, c in self.t.items()})
    def __sub__(self, o): return self + (-o if isinstance(o, Q) else Q.const(-o))
    def __mul__(self, o):
        if not isinstance(o, Q):
            o = F(o); return Q({m: c * o for m, c in self.t.items()}) if o else Q()
        t = {}
        for m1, c1 in self.t.items():
            for m2, c2 in o.t.items():
                m, s = mono_mul(m1, m2)
                if s == 0: continue
                c = t.get(m, 0) + s * c1 * c2
                if c: t[m] = c
                else: t.pop(m)
        return Q(t)
    __rmul__ = lambda s, o: s * o
    def iszero(self): return not self.t

def mono_q(m):
    return Q({m: F(1)})

def derivation(q, act, parity):
    """apply the derivation with values act(v) (a Q) on generators, of given
    parity (0/1), to q; sign (-1)^{parity * p(prefix)}."""
    r = Q()
    for (ev, od), c in q.t.items():
        # even generators: no sign issues for position; put derivative in front
        oddpart = mono_q(((), od))
        for v, a in ev:
            av = act(v)
            if av is None or av.iszero(): continue
            ev2 = dict(ev); ev2[v] -= 1
            if not ev2[v]: del ev2[v]
            rest = mono_q((tuple(sorted(ev2.items())), ()))
            # X(e^a) = a e^{a-1} X(e): X(e) placed in front of everything (sign (+) since prefix empty... )
            r = r + (av * rest * oddpart) * (c * a)
        evpart = mono_q((ev, ()))
        for i, v in enumerate(od):
            av = act(v)
            if av is None or av.iszero(): continue
            pre = mono_q(((), od[:i])); post = mono_q(((), od[i + 1:]))
            sgn = -1 if (parity and i % 2) else 1
            r = r + (evpart * pre * av * post) * (c * sgn)
    return r

def Dx(q):
    def act(v):
        if v == ('x',): return Q.const(1)
        if len(v) == 3: return Q.var((v[0], v[1] + 1, v[2]))
        return None
    return derivation(q, act, 0)
def Dt(q):
    def act(v):
        if v == ('t',): return Q.const(1)
        if len(v) == 3: return Q.var((v[0], v[1], v[2] + 1))
        return None
    return derivation(q, act, 0)

# ---- build the system from system.py's component equations ----
def conv(p):
    """spdo.P with variables (name,k) -> Q with (name,k,0)."""
    r = Q()
    for (ev, od), c in p.t.items():
        m = Q.const(c)
        for v, a in ev:
            for _ in range(a): m = m * Q.var((v[0], v[1], 0))
        for v in od: m = m * Q.var((v[0], v[1], 0))
        r = r + m
    return r

def build():
    import system as S_
    for nm, p in spdo.PAR.items(): PAR[nm] = p
    l4, l6 = S_.lead4, S_.lead6
    RHS = {}
    for i in (1, 2):
        for comp in (0, 1):
            nm = f'U{i}{comp}'
            RHS[nm] = conv(l4[i][comp])
    for i in (1, 2):
        for comp in (0, 1):
            # U_{i+2}_y = (2/3)(U_i_t - (t6(U_i) - 3/2 t4(U_{i+2})))
            nm = f'U{i+2}{comp}'
            G = conv(l6[i][comp]) - conv(l4[i + 2][comp]) * F(3, 2)
            RHS[nm] = (Q.var((f'U{i}{comp}', 0, 1)) - G) * F(2, 3)
    for nm, q in RHS.items():
        for (ev, od) in q.t:
            for v in [e[0] for e in ev] + list(od):
                assert v[0][:2] in ('U1', 'U2', 'U3', 'U4'), (nm, v)
    return RHS

_cache = {}
def Dy_var(v, RHS):
    key = v
    if key in _cache: return _cache[key]
    nm, i, j = v
    q = RHS[nm]
    for _ in range(i): q = Dx(q)
    for _ in range(j): q = Dt(q)
    _cache[key] = q
    return q
def Dy(q, RHS):
    def act(v):
        if v == ('y',): return Q.const(1)
        if len(v) == 3: return Dy_var(v, RHS)
        return None
    return derivation(q, act, 0)

if __name__ == '__main__':
    RHS = build()
    for k, v in RHS.items(): print(k, len(v.t))
    # compatibility sanity: D_t of U1 equation vs ... (trivially fine)
