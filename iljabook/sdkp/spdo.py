"""Super pseudodifferential operators in D = d_theta + theta d_x, D^2 = d_x,
with superfield coefficients written in components U = U0 + theta U1.
Exact arithmetic over Q; Grassmann jets handled by a supercommutative
polynomial algebra.  Self-tests at the bottom."""
from fractions import Fraction
from itertools import product

# ---------- supercommutative polynomials in jet variables ----------
# a variable is (name, k) = k-th x-derivative of the component 'name';
# PAR[name] = 0/1
PAR = {}
def par_var(v): return PAR[v[0]]

def mono_mul(m1, m2):
    e1, o1 = m1; e2, o2 = m2
    if set(o1) & set(o2): return None, 0
    ev = dict(e1)
    for v, a in e2: ev[v] = ev.get(v, 0) + a
    seq = list(o1) + list(o2)
    # sign of sorting permutation
    sgn = 1
    s = seq[:]
    for i in range(len(s)):
        for j in range(len(s) - 1 - i):
            if s[j] > s[j + 1]:
                s[j], s[j + 1] = s[j + 1], s[j]; sgn = -sgn
    return (tuple(sorted(ev.items())), tuple(s)), sgn

class P:
    __slots__ = ('t',)
    def __init__(self, t=None): self.t = t or {}
    @staticmethod
    def var(name, k=0):
        v = (name, k)
        if PAR[name]: return P({((), (v,)): Fraction(1)})
        return P({(((v, 1),), ()): Fraction(1)})
    @staticmethod
    def const(c):
        return P({((), ()): Fraction(c)}) if c else P()
    def __add__(self, o):
        if not isinstance(o, P): o = P.const(o)
        t = dict(self.t)
        for m, c in o.t.items():
            c2 = t.get(m, 0) + c
            if c2: t[m] = c2
            else: t.pop(m, None)
        return P(t)
    __radd__ = __add__
    def __neg__(self): return P({m: -c for m, c in self.t.items()})
    def __sub__(self, o): return self + (-o if isinstance(o, P) else P.const(-o))
    def __mul__(self, o):
        if not isinstance(o, P):
            o = Fraction(o)
            return P({m: c * o for m, c in self.t.items()}) if o else P()
        t = {}
        for m1, c1 in self.t.items():
            for m2, c2 in o.t.items():
                m, s = mono_mul(m1, m2)
                if s == 0: continue
                c = t.get(m, 0) + s * c1 * c2
                if c: t[m] = c
                else: t.pop(m)
        return P(t)
    __rmul__ = lambda self, o: self * o
    def iszero(self): return not self.t
    def parity(self):
        ps = {len(m[1]) % 2 for m in self.t}
        assert len(ps) <= 1, "inhomogeneous"
        return ps.pop() if ps else 0
    def dx(self):
        # total x-derivative (even derivation)
        r = P()
        for (ev, od), c in self.t.items():
            for i, (v, a) in enumerate(ev):
                ev2 = dict(ev); ev2[v] -= 1
                if not ev2[v]: del ev2[v]
                m, s = mono_mul((tuple(sorted(ev2.items())), ()),
                                ((((v[0], v[1] + 1), 1),), ()))
                m, s2 = mono_mul(m, ((), od))
                r = r + P({m: c * a * s * s2})
            for i, v in enumerate(od):
                od2 = list(od); od2[i] = (v[0], v[1] + 1)
                # replacing in place keeps the position; re-sort
                m, s = mono_mul((ev, ()), ((), ()))
                seq = od2; sgn = 1; s_ = seq[:]
                if len(set(s_)) < len(s_): continue
                for a_ in range(len(s_)):
                    for b_ in range(len(s_) - 1 - a_):
                        if s_[b_] > s_[b_ + 1]:
                            s_[b_], s_[b_ + 1] = s_[b_ + 1], s_[b_]; sgn = -sgn
                r = r + P({(ev, tuple(s_)): c * sgn})
        return r
    def __repr__(self):
        if not self.t: return '0'
        out = []
        for (ev, od), c in sorted(self.t.items(), key=str):
            f = '*'.join([f'{v[0]}{"x"*v[1]}' + (f'^{a}' if a > 1 else '') for v, a in ev] +
                         [f'{v[0]}{"x"*v[1]}' for v in od])
            out.append(f'{c}' + ('*' + f if f else ''))
        return ' + '.join(out)

# ---------- superfields U = U0 + theta U1 ----------
class S:
    """superfield (a, b) = a + theta b; parity p = parity of a."""
    __slots__ = ('a', 'b', 'p')
    def __init__(self, a, b, p): self.a, self.b, self.p = a, b, p
    @staticmethod
    def field(name, p):
        PAR[name + '0'] = p; PAR[name + '1'] = 1 - p
        return S(P.var(name + '0'), P.var(name + '1'), p)
    @staticmethod
    def const(c): return S(P.const(c), P(), 0)
    def __add__(self, o):
        if o.iszero(): return self
        if self.iszero(): return o
        assert self.p == o.p
        return S(self.a + o.a, self.b + o.b, self.p)
    def __neg__(self): return S(-self.a, -self.b, self.p)
    def __sub__(self, o): return self + (-o)
    def scale(self, c): return S(self.a * c, self.b * c, self.p)
    def __mul__(self, o):
        # (a + th b)(c + th d) = ac + th(b c + (-1)^{p(a)} a d)
        s = -1 if self.p else 1
        return S(self.a * o.a, self.b * o.a + (self.a * o.b) * s, (self.p + o.p) % 2)
    def D(self):
        return S(self.b, self.a.dx(), 1 - self.p)
    def dx(self): return S(self.a.dx(), self.b.dx(), self.p)
    def iszero(self): return self.a.iszero() and self.b.iszero()
    def __repr__(self): return f'[{self.a} | th: {self.b}]'

ZERO = lambda p=0: S(P(), P(), p)

def gbinom(n, k):
    # generalized binomial, k >= 0
    r = Fraction(1)
    for j in range(k): r = r * (n - j) / (j + 1)
    return r

def sbinom(k, i):
    if k % 2 == 0 and i % 2 == 1: return Fraction(0)
    return gbinom(k // 2, i // 2)

def Dpow(f, i):
    for _ in range(i): f = f.D()
    return f

# ---------- super PDO: dict power -> superfield ----------
class Op:
    def __init__(self, t=None, kmin=-12): self.t = t or {}; self.kmin = kmin
    def __add__(self, o):
        t = dict(self.t)
        for k, f in o.t.items():
            t[k] = t[k] + f if k in t else f
            if t[k].iszero(): del t[k]
        return Op(t, min(self.kmin, o.kmin))
    def __neg__(self): return Op({k: -f for k, f in self.t.items()}, self.kmin)
    def __sub__(self, o): return self + (-o)
    def __mul__(self, o):
        kmin = max(self.kmin, o.kmin)
        t = {}
        for k, f in self.t.items():
            for l, g in o.t.items():
                # f D^k g D^l = sum_i [k,i] (-1)^{p(g)(k-i)} f (D^i g) D^{k-i+l}
                i = 0
                while k - i + l >= kmin:
                    c = sbinom(k, i)
                    if k >= 0 and i > k: break
                    if c:
                        sg = -1 if (g.p * (k - i)) % 2 else 1
                        term = (f * Dpow(g, i)).scale(c * sg)
                        n = k - i + l
                        if not term.iszero():
                            t[n] = t[n] + term if n in t else term
                            if t[n].iszero(): del t[n]
                    i += 1
        return Op(t, kmin)
    def plus(self): return Op({k: f for k, f in self.t.items() if k >= 0}, self.kmin)
    def minus(self): return Op({k: f for k, f in self.t.items() if k < 0}, self.kmin)
    def parity(self):
        ps = {(f.p + k) % 2 for k, f in self.t.items()}
        assert len(ps) <= 1; return ps.pop() if ps else 0

def scomm(A, B):
    pa, pb = A.parity(), B.parity()
    return A * B + B * A if (pa and pb) else A * B - B * A

def Dop(k, kmin=-12): return Op({k: S.const(1)}, kmin)

if __name__ == '__main__':
    # self-tests
    f = S.field('f', 0); g = S.field('g', 1)
    for h in (f, g):
        lhs = Dop(1) * (Dop(-1) * Op({0: h}))
        ok = (lhs - Op({0: h})).t
        assert all(k < -8 for k in ok), ok
        lhs = Dop(2) * Op({0: h}); rhs = Op({0: h.dx(), 2: h})
        assert not (lhs - rhs).t
        # associativity
        A = Op({1: f, -2: g}); B = Op({0: g, -1: f}); C = Op({-1: f})
        d = (A * B) * C - A * (B * C)
        assert all(k < -8 for k in d.t), sorted(d.t)
    # D^2 = d_x on superfields
    assert ((f.D().D()) - f.dx()).iszero() and (g.D().D() - g.dx()).iszero()
    print('spdo self-tests pass')
