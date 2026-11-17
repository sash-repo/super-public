"""Superfields on the (1|n)-dimensional superline, used for Subsecs. 5.10, 7.1 and 9.4 of the text.

Elements of Lambda(gens) (x) sympy functions of x: the odd coordinates theta (named 'ta', 'tb',
'tc', 'th') and odd constants e1, e2, ... (C-points: a superfield of either parity is made even by
multiplying it with an odd constant).  D(F, t) = d_t F + t F' is the operator D_theta = theta d_x +
d_theta, so D_theta^2 = d_x.

Run directly, it checks the cocycle property used in Proposition 5.10.1 and in Statement 5.10.3:
c_n = St_n is a 1-cocycle of k(1|n) with values in F_{1-n/2} for n = 0, 1, 2 (and c_2' = H' as
well), D_theta1 D_theta2 D_theta3 is a 1-cocycle with values in F_{1/2}, and the n = 0 Miura map
is Hamiltonian, (1/2) A d A^* = ad^*(U, -1).  Run:  python3 nsgrass.py
"""
import itertools, sympy as sp
x = sp.Symbol('x')

class G:
    """element of Lambda(gens) (x) sympy-functions-of-x; gens are strings, ordered by name"""
    def __init__(s, d=None):
        s.d = {k: v for k, v in (d or {}).items() if sp.simplify(v) != 0}
    @staticmethod
    def c(v): return G({(): sp.sympify(v)})
    @staticmethod
    def g(name, coef=1): return G({(name,): sp.sympify(coef)})
    def __add__(s, o):
        o = o if isinstance(o, G) else G.c(o); d = dict(s.d)
        for k, v in o.d.items(): d[k] = d.get(k, 0) + v
        return G(d)
    __radd__ = __add__
    def __neg__(s): return G({k: -v for k, v in s.d.items()})
    def __sub__(s, o): return s + (-(o if isinstance(o, G) else G.c(o)))
    def __rsub__(s, o): return (-s) + o
    def __mul__(s, o):
        o = o if isinstance(o, G) else G.c(o); d = {}
        for k1, v1 in s.d.items():
            for k2, v2 in o.d.items():
                if set(k1) & set(k2): continue
                seq = list(k1) + list(k2); sign = 1
                # bubble sort
                for i in range(len(seq)):
                    for j in range(len(seq) - 1 - i):
                        if seq[j] > seq[j + 1]:
                            seq[j], seq[j + 1] = seq[j + 1], seq[j]; sign = -sign
                k = tuple(seq); d[k] = d.get(k, 0) + sign * v1 * v2
        return G(d)
    def __rmul__(s, o): return G.c(o) * s
    def dx(s, n=1): return G({k: sp.diff(v, x, n) for k, v in s.d.items()})
    def dth(s, t):  # left derivative d/d t
        d = {}
        for k, v in s.d.items():
            if t in k:
                i = k.index(t); d[tuple(k[:i] + k[i + 1:])] = d.get(tuple(k[:i] + k[i + 1:]), 0) + (-1) ** i * v
        return G(d)
    def iszero(s): return all(sp.simplify(sp.expand(v)) == 0 for v in s.d.values())
    def __repr__(s): return repr({k: sp.simplify(v) for k, v in s.d.items()})

def D(F, t): return F.dth(t) + G.g(t) * F.dx()
def ok(name, cond): print(('PASS ' if cond else 'FAIL ') + name)

f = [sp.Function('f%d' % i)(x) for i in range(10)]

def Kact(H, F, lam, ths):  # K_H acting on F alpha^lam ; K_H = H d + 1/2 (-1)^{p(H)} sum D_i(H) D_i  (H homogeneous)
    pH = parity(H)
    r = H * F.dx() + G.c(lam) * H.dx() * F
    for th in ths:
        r = r + G.c(sp.Rational(1, 2) * (-1) ** pH) * D(H, th) * D(F, th)
    return r
def parity(F):
    ps = {sum(1 for g in k) % 2 for k in F.d}
    assert len(ps) <= 1, ps
    return ps.pop() if ps else 0
def bracket(H, Gg, ths):  # K_H(G) - G'H... {H,G} = H G' - H' G + 1/2 (-1)^{pH} sum D_iH D_iG
    pH = parity(H)
    r = H * Gg.dx() - H.dx() * Gg
    for th in ths:
        r = r + G.c(sp.Rational(1, 2) * (-1) ** pH) * D(H, th) * D(Gg, th)
    return r
import random
random.seed(1)
def rpoly(deg=3): return sum(random.randint(-3, 3) * x ** j for j in range(deg + 1))
def rsf(ths, odd, eps):  # random superfield of given total parity using odd constant eps
    F = G()
    for r in range(len(ths) + 1):
        for comb in itertools.combinations(ths, r):
            mon = G.c(1)
            for th in comb: mon = mon * G.g(th)
            if (r % 2) != odd: mon = mon * G.g(eps)
            F = F + mon * G.c(rpoly())
    return F
def cocycle_test(ths, cop, lam, name):
    allok = True
    for (oH, oG) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        H = rsf(ths, 0, 'e7') if oH == 0 else rsf(ths, 1, 'e7')
        Gg = rsf(ths, 0, 'e8') if oG == 0 else rsf(ths, 1, 'e8')
        # C-points: an odd superfield is made even by an odd constant
        He = H if oH == 0 else H * G.g('e5')
        Ge = Gg if oG == 0 else Gg * G.g('e6')
        lhs = cop(bracket(He, Ge, ths))
        rhs = Kact(He, cop(Ge), lam, ths) - Kact(Ge, cop(He), lam, ths)
        allok = allok and (lhs - rhs).iszero()
    ok(name, allok)

if __name__ == '__main__':
    # the n = 0 Miura map U = -V' - V^2 is Hamiltonian: (1/2) A d A^* = ad^*(U, -1)
    V = f[0]; h = f[1]
    A = lambda w: -sp.diff(w, x) - 2 * V * w
    As = lambda w: sp.diff(w, x) - 2 * V * w
    U = -sp.diff(V, x) - V ** 2
    lhs = sp.Rational(1, 2) * A(sp.diff(As(h), x))
    rhs = -sp.Rational(1, 2) * sp.diff(h, x, 3) - U * sp.diff(h, x) - sp.diff(U * h, x)
    ok('n=0: (1/2) A d A^* = ad^*(U,-1)', sp.simplify(lhs - rhs) == 0)
    cocycle_test([], lambda H: H.dx(2), 1, "n=0: c_0(H)=H'' is a 1-cocycle with values in F_1")
    cocycle_test(['ta'], lambda H: D(H.dx(), 'ta'), sp.Rational(1, 2), "n=1: c_1(H)=D(H') is a 1-cocycle in F_{1/2}")
    cocycle_test(['ta', 'tb'], lambda H: D(D(H, 'tb'), 'ta'), 0, 'n=2: c_2(H)=D1D2(H) is a 1-cocycle in F_0')
    cocycle_test(['ta', 'tb'], lambda H: H.dx(), 0, "n=2: H -> H' is a 1-cocycle in F_0")
    th3 = ['ta', 'tb', 'tc']
    cocycle_test(th3, lambda H: D(D(D(H, 'tc'), 'tb'), 'ta'), sp.Rational(1, 2),
                 'n=3: D1D2D3(H) is a 1-cocycle in F_{1/2}')
