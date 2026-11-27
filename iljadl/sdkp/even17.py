"""The even symmetries of order at most 1 found by the census (symm1.py, sigma = 0, coefficients
of degree <= 3 in x, y, t): a space of dimension 17, the sum of the fields T_f, S_f
(f in {1, t, t^2, t^3}), of two families f(t) d_t + 2/3 f'(t) y d_y + ... and
g(t) d_y + 2/3 g'(t) y d_x + ... with f, g of degree <= 3, and of one more scaling.  Needs sols_s0_c3_21.pkl and
sols_s1_c3_21.pkl (run symm1.py first).
A point field f d_t (resp. g d_y + c y d_x) has the evolutionary form phi = -f U_t
(resp. -g U_y - c y U_x)."""
import sys; sys.path.insert(0, '.')
from fractions import Fraction as F
import sympy as sp
from brackets import S0, E, express, Q
from symm import condition
X, Y, T = sp.symbols('x y t')
V = lambda c, i=0, j=0: Q.var((c, i, j))
t = Q.var(('t',))
COMPS = [f'U{i}{c}' for i in range(1, 5) for c in (0, 1)]
def tp(k, d=0):
    c = 1
    for i in range(d): c *= (k - i)
    if c == 0 or k - d < 0: return Q()
    r = Q.const(c)
    for _ in range(k - d): r = r * t
    return r
def Tf(k):
    p = {c: tp(k) * V(c, 1) for c in COMPS}
    p['U31'] = p['U31'] + tp(k, 1) * F(1, 3)
    return p
def Sf(k): return {'U31': tp(k) * 2, 'U40': -tp(k)}
TS = [Tf(k) for k in range(4)] + [Sf(k) for k in range(4)]
print('census: even solutions', len(E))
print('T_f, S_f (f = 1, t, t^2, t^3) are symmetries:', all(len(condition(p, 0)) == 0 for p in TS))
print('T_f, S_f lie in the span of the census:', all(express(p, E) is not None for p in TS))
def mono(ce): return sp.Mul(*[{'x': X, 'y': Y, 't': T}[k] ** a for k, a in ce])
rows, ok_t, ok_xi, ok_y, ok_sc = [], True, True, True, True
for key, sol in S0:
    f = g = cy = 0
    for (kind, comp, ce, u, j), c in sol:
        if kind == 'xiy' and not u: g += c * mono(ce)          # phi contains -g U_y
        if kind == 'jet' and comp == 'U10' and not u and j == ('U10', 't'): f -= c * mono(ce)
        if kind == 'jet' and comp == 'U10' and not u and j == ('U10', 'x'):
            m = mono(ce)
            if sp.degree(m, Y) == 1 and sp.degree(m, X) == 0: cy -= c * m / Y   # phi contains -cy y U_x
    f, g = sp.Poly(sp.expand(f), X, Y, T), sp.Poly(sp.expand(g), X, Y, T)
    ok_t &= all(mm[0] == 0 and mm[1] == 0 and mm[2] <= 3 for mm in f.monoms())
    ok_xi &= all(mm[0] == 0 and mm[1] <= 1 and mm[2] <= 3 for mm in g.monoms())
    gt = sum(cf * T ** mm[2] for mm, cf in zip(g.monoms(), g.coeffs()) if mm[1] == 0)
    hy = sum(cf * T ** mm[2] for mm, cf in zip(g.monoms(), g.coeffs()) if mm[1] == 1)
    ft = f.as_expr()
    sc = sp.expand(hy - sp.Rational(2, 3) * sp.diff(ft, T))   # the y d_y part beyond 2/3 f' y d_y
    ok_sc &= sp.diff(sc, T) == 0
    ok_y &= sp.expand(cy - sp.Rational(2, 3) * sp.diff(gt, T)) == 0
    rows.append([f.coeff_monomial(T ** k) for k in range(4)] + [sp.Poly(gt, T).coeff_monomial(T ** k) if gt != 0 else 0 for k in range(4)] + [sc])
r = sp.Matrix(rows).rank()
print('coefficient of d_t is f(t) of degree <= 3:', ok_t)
print('coefficient of d_y is g(t) + y h(t), g, h of degree <= 3:', ok_xi)
print('h - 2/3 f\' is a constant (the multiple of the scaling y d_y + ...):', ok_sc)
print('the coefficient of y d_x is 2/3 g\'(t):', ok_y)
print('rank of the map to (f, g, that constant):', r, 'of 9')
print('dim = 8 (T_f, S_f) + 9:', len(E) == 8 + r and r == 9)
