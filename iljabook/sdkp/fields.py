"""The symmetries of the super dKP system for polynomial functions of t, as evolutionary fields
(dicts: component -> value): Xi_h, Xi'_h, A_a, B_b, T_f, S_f with h = a = b = f = t^k."""
import sys; sys.path.insert(0, '.')
from fractions import Fraction as F
from jets import Q
V = lambda c, i=0, j=0: Q.var((c, i, j))
t, x, y = Q.var(('t',)), Q.var(('x',)), Q.var(('y',))
COMPS = [f'U{i}{c}' for i in range(1, 5) for c in (0, 1)]
def tp(k, d=0):
    """the d-th derivative of t^k"""
    c = 1
    for i in range(d): c *= (k - i)
    if c == 0 or k - d < 0: return Q()
    r = Q.const(c)
    for _ in range(k - d): r = r * t
    return r
def Xi(k):
    p = {}
    for i in range(1, 5):
        p[f'U{i}0'] = tp(k) * V(f'U{i}1'); p[f'U{i}1'] = tp(k) * V(f'U{i}0', 1)
    p['U41'] = p['U41'] + tp(k, 1) * F(1, 3)
    return p
def Xip(k):
    u = V
    p = {'U10': u('U11') + u('U20'), 'U11': -u('U21'), 'U20': u('U21'), 'U21': Q(),
         'U30': u('U31') + u('U40') + u('U11') * u('U20') + u('U20') * u('U20'),
         'U31': -u('U41') - u('U11') * u('U21') - u('U20') * u('U21') * 2 - u('U20') * V('U10', 1),
         'U40': u('U41') + u('U20') * u('U21') + u('U20') * V('U10', 1),
         'U41': -(u('U21') * V('U10', 1)) - u('U20') * V('U11', 1) - u('U20') * V('U20', 1)}
    p = {c: tp(k) * q for c, q in p.items()}
    p['U10'] = p['U10'] + tp(k, 1) * y * F(2, 3)
    p['U30'] = p['U30'] + tp(k, 1) * x * F(1, 3) + tp(k, 2) * y * y * F(2, 9)
    return p
def A(k): return {'U10': tp(k), 'U30': tp(k, 1) * y * F(2, 3)}
def B(k): return {'U30': tp(k)}
def Tf(k):
    p = {c: tp(k) * V(c, 1) for c in COMPS}
    p['U31'] = p['U31'] + tp(k, 1) * F(1, 3)
    return p
def Sf(k): return {'U31': tp(k) * 2, 'U40': -tp(k)}
def lin(*terms):
    """linear combination of fields: lin((c1, f1), (c2, f2), ...)"""
    r = {}
    for c0, f in terms:
        for comp, q in f.items(): r[comp] = r.get(comp, Q()) + q * c0
    return r
