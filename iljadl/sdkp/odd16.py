"""The odd symmetries of order at most 1 found by the census (symm1.py, sigma = 1, coefficients
of degree <= 3 in x, y, t) span exactly the 16 fields Xi_h, Xi'_h, A_a, B_b with h, a in
{1, t, t^2, t^3}.  Needs sols_s1_c3_21.pkl and sols_s0_c3_21.pkl (run symm1.py first)."""
import sys; sys.path.insert(0, '.')
from fractions import Fraction as F
from brackets import O, express, Q
from symm import condition
V = lambda c, i=0, j=0: Q.var((c, i, j))
t, x, y = Q.var(('t',)), Q.var(('x',)), Q.var(('y',))
def tp(k, d=0):
    """d-th derivative of t^k as an element of Q"""
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
G = [f(k) for f in (Xi, Xip, A, B) for k in range(4)]
print('census: odd solutions', len(O))
print('the 16 fields are symmetries:', all(len(condition(g, 1)) == 0 for g in G))
print('every census solution is a combination of the 16 fields:', all(express(o, G) is not None for o in O))
print('every one of the 16 fields is a combination of the census solutions:', all(express(g, O) is not None for g in G))
# control: a field outside the family is not in the span of the census
bad = dict(B(0)); bad['U10'] = Q.var(('t',))
print('control (B_1 plus t in U_1^0) is not in the span:', express(bad, O) is None)
