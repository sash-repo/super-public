"""The reparametrizations of the time R_f: R_f is a symmetry of the super dKP system for an
arbitrary function f(t); its brackets with Xi_h, Xi'_h, A_a, B_b (f, h arbitrary); for
f = 1, t, t^2, t^3 the field R_f lies in the span of the even census (symm1.py, sigma = 0).
The functions f(t), h(t) and their derivatives are free jet variables.
Needs sols_s0_c3_21.pkl and sols_s1_c3_21.pkl (run symm1.py first)."""
import sys; sys.path.insert(0, '.')
from general import *          # Xi('h'), Xip('h') for an arbitrary h(t); bracket, condition
import fields
jets.PAR['f'] = 0
_Dyv = jets.Dy_var
jets.Dy_var = lambda v, R_: Q() if v[0] == 'f' else _Dyv(v, R_)
def Dx3(q):
    def act(v):
        if v == ('x',): return Q.const(1)
        if len(v) == 3 and v[0] in ('h', 'g', 'f'): return None
        if len(v) == 3: return Q.var((v[0], v[1] + 1, v[2]))
    return jets.derivation(q, act, 0)
jets.Dx = Dx3; symm.Dx = Dx3
fk = lambda k: Q.var(('f', 0, k))
H = lambda k: Q.var(('h', 0, k))
Xc, Yc = Q.var(('x',)), Q.var(('y',))
W = {'U10': F(1, 6), 'U11': F(1, 3), 'U20': F(1, 3), 'U21': F(1, 2), 'U30': F(1, 2), 'U31': F(2, 3), 'U40': F(2, 3), 'U41': F(5, 6)}
def R():
    """phi(U_i^c) = f U_t + 1/3 f' x U_x + 2/9 f'' y^2 U_x + 2/3 f' y U_y + (i+c)/6 f' U + rho_f(U)"""
    p = {}
    for c in COMPS:
        p[c] = fk(0) * V(c, 0, 1) + fk(1) * Xc * V(c, 1) * F(1, 3) + fk(2) * Yc * Yc * V(c, 1) * F(2, 9) \
               + fk(1) * V(c) * W[c] + fk(1) * Yc * RHS[c] * F(2, 3)
    e = fk(2) * Yc * F(1, 9)
    p['U20'] += e; p['U30'] += e * V('U10'); p['U31'] += e * V('U11')
    p['U40'] -= e * (V('U11') + V('U20')); p['U41'] -= e * (V('U21') + V('U10', 1))
    p['U40'] += fk(3) * Yc * Yc * F(1, 27) + fk(2) * Xc * F(1, 18)
    return p
print('R_f is a symmetry for arbitrary f(t):', len(condition(R(), 0)) == 0)
p = R(); p['U40'] += fk(3) * Yc * Yc * F(1, 54)
print('control (perturbed R_f) fails:', len(condition(p, 0)) > 0)

def ddt(q):
    """d/dt on polynomials in the jets of f and h"""
    def act(v):
        if len(v) == 3 and v[0] in ('f', 'h'): return Q.var((v[0], 0, v[2] + 1))
        return Q()
    return jets.derivation(q, act, 0)
def in_h(field, g):
    """the field `field`, written for the function h, taken at the function g (a polynomial in jets)"""
    der = [g]
    for _ in range(4): der.append(ddt(der[-1]))
    out = {}
    for cc, q in field.items():
        r = Q()
        for (ev, od), c0 in q.t.items():
            m = Q.const(c0); hj = None
            for v, a in ev:
                if len(v) == 3 and v[0] == 'h': hj = v[2]; continue
                for _ in range(a): m = m * Q.var(v)
            for v in od: m = m * Q.var(v)
            r = r + (der[hj] * m if hj is not None else m)
        out[cc] = r
    return out
def same(a, b): return all((a.get(c, Q()) - b.get(c, Q())).iszero() for c in COMPS)
Ah = {'U10': H(0), 'U30': H(1) * Yc * F(2, 3)}
Bh = {'U30': H(0)}
print('A_h, B_h are symmetries for arbitrary h(t):', len(condition(Ah, 1)) == 0, len(condition(Bh, 1)) == 0)
for name, fld, lam in (('Xi', Xi('h'), F(-1, 6)), ("Xi'", Xip('h'), F(-1, 6)), ('A', Ah, F(1, 6)), ('B', Bh, F(1, 2))):
    g = fk(0) * H(1) + fk(1) * H(0) * lam
    rhs = {c: -q for c, q in in_h(fld, g).items()}
    ok = same(bracket(R(), 0, fld, 1), rhs)
    print(f'[R_f,{name}_h] = -{name}_(f h\' + ({lam}) f\' h):', ok)
    # control: another weight fails
    rhs2 = {c: -q for c, q in in_h(fld, fk(0) * H(1) + fk(1) * H(0) * (lam + 1)).items()}
    print(f'  control (weight {lam + 1}) fails:', not same(bracket(R(), 0, fld, 1), rhs2))

# R_f for f = t^k lies in the span of the even census
from brackets import E, express
def R_at(k):
    """R_f at f = t^k: replace the jets of f by the derivatives of t^k"""
    out = {}
    for cc, q in R().items():
        r = Q()
        for (ev, od), c0 in q.t.items():
            m = Q.const(c0)
            for v, a in ev:
                for _ in range(a): m = m * (fields.tp(k, v[2]) if len(v) == 3 and v[0] == 'f' else Q.var(v))
            for v in od: m = m * Q.var(v)
            r = r + m
        out[cc] = r
    return out
print('R_f for f = 1, t, t^2, t^3 is a symmetry and lies in the span of the even census:',
      all(len(condition(R_at(k), 0)) == 0 and express(R_at(k), E) is not None for k in range(4)))
