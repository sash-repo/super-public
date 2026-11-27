"""The solutions of the super dKP system invariant under an odd symmetry Theta.

For each Theta below: (a) Theta vanishes on the stated conditions on four of the eight
components; (b) substituting the conditions into the system solved for the derivatives in y
gives the reduced system for the four other components; (c) the derivatives in y of the
conditions vanish on the reduced system (compatibility); (d) the stated change of the
unknowns carries the solutions (u, v, psi, chi) of the system

    u_y = 2 u u_x + 2 v_x,   v_y = 2/3 u_t - 2 (u v)_x - 2 u^2 u_x,
    psi_y = 2 chi_x,         chi_y = 2/3 psi_t - 2 v psi_x

to the solutions of the reduced system.  Everything is computed in the supercommutative algebra
of the jets (U_i^c is odd for i + c odd); the functions k, a, b of t are free jet variables.
Components: Uic = U_i^c, e.g. U20 = U_2^0."""
import sys; sys.path.insert(0, '.')
import jets
from jets import Q, derivation, F
RHS = jets.build()
FUN = ('k', 'a', 'b')
for f in FUN: jets.PAR[f] = 0
for f, p in (('u', 0), ('v', 0), ('p', 1), ('c', 1)): jets.PAR[f] = p
COMPS = sorted(RHS)
isf = lambda v: len(v) == 3 and v[0] in FUN
def Dx(q):
    def act(v):
        if v == ('x',): return Q.const(1)
        if isf(v): return None
        if len(v) == 3: return Q.var((v[0], v[1] + 1, v[2]))
    return derivation(q, act, 0)
def Dt(q):
    def act(v):
        if v == ('t',): return Q.const(1)
        if len(v) == 3: return Q.var((v[0], v[1], v[2] + 1))
    return derivation(q, act, 0)
V = lambda c, i=0, j=0: Q.var((c, i, j))
X, Y = Q.var(('x',)), Q.var(('y',))
k0, k1 = V('k'), V('k', 0, 1)
a0, a1, b0 = V('a'), V('a', 0, 1), V('b')

def jet_of(q, i, j):
    for _ in range(i): q = Dx(q)
    for _ in range(j): q = Dt(q)
    return q
def subst(q, table):
    """replace every jet (name, i, j) with name in table by the (i, j)-th derivative of table[name]"""
    r = Q()
    for (ev, od), c in q.t.items():
        m = Q.const(c)
        for v, e in ev:
            for _ in range(e): m = m * (jet_of(table[v[0]], v[1], v[2]) if len(v) == 3 and v[0] in table else Q.var(v))
        for v in od: m = m * (jet_of(table[v[0]], v[1], v[2]) if v[0] in table else Q.var(v))
        r = r + m
    return r
def Dy_on(system):
    """total derivative in y on the solutions of `system` (dict name -> right-hand side)"""
    def D(q):
        def act(v):
            if v == ('y',): return Q.const(1)
            if isf(v) or v in (('x',), ('t',)): return Q()
            if len(v) == 3: return jet_of(system[v[0]], v[1], v[2])
        return derivation(q, act, 0)
    return D

# the target system (u, v, psi, chi) = (u, v, p, c)
u, v, p, c = V('u'), V('v'), V('p'), V('c')
TARGET = {'u': u * Dx(u) * 2 + Dx(v) * 2,
          'v': Dt(u) * F(2, 3) - Dx(u * v) * 2 - u * u * Dx(u) * 2,
          'p': Dx(c) * 2,
          'c': Dt(p) * F(2, 3) - v * Dx(p) * 2}

# the odd symmetries (values on the components)
def Xip1():
    U = V
    return {'U10': U('U11') + U('U20'), 'U11': -U('U21'), 'U20': U('U21'), 'U21': Q(),
            'U30': U('U31') + U('U40') + U('U11') * U('U20') + U('U20') * U('U20'),
            'U31': -U('U41') - U('U11') * U('U21') - U('U20') * U('U21') * 2 - U('U20') * V('U10', 1),
            'U40': U('U41') + U('U20') * U('U21') + U('U20') * V('U10', 1),
            'U41': -(U('U21') * V('U10', 1)) - U('U20') * V('U11', 1) - U('U20') * V('U20', 1)}
def Xi1():
    p_ = {}
    for i in range(1, 5): p_[f'U{i}0'] = V(f'U{i}1'); p_[f'U{i}1'] = V(f'U{i}0', 1)
    return p_
def add(*fields):
    r = {}
    for coef, f in fields:
        for comp, q in f.items(): r[comp] = r.get(comp, Q()) + q * coef
    return r
XipH = add((1, Xip1()), (1, {'U10': k0 * Y * F(2, 3),
                             'U30': k0 * X * F(1, 3) + (k1 + k0 * k0) * Y * Y * F(2, 9)}))  # h^{-1} Xi'_h, k = h'/h
A = {'U10': a0, 'U30': a1 * Y * F(2, 3)}
B = {'U30': b0}

CASES = [
    ("Xi'_1", Xip1(),
     {'U21': Q(), 'U11': -V('U20'), 'U31': -V('U40'), 'U41': -(V('U20') * V('U10', 1))},
     {'U20': u, 'U40': v, 'U10': p, 'U30': c}),
    ("h^{-1} Xi'_h, k = h'/h (reduced system only)", XipH,
     {'U21': Q(), 'U11': -V('U20') - k0 * Y * F(2, 3), 'U41': -(V('U20') * V('U10', 1)),
      'U31': -V('U40') + k0 * Y * V('U20') * F(2, 3) - k0 * X * F(1, 3) - (k1 + k0 * k0) * Y * Y * F(2, 9)},
     None),
    ("Xi'_1 + A_a + B_b", add((1, Xip1()), (1, A), (1, B)),
     {'U21': Q(), 'U11': -V('U20') - a0, 'U31': -V('U40') + a0 * V('U20') - a1 * Y * F(2, 3) - b0,
      'U41': -(V('U20') * V('U10', 1))},
     {'U20': u + a0, 'U40': v + b0 + a1 * Y * F(2, 3), 'U10': p, 'U30': c + a0 * p}),
    ("Xi_1 - Xi'_1", add((1, Xi1()), (-1, Xip1())),
     {'U20': Q(), 'U40': Q(), 'U21': -V('U10', 1), 'U41': -V('U30', 1) + V('U11') * V('U10', 1)},
     {'U11': u, 'U31': v, 'U10': p, 'U30': c}),
    ("Xi_1 - Xi'_1 + A_a + B_b", add((1, Xi1()), (-1, Xip1()), (1, A), (1, B)),
     {'U20': a0, 'U40': a1 * Y * F(2, 3) + b0 - a0 * V('U11') - a0 * a0, 'U21': -V('U10', 1),
      'U41': -V('U30', 1) + (V('U11') + a0) * V('U10', 1)},
     {'U11': u - a0 * 2, 'U31': v + a0 * u - a0 * a0 - b0 * 2 - a1 * Y * F(4, 3), 'U10': p, 'U30': c - a0 * p}),
]

def show(q):
    out = []
    for (ev, od), cf in sorted(q.t.items(), key=str):
        f = '*'.join([(w[0] + 'x' * w[1] + 't' * w[2] if len(w) == 3 else w[0]) + (f'^{e}' if e > 1 else '') for w, e in ev]
                     + [w[0] + 'x' * w[1] + 't' * w[2] for w in od])
        out.append(f'{cf}*{f}' if f else f'{cf}')
    return ' + '.join(out) or '0'

allok = True
for name, theta, sub, change in CASES:
    print(f'=== {name}')
    vanish = all(subst(theta.get(comp, Q()), sub).iszero() for comp in COMPS)
    red = {comp: subst(RHS[comp], sub) for comp in COMPS}
    free = [comp for comp in COMPS if comp not in sub]
    for comp in free: print(f'  {comp}_y =', show(red[comp]))
    Dy = Dy_on(red)
    consistent = all((red[nm] - Dy(q)).iszero() for nm, q in sub.items())
    print('  Theta vanishes on the conditions:', vanish)
    print('  the conditions are compatible with the system:', consistent)
    allok &= vanish and consistent
    if name.startswith('h^{-1}'):
        uu, vv, pp, cc = V('U20'), V('U40'), V('U10'), V('U30')
        kk = k0 * k0 - k1
        expected = {'U20': uu * Dx(uu) * 2 + Dx(vv) * 2 - k0 * Y * Dx(uu) * F(4, 3),
                    'U40': Dt(uu) * F(2, 3) - Dx(uu * vv) * 2 - uu * uu * Dx(uu) * 2
                           + k0 * Y * (Dx(vv) + uu * Dx(uu) * 2) * F(4, 3) + k0 * Dx(X * uu) * F(2, 3)
                           - kk * Y * Y * Dx(uu) * F(4, 9),
                    'U10': Dx(cc) * 2 - k0 * Y * Dx(pp) * F(4, 3),
                    'U30': Dt(pp) * F(2, 3) - vv * Dx(pp) * 2 + k0 * Y * Dx(cc) * F(4, 3) + k0 * X * Dx(pp) * F(2, 3)
                           - kk * Y * Y * Dx(pp) * F(4, 9)}
        ok = all((red[comp] - expected[comp]).iszero() for comp in free)
        print('  the reduced system is the one printed (u, v, psi, chi = U20, U40, U10, U30):', ok)
        allok &= ok
    if change is not None:
        Dyt = Dy_on(TARGET)
        ok = all((Dyt(change[comp]) - subst(red[comp], change)).iszero() for comp in free)
        print('  the change of the unknowns carries the target system to the reduced one:', ok)
        allok &= ok
print('all checks passed:', allok)
