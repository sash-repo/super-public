"""The N=2 super KdV family U_t = 2(UF)_x + 2UF_x - sum_i D_i(U D_i F) + D_1 D_2 F_x,
F = a D_1 D_2 U + b U^2, with D_i = d_theta_i + theta_i d_x and U = u + theta_1 f_1 + theta_2 f_2
+ theta_1 theta_2 m even: the bosonic sector, the terms in the fermions, the supersymmetry
Xi_1(U) = Q_1 U (Q_i = d_theta_i - theta_i d_x), and the fermionic partners f_1 = u_x, f_2 = -m."""
from model import *
N = 2
U = gen(N, 'U', 0)
REN = {'U0': 'u', 'U1': 'f1', 'U2': 'f2', 'U12': 'm'}
def rhs(U, a, b):
    D = lambda X, i: X.D(i)
    F = D(D(U, 2), 1).scale(a) + (U * U).scale(b)
    r = (U * F).dx().scale(2) + (U * F.dx()).scale(2)
    for i in (1, 2): r = r - D(U * D(F, i), i)
    return r + D(D(F.dx(), 2), 1)
v = lambda n, k=0: dxk(P.var(n), k)
u = lambda k=0: v('U0', k); m = lambda k=0: v('U12', k)
f1 = lambda k=0: v('U1', k); f2 = lambda k=0: v('U2', k)
def part(p, d): return split_by_odd(p).get(d, P())
sa, sb = system(U, rhs(U, 1, 0)), system(U, rhs(U, 0, 1))     # the a- and the b-part
ok = (part(sa['U0'], 0) + (u(3) + (u() * m()).dx() * 2)).iszero() \
     and (part(sb['U0'], 0) - ((u() * u() * u()).dx() * 2 - (u() * m()).dx() * 2)).iszero() \
     and (part(sa['U12'], 0) + (m(3) + m() * m(1) * 6 - u() * u(3) * 2)).iszero() \
     and (part(sb['U12'], 0) - ((u() * u() * m()).dx() * 6 + (u() * u(2)).dx() * 2 + (u(1) * u(1)).dx() * 2)).iszero()
print('bosonic sector: u_t = -a(u_xxx + 2(um)_x) + 2b(u^3 - um)_x, m_t = -a(m_xxx + 6mm_x - 2uu_xxx) + 2b(3u^2m + uu_xx + u_x^2)_x:', ok)
print('the equations for f_1, f_2 are linear in the fermions (a- and b-part):',
      all(set(split_by_odd(s_[n])) <= {1} for s_ in (sa, sb) for n in ('U1', 'U2')))
f12x = (f1() * f2()).dx()
print('fermionic terms of the u-equation: (2b - a)(f_1 f_2)_x:',
      (part(sa['U0'], 2) + f12x).iszero() and (part(sb['U0'], 2) - f12x * 2).iszero())
print('fermionic terms of the m-equation: 3a(f_1 f_1xx + f_2 f_2xx) - 12b(u f_1 f_2)_x:',
      (part(sa['U12'], 2) - (f1() * f1(2) + f2() * f2(2)) * 3).iszero()
      and (part(sb['U12'], 2) + (u() * f1() * f2()).dx() * 12).iszero())
Xi1 = evol_char(U, U.Q(1), 1)
print('Xi_1: u -> f1, f1 -> u_x, f2 -> -m, m -> -f2_x:',
      (Xi1['U0'] - f1()).iszero() and (Xi1['U1'] - u(1)).iszero() and (Xi1['U2'] + m()).iszero() and (Xi1['U12'] + f2(1)).iszero())
print('Xi_1, Xi_2 are symmetries (a- and b-part):', all(is_symmetry(s_, evol_char(U, U.Q(i), 1), 1)[0] for s_ in (sa, sb) for i in (1, 2)))
print('control: U -> D_1 U is not a symmetry (a- and b-part):', all(not is_symmetry(s_, evol_char(U, U.D(1), 1), 1)[0] for s_ in (sa, sb)))
# f_1 = u_x, f_2 = -m solve the linear equations for the fermions whenever (u, m) solves the bosonic sector
for lab, s_ in (('a', sa), ('b', sb)):
    sub = {'U1': u(1), 'U2': -m()}
    def at(p):
        """substitute f_1 = u_x, f_2 = -m into a polynomial linear in f_1, f_2"""
        r = P()
        for (ev, od), c in p.t.items():
            mm = P.const(c)
            for w, e in ev:
                for _ in range(e): mm = mm * dxk(P.var(w[0]), w[1])
            for w in od: mm = mm * dxk(sub[w[0]], w[1])
            r = r + mm
        return r
    bos = {n: part(s_[n], 0) for n in ('U0', 'U12')}
    lhs1 = dxk(bos['U0'], 1)                    # d/dt (u_x) on the bosonic sector
    lhs2 = -bos['U12']                          # d/dt (-m)
    print(f'{lab}-part: f_1 = u_x, f_2 = -m solve the fermion equations:',
          (at(part(s_['U1'], 1)) - lhs1).iszero() and (at(part(s_['U2'], 1)) - lhs2).iszero())
