"""Odd tangent covering Pi T E: the Cartan calculus d_psi, iota_s, L_s (Theorem), on KdV."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sdkp'))
from spdo import P, PAR
from evo import *
PAR.update(u=0, p=1)
V = lambda n, k=0: dxk(P.var(n), k)
u, p = V('u'), V('p')
NM = ['u', 'p']
d = {'u': p, 'p': P()}
ell = lambda s: apply(d, 1, s)          # l_s(psi) = d_psi(s)
iota = lambda s: {'u': P(), 'p': s}
L = lambda s: {'u': s, 'p': ell(s)}
K = -dxk(u,3) + u*u.dx()*6
sysK = {'u': K, 'p': ell(K)}            # Pi T E for KdV: psi_t = l_K(psi)
print('l_K(psi) = -psi_xxx + 6(u psi)_x:', (ell(K) - (-dxk(p,3) + (u*p).dx()*6)).iszero())
K5 = dxk(u,5) - (u*dxk(u,3)*10 + u.dx()*dxk(u,2)*20 - u*u*u.dx()*30)   # 5th-order flow of u_t=-u_xxx+6uu_x (up to scale)
syms = {'u_x': u.dx(), 'K': K, 'K5': K5}
from evo import is_sym
print('K5 commutes with K (so it is a symmetry of KdV):', is_sym({'u': K}, {'u': K5}, 0, ['u']))
for nm, s in syms.items():
    print(f'item 1) s={nm}: d_psi, iota_s, L_s symmetries of Pi T E:',
          is_sym(sysK, d, 1, NM), is_sym(sysK, iota(s), 1, NM), is_sym(sysK, L(s), 0, NM))
print('control: iota_s for s=u (not a symmetry) fails:', not is_sym(sysK, iota(u), 1, NM))
eqd = lambda A, B: all((A.get(n, P()) - B.get(n, P())).iszero() for n in NM)
zero = {}
br = lambda X, sx, Y, sy: bracket(X, sx, Y, sy, NM)
# item 2) on the free jet space with arbitrary s, s'
S = [u*u.dx(), dxk(u,3), u*u, u.dx()*dxk(u,2) + u*u*u]
jac = lambda s, s2: ell(s2).__class__  # placeholder
def jb(s, s2):   # {s,s'} = l_{s'}(s) - l_s(s')  with l_f(g) = linearization of f applied to g (even)
    lin = lambda f, g: apply({'u': g}, 0, f)
    return lin(s2, s) - lin(s, s2)
ok = eqd({n: apply(d, 1, d[n]) for n in NM}, zero)
print('d_psi^2 = 0:', ok)
res = True
for s in S:
    res &= eqd(br(d, 1, iota(s), 1), L(s))
    res &= eqd(br(d, 1, L(s), 0), zero)
    for s2 in S:
        res &= eqd(br(iota(s), 1, iota(s2), 1), zero)
        res &= eqd(br(L(s), 0, iota(s2), 1), iota(jb(s, s2)))
        res &= eqd(br(L(s), 0, L(s2), 0), L(jb(s, s2)))
print('item 2) all brackets (4 generating functions, all pairs):', res)
# item 3) Xi_s^2 = L_s
for s in S:
    Xi = {n: d[n] + iota(s)[n] for n in NM}
    print('Xi_s^2 = L_s:', eqd({n: apply(Xi, 1, Xi[n]) for n in NM}, L(s)))
# Super KdV N=1 is not an odd tangent covering: l_K has 6, MR fermion eq has 3
