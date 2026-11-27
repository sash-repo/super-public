"""Super KdV: N=1 (Manin-Radul) and the N=3 family (D_i = d_theta_i - theta_i d_x)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sdkp'))
from spdo import P, PAR
from evo import *
from fractions import Fraction as F
from itertools import combinations
V = lambda n, k=0: dxk(P.var(n), k)
# ---- N = 1
PAR.update(u=0, f=1)
u, f = V('u'), V('f')
K = {'u': -dxk(u,3) + u*u.dx()*6 - f*dxk(f,2)*3, 'f': -dxk(f,3) + (u*f).dx()*3}
Xi = {'u': f.dx(), 'f': u}
print('N=1: Xi symmetry:', is_sym(K, Xi, 1, ['u','f']))
sq = {n: apply(Xi, 1, Xi[n]) for n in Xi}
print('N=1: Xi^2 = d_x:', (sq['u'] - u.dx()).iszero() and (sq['f'] - f.dx()).iszero())
# negative control
Xb = {'u': f.dx(), 'f': u*2}
print('N=1: control (f->2u) fails:', not is_sym(K, Xb, 1, ['u','f']))

# ---- N = 3 superfields
N = 3
def tmul(I, J):
    if set(I) & set(J): return 0, None
    seq = list(I) + list(J); s = 1
    for a in range(len(seq)):
        for b in range(len(seq) - 1 - a):
            if seq[b] > seq[b+1]: seq[b], seq[b+1] = seq[b+1], seq[b]; s = -s
    return s, tuple(seq)
class SF:
    def __init__(s, c, p): s.c = {I: v for I, v in c.items() if not v.iszero()}; s.p = p
    def cp(s, I): return (s.p + len(I)) % 2
    def __add__(s, o):
        c = dict(s.c)
        for I, v in o.c.items(): c[I] = c[I] + v if I in c else v
        return SF(c, s.p if s.c else o.p)
    def __neg__(s): return SF({I: -v for I, v in s.c.items()}, s.p)
    def __sub__(s, o): return s + (-o)
    def sc(s, k): return SF({I: v*k for I, v in s.c.items()}, s.p)
    def __mul__(s, o):
        c = {}
        for I, a in s.c.items():
            for J, b in o.c.items():
                sg, K_ = tmul(I, J)
                if not sg: continue
                if s.cp(I) and len(J) % 2: sg = -sg   # move theta_J left past a
                v = a*b*sg; c[K_] = c[K_] + v if K_ in c else v
        return SF(c, (s.p + o.p) % 2)
    def dx(s): return SF({I: v.dx() for I, v in s.c.items()}, s.p)
    def dth(s, i):
        c = {}
        for I, v in s.c.items():
            if i in I:
                pos = I.index(i); J = I[:pos] + I[pos+1:]
                c[J] = c.get(J, P()) + v*(-1 if pos % 2 else 1)
        return SF(c, (s.p + 1) % 2)
    def thmul(s, i):
        c = {}
        for I, v in s.c.items():
            sg, K_ = tmul((i,), I)
            if sg: c[K_] = c.get(K_, P()) + v*sg
        return SF(c, (s.p + 1) % 2)
    def D(s, i): return s.dth(i) - s.dx().thmul(i)      # D_i = d_theta_i - theta_i d_x
    def Q(s, i): return s.dth(i) + s.dx().thmul(i)      # Q_i = d_theta_i + theta_i d_x
    def zero(s): return all(v.iszero() for v in s.c.values())
names = {(): 'psi', (1,): 'J1', (2,): 'J2', (3,): 'J3', (1,2): 'c12', (1,3): 'c13', (2,3): 'c23', (1,2,3): 'u'}
for I, n in names.items(): PAR[n] = (1 + len(I)) % 2
U = SF({I: V(n) for I, n in names.items()}, 1)
# operator identities
print('N=3: D_i^2 = -d_x:', all((U.D(i).D(i) + U.dx()).zero() for i in (1,2,3)))
print('N=3: Q_i^2 = d_x:', all((U.Q(i).Q(i) - U.dx()).zero() for i in (1,2,3)))
print('N=3: Q_i D_j + D_j Q_i = 0:', all((U.D(j).Q(i) + U.Q(i).D(j)).zero() for i in (1,2,3) for j in (1,2,3)))
def rhs(a, b):
    Fs = U.D(3).D(2).D(1).sc(a)          # D1 D2 D3 U  (D1 applied last)
    for i in (1,2,3): Fs = Fs + (U.D(i)*U.D(i)).sc(b)
    R = Fs.D(3).D(2).D(1) - (U.dx()*Fs).sc(2) - U*Fs.dx()
    for i in (1,2,3): R = R + U.D(i)*Fs.D(i)
    return {names[I]: R.c.get(I, P()) for I in names}
Ka, Kb = rhs(1, 0), rhs(0, 1)
def body(p): return P({m: c for m, c in p.t.items() if not m[1]})
def nodd(p): return {len(m[1]) for m in p.t}
J = [None, V('J1'), V('J2'), V('J3')]; Jd = lambda i, k: V('J%d' % i, k); ud = lambda k: V('u', k)
dot = lambda k1, k2: sum((Jd(i,k1)*Jd(i,k2) for i in (1,2,3)), P())
def cross(i, k):
    j, l = {1:(2,3), 2:(3,1), 3:(1,2)}[i]
    return Jd(j,0)*Jd(l,k) - Jd(l,0)*Jd(j,k)
Ea_u = ud(3) + ud(0)*ud(1)*6 - dot(0,3)*2
Eb_u = -((dot(0,0)*ud(0)*3 + dot(0,2) + dot(1,1))*2).dx()
okE0 = (body(Ka['u']) - Ea_u).iszero() and (body(Kb['u']) - Eb_u).iszero()
for i in (1,2,3):
    Ea = Jd(i,3) + (ud(0)*Jd(i,0)).dx()*2 - cross(i,2)
    Eb = ((ud(0)*Jd(i,0)).dx() - Jd(i,0)*dot(0,1) - dot(0,0)*Jd(i,1)*2 - cross(i,2))*2
    okE0 &= (body(Ka['J%d'%i]) - Ea).iszero() and (body(Kb['J%d'%i]) - Eb).iszero()
print('N=3 item 1) bosonic sector (E0N3), a and b parts:', okE0)
print('N=3 item 2) numbers of odd factors in the monomials:')
for lab, K_ in (('a', Ka), ('b', Kb)):
    print('  ', lab, {n: sorted(nodd(K_[n])) for n in K_})
def atJ0_lin(p):
    return P({m: c for m, c in p.t.items() if len(m[1]) == 1 and not any(v[0][0] == 'J' for v, _ in m[0])})
psi = V('psi')
lin_psi = (dxk(psi,3) + ud(0)*psi.dx()*2 + ud(1)*psi, ud(0)*psi.dx()*6)
ok3 = (atJ0_lin(Ka['psi']) - lin_psi[0]).iszero() and (atJ0_lin(Kb['psi']) - lin_psi[1]).iszero()
for n in ('c12','c13','c23'):
    ch = V(n)
    ok3 &= (atJ0_lin(Ka[n]) - dxk(ch,3) - (ud(0)*ch).dx()*3).iszero() and atJ0_lin(Kb[n]).iszero()
print('N=3 item 3) linear part at J=0 (linN3):', ok3)
# item 4)
# an odd field passes theta_I with the sign (-1)^|I|: Xi(c_I) = (-1)^|I| (Q_i U)_I
Xi = [None] + [{names[I]: U.Q(i).c.get(I, P())*(-1)**len(I) for I in names} for i in (1,2,3)]
print('N=3 item 4) Xi_1 on the unknowns:', {n: repr(v) for n, v in Xi[1].items()})
NM = list(names.values())
print('N=3 item 4) Xi_i symmetries (a and b parts):', all(is_sym(K_, Xi[i], 1, NM) for i in (1,2,3) for K_ in (Ka, Kb)))
br = lambda i, j: bracket(Xi[i], 1, Xi[j], 1, NM)
print('N=3 item 4) [Xi_i,Xi_j] = -2 delta_ij d_x:', all(all((br(i,j)[n] + (V(n).dx()*2 if i == j else P())).iszero() for n in NM) for i in (1,2,3) for j in (1,2,3)))
# D_i in place of Q_i: not symmetries (remark in the text)
XiD = {names[I]: U.D(1).c.get(I, P())*(-1)**len(I) for I in names}
print('N=3: D_1 in place of Q_1 is not a symmetry:', not is_sym(Ka, XiD, 1, NM))
# conclusion: (psi,c12,c13,c23) = (J1, J2x, J3x, u) solves the linear part in the fermions, for (u,J) solving E0
# for every (a, b) != (0, 0), U -> D_i U is not a symmetry: the defects [K, X] of the a-part and
# of the b-part are linearly independent (the defect of a Ka + b Kb is a Da + b Db), i = 1, 2, 3
from fractions import Fraction as _Fr
def _rank2(d1, d2):
    keys = sorted({(n, m) for d in (d1, d2) for n, p in d.items() for m in p.t}, key=str)
    rows = [[d1[n].t.get(m, 0), d2[n].t.get(m, 0)] for n, m in keys]
    nz = [r for r in rows if r != [0, 0]]
    if not nz: return 0
    a0, b0 = nz[0]
    return 2 if any(_Fr(a0) * r[1] != _Fr(b0) * r[0] for r in nz) else 1
ok5 = True
for i in (1, 2, 3):
    XiDi = {names[I]: U.D(i).c.get(I, P())*(-1)**len(I) for I in names}
    Da, Db = bracket(Ka, 0, XiDi, 1, NM), bracket(Kb, 0, XiDi, 1, NM)
    ok5 &= _rank2(Da, Db) == 2
print('N=3 item 5) for every (a,b) != (0,0), U -> D_i U is not a symmetry (i = 1, 2, 3):', ok5)
