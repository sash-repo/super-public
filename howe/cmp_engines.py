from fractions import Fraction as F
import omcheck as O, slcheck as S, slbivec as BV
from le import D as VF_of

n = 2
def lift(m):        # omcheck monomial -> slcheck monomial
    A, B = m
    return ((0,)*n, (), A, B)
def drop(e):        # slcheck element (coeff-free part) -> omcheck element
    out = {}
    for (a,Ss,A,B), c in e.items():
        assert sum(a)==0 and not Ss, (a,Ss)
        out[(A,B)] = c
    return out

mons = []
for i in range(4): mons += O.basis(n, i)

# 1. compare Y_-
bad = [m for m in mons if BV.Yminus_const(n,{m:F(1)}) != drop(S.Yminus(n,{lift(m):F(1)}))]
print("Y_- differs on", len(bad), "of", len(mons), bad[:4])

# 2. compare the action of gl(i,j)  <->  f = x_i xi_j
for i in range(n):
    for j in range(n):
        g = O.Gen(n,'gl',i,j)
        a = [0]*n; a[i] = 1
        f = {(tuple(a),(j,)): F(1)}
        v = VF_of(f, n)
        diff = [m for m in mons if g.act({m:F(1)}) != drop(S.lie(n,v,{lift(m):F(1)}))]
        print("gl(%d,%d) f=x%d xi%d : par %d/%d, differs on %d  %s"
              % (i+1,j+1,i+1,j+1,g.pD,v.par,len(diff),diff[:3]))
        for m in diff[:2]:
            print("     ", m, "omcheck:", g.act({m:F(1)}), " slcheck:", drop(S.lie(n,v,{lift(m):F(1)})))
