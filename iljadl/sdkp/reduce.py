"""Xi'-invariant solutions: impose U21=0, U11=-U20, U31=-U40, U41=-U20*U10_x
and read off the reduced system for U10, U20, U30, U40."""
import sys; sys.path.insert(0,'.')
from symm import *
from jets import Dx, Dt
sub = {'U21': Q(), 'U11': -Q.var(('U20',0,0)), 'U31': -Q.var(('U40',0,0)),
       'U41': -(Q.var(('U20',0,0))*Q.var(('U10',1,0)))}
def subst_var(v):
    nm,i,j=v
    if nm not in sub: return Q.var(v)
    q=sub[nm]
    for _ in range(i): q=Dx(q)
    for _ in range(j): q=Dt(q)
    return q
def subst(q):
    r=Q()
    for (ev,od),c in q.t.items():
        m=Q.const(c)
        for v,a in ev:
            for _ in range(a): m=m*(subst_var(v) if len(v)==3 else Q.var(v))
        for v in od: m=m*subst_var(v)
        r=r+m
    return r
def show(q):
    out=[]
    for (ev,od),c in sorted(q.t.items(),key=str):
        f='*'.join([f'{v[0]}'+'x'*v[1]+'t'*v[2]+(f'^{a}' if a>1 else '') for v,a in ev]+[v[0]+'x'*v[1]+'t'*v[2] for v in od])
        out.append(f'{c}*{f}')
    return ' + '.join(out) or '0'
red={c:subst(RHS[c]) for c in COMPS}
for c in COMPS: print(f'{c}_y =', show(red[c]))
# consistency: the y-derivative of each constraint must vanish on the reduced system
print('--- consistency of the constraints (should all be 0):')
yv={c:red[c] for c in COMPS}
def Dy_red(q):
    from jets import derivation
    def act(v):
        if v==('y',): return Q.const(1)
        if len(v)==3:
            r=yv[v[0]]
            for _ in range(v[1]): r=Dx(r)
            for _ in range(v[2]): r=Dt(r)
            return r
        return None
    return derivation(q,act,0)
for nm,q in sub.items():
    lhs=yv[nm]; rhs=Dy_red(q)
    print(nm, show(lhs-rhs))
