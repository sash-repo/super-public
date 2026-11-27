"""The odd families Xi_h, Xi'_h for an ARBITRARY function h(t): h and its
derivatives are jet variables ('h',0,k) with D_x h = D_y h = 0."""
import sys; sys.path.insert(0,'.')
import jets
from symm import *
from brackets import phi_of, S0, S1, bracket, express, E
jets.PAR['h']=0; jets.PAR['g']=0
_Dx, _Dt = jets.Dx, jets.Dt
def Dx2(q):
    def act(v):
        if v==('x',): return Q.const(1)
        if len(v)==3 and v[0] in ('h','g'): return None
        if len(v)==3: return Q.var((v[0],v[1]+1,v[2]))
    return jets.derivation(q,act,0)
jets.Dx=Dx2
import symm; symm.Dx=Dx2
_Dyv=jets.Dy_var
def Dy_var2(v,R):
    if v[0] in ('h','g'): return Q()
    return _Dyv(v,R)
jets.Dy_var=Dy_var2
def hk(k,n='h'): return Q.var((n,0,k))
V=lambda c,i=0,j=0: Q.var((c,i,j))
def Xi(n='h'):
    p={}
    for i in range(1,5):
        p[f'U{i}0']=hk(0,n)*V(f'U{i}1'); p[f'U{i}1']=hk(0,n)*V(f'U{i}0',1)
    p['U41']=p['U41']+hk(1,n)*F(1,3)
    return p
def Xip1():
    u=lambda c:V(c)
    p={'U10':u('U11')+u('U20'),'U11':-u('U21'),'U20':u('U21'),'U21':Q(),
       'U30':u('U31')+u('U40')+u('U11')*u('U20')+u('U20')*u('U20'),
       'U31':-u('U41')-u('U11')*u('U21')-u('U20')*u('U21')*2-u('U20')*V('U10',1),
       'U40':u('U41')+u('U20')*u('U21')+u('U20')*V('U10',1),
       'U41':-(u('U21')*V('U10',1))-u('U20')*V('U11',1)-u('U20')*V('U20',1)}
    return p
def Xip(n='h'):
    p={c:hk(0,n)*q for c,q in Xip1().items()}
    p['U10']=p['U10']+hk(1,n)*Q.var(('y',))*F(2,3)
    p['U30']=p['U30']+hk(1,n)*Q.var(('x',))*F(1,3)+hk(2,n)*Q.var(('y',))*Q.var(('y',))*F(2,9)
    return p
def report():
    """Xi_h and Xi'_h are symmetries for an arbitrary h(t); perturbed versions are not"""
    for nm,p in (('Xi_h',Xi()),("Xi'_h",Xip())):
        c=condition(p,1); print(nm,'is a symmetry for arbitrary h(t):',len(c)==0, len(c))
    # non-vacuity: perturbed coefficients must FAIL
    p=Xip(); p['U30']=p['U30']+hk(2)*Q.var(('y',))*Q.var(('y',))*F(1,9)
    print('perturbed Xi\'_h fails:', len(condition(p,1))>0)
    p=Xi(); p['U41']=p['U41']+hk(1)*F(1,6)
    print('perturbed Xi_h fails:', len(condition(p,1))>0)
if __name__ == '__main__':
    report()
