import sys, os, pickle; sys.path.insert(0,'.'); sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','lib'))
from symm1 import *
from exactlin import solve
def phi_of(sol):
    phi={}
    for lab,c in sol:
        kind,comp,ce,u,j=lab; ce=dict(ce)
        if kind=='xiy':
            xi=qmono(ce,u)
            for cc in COMPS: phi[cc]=phi.get(cc,Q())-(xi*RHS[cc])*c
        else:
            q=qmono(ce,u)
            if j: q=q*Q.var(jetvar(*j))
            phi[comp]=phi.get(comp,Q())+q*c
    return phi
def bracket(p1,s1,p2,s2):
    out={}
    sign = 1 if (s1 and s2) else -1
    for cc in COMPS:
        out[cc]=X_apply(p1,p2.get(cc,Q()),s1)+X_apply(p2,p1.get(cc,Q()),s2)*sign
    return out
S0=pickle.load(open('sols_s0_c3_21.pkl','rb')); S1=pickle.load(open('sols_s1_c3_21.pkl','rb'))
E=[phi_of(s) for k,s in S0]; O=[phi_of(s) for k,s in S1]
def express(br, basis):
    keys=sorted({(cc,m) for ph in basis+[br] for cc,q in ph.items() for m in q.t},key=str)
    A=[[ph.get(cc,Q()).t.get(m,0) for ph in basis] for cc,m in keys]
    b=[br.get(cc,Q()).t.get(m,0) for cc,m in keys]
    return solve(A,b)
def name(i, lst): return f'{lst}{i}{[str(a) for a in (S0 if lst=="E" else S1)[i][0]]}'
if __name__=='__main__':
    for i,(k,s) in enumerate(S0): print('E',i,[str(a) for a in k], len(s))
    for i,(k,s) in enumerate(S1): print('O',i,[str(a) for a in k], len(s))
    import itertools
    for i,j in itertools.combinations_with_replacement(range(len(O)),2):
        br=bracket(O[i],1,O[j],1)
        if all(q.iszero() for q in br.values()): continue
        sol=express(br,E)
        print(f'[O{i},O{j}] =', 'NOT IN SPAN' if sol is None else ' + '.join(f'{c}*E{n}' for n,c in enumerate(sol) if c))
