"""Weight-0 1-cocycles of the polynomial contact superalgebra k(1|n) with values in F_lam,
in a window of degrees; computes dim(cocycles) and dim(coboundaries).  Statement 5.10.3: for n = 3, lam = -1/2 every
weight-0 cocycle is one of the 3 coboundaries; the other rows are the controls (n = 0, 1, 2 give
the cocycles c_n = St_n of Proposition 5.10.1, and n = 3, lam = 1/2 the cocycle D_th1 D_th2 D_th3).
Run:  python3 h1kn.py"""
import sys, itertools
from fractions import Fraction as Fr
import os; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"), os.path.dirname(os.path.abspath(__file__))]
from exactlin import rank, nullspace
def mul(P,Q):
    R={}
    for (a,S),u in P.items():
        for (b,T),v in Q.items():
            if set(S)&set(T): continue
            seq=list(S)+list(T); s=1
            for i in range(len(seq)):
                for j in range(len(seq)-1-i):
                    if seq[j]>seq[j+1]: seq[j],seq[j+1]=seq[j+1],seq[j]; s=-s
            k=(a+b,tuple(seq)); R[k]=R.get(k,0)+s*u*v
    return {k:v for k,v in R.items() if v}
def add(*Ps):
    R={}
    for P in Ps:
        for k,v in P.items(): R[k]=R.get(k,0)+v
    return {k:v for k,v in R.items() if v}
def sc(c,P): return {k:c*v for k,v in P.items() if c*v}
def dx(P): return {(a-1,S):a*v for (a,S),v in P.items() if a>0}
def dth(P,i):
    R={}
    for (a,S),v in P.items():
        if i in S:
            j=S.index(i); R[(a,S[:j]+S[j+1:])]=(-1)**j*v
    return R
def D(P,i): return add(dth(P,i), mul({(0,(i,)):1},dx(P)))
def par(P):
    ps={len(S)%2 for (a,S) in P}; assert len(ps)<=1; return ps.pop() if ps else 0
def K(H,F,lam,n):
    pH=par(H); r=add(mul(H,dx(F)), sc(lam,mul(dx(H),F)))
    for i in range(n): r=add(r, sc(Fr((-1)**pH,2), mul(D(H,i),D(F,i))))
    return r
def br(H,G,n):
    pH=par(H); r=add(mul(H,dx(G)), sc(-1,mul(dx(H),G)))
    for i in range(n): r=add(r, sc(Fr((-1)**pH,2), mul(D(H,i),D(G,i))))
    return r
def deg(a,S): return Fr(2*a+len(S),2)
def monos(n, d):  # monomials x^a theta^S of degree d
    out=[]
    for r in range(n+1):
        for S in itertools.combinations(range(n),r):
            a=d-Fr(r,2)
            if a>=0 and a.denominator==1: out.append((int(a),S))
    return out
def run(n, lam, shift, Dmax, cpar):
    """cochains c: x^a th^S (deg d) -> F_lam of degree d+shift; parity cpar of c"""
    degs=[Fr(k,2) for k in range(0,int(2*Dmax)+1)]
    basis=[(m,d) for d in degs for m in monos(n,d)]
    # unknown index: (source mono, target mono)
    unk=[]; idx={}
    for (m,d) in basis:
        for t in monos(n, d+shift) if d+shift>=0 else []:
            if (len(m[1])+len(t[1])+cpar)%2==0:
                idx[(m,t)]=len(unk); unk.append((m,t))
    def capply(Hmono_poly):  # returns dict target-> linear form (dict unk->coef)
        R={}
        for mono,v in Hmono_poly.items():
            for t in monos(n, deg(*mono)+shift) if deg(*mono)+shift>=0 else []:
                if (mono,t) in idx:
                    R.setdefault(t,{}); R[t][idx[(mono,t)]]=R[t].get(idx[(mono,t)],0)+v
        return R
    def act(H, lin):  # H acting on a linear-form-valued element
        R={}
        for t,form in lin.items():
            img=K(H,{t:1},lam,n)
            for t2,v in img.items():
                for j,cf in form.items():
                    R.setdefault(t2,{}); R[t2][j]=R[t2].get(j,0)+v*cf
        return R
    rows=[]
    for (m1,d1) in basis:
        for (m2,d2) in basis:
            if d1+d2-1>Dmax: continue
            H={m1:1}; G={m2:1}; pH,pG=par(H),par(G)
            lhs=capply(br(H,G,n))
            # c([H,G]) = (-1)^{pH cpar} H.c(G) - (-1)^{pG(pH+cpar)} G.c(H)
            r1=act(H,capply(G)); r2=act(G,capply(H))
            tot={}
            for t,f in lhs.items():
                for j,v in f.items(): tot.setdefault(t,{}); tot[t][j]=tot[t].get(j,0)+v
            for t,f in r1.items():
                for j,v in f.items(): tot.setdefault(t,{}); tot[t][j]=tot[t].get(j,0)-(-1)**(pH*cpar)*v
            for t,f in r2.items():
                for j,v in f.items(): tot.setdefault(t,{}); tot[t][j]=tot[t].get(j,0)+(-1)**(pG*(pH+cpar))*v
            for t,f in tot.items():
                row=[0]*len(unk)
                for j,v in f.items(): row[j]+=v
                if any(row): rows.append(row)
    ncoc=len(unk)-(rank(rows) if rows else 0)
    # coboundaries: c(H)=H.v for v in F_lam of degree shift (weight-0), parity cpar
    cob=[]
    for v in (monos(n,-lam) if -lam>=0 else []):
        if len(v[1])%2!=cpar: continue
        vec=[0]*len(unk)
        for (m,d) in basis:
            for t,cf in K({m:1},{v:1},lam,n).items():
                if (m,t) in idx: vec[idx[(m,t)]]+=cf
        cob.append(vec)
    ncob=rank(cob) if cob else 0
    return ncoc, ncob
if __name__=='__main__':
    # calibration: n=0 lam=1 (H''), n=1 lam=1/2 (D H'), n=2 lam=0 (two)
    for n,lam,Dm in [(0,Fr(1),5),(1,Fr(1,2),4),(2,Fr(0),3),(3,Fr(-1,2),3),(3,Fr(1,2),3)]:
        print('n=%d lam=%s: (cocycles, coboundaries) by parity of c:'%(n,lam), [run(n,lam,-1-lam,Dm,p) for p in (0,1)])
