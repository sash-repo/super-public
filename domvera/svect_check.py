"""Checks of several claims of Sect. 8.3 of the text.
A) k^L(1|5) (the Remark after Lemma 8.3.23): [t, phi theta t^k], the roots, and Phi/t = -[t^k theta, t^{-k-1} Phi theta].
B) k^L(1|4), Proposition 8.3.29: the reflection in an odd simple root beta is a linear map.
C) svect^L_lambda(1|2): the root vector t^{k+1} xi1 xi2 d_t lies in g if and only if k=-lambda-1.
D) k^L(1|2), Proposition 8.3.9: sigma phi sigma = phi^{-1}; phi^{2n+1} sigma = phi^n (phi sigma) phi^{-n}.
E) svect^L_lambda(1|2), Subsec. 8.3.18: the Cartan matrices G1, G2, G3 vs. svect_refl."""
import sys, itertools, random
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from fractions import Fraction as F
import kL
from kL import add, sm, m, par
import sympy as sp
N=0
def ok(c,msg):
    global N; N+=1
    if not c: print("FAIL",msg); sys.exit(1)
# ---------- A) k^L(1|5): odd labels 0=xi1,1=eta1,2=xi2,3=eta2,4=theta
kL.SIGN=-1; kL.PAIRS=[(0,1),(1,0),(2,3),(3,2),(4,4)]
br=lambda f,g: kL.br(f,g,5)
Phi=(0,1,2,3)
for k in range(-6,7):
    ok(br(m(k,4),m(-k-1,0,1,2,3,4))==m(-1,0,1,2,3,c=-1), f"Phi/t k={k}")
for k in range(-4,5):
    for r in range(0,5):
        for S in itertools.combinations(range(4),r):
            f=m(k,*S,4); ok(br(m(1),f)==sm(2*k+r-1,f), f"[t,f] {k}{S}")
# Phi/t in k^L(1|4): never a bracket (coefficient check on random pairs) vs. in k^L(1|5) it is
random.seed(7)
def rnd(Nv):
    f={}
    for a in range(-2,3):
        for r in range(Nv+1):
            for S in itertools.combinations(range(Nv),r):
                if random.random()<.3: f=add(f,{(a,S):F(random.randint(-3,3))})
    return f
for _ in range(60):
    f,g=rnd(5),rnd(5)
    # Jacobi spot check in k^L(1|5) on homogeneous parts
    f={k:v for k,v in f.items() if len(k[1])%2==0}; g={k:v for k,v in g.items() if len(k[1])%2==1}
    h=m(random.randint(-2,2),*random.choice([(4,),(0,1),(2,),(0,1,4)]))
    ph,pg,pf=par(h),par(g),par(f)
    J=add(br(f,br(g,h)),sm(-1,br(br(f,g),h)),sm(-(-1)**(pf*pg),br(g,br(f,h))))
    ok(J=={},"Jacobi k^L(1|5)")
kL.PAIRS=[(0,1),(1,0),(2,3),(3,2)]
for _ in range(300):
    a,b=random.randint(-3,3),random.randint(-3,3)
    S=random.choice([S for r in range(5) for S in itertools.combinations(range(4),r)])
    T=tuple(sorted(set(range(4))-set(S)))
    for (x,y) in ((S,T),):
        B=kL.br(m(a,*x),m(b,*y),4)
        ok(B.get((-1,Phi),0)==0,"k14 Phi/t never a bracket")
print("A done",N)
# ---------- B) k^L(1|4) reflections, combinatorics on a window
# root: (j, d1, d2) meaning d1 delta1 + d2 delta2 + j eps; multiplicity & parity from k14table
def mult(r):
    j,a,b=r
    if (a,b)==(0,0): return (4,0) if j%2==0 and j!=0 else None
    if abs(a)+abs(b)==1: return (0,2) if j%2 else None
    if abs(a)==1 and abs(b)==1: return (1,0) if j%2==0 else None
    return None
K=30
R=[(j,a,b) for j in range(-K,K+1) for a in (-1,0,1) for b in (-1,0,1) if mult((j,a,b))]
inner=lambda r: abs(r[0])<=K-12
def coeffs(B,r):
    M=sp.Matrix([list(x) for x in B]).T; return list(M.LUsolve(sp.Matrix(r)))
def pos(B): return {r for r in R if all(c>=0 for c in coeffs(B,r))}
def simple(P):
    S=[]
    for r in P:
        if not inner(r): continue
        dec=any(tuple(x-y for x,y in zip(r,s)) in P for s in P if s!=r and inner(s))
        if not dec: S.append(r)
    return set(S)
bases=[]
for a0 in [(1,1),(1,-1),(-1,1),(-1,-1)]:
    for b0 in [(1,0),(-1,0),(0,1),(0,-1)]:
        if a0[0]*b0[0]+a0[1]*b0[1]==-1: bases.append((a0,b0))
ok(len(bases)==8,"8 bases")
cnt=0
for (a0,b0) in bases:
  for _ in range(4):
    a=2*random.randint(-3,3); b=2*random.randint(-3,3)+1; c=2-a-2*b
    if abs(c)>8: continue
    g0=(-a0[0]-2*b0[0],-a0[1]-2*b0[1])
    al,be,ga=(a,)+a0,(b,)+b0,(c,)+g0
    B=[al,be,ga]; P=pos(B)
    ok(simple(P)=={al,be,ga},"simple")
    def nb(r): return tuple(-x for x in r)
    def ad(r,s,k=1): return tuple(x+k*y for x,y in zip(r,s))
    # odd beta: flip beta only
    Pb=(P-{be})|{nb(be)}
    newB=[ad(al,be,2),nb(be),ad(ga,be,2)]
    ok({r for r in pos(newB) if inner(r)}=={r for r in Pb if inner(r)},"odd refl = flip beta")
    ok(simple(Pb)=={r for r in newB},"new simple roots {-b,a+2b,g+2b}")
    ok(ad(al,be) in P and ad(al,be) not in simple(Pb),"alpha+beta not simple after")
    # linear map S_beta: fixes eps, permutes roots with mult & parity
    M=sp.Matrix([list(x) for x in B]).T; Mn=sp.Matrix([list(x) for x in newB]).T; S=Mn*M.inv()
    ok(S*sp.Matrix([1,0,0])==sp.Matrix([1,0,0]),"eps fixed")
    for r in R:
        if not inner(r): continue
        s=tuple(int(x) for x in S*sp.Matrix(r)); ok(mult(s)==mult(r),"S_beta keeps mult/parity")
    # even alpha: Weyl reflection beta->beta+alpha, gamma->gamma
    for e,o1,o2 in ((al,be,ga),(ga,be,al)):
        Pa=(P-{e})|{nb(e)}
        nB=[nb(e),ad(o1,e),o2]
        ok({r for r in pos(nB) if inner(r)}=={r for r in Pa if inner(r)},"even refl")
        ok(simple(Pa)==set(nB),"even simple")
    cnt+=1
print("B done",N,"systems",cnt)
# ---------- C) svect^L_lambda(1|2): weight spaces
import svect
from svect import lam, mono, div
def space(lv, w):
    """basis count of the elements of tilde-weight w=(e,d1,d2), lambda=lv."""
    mons=[]
    for c in ('t',1,2):
        for S in [(),(1,),(2,),(1,2)]:
            d1=(1 in S)-(c==1); d2=(2 in S)-(c==2)
            if (d1,d2)!=w[1:]: continue
            a=w[0]+(1 if c=='t' else 0)
            mons.append(mono(c,a,S))
    if not mons: return 0
    # split by parity
    tot=0
    for p in (0,1):
        ms=[x for x in mons if svect.parity(x)==p]
        if not ms: continue
        keys=sorted({k for x in ms for k in div(x)},key=str)
        M=sp.Matrix(len(keys),len(ms),lambda i,j: div(ms[j]).get(keys[i],0)).subs(lam,lv)
        tot+=len(ms)-(M.rank() if keys else 0)
    return tot
expected={}
for lv in (0,1,-1,-3,sp.Rational(1,2),sp.Rational(-5,3)):
    for k in range(-5,6):
        for d in [(1,0),(-1,0),(0,1),(0,-1),(-1,1),(1,-1),(1,1),(-1,-1),(0,0)]:
            n=space(lv,(k,)+d)
            if d==(1,1):
                want=1 if (lv.is_integer if hasattr(lv,'is_integer') else True) and k==-lv-1 else 0
                ok(n==want,f"d1+d2 lam={lv} k={k}: {n}")
            elif d==(-1,-1): ok(n==0,"-d1-d2")
            elif d==(0,0): ok(n==(2 if k else 2),f"k eps {lv} {k} {n}")
            else: ok(n==1,f"{d} {lv} {k} {n}")
print("C done",N)
# ---------- D) phi, sigma on k^L(1|2): labels 0=xi,1=eta
def phi(f,inv=False):
    s=-1 if inv else 1; o={}
    for (a,S),c in f.items():
        if S==(1,): o=add(o,{(a+s,S):c})
        elif S==(0,): o=add(o,{(a-s,S):c})
        elif S==(0,1): o=add(o,{(a,S):c})
        else: o=add(o,{(a,()):c},{(a-1,(0,1)):-2*s*c})
    return o
def sig(f):
    o={}
    for (a,S),c in f.items():
        if S==(0,): o=add(o,{(a,(1,)):c})
        elif S==(1,): o=add(o,{(a,(0,)):c})
        elif S==(0,1): o=add(o,{(a,S):-c})
        else: o=add(o,{(a,S):c})
    return o
def pw(f,n):
    for _ in range(abs(n)): f=phi(f,inv=n<0)
    return f
mons=[{(a,S):F(1)} for a in range(-5,6) for S in [(),(0,),(1,),(0,1)]]
for f in mons:
    ok(sig(phi(sig(f)))==phi(f,inv=True),"sigma phi sigma = phi^-1")
    for n in range(-3,4):
        ok(pw(sig(f),2*n+1)==pw(phi(sig(pw(f,-n))),n),"conjugate")
kL.PAIRS=[(0,1),(1,0)]
for f in mons:
    for g in mons:
        ok(kL.br(phi(f),phi(g),2)==phi(kL.br(f,g,2)),"phi automorphism")
        ok(kL.br(sig(f),sig(g),2)==sig(kL.br(f,g,2)),"sigma automorphism")
print("D done",N)
# ---------- E) G1,G2,G3 (transposed convention: A_ij = alpha_j(H_i), i.e. transpose of svect_refl's)
L=sp.Symbol('lam')
Ours={'A':sp.Matrix([[2,1-L,L+1],[-1,0,-L],[-1,L,0]]),
      'B':sp.Matrix([[0,1-L,-L**2],[1-L,0,-L**2],[L-2,L,2*L**2]]),
      'C':sp.Matrix([[0,L**2+2*L,L+1],[-L,-2*L**2-4*L,-L-2],[L+1,L**2+2*L,0]])}
G1=sp.Matrix([[2,-1,-1],[1-L,0,L],[1+L,-L,0]])
G3=sp.Matrix([[0,-L,L+1],[-1,2,-1],[1+L,-L-2,0]])
G2crossed=sp.Matrix([[0,L+1,-2+L],[1-L,0,L],[-1,-1,2]])
def rowprop(M,Nm):  # rows proportional (nonzero factors)
    for i in range(3):
        r,s=M.row(i),Nm.row(i)
        if sp.simplify(sp.Matrix.hstack(r.T,s.T).rank(simplify=True))!=1: return False
    return True
def colscale(M):  # normalize columns: diag 2 or col scaled to have a -1 in first off-diag... use A^T rows
    return M
At={k:v.T for k,v in Ours.items()}
ok(At['A']==G1,"G1 exact")
ok(rowprop(At['C'],G3),"G3 up to the normalization of H_2")
ok(not rowprop(At['B'],G2crossed),"G2 crossed out rightly")
C1=lambda mu: sp.Matrix([[2,-1,-1],[-1+1/mu,0,1],[1+1/mu,-1,0]])
found={}
for k in ('B','C'):
    for p in itertools.permutations(range(3)):
        M=At[k].extract(list(p),list(p))
        for mu in (L+1,L-1,1-L,-L,-1-L):
            if rowprop(M,C1(mu)): found.setdefault(k,[]).append((p,mu))
print("E: G2,G3 vs C1(mu):",found)
ok(any(mu==L+1 for p,mu in found['C']),"C3=C1(lam+1)")
print("checks:",N)
