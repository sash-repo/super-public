"""k^L(1|4), Subsecs. 8.3.22-8.3.26 of the text: the root table (formula (49)), the root
multiplicities, Lemma 8.3.23 (H = t^{-1} xi1 eta1 xi2 eta2 is not in the derived algebra) and the
brackets used there. Odd labels 0=xi1, 1=eta1, 2=xi2, 3=eta2; contact form
dt+sum(xi_i deta_i+eta_i dxi_i); the bracket of formula (26). Run: python3 k14.py"""
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
import sys, itertools, random
import kL; kL.SIGN=-1; kL.PAIRS=[(0,1),(1,0),(2,3),(3,2)]
from kL import br, m, add, sm, par
from fractions import Fraction as F
N=4; X1,E1,X2,E2=0,1,2,3
B=lambda f,g: br(f,g,N)
cnt=[0,0]
def ok(c,msg):
    cnt[0]+=1
    if not c: cnt[1]+=1; print("FAIL",msg)
def eq(a,b,msg): ok(add(a,sm(-1,b))=={},f"{msg}: {a} vs {b}")
# Jacobi
random.seed(1)
def rnd(p):
    f={}
    for a in range(-2,3):
        for r in range(5):
            if r%2!=p: continue
            for S in itertools.combinations(range(4),r): f=add(f,{(a,S):F(random.randint(-3,3))})
    return f
for _ in range(40):
    ps=[random.randint(0,1) for _ in range(3)]; f,g,h=[rnd(p) for p in ps]
    eq(B(f,B(g,h)),add(B(B(f,g),h),sm((-1)**(ps[0]*ps[1]),B(g,B(f,h)))),"Jacobi")
# roots: weight of monomial (a,S) w.r.t. t, xi1eta1, xi2eta2: ad_t eigen = 2a-2+|S|
T,H1,H2=m(1),m(0,X1,E1),m(0,X2,E2)
def weight(a,S):
    f=m(a,*S)
    res=[]
    for h in (T,H1,H2):
        g=B(h,f); c=F(0) if not g else g[next(iter(f))]
        eq(g,sm(c,f),f"weight vector {a}{S}"); res.append(c)
    return tuple(res)   # (eps-coef, then delta-coefficients as eigenvalues of xi_i eta_i)
mons=[(a,S) for a in range(-4,5) for r in range(5) for S in itertools.combinations(range(4),r)]
W={}
for a,S in mons: W.setdefault(weight(a,S),[]).append((a,S))
# the root table (49): delta_i(xi_i eta_i)=1, so weight of eta_i is +delta_i
def dl(a,S):
    e=2*a-2+len(S); d=[0,0]
    for i,(x,y) in enumerate(((X1,E1),(X2,E2))):
        d[i]=(y in S)-(x in S)
    return (e,d[0],d[1])
for a,S in mons:
    w=weight(a,S); ok(w==(F(dl(a,S)[0]),F(dl(a,S)[1]),F(dl(a,S)[2])),f"table {a}{S} {w} vs {dl(a,S)}")
# multiplicities: +-delta_i+(2n+1)e:2, 2n e:4, +-d1+-d2+2ne:1
for n in range(-2,2):
    for s in (1,-1):
        ok(len(W[(F(2*n+1),F(s),F(0))])==2 and len(W[(F(2*n+1),F(0),F(s))])==2,"mult 2")
        for s2 in (1,-1): ok(len(W[(F(2*n),F(s),F(s2))])==1,"mult 1")
    ok(len(W[(F(2*n),F(0),F(0))])==4,"mult 4")
# derived algebra: coefficient of H=t^{-1}xi1eta1xi2eta2 vanishes on every bracket (Lemma 8.3.23)
Hk=(-1,(0,1,2,3))
for (w,L) in W.items():
    mw=(-w[0],-w[1],-w[2])
    if mw not in W: continue
    for u in L:
        for v in W[mw]:
            ok(B(m(*u[:1],*u[1]),m(*v[:1],*v[1])).get(Hk,0)==0,f"H in [g,g]? {u},{v}")
H=m(-1,X1,E1,X2,E2)
eq(B(T,H),{},"[t,H]=0"); eq(B(m(1,X1,E1),H),{},"[xi1eta1 t,H]=0"); eq(B(m(1,X2,E2),H),{},"[xi2eta2 t,H]=0")
eq(B(H1,H),{},"[xi1eta1,H]"); eq(B(H2,H),{},"[xi2eta2,H]")
for k in range(-3,4):
    eq(B(H,m(k)),sm(2-2*k,m(k-2,X1,E1,X2,E2)),f"[H,t^k] k={k}")
    g=B(H,m(k,X1)); ok(g!={} and set(g)=={(k-1,(0,2,3))},f"[H,xi1 t^k] {g}")
    for r in range(2,5):
        for S in itertools.combinations(range(4),r): eq(B(H,m(k,*S)),{},"[H,phi t^k]=0 d>=2")
# formulas [t^2, phi t^k] = 2(k+d-2) t phi t^k, [t^n,f]=t^{n-1}f(2k-2n+n d)
for k in range(-3,4):
    for r in range(5):
        for S in itertools.combinations(range(4),r):
            f=m(k,*S)
            for nn in range(-2,4): eq(B(m(nn),f),sm(2*k-2*nn+nn*r,m(nn+k-1,*S)),f"[t^n,f] n={nn}")
print(f"part 1: {cnt[0]} checks, {cnt[1]} failures")
