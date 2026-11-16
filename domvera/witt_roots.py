"""witt/vir, Subsec. 8.3.3 of the text.  Elements: dict {n: coeff} for L_n := t^{n+1} (f in C[t^-1,t],
bracket [f,g]=f g'-f' g, so [L_m,L_n]=(n-m)L_{m+n}), plus central z with cocycle c(L_m,L_n)=(m^3-m)delta_{m+n,0}
(= Res f''' g).  Prints every check; exits non-zero on failure."""
from fractions import Fraction as F
import itertools, sys
Z='z'
def add(*fs):
    r={}
    for f in fs:
        for k,v in f.items(): r[k]=r.get(k,0)+v
    return {k:v for k,v in r.items() if v}
def sm(c,f): return {k:c*v for k,v in f.items() if c*v}
def L(n,c=1): return {n:F(c)}
def br(f,g,vir=True):
    r={}
    for m,a in f.items():
        if m==Z: continue
        for n,b in g.items():
            if n==Z: continue
            r=add(r,{m+n:(n-m)*a*b})
            if vir and m+n==0: r=add(r,{Z:(m**3-m)*a*b})
    return r
fails=0; checks=0
def ok(cond,msg):
    global fails,checks; checks+=1
    if not cond: fails+=1; print('FAIL',msg)
# Jacobi + antisymmetry in a window
W=range(-5,6)
for a,b,c in itertools.product(W,W,W):
    x,y,w=L(a),L(b),L(c)
    ok(add(br(x,br(y,w)),br(y,br(w,x)),br(w,br(x,y)))=={}, ('Jac',a,b,c))
    ok(add(br(x,y),br(y,x))=={}, ('anti',a,b))
# system I: X1+=t^2=L1, X1-=1=L_{-1}, X2+=t^3=L2, X2-=t^{-1}=L_{-2}
H1=br(L(-1),L(1)); H2=br(L(-2),L(2))  # H_i=[X_i^-,X_i^+]
print('vir I: H1=[X1+,X1-]=',H1,' H2=',H2)
H1w=br(L(1),L(-1),False); H2w=br(L(2),L(-2),False)
print('witt I: H1=',H1w,' H2=',H2w)
# roots: [H,X_j^+]=alpha_j(H) X_j^+
def ev(H,n):  # [H,L_n] = c L_n
    r=br(H,L(n)); return r.get(n,0) if set(r)<= {n} else None
A=[[ev(H,n) for n in (1,2)] for H in (H1,H2)]
print('A_{ij}=alpha_j(H_i):',A)
print('cross [X1+,X2-]=',br(L(1),L(-2)),' [X2+,X1-]=',br(L(2),L(-1)))
# system II: X1+=1=L_{-1}, X1-=t^2=L1, X2+=t^4=L3, X2-=-t^{-2}=-L_{-3}
print('II: H1=',br(L(1),L(-1)),' H2=',br(L(-3,-1),L(3)),' (expected: -2t, -2t)')
# generation by II's X+ : L_{-1}, L_3
S={(-1,),(3,)}; span={-1:L(-1),3:L(3)}; frontier=[L(-1),L(3)]; got={-1,3}
for _ in range(4):
    new=[]
    for f in frontier:
        for g in (L(-1),L(3)):
            r=br(f,g,False)
            for k in r:
                if k not in got and abs(k)<8: got.add(k); new.append(L(k))
    frontier=new
print('degrees reached by the X+ of II:',sorted(got))
ok(0 in got,'II X+ reach degree 0')
# the flip: t -> 1/t sends L_n -> -L_{-n}; on vir z -> -z
def phi(f): 
    r={}
    for k,v in f.items():
        r=add(r,{Z:-v} if k==Z else {-k:-v})
    return r
for a,b in itertools.product(W,W):
    ok(phi(br(L(a),L(b)))==br(phi(L(a)),phi(L(b))),('phi',a,b))
# z -> +z fails:
def phi2(f):
    r={}
    for k,v in f.items(): r=add(r,{Z:v} if k==Z else {-k:-v})
    return r
bad=sum(phi2(br(L(a),L(b)))!=br(phi2(L(a)),phi2(L(b))) for a,b in itertools.product(W,W))
print('with z->+z the flip fails on',bad,'pairs (expected > 0)'); ok(bad>0,'control')
# ad L_1 not locally nilpotent: coefficient of [L1,L_n] = (n-1)
ok(all(br(L(1),L(n),False)=={n+1:n-1} for n in W if n!=1),"adL1")
# only two systems of positive roots: closed P in Z\0 with P u -P = Z\0, |roots|<=N
N=6
cnt=0
for bits in itertools.product([1,-1],repeat=N):
    P={s*k for s,k in zip(bits,range(1,N+1))}
    closed=all((a+b) in P for a in P for b in P if 0<abs(a+b)<=N)
    cnt+=closed
print('closed halves of {+-1..+-%d}:'%N,cnt); ok(cnt==2,'two Borels')
# generation: L1 alone does not generate witt_+, L1,L2 do
ok(br(L(1),L(1),False)=={},'L1 alone')
print(checks,'checks,',fails,'failures'); sys.exit(1 if fails else 0)
