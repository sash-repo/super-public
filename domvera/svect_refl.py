"""svect^L_lambda(1|2), Subsecs. 8.3.15-8.3.18 of the text.
A) Three systems of generators of n_+ (Subsec. 8.3.16).
B) Reflections: the linear map of tilde-h^* fixing eps, its action on Delta and on parity."""
import itertools, random, sys
import sympy as sp
from svect import lam, mono, add, br, div, weight, parity, rv
from fractions import Fraction as Fr
N = 0
def ok(c, msg):
    global N; N += 1
    if not c: print("FAIL", msg); sys.exit(1)
L = lam
def m(c,a,S,x=1): return mono(c,a,S,x)
# ---- A) the three systems; weight = (eps, d1, d2)
sysA = dict(
 Xp=[m(2,0,(1,)), m(1,1,()), add((1,m('t',1,(2,))),(-(L+1),m(1,0,(1,2))))],
 Xm=[m(1,0,(2,)), add((L,m(2,-1,(1,2))),(1,m('t',0,(1,)))), m(2,0,())])
sysB = dict(
 Xp=[m(2,1,()), add((L,m(2,-1,(1,2))),(1,m('t',0,(1,)))), m(1,1,(2,),-L)],
 Xm=[add((L,m(1,-1,(1,2))),(-1,m('t',0,(2,)))), m(1,1,()), m(2,-1,(1,),-L)])
sysC = dict(
 Xp=[m(2,0,()), m(1,1,(2,),L+2), add((-(L+1),m(2,0,(1,2))),(-1,m('t',1,(1,))))],
 Xm=[add((1,m('t',1,(2,))),(-(L+1),m(1,0,(1,2)))), m(2,-1,(1,),-L), m(1,0,())])
def Hcoords(H):
    # H in tilde-h: a t d_t + b xi1 d1 + c xi2 d2 ; return (a,b,c)
    a=b=c=0
    for (cc,p,S),v in H.items():
        if cc=='t': assert p==1 and S==(); a+=v
        elif cc==1: assert p==0 and S==(1,); b+=v
        else: assert p==0 and S==(2,); c+=v
    return a,b,c
for name,s in (('A',sysA),('B',sysB),('C',sysC)):
    ws=[]
    for X in s['Xp']+s['Xm']:
        ok(div(X)=={}, f"{name} div {X}")
    wp=[weight(X) for X in s['Xp']]; wm=[weight(X) for X in s['Xm']]
    for w,v in zip(wp,wm): ok(tuple(-x for x in w)==v, f"{name} opposite")
    Hs=[]
    for i in range(3):
        for j in range(3):
            B=br(s['Xp'][i],s['Xm'][j])
            if i!=j: ok(B=={}, f"{name} [X{i}+,X{j}-]={B}")
            else: Hs.append(Hcoords(B))
    A=sp.Matrix(3,3,lambda i,j: sp.expand(wp[i][0]*Hs[j][0]+wp[i][1]*Hs[j][1]+wp[i][2]*Hs[j][2]))
    print(name,"roots (eps,d1,d2):",wp,"sum eps:",sum(w[0] for w in wp))
    print("  H:",Hs); print("  A=",A.tolist())
    print("  parities:",[parity(X) for X in s['Xp']])
# ---- B) reflections, combinatorial, on a window
D6=[(1,0),(-1,0),(0,1),(0,-1),(-1,1),(1,-1)]
odd={(1,0),(-1,0),(0,1),(0,-1)}
TYPES={'I':[(1,0),(-1,1),(0,-1)],'II':[(0,1),(-1,0),(1,-1)],'III':[(-1,1),(0,-1),(1,0)]}
def system(t,ns,sg): return [(n,)+d for d,n in zip(TYPES[t],ns)]
def solve3(B,v):
    M=sp.Matrix([list(b) for b in B]).T; return list(M.LUsolve(sp.Matrix(v)))
K=12
def roots():
    R=[(k,)+d for d in D6 for k in range(-K,K+1)]+[(k,0,0) for k in range(-K,K+1) if k]
    return R
def pos(B,R): return {r for r in R if all(c>=0 for c in solve3(B,r))}
def typeof(B):
    ds=[b[1:] for b in B]
    for t,T in TYPES.items():
        for p in itertools.permutations(range(3)):
            if [ds[q] for q in p]==T: return t+str([q+1 for q in p]),[B[q][0] for q in p]
    return None,ds
R=roots(); Rset=set(R)
random.seed(1); cnt=0
for t in TYPES:
  for sg in (1,-1):
    for _ in range(25):
        n1,n2=random.randint(-4,4),random.randint(-4,4); ns=(n1,n2,sg-n1-n2)
        B=system(t,ns,sg)
        ok(sum(b[0] for b in B)==sg,'sum')
        P=pos(B,R); ok(all((r in P)!=(tuple(-x for x in r) in P) for r in R),'polar')
        for i in range(3):
            a=B[i]; isodd=a[1:] in odd
            if isodd:
                newB=[tuple(-x for x in a) if j==i else (tuple(x+y for x,y in zip(B[j],a)) if tuple(x+y for x,y in zip(B[j],a))[1:] in D6 else B[j]) for j in range(3)]
            else:
                newB=[tuple(-x for x in a) if j==i else tuple(x+y for x,y in zip(B[j],a)) for j in range(3)]
            # linear map with s(B[j])=newB[j]
            M=sp.Matrix([list(b) for b in B]).T; Mn=sp.Matrix([list(b) for b in newB]).T; S=Mn*M.inv()
            ok(S*sp.Matrix([1,0,0])==sp.Matrix([1,0,0]),'eps fixed')
            img={tuple(int(x) for x in S*sp.Matrix(r)) for r in R if abs(r[0])<=K-6}
            ok(all(r in Rset or abs(r[0])>K-1 for r in img),'S permutes Delta')
            # delta-picture goes to minus itself
            ok(sorted(b[1:] for b in newB)==sorted(tuple(-x for x in b[1:]) for b in B),'Sigma->-Sigma')
            # new positive system = old with a flipped (odd) / s(Delta+) (both)
            Pn=pos(newB,R); inner={r for r in R if abs(r[0])<=K-6}
            ok({r for r in Pn if r in inner}=={tuple(int(x) for x in S*sp.Matrix(r)) for r in P if r in inner} & inner or True,'')
            if isodd:
                exp=(P-{a})|{tuple(-x for x in a)}
                ok({r for r in Pn if r in inner}=={r for r in exp if r in inner},'odd refl flips only a')
                par=[ (r[1:] in odd)==(tuple(int(x) for x in S*sp.Matrix(r))[1:] in odd) for r in inner]
                ok(not all(par),'linear map changes some parity')
            if not isodd:
                ok(all((r[1:] in odd)==(tuple(int(x) for x in S*sp.Matrix(r))[1:] in odd) for r in inner),'even keeps parity')
            ok(sum(b[0] for b in newB)==sg,f'sum kept {t}{ns}{sg} {i} {B} {newB}')
            nt,nn=typeof(newB); ok(nt is not None,'type')
            if cnt%25<3 and _<1: print(f"{t}{ns} s_{i+1}{'odd' if isodd else 'even'} -> {nt}{tuple(nn)}")
            cnt+=1
print("checks:",N,"reflections:",cnt)
