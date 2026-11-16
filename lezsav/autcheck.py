# Remark 5.3.2 b): Aut k^L(1|2) acting on the embeddings L^{a,b}_m = L_m+(a+bm)J_m.
# Centerless N=2 NS algebra: [L_m,L_n]=(m-n)L_{m+n}, [L_m,J_n]=-nJ_{m+n},
# [L_m,G_r]=(m/2-r)G_{m+r}, [L_m,H_r]=(m/2-r)H_{m+r}, [J_m,G_r]=G_{m+r},
# [J_m,H_r]=-H_{m+r}, [G_r,H_s]=2L_{r+s}+(r-s)J_{r+s}, [G,G]=[H,H]=[J,J]=0.
import itertools, sympy as sp
from fractions import Fraction as F
h=F(1,2)
par={'L':0,'J':0,'G':1,'H':1}
def add(d,k,v):
    d[k]=sp.expand(d.get(k,0)+v)
    if d[k]==0: del d[k]
def gbr(a,i,b,j):
    out={}
    if a=='L' and b=='L': add(out,('L',i+j),i-j)
    elif a=='L' and b=='J': add(out,('J',i+j),-j)
    elif a=='L' and b in 'GH': add(out,(b,i+j),i*h-j)
    elif a=='J' and b=='G': add(out,('G',i+j),1)
    elif a=='J' and b=='H': add(out,('H',i+j),-1)
    elif a=='G' and b=='H': add(out,('L',i+j),2); add(out,('J',i+j),i-j)
    elif a=='H' and b=='G': return gbr('G',j,'H',i)
    elif b=='L' or (b=='J' and a in 'GH'):
        s=-1 if not (par[a] and par[b]) else 1
        return {k:s*v for k,v in gbr(b,j,a,i).items()}
    return out
def br(x,y):
    out={}
    for (a,i),u in x.items():
        for (b,j),w in y.items():
            for k,v in gbr(a,i,b,j).items(): add(out,k,u*w*v)
    return out
def comb(*ts):
    out={}
    for c,x in ts:
        for k,v in x.items(): add(out,k,c*v)
    return out
def basis(K):
    for m in range(-K,K+1):
        yield ('L',m); yield ('J',m); yield ('G',m+h); yield ('H',m+h)
B=list(basis(3))
bad=0;n=0
for x,y,z in itertools.product(B,repeat=3):
    X,Y,Z={x:1},{y:1},{z:1}
    s=lambda p,q: (-1)**(par[p[0]]*par[q[0]])
    lhs=br(X,br(Y,Z)); r1=br(br(X,Y),Z); r2=br(Y,br(X,Z))
    tot=comb((1,lhs),(-1,r1),(-s(x,y),r2)); n+=1
    if tot: bad+=1
print("Jacobi triples",n,"bad",bad)
def check_aut(phi,name):
    bad=0;n=0
    for x,y in itertools.product(B,repeat=2):
        lhs=comb(*[(v,phi(k)) for k,v in br({x:1},{y:1}).items()])
        rhs=br(phi(x),phi(y)); n+=1
        if comb((1,lhs),(-1,rhs)): bad+=1
    print(name,"pairs",n,"bad",bad)
# spectral flow by 1
def U(k):
    a,i=k
    return {'L':{('L',i):1,('J',i):1},'J':{('J',i):1},'G':{('G',i+1):1},'H':{('H',i-1):1}}[a]
check_aut(U,"spectral flow U_1")
# inversion t->1/t: L_m->-L_{-m}, J_m->J_{-m}, G_r->G_{-r}, H_r->-H_{-r}
def P(k):
    a,i=k
    return {'L':{('L',-i):-1},'J':{('J',-i):1},'G':{('G',-i):1},'H':{('H',-i):-1}}[a]
check_aut(P,"inversion P")
# mirror: J->-J, G<->H
def Mi(k):
    a,i=k
    return {'L':{('L',i):1},'J':{('J',i):-1},'G':{('H',i):1},'H':{('G',i):1}}[a]
check_aut(Mi,"mirror M")
al,be=sp.symbols('alpha beta')
Lab=lambda m,a,b:{('L',m):1,('J',m):a+b*m}
def img(phi,x):
    return comb(*[(v,phi(k)) for k,v in x.items()])
for name,phi,target,sgn,inv in [("U",U,(al+1,be),1,1),("P",P,(-al,be),-1,-1),("M",Mi,(-al,-be),1,1)]:
    ok=all(not comb((1,img(phi,Lab(m,al,be))),(-sgn,Lab(inv*m,*target))) for m in range(-3,4))
    print(name,"maps L^{a,b} to",target,"(sign %d, m->%dm):"%(sgn,inv),ok)
# odd module of the image: [L^{ab}_m, G_r], [L^{ab}_m,H_r]
m,r=sp.symbols('m r')
print("[L^ab_m,G_r] coeff:", sp.expand((m/2-r)+(al+be*m)), " [L^ab_m,H_r]:",sp.expand((m/2-r)-(al+be*m)))
