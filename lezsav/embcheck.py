# Embeddings of witt into witt + current (centrally extended); Remark 5.3.2, the family (Lab).
# Relations: [L_m,L_n]=(m-n)L_{m+n}+c/12(m^3-m)d, [L_m,J_a]=-aJ_{m+a}, [J_a,J_b]=c/3 a d.
from fractions import Fraction as F
import itertools, sympy as sp
al,be,c=sp.symbols('alpha beta c')
def br(x,y):
    # x,y: dicts {('L',m):coef,('J',a):coef,('C',0):coef}
    out={}
    def add(k,v):
        out[k]=sp.expand(out.get(k,0)+v)
    for (tx,i),u in x.items():
        for (ty,j),w in y.items():
            if tx=='C' or ty=='C': continue
            if tx=='L' and ty=='L':
                add(('L',i+j),(i-j)*u*w)
                if i+j==0: add(('C',0),c*(i**3-i)/12*u*w)
            elif tx=='L' and ty=='J': add(('J',i+j),-j*u*w)
            elif tx=='J' and ty=='L': add(('J',i+j),i*u*w)
            else:
                if i+j==0: add(('C',0),c*i/3*u*w)
    return {k:v for k,v in out.items() if v!=0}
Lab=lambda m:{('L',m):1,('J',m):al+be*m}
bad=0;n=0
for m,k in itertools.product(range(-4,5),repeat=2):
    lhs=br(Lab(m),Lab(k)); 
    rhs={kk:sp.expand((m-k)*v) for kk,v in Lab(m+k).items()}
    for kk in set(lhs)|set(rhs):
        if kk[0]=='C': continue
        if sp.simplify(lhs.get(kk,0)-rhs.get(kk,0))!=0: bad+=1
    n+=1
# central term of [L^ab_m, L^ab_{-m}]
cen=[sp.factor(br(Lab(m),Lab(-m)).get(('C',0),0)) for m in range(1,5)]
# fit A m^3 + B m
A,B=sp.symbols('A B')
sol=sp.solve([cen[i]-(A*(i+1)**3+B*(i+1)) for i in range(4)],[A,B])
print("pairs",n,"bad",bad,"m^3 coeff*12 =",sp.factor(12*sol[A]))
