"""br(2;eps), p=3: Casimir on the highest weight vectors, symbolically in eps.
Positive roots alpha1, alpha2, alpha1+alpha2, alpha1+2alpha2 (the o(5) shape);
x_4 = alpha1+2alpha2 is the highest one (V_2 = x_4^2 has weight 2wt(x_4) as printed).
Form: (a_i,a_j) = d_i A_ij; the h_i <-> a_i/d_i consistency holds only mod 3."""
import sympy as sp
e=sp.symbols('eps')
A=sp.Matrix([[2,-1],[-2,1-e]])
d=[sp.Integer(1),sp.Rational(1,2)*A[0,1]/A[1,0]*2]   # d1 A12 = d2 A21 -> d2 = A12/A21 = 1/2
d=[sp.Integer(1),sp.Rational(1,2)]
def wt(v): return [sum(A[i,j]*v[j] for j in range(2)) for i in range(2)]
pos=[[1,0],[0,1],[1,1],[1,2]]
two_rho=[sum(wt(v)[i] for v in pos) for i in range(2)]
def B(lam,mu):
    c=A.LUsolve(sp.Matrix(lam))
    return sp.simplify(sum(c[j]*d[j]*mu[j] for j in range(2)))
def cas(lam): return sp.simplify(B(lam,[lam[i]+two_rho[i] for i in range(2)]))
def red3(x):
    n,dd=sp.fraction(sp.together(x))
    return sp.Poly(n,e,modulus=3), sp.Poly(dd,e,modulus=3)
top=wt([1,2]); V1=[a+b for a,b in zip(wt([1,0]),wt([1,2]))]; V2=[2*a for a in top]
print("2rho",two_rho,"wt top",top,"V1",V1,"V2",V2)
# quadratic part of the Casimir on h-values (control; the full Casimir is cas()):
def Q(l): h1,h2=l; return (1+2*e)/e*h1**2+h2**2/e+2/e*h1*h2
for nm,l in [("g",top),("V1",V1),("V2",V2)]:
    print(nm,"Q",sp.simplify(Q(l)),"  B(l,l)",sp.simplify(B(l,l)),"  cas",cas(l),"  mod3",red3(cas(l)))
t=cas(top)/2; al=(4*t-cas(V1))/2; be=(4*t-cas(V2))/2; ga=t-al-be
for nm,x in [("t",t),("alpha",al),("beta",be),("gamma",ga)]: print(nm,sp.simplify(x),red3(x))
