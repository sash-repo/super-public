# Indecomposability of the Lambda(2)-modules of the type II diagrams (a^2=b^2=ab=ba=0).
# Module: bottom vertices B_1..B_v, top T_1..T_v, each C^m; a:B_i->T_i = I,
# b:B_{i+1}->T_i = I (i<v), b:B_1->T_v = J_mu(m).  End = commutant of a,b (block maps
# preserving the B/T splitting, which is forced: B = complement, T = aV+bV).
# Indecomposable <=> End local <=> End/rad has dim 1; we test: dim of the
# commutant and whether a generic element has a single eigenvalue.
import sympy as sp, itertools, random
def module(v,m,mu):
    N=2*v*m; a=sp.zeros(N); b=sp.zeros(N)
    B=lambda i:i*m; T=lambda i:v*m+i*m
    J=sp.Matrix(m,m,lambda r,c: mu if r==c else (1 if c==r+1 else 0))
    for i in range(v):
        a[T(i):T(i)+m,B(i):B(i)+m]=sp.eye(m)
    for i in range(v-1):
        b[T(i):T(i)+m,B(i+1):B(i+1)+m]=sp.eye(m)
    b[T(v-1):T(v-1)+m,B(0):B(0)+m]=J
    return a,b,N
def test(v,m,mu):
    a,b,N=module(v,m,mu)
    X=sp.Matrix(N,N,sp.symbols(f'x0:{N*N}'))
    eqs=list(X*a-a*X)+list(X*b-b*X)
    sol=sp.solve(eqs,list(X),dict=True)[0]
    Xs=X.subs(sol); free=sorted(Xs.free_symbols,key=str)
    random.seed(1); g=Xs.subs({s:random.randint(-9,9) for s in free})
    ev=set(sp.Matrix(g).eigenvals().keys())
    return len(free), len(ev)
for v,m,mu in [(1,1,1),(1,2,3),(1,3,2),(2,1,1),(2,1,4),(3,1,1),(2,2,1),(1,2,0),(2,1,0),(3,1,0)]:
    d,ne=test(v,m,mu)
    print(f"v={v} m={m} mu={mu}: dim End={d}, eigenvalues of a generic endomorphism={ne} ->",
          "indecomposable" if ne==1 else "DECOMPOSABLE")
