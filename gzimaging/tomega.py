"""Two identities of the imaging text, checked symbolically (a computation of H. Andersen).

(1) Remark re:TOmega: on T*R (m = 1), hGz vanishes on pull-backs, and on the Sasaki-type
    lift Lambda(A) = a dq^2 + lam a dp^2 it gives the displayed formula.
(2) Remark re:poisson: the Poisson bracket of T*R^2 on two normal-curvature functions is
    -1/|p| times the pairing of Statement st:brk.

Run:  python3 tomega.py
"""
import sympy as sp
# ---- (1) T*R (m=1): Sasaki-type lift  Lambda(A) = a dq^2 + lam a dp^2  through hGz (m = 1) with (x,y)=(q,p)
x,y,lam=sp.symbols('q p lam',real=True)
a=sp.Function('a')(x); b=sp.Function('b')(x)
def PB(f,h): return sp.diff(f,x)*sp.diff(h,y)-sp.diff(f,y)*sp.diff(h,x)
def hGz(A,B):
    a_,b_,c_=A; a2,b2,c2=B; g=a_*c2-2*b_*b2+c_*a2
    D2A=sp.diff(a_,y,2)-2*sp.diff(b_,x,y)+sp.diff(c_,x,2); D2B=sp.diff(a2,y,2)-2*sp.diff(b2,x,y)+sp.diff(c2,x,2)
    return [sp.expand(sp.diff(g,x,2)+D2A*a2+D2B*a_+(PB(a_,b2)-PB(b_,a2))),
            sp.expand(sp.diff(g,x,y)+D2A*b2+D2B*b_+(PB(a_,c2)-PB(c_,a2))/2),
            sp.expand(sp.diff(g,y,2)+D2A*c2+D2B*c_+(PB(b_,c2)-PB(c_,b2)))]
print("pull-backs only (lam=0): hGz(pi^*A, pi^*B) =",hGz((a,0,0),(b,0,0)))
H=hGz((a,0,lam*a),(b,0,lam*b))
print("Sasaki lift: horizontal (dq^2) block =",sp.factor(H[0]))
print("             mixed (dq dp) block     =",H[1])
print("             vertical (dp^2) block   =",sp.factor(H[2]))
print("depends on p?",any(sp.diff(h,y)!=0 for h in H))
# ---- (2) Poisson bracket on T*R^2 of normal-curvature functions vs Statement 3.11 bracket
q1,q2,p1,p2=sp.symbols('q1 q2 p1 p2',real=True)
def rsym(seed):
    import random; random.seed(seed)
    f=lambda: sum(random.randint(-3,3)*q1**i*q2**j for i in range(3) for j in range(3-i))
    u,v,w=f(),f(),f(); return sp.Matrix([[u,v],[v,w]])
A,B=rsym(1),rsym(2)
# the line element is ker p; v = (p2,-p1)/|p| spans it (metric only in the normalisation)
r2=p1**2+p2**2
def kappa(M): v=sp.Matrix([p2,-p1]); return (v.T*M*v)[0]/r2
kA,kB=kappa(A),kappa(B)
pois=sp.simplify(sp.diff(kA,p1)*sp.diff(kB,q1)+sp.diff(kA,p2)*sp.diff(kB,q2)-sp.diff(kA,q1)*sp.diff(kB,p1)-sp.diff(kA,q2)*sp.diff(kB,p2))
# Statement 3.11 bracket in the metric frame: theta = direction angle of v, e2 = unit vector along v
th,r=sp.symbols('theta r',real=True)
sub={p1:-r*sp.sin(th),p2:r*sp.cos(th)}       # then v/|v| = (cos th, sin th)
kAt=sp.simplify(kA.subs(sub)); kBt=sp.simplify(kB.subs(sub))
e2=lambda F: sp.cos(th)*sp.diff(F,q1)+sp.sin(th)*sp.diff(F,q2)
brk=sp.diff(kAt,th)*e2(kBt)-e2(kAt)*sp.diff(kBt,th)
ratio=sp.simplify(pois.subs(sub)/brk)
print("Poisson bracket on T*R^2 / Statement-3.11 bracket =",ratio)
