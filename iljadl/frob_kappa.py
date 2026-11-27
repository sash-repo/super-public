"""Frobenius systems u_x=a, u_y=b in two independent variables.  Exact checks:
alpha^dalpha = -c dx^dy^du; [X,Z]=k1 Z, [Y,Z]=k2 Z for Z=[X,Y]=c d_u,
k1 = X(c)/c - a_u, k2 = Y(c)/c - b_u; closedness of x*al2+al3 <=> k1=k2=0; the examples;
the even fields on (x,y,u) commuting with X and Y for the system u_x=0, u_y=x."""
import sympy as sp
x, y, u = sp.symbols('x y u')
a, b = sp.Function('a')(x, y, u), sp.Function('b')(x, y, u)
def vf(F, f): return F[0]*sp.diff(f, x) + F[1]*sp.diff(f, y) + F[2]*sp.diff(f, u)
def br(F, G): return [sp.simplify(vf(F, G[i]) - vf(G, F[i])) for i in range(3)]
X = [1, 0, a]; Y = [0, 1, b]
c = sp.diff(b, x) + a*sp.diff(b, u) - sp.diff(a, y) - b*sp.diff(a, u)
Z = br(X, Y); assert sp.simplify(Z[0]) == 0 and sp.simplify(Z[1]) == 0 and sp.simplify(Z[2] - c) == 0
k1 = vf(X, c)/c - sp.diff(a, u); k2 = vf(Y, c)/c - sp.diff(b, u)
XZ = br(X, Z); YZ = br(Y, Z)
assert all(sp.simplify(XZ[i] - k1*Z[i]) == 0 for i in range(3))
assert all(sp.simplify(YZ[i] - k2*Z[i]) == 0 for i in range(3))
print('[X,Y]=c d_u, [X,Z]=k1 Z, [Y,Z]=k2 Z: True')
# alpha ^ d alpha, alpha = du - a dx - b dy, via the 3x3 formula alpha^dalpha = (A . curl A) dV
A = sp.Matrix([-a, -b, 1])     # components on dx, dy, du
curl = sp.Matrix([sp.diff(A[2], y) - sp.diff(A[1], u), sp.diff(A[0], u) - sp.diff(A[2], x),
                  sp.diff(A[1], x) - sp.diff(A[0], y)])
print('alpha^dalpha / (dx^dy^du) + c =', sp.simplify(A.dot(curl) + c))
# closedness of omega = x al2 + al3 on the frame (X,Y,Z): d omega(U,V)=U(om(V))-V(om(U))-om([U,V])
# dual coframe values: om(X)=0, om(Y)=x, om(Z)=1
om = {'X': 0, 'Y': x, 'Z': 1}
fr = {'X': X, 'Y': Y, 'Z': Z}
def omega_of(F):  # write F in the frame (X,Y,Z): F = p X + q Y + r Z
    p, q = F[0], F[1]; r = sp.simplify((F[2] - p*a - q*b)/c)
    return q*x + r
for U, V in (('X', 'Y'), ('X', 'Z'), ('Y', 'Z')):
    val = vf(fr[U], om[V]) - vf(fr[V], om[U]) - omega_of(br(fr[U], fr[V]))
    print(f'd(x al2 + al3)({U},{V}) =', sp.simplify(val))
# the examples
for name, aa, bb, v in (('u_x=0,u_y=x', 0, x, None), ('u_x=u,u_y=xu', u, x*u, sp.log(u) - x), ('a=1,b=u^2', 1, u**2, None)):
    cc = sp.diff(bb, x) + aa*sp.diff(bb, u) - sp.diff(aa, y) - bb*sp.diff(aa, u)
    X_ = [1, 0, aa]; Y_ = [0, 1, bb]
    K1 = sp.simplify(vf(X_, cc)/cc - sp.diff(aa, u)); K2 = sp.simplify(vf(Y_, cc)/cc - sp.diff(bb, u))
    line = f'{name}: c={cc}, k1={K1}, k2={K2}'
    if v is not None:
        line += f';  v={v}: X(v)={sp.simplify(vf(X_, v))}, Y(v)={sp.simplify(vf(Y_, v))}'
    print(line)
# u_x=0, u_y=x: even fields V=f d_x+g d_y+h d_u with [X,V]=[Y,V]=0, general solution by the ODE argument;
# check that the claimed 3-dim family works and is the general polynomial solution up to degree 4
X0 = [1, 0, 0]; Y0 = [0, 1, x]
cs = sp.symbols('c0:200'); k = 0; V = []
mons = [x**i*y**j*u**l for i in range(5) for j in range(5) for l in range(3) if i+j+l <= 4]
for comp in range(3):
    expr = 0
    for m in mons: expr += cs[k]*m; k += 1
    V.append(expr)
eqs = []
for F in (X0, Y0):
    for e in br(F, V): eqs += sp.Poly(sp.expand(e), x, y, u).coeffs()
sol = sp.solve(eqs, cs[:k], dict=True)[0]
Vs = [sp.expand(e.subs(sol)) for e in V]
print('general V commuting with X,Y (deg<=4 ansatz):', Vs)
# the closedness conditions are exactly -k1, -k2
for (U, V), kk in ((('X', 'Z'), k1), (('Y', 'Z'), k2)):
    val = vf(fr[U], om[V]) - vf(fr[V], om[U]) - omega_of(br(fr[U], fr[V]))
    print(f'd(x al2+al3)({U},{V}) + k =', sp.simplify(val + kk))
