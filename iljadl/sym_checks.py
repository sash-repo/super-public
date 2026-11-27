"""Exact (sympy) checks: the dispersionless mKP equation and the KZ equation (Miura map),
supersymmetric quantum mechanics (Darboux, Crum, Baecklund), the curvature of an operator
and of a derivation."""
import sympy as sp
x, y, t = sp.symbols('x y t')
def ok(name, e):
    if isinstance(e, sp.MatrixBase):
        print(f'{name}:', all(sp.simplify(z) == 0 for z in e))
    else:
        print(f'{name}:', sp.simplify(e) == 0)

# ---- 1. dispersionless mKP -> KZ (Miura): the solutions invariant under Xi'_1
phi = sp.Function('phi')(x, y, t)
R = sp.Rational(3,4)*phi.diff(y,2) + sp.Rational(3,2)*phi.diff(y)*phi.diff(x,2) - sp.Rational(3,2)*phi.diff(x)**2*phi.diff(x,2)
def onshell(e):
    # replace every derivative containing x and t at least once by derivatives of R
    e = sp.expand(e)
    for d in sorted(e.atoms(sp.Derivative), key=lambda d: -len(d.variables)):
        vs = list(d.variables)
        if vs.count(x) >= 1 and vs.count(t) >= 1:
            vs2 = list(vs); vs2.remove(x); vs2.remove(t)
            r = R
            for v in vs2: r = r.diff(v)
            e = e.subs(d, r)
    return sp.expand(e)
u = phi.diff(x); v = (phi.diff(y) - phi.diff(x)**2)/2
ok('u_y = 2uu_x + 2v_x (identity)', u.diff(y) - 2*u*u.diff(x) - 2*v.diff(x))
ok('v_y = 2/3u_t - 2(uv)_x - 2u^2u_x  <=>  d_x of it is dmKP', onshell(sp.diff(v.diff(y) - sp.Rational(2,3)*u.diff(t) + 2*(u*v).diff(x) + 2*u**2*u.diff(x), x)))
# the undifferentiated form: v_y - 2/3 u_t + ... equals -2/3*(phi_xt - R)
e = v.diff(y) - sp.Rational(2,3)*u.diff(t) + 2*(u*v).diff(x) + 2*u**2*u.diff(x) + sp.Rational(2,3)*(phi.diff(x,t) - R)
ok('second equation == -2/3*(dmKP)', e)
KZ = v.diff(x,t) - sp.Rational(3,4)*v.diff(y,2) - 3*(v*v.diff(x)).diff(x)
ok('Miura image solves v_xt = 3/4 v_yy + 3(v v_x)_x', onshell(KZ))

# ---- 2. SQM
W = sp.Function('W')(x); f = sp.Function('f')(x)
A = lambda g: g.diff(x) + W*g
Ad = lambda g: -g.diff(x) + W*g
Hm = lambda g: -g.diff(x,2) + (W**2 - W.diff(x))*g
Hp = lambda g: -g.diff(x,2) + (W**2 + W.diff(x))*g
ok('A^dag A = H_-', Ad(A(f)) - Hm(f)); ok('A A^dag = H_+', A(Ad(f)) - Hp(f))
ok('H_+ A = A H_-  ([H,Xi]=0)', Hp(A(f)) - A(Hm(f)))
k = sp.symbols('kappa', positive=True)
psi = sp.cosh(k*(x - 4*k**2*t)); um = 0; lam = -k**2
P = lambda g, uu: -4*g.diff(x,3) + 6*uu*g.diff(x) + 3*sp.diff(uu,x)*g
ok('L psi = lam psi', -psi.diff(x,2) + um*psi - lam*psi); ok('psi_t = P psi', psi.diff(t) - P(psi, um))
Wd = -psi.diff(x)/psi
ok('Riccati W^2 - W\' = u_- - lam', Wd**2 - Wd.diff(x) - (um - lam))
up = um + 2*Wd.diff(x)
ok('u_+ = -2k^2 cosh^-2', up + 2*k**2/sp.cosh(k*(x-4*k**2*t))**2)
ok('u_+ solves KdV u_t=-u_xxx+6uu_x', up.diff(t) + up.diff(x,3) - 6*up*up.diff(x))
c = sp.symbols('c'); psi0 = x + c; W0 = -psi0.diff(x)/psi0; up0 = 2*W0.diff(x)
ok('lambda=0, vacuum: u_+ = 2/(x+c)^2', up0 - 2/(x+c)**2)
ok('... solves KdV', up0.diff(t) + up0.diff(x,3) - 6*up0*up0.diff(x))
# Baecklund in general: u_- KdV with Lax pair => u_+ KdV; check on the 1-soliton as u_- and lam = -m^2
m = sp.symbols('m', positive=True)
um1 = -2*k**2/sp.cosh(k*(x-4*k**2*t))**2
# eigenfunction: psi = L-eigenfunction with L psi = -m^2 psi and psi_t = P psi: use Darboux of vacuum
# vacuum eigenfunctions: e^{+-m(x-4m^2 t)}; Crum: psi1 = cosh k(...), phi2 = sinh m(...) -> Darboux image W(psi1,phi2)/psi1
ph = sp.sinh(m*(x - 4*m**2*t))
psi2 = sp.simplify(ph.diff(x) + Wd*ph)   # Darboux transform A with W = -psi_x/psi: A ph = ph' + W ph
# exact test: with ea = exp(kappa(x-4kappa^2 t)), eb = exp(m(x-4m^2 t)) as symbols, every function
# here is rational in ea, eb, and d/dx, d/dt act as the derivations ea -> kappa ea, eb -> m eb,
# ea -> -4kappa^3 ea, eb -> -4m^3 eb
ea, eb = sp.symbols('ea eb', positive=True)
Dxe = lambda e: sp.cancel(k*ea*sp.diff(e, ea) + m*eb*sp.diff(e, eb))
Dte = lambda e: sp.cancel(-4*k**3*ea*sp.diff(e, ea) - 4*m**3*eb*sp.diff(e, eb))
che = lambda z: (z + 1/z)/2; she = lambda z: (z - 1/z)/2
Wde = sp.cancel(-Dxe(che(ea))/che(ea)); ume = sp.cancel(-2*k**2/che(ea)**2)
psi2e = sp.cancel(Dxe(she(eb)) + Wde*she(eb))
Pe = lambda g, uu: -4*Dxe(Dxe(Dxe(g))) + 6*uu*Dxe(g) + 3*Dxe(uu)*g
ok('Darboux image is an eigenfunction of L_+ (lam=-m^2)', sp.cancel(-Dxe(Dxe(psi2e)) + ume*psi2e + m**2*psi2e))
ok('... and evolves by P_+', sp.cancel(Dte(psi2e) - Pe(psi2e, ume)))
W2 = -psi2.diff(x)/psi2; up2 = um1 + 2*W2.diff(x)
ok('2-soliton (Crum) solves KdV', sp.simplify(up2.diff(t) + up2.diff(x,3) - 6*up2*up2.diff(x)))

# ---- 3. curvature of an operator: connection (n=2, matrices 2x2), Riemann (n=2,3), linear Frobenius link
X = sp.symbols('x1:4')
def G(name, n, r): return [sp.Matrix(r, r, lambda a, b: sp.Function(f'{name}{i}{a}{b}')(*X[:n])) for i in range(n)]
n, r = 2, 2
Gam = G('g', n, r); uu = sp.Matrix(r, 1, lambda a, b: sp.Function(f'u{a}')(*X[:n]))
q = {(i, j): Gam[i].diff(X[j])*uu - Gam[i]*Gam[j]*uu for i in range(n) for j in range(n)}
F01 = q[(1,0)] - q[(0,1)]   # coefficient of dx_0^dx_1 in sum q_{ij} dx_j ^ dx_i
ok('kappa_nabla = d_iG_j - d_jG_i + [G_i,G_j]', (F01 - (Gam[1].diff(X[0]) - Gam[0].diff(X[1]) + Gam[0]*Gam[1] - Gam[1]*Gam[0])*uu).applyfunc(sp.simplify))
# rank-1 link with Example Frobenius: u_x = al u, u_y = be u  => Gamma = (-al,-be); c = be_x - al_y (times u)
al, be = sp.Function('al')(X[0], X[1]), sp.Function('be')(X[0], X[1]); U = sp.symbols('U')
a_, b_ = al*U, be*U
cF = b_.diff(X[0]) + a_*b_.diff(U) - a_.diff(X[1]) - b_*a_.diff(U)
kap = (-be).diff(X[0]) - (-al).diff(X[1])
ok('kappa_nabla(u) = -c dx^dy for u_x=al u, u_y=be u', kap*U + cF)

for n in (2, 3):
    Xn = X[:n]
    Gm = [[[sp.Function(f'G{m}{min(i,j)}{max(i,j)}')(*Xn) for j in range(n)] for i in range(n)] for m in range(n)]  # Gm[m][i][j]=Gamma^m_ij sym
    fs = sp.symbols(f'f0:{n}')
    T = lambda i, j, l: -sum(Gm[m][i][j].diff(Xn[l]) + sum(Gm[kk][i][j]*Gm[m][kk][l] for kk in range(n)) for m in range(n) for _ in [0]) if False else -sum((Gm[m][i][j].diff(Xn[l]) + sum(Gm[kk][i][j]*Gm[m][kk][l] for kk in range(n)))*fs[m] for m in range(n))
    # R(d_l,d_j)d_i = nabla_l nabla_j d_i - nabla_j nabla_l d_i ; nabla_a d_b = sum Gamma^m_ab d_m
    def Rm(m, i, l, j):
        return (Gm[m][j][i].diff(Xn[l]) - Gm[m][l][i].diff(Xn[j])
                + sum(Gm[kk][j][i]*Gm[m][l][kk] - Gm[kk][l][i]*Gm[m][j][kk] for kk in range(n)))
    allok = all(sp.simplify(T(i,j,l) - T(i,l,j) + sum(Rm(m,i,l,j)*fs[m] for m in range(n))) == 0
                for i in range(n) for j in range(n) for l in range(n))
    print(f'Riemann: T_ij,l - T_il,j = -sum R^m_ilj f_m  (n={n}):', allok)
    # rank of Coker: n(n^2-1)/3 = dim(Omega x S^2) - dim S^3
    print(f'  rank check n={n}:', n*n*(n+1)//2 - n*(n+1)*(n+2)//6 == sp.Rational(n*(n*n-1),3))

# Riemann: check also the preimage claim f_ij = sum Gamma^k_ij f_k for R_2 of H(f) = nabla d f
# (H(f)_ij = f_ij - sum Gamma^k_ij f_k : nabla_{d_i} df evaluated on d_j)  -> trivially by definition

# ---- 4. curvature of a derivation: Theorem, item 3), n=3 generic
n = 3; Xn = X[:n]
Mm = sp.Matrix(n, n, lambda i, k: sp.Symbol(f'M{i}{k}'))
Qd = [sum(sp.Symbol(f'q{i}{a}{b}')*Xn[a]*Xn[b] for a in range(n) for b in range(a, n)) for i in range(n)]
aa = [sum(Mm[i, k]*Xn[k] for k in range(n)) + Qd[i] for i in range(n)]
# kappa(alpha) = sum_{i<j} (d_i b_j - d_j b_i) abar_i ^ abar_j, abar_i in m/m^2 = linear parts
lin = lambda f: [sp.diff(f, Xn[k]).subs({X_: 0 for X_ in Xn}) for k in range(n)]  # class mod m^2 in basis xbar_k
def wedge(v, w):  # in basis xbar_a ^ xbar_b, a<b
    return {(a, b): v[a]*w[b] - v[b]*w[a] for a in range(n) for b in range(a+1, n)}
def kappa(bvec):
    tot = {(a, b): 0 for a in range(n) for b in range(a+1, n)}
    for i in range(n):
        for j in range(i+1, n):
            cf = sp.diff(bvec[j], Xn[i]) - sp.diff(bvec[i], Xn[j])
            # cf mod I: I = m, so take value at 0
            cf0 = cf.subs({X_: 0 for X_ in Xn})
            wv = wedge(lin(aa[i]), lin(aa[j]))
            for kk in tot: tot[kk] += cf0*wv[kk]
    return tot
good = True
for i in range(n):
    for j in range(i+1, n):
        bvec = [0]*n; bvec[j] += aa[i]; bvec[i] -= aa[j]   # eps_ij = a_i dx_j - a_j dx_i
        lhs = kappa(bvec)
        Tai = [sum(Mm[i, k]*lin(aa[k])[c] for k in range(n)) for c in range(n)]
        Taj = [sum(Mm[j, k]*lin(aa[k])[c] for k in range(n)) for c in range(n)]
        rhs1 = wedge(Tai, lin(aa[j])); rhs2 = wedge(lin(aa[i]), Taj)
        good &= all(sp.expand(lhs[kk] - rhs1[kk] - rhs2[kk]) == 0 for kk in lhs)
print('curvature of a derivation, item 3) (n=3, generic quadratic field):', good)
# item 4): matrix of the derivation-extension of T on Lambda^2 has eigenvalues lam_i+lam_j
Mn = sp.Matrix([[1, 2, 0], [0, -3, 1], [4, 0, 5]])
ext = sp.zeros(3, 3); pairs = [(0,1), (0,2), (1,2)]
for c, (i, j) in enumerate(pairs):
    for k in range(3):
        for (ii, jj, s) in ((k, j, Mn[i, k]), (i, k, Mn[j, k])):
            if ii == jj: continue
            key, sg = ((ii, jj), 1) if ii < jj else ((jj, ii), -1)
            ext[pairs.index(key), c] += sg*s
ev = Mn.eigenvals(); lam = [l for l, mlt in ev.items() for _ in range(mlt)]
print('eigenvalues of Lambda^2 T = lam_i+lam_j:', sp.simplify(ext.charpoly().as_expr() - sp.prod([sp.Symbol('lambda') - (lam[a]+lam[b]) for a, b in pairs])) == 0)
