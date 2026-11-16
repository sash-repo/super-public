"""Symbolic block equations for B = sum_{i+j=k} c_ij B_ij, B_ij(phi,psi)=(-1)^{j p(phi)} D^i phi D^j psi,
invariance under X = K_{t theta}: L^c_X = t K_theta + c theta (osp1 convention, K_theta = theta d_t - d_theta).
Functions are pairs (u0,u1) = u0 + u1 theta, u_i in Q[a,b,c][t]."""
import sympy as sp
t, a, b, c = sp.symbols('t a b c')
def D(f): return (sp.diff(f[1]*0, t) + f[1], sp.diff(f[0], t))   # D(u0+u1 th) = u1 + u0' th
def Dn(f, n):
    for _ in range(n): f = D(f)
    return f
def L(f, lam): return (sp.expand(-t*f[1]), sp.expand(t*sp.diff(f[0], t) + lam*f[0]))
def mul(f, g): return (sp.expand(f[0]*g[0]), sp.expand(f[0]*g[1] + f[1]*g[0]))
def add(*fs):
    return (sp.expand(sum(f[0] for f in fs)), sp.expand(sum(f[1] for f in fs)))
def sc(s, f): return (s*f[0], s*f[1])

def Bij(i, j, phi, pphi, psi):
    return sc((-1)**(j*pphi), mul(Dn(phi, i), Dn(psi, j)))

def equations(k, R=None):
    R = R or k + 3
    cs = {(i, k-i): sp.Symbol(f'c{i}_{k-i}') for i in range(k+1)}
    pi = k % 2
    eqs = []
    for pphi in (0, 1):
        for ppsi in (0, 1):
            for r in range(R):
                for s in range(R):
                    phi = (t**r, 0) if pphi == 0 else (0, t**r)
                    psi = (t**s, 0) if ppsi == 0 else (0, t**s)
                    lhs = (0, 0); rhs = (0, 0)
                    for (i, j), cij in cs.items():
                        lhs = add(lhs, sc(cij, L(Bij(i, j, phi, pphi, psi), c)))
                        rhs = add(rhs, sc(cij, Bij(i, j, L(phi, a), 1-pphi, psi)),
                                  sc(cij*(-1)**pphi, Bij(i, j, phi, pphi, L(psi, b))))
                    d = add(lhs, sc(-(-1)**pi, rhs))
                    for comp in d:
                        for co in sp.Poly(comp, t).coeffs() if comp != 0 else []:
                            eqs.append(sp.expand(co))
    return cs, [e for e in set(eqs) if e != 0]

if __name__ == "__main__":
    import sys
    for k in range(int(sys.argv[1]) + 1):
        cs, eqs = equations(k)
        sol = sp.solve(eqs, list(cs.values()) , dict=True)
        print("k =", k, "#eqs", len(eqs))
        for e in sorted(set(sp.factor(e) for e in eqs), key=str)[:12]: print("   ", e)
