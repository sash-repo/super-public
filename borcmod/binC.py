"""Over Q: which supertransvectants J_k(a,b) (block k, c=a+b+k, the osp(1|2)-invariant) are invariant under
K_{t^2 theta} (hence under k(1|1)), as polynomial conditions on a,b.
Usage: python3 binC.py kmax"""
import sympy as sp, sys
from binsym import Dn, mul, add, sc, t, a, b
def L2(f, lam): # L^lam_{K_{t^2 theta}} = t^2 K_theta + 2 lam t theta ; K_theta(u0+u1 th) = -u1 + u0' th
    return (sp.expand(-t**2*f[1]), sp.expand(t**2*sp.diff(f[0], t) + 2*lam*t*f[0]))
def J(k):
    # the closed form of Corollary 7.6 (coefficients c_ij of J_k)
    cs = {}
    if k % 2 == 0:
        m = k//2
        for i in range(m+1):
            j = m-i; cs[(2*i, 2*j)] = (-1)**j*sp.binomial(m, i)*sp.rf(a+i, j)*sp.rf(b+j, i)
        for i in range(m):
            j = m-1-i; cs[(2*i+1, 2*j+1)] = (-1)**j*m*sp.binomial(m-1, i)*sp.rf(a+i+1, j)*sp.rf(b+j+1, i)
    else:
        m = k//2
        for i in range(m+1):
            j = m-i
            cs[(2*i+1, 2*j)] = (-1)**j*sp.binomial(m, i)*sp.rf(a+i+1, j)*sp.rf(b+j, i+1)
            cs[(2*i, 2*j+1)] = (-1)**(j+1)*sp.binomial(m, i)*sp.rf(a+i, j+1)*sp.rf(b+j+1, i)
    return cs
def apply(cs, phi, pphi, psi):
    out = (0, 0)
    for (i, j), v in cs.items():
        out = add(out, sc(v*(-1)**(j*pphi), mul(Dn(phi, i), Dn(psi, j))))
    return out
def conditions(k, gen=L2):
    cs = J(k); c = a+b+k; pi = k % 2; eqs = set()
    R = k+3
    for pphi in (0, 1):
        for ppsi in (0, 1):
            for r in range(R):
                for s in range(R):
                    phi = (t**r, 0) if pphi == 0 else (0, t**r)
                    psi = (t**s, 0) if ppsi == 0 else (0, t**s)
                    lhs = gen(apply(cs, phi, pphi, psi), c)
                    rhs = add(apply(cs, gen(phi, a), 1-pphi, psi), sc((-1)**pphi, apply(cs, phi, pphi, gen(psi, b))))
                    d = add(lhs, sc(-(-1)**pi, rhs))
                    for comp in d:
                        if comp != 0:
                            for co in sp.Poly(comp, t).coeffs(): eqs.add(sp.factor(co))
    return eqs
if __name__ == "__main__":
    # sanity: J_k is osp-invariant (K_{t theta})
    from binsym import L as L1
    for k in range(int(sys.argv[1])+1):
        assert not conditions(k, gen=L1), k
        eqs = conditions(k)
        g = sp.gcd_list(list(eqs)) if eqs else 0
        print("k =", k, ": K_{t^2 theta}-invariant iff", sp.factor(g) if eqs else "always", "== 0", flush=True)
