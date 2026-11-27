"""The geometry of an odd metric (Section 5.A): Levi-Civita connection, curvature, Ricci tensor.
Odd metric g on R^{n|n} (prefix convention: g(fu,v)=(-1)^{p(f)} f g(u,v),
g(u,fv)=(-1)^{p(f)(1+p(u))} f g(u,v); symmetric: g(u,v)=(-1)^{p(u)p(v)} g(v,u)).
L_g = Rp + TM + Rq, p even, q odd, odd form <p,q>=1, <X,Y>=g(X,Y).
Tests: 1) Levi-Civita (super Koszul with K_abc=(-1)^{p(a)} d_a g_bc), torsion-free + metric;
2) which signs in (4.1) make the tractor connection metric; 3) for which kappa in
rho^0 = -1/2 rho + kappa S g the map psi of (4.2) intertwines L_{lam^2 g} -> L_g;
4) with q -> q + c p (c odd) the gauge freedom, |grad lam|^2 = 0.  Exact, sympy."""
import sys, itertools, sympy as sp
from stractor import Alg

def pairing(A, Bpar, B, par, u, v):
    """<u,v> for sections u,v (dicts basis->function), odd form (prefix), basis parities par, Gram B."""
    tot = {}
    for i, f in u.items():
        if not f: continue
        for j, h in v.items():
            if not h or not B.get((i, j)): continue
            t = A.mul(A.sgn(Bpar, f), A.mul(A.sgn(Bpar + par[i], h), B[(i, j)]))
            tot = A.add(tot, t)
    return tot

def geometry_odd(A, g):
    N = A.N; P = A.par
    Gi = A.matinv(g)
    # check right inverse
    I = A.matmul(g, Gi)
    assert all(A.add(I[i][j], A.const(-1 if i == j else 0)) == {} for i in range(N) for j in range(N))
    K = {(a, b, c): A.sc((-1)**P(a), A.d(a, g[b][c])) for a, b, c in itertools.product(range(N), repeat=3)}
    Gam1 = {}
    for a, b, c in itertools.product(range(N), repeat=3):
        Gam1[a, b, c] = A.sc(sp.Rational(1, 2), A.add(K[a, b, c], A.sc((-1)**(P(a)*P(b)), K[b, a, c]),
                                                       A.sc(-(-1)**(P(c)*(P(a)+P(b))), K[c, a, b])))
    Gam = {}
    for a, b in itertools.product(range(N), repeat=2):
        v = A.matmul([[Gam1[a, b, c] for c in range(N)]], Gi)[0]
        for d in range(N): Gam[a, b, d] = A.sc((-1)**(P(a)+P(b)+P(d)), v[d])
    for a, b, e in itertools.product(range(N), repeat=3):
        assert A.add(Gam[a, b, e], A.sc(-(-1)**(P(a)*P(b)), Gam[b, a, e])) == {}, "torsion"
    def nabla(a, vec):
        out = [A.d(a, vec[e]) for e in range(N)]
        for d in range(N):
            f = A.sgn(P(a), vec[d])
            for e in range(N): out[e] = A.add(out[e], A.mul(f, Gam[a, d, e]))
        return out
    # metric check with the odd-form pairing
    B = {(i, j): g[i][j] for i in range(N) for j in range(N)}
    par = {i: P(i) for i in range(N)}
    for a, b, c in itertools.product(range(N), repeat=3):
        u = {d: Gam[a, b, d] for d in range(N)}; v = {d: Gam[a, c, d] for d in range(N)}
        rhs = A.add(A.sc((-1)**P(a), pairing(A, 1, B, par, u, {c: A.const(1)})),
                    A.sc((-1)**(P(a)*(1+P(b))), pairing(A, 1, B, par, {b: A.const(1)}, v)))
        assert A.add(A.d(a, g[b][c]), A.neg(rhs)) == {}, ("metric", a, b, c)
    R = {}
    for a, b, c in itertools.product(range(N), repeat=3):
        v1 = nabla(a, [Gam[b, c, e] for e in range(N)]); v2 = nabla(b, [Gam[a, c, e] for e in range(N)])
        s = (-1)**(P(a)*P(b))
        for e in range(N): R[a, b, c, e] = A.add(v1[e], A.sc(-s, v2[e]))
    rs = lambda pa, pb, pc: (-1)**(pa*(1+pb+pc))       # convention B (validated in stractor)
    Ric = [[A.add(*[A.sc(rs(P(a), P(b), P(c)), R[a, b, c, a]) for a in range(N)]) for c in range(N)] for b in range(N)]
    return dict(Gi=Gi, Gam=Gam, Ric=Ric, R=R, nabla=nabla, B=B, par=par)

def r_of(A, form, Gi):
    """operator r (odd if form even) with form(X,Y)=g(rX,Y): r[a][c] = coefficient of d_c in r(d_a)."""
    N = A.N; P = A.par
    out = []
    for a in range(N):
        w = A.matmul([form[a]], Gi)[0]
        # form_ab = sum_c (-1)^{p(r_a^c)} r_a^c g_cb, p(r_a^c) = p(a)+p(c)+1 for even form
        out.append([A.sc((-1)**(P(a)+P(c)+1), w[c]) for c in range(N)])
    return out

def str_op(A, r):
    # str of the operator d_a -> sum_c r[a][c] d_c ; (-1)^{p(a)} r_a^a
    return A.add(*[A.sc((-1)**A.par(a), r[a][a]) for a in range(A.N)])

def run(A, g, lam, c_gauge=None):
    N = A.N; P = A.par
    geo = geometry_odd(A, g)
    convs = {'A': lambda pa, pb, pc: (-1)**pa, 'B': lambda pa, pb, pc: (-1)**(pa*(1+pb+pc)),
             'C': lambda pa, pb, pc: (-1)**(pa*(pb+pc)), 'D': lambda pa, pb, pc: 1}
    for nm, rs in convs.items():
        Ric = [[A.add(*[A.sc(rs(P(a), P(b), P(c)), geo['R'][a, b, c, a]) for a in range(N)]) for c in range(N)] for b in range(N)]
        for ss in (1, -1):
            ok = all(A.add(Ric[b][c], A.sc(-ss*(-1)**(P(b)*P(c)), Ric[c][b])) == {} for b in range(N) for c in range(N))
            nz = sum(1 for b in range(N) for c in range(N) if Ric[b][c])
            print(f"  conv {nm}, symmetry sign {ss}: {ok}  (nonzero entries {nz})", flush=True)
    sym = all(A.add(geo['Ric'][b][c], A.sc(-(-1)**(P(b)*P(c)), geo['Ric'][c][b])) == {} for b in range(N) for c in range(N))
    r = r_of(A, geo['Ric'], geo['Gi']); S = str_op(A, r)
    rg = r_of(A, g, geo['Gi'])  # this is for an even form; g is odd: just a sanity print
    print(f"  LC ok; Ric symmetric: {sym}; S = {S}", flush=True)
    return geo, S

if __name__ == "__main__":
    x, y = sp.symbols('x y')
    A = Alg([x, y], 2)          # sdim (2|2), odd coords xi=2, eta=3 -> generators 0,1
    # odd metric: mixed blocks even, even-even & odd-odd blocks odd functions
    g = [[{} for _ in range(4)] for _ in range(4)]
    g[0][2] = g[2][0] = {(): 1 + x**2, (0, 1): y}
    g[1][3] = g[3][1] = {(): 1 + x*y}
    g[0][3] = g[3][0] = {(): x}
    g[0][0] = {(0,): x*y}
    g[0][1] = g[1][0] = {(1,): 1 + x}
    g[2][3] = {(0,): x}; g[3][2] = A.neg(g[2][3])
    run(A, g, None)
