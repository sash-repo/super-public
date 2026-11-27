"""An explicit odd metric g on R^{2|2} and the contractions of its curvature tensor (Section 5.A).
1) Ricci (str-contraction, convention B): anti-symmetric; S.
2) Candidate 'second Ricci' contractions T(b,d) = sum_{a,c} sign * Gi[.][.] * <R(d_a,d_b)d_c, d_d>;
   keep the symmetric nonzero ones.
3) Tractor L_g = Rp+TM+Rq (p even, q odd, <p,q>=1): signs s1,s2 of (4.1) from metric compatibility.
4) Solve for rho^0 = c1*T + c2*(str T)g making psi of (4.2) intertwine L_{g'} -> L_g, over psi sign variants."""
import sys, itertools, pickle, sympy as sp
from stractor import Alg
from oddgeom import geometry_odd, pairing, r_of, str_op

x, y = sp.symbols('x y')
A = Alg([x, y], 2); N = A.N; P = A.par

def metric():
    g = [[{} for _ in range(4)] for _ in range(4)]
    g[0][2] = g[2][0] = {(): 1 + x**2, (0, 1): y}
    g[1][3] = g[3][1] = {(): 1 + x*y}
    g[0][3] = g[3][0] = {(): x}
    g[0][0] = {(0,): x*y}
    g[0][1] = g[1][0] = {(1,): 1 + x}
    g[2][3] = {(0,): x}; g[3][2] = A.neg(g[2][3])
    return g

PT = {x: sp.Rational(3, 7), y: sp.Rational(-5, 11)}
def ev(f): return A.clean({k: sp.sympify(v).subs(PT) for k, v in f.items()})
def evM(M): return [[ev(e) for e in r] for r in M]

def rfull(geo, g):
    R = geo['R']; out = {}
    for a, b, c, d in itertools.product(range(N), repeat=4):
        t = {}
        for e in range(N):
            f = R[a, b, c, e]
            if f and g[e][d]: t = A.add(t, A.mul(A.sgn(1, ev(f)), ev(g[e][d])))
        out[a, b, c, d] = t
    return out

def contractions(geo, g):
    Rf = rfull(geo, g); Gi = evM(geo['Gi']); res = {}
    for tr, xs, ys, zs, ws in itertools.product((0, 1), repeat=5):
        T = [[{} for _ in range(N)] for _ in range(N)]
        for b, d in itertools.product(range(N), repeat=2):
            acc = {}
            for a, c in itertools.product(range(N), repeat=2):
                h = Gi[c][a] if tr else Gi[a][c]
                if not h or not Rf[a, b, c, d]: continue
                s = (-1)**(xs*P(a) + ys*P(c) + zs*P(a)*P(b) + ws*P(c)*P(b))
                acc = A.add(acc, A.sc(s, A.mul(h, Rf[a, b, c, d])))
            T[b][d] = acc
        res[tr, xs, ys, zs, ws] = T
    return res

def is_sym(T, sgn=1):
    return all(A.add(T[b][d], A.sc(-sgn*(-1)**(P(b)*P(d)), T[d][b])) == {} for b in range(N) for d in range(N))

if __name__ == "__main__":
    g = metric()
    lam = {(): 1 + x**2 + y, (0, 1): 1 + x}
    gp = [[A.mul(A.mul(lam, lam), g[i][j]) for j in range(N)] for i in range(N)]
    geo = geometry_odd(A, g)
    print("geometry done", flush=True)
    Ric = geo['Ric']
    print("Ric anti-symmetric:", is_sym(Ric, -1), " symmetric:", is_sym(Ric, 1), flush=True)
    C = contractions(geo, g)
    good = []
    for k, T in C.items():
        nz = sum(1 for b in range(N) for d in range(N) if T[b][d])
        print(k, "sym", is_sym(T, 1), "anti", is_sym(T, -1), "nonzero", nz, flush=True)
        if is_sym(T, 1) and nz: good.append(k)
    print("symmetric nonzero contractions:", good, flush=True)
