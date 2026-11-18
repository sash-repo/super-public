"""Vogel-type operators for algebras with no quadratic Casimir.

1) pe(n) = p(n) in gl(n|n): Serganova's (BDE+, arXiv:1610.08470) fake Casimir
   Omega = 2 sum X_i (x) X^i,  X_i basis of g=p(n), X^i dual basis of g'=p(n)^perp
   w.r.t. str(XY).  Acts on M (x) N, N a gl(n|n)-module.
   We take M = g (adjoint), N = V (tautological) and N = gl(n|n) (adjoint);
   on g (x) gl(n|n) = g(x)g + g(x)g', Omega swaps the two halves, Omega^2 preserves g(x)g.
2) psq(n) with its ODD invariant form otr(XY): the mixed operator
   D = sum (+-) ad x_i (x) ad x^i on g (x) g; D is odd; D^2 even.

Everything exact (Fraction); char. polys via python-flint.
"""
import sys, itertools
from fractions import Fraction as F
import flint

def zeros(N): return [[F(0)]*N for _ in range(N)]

class GL:
    def __init__(s, n):
        s.n = n; s.N = 2*n
    def par(s, i): return 0 if i < s.n else 1
    def E(s, i, j):
        M = zeros(s.N); M[i][j] = F(1); return M

def mul(A, B):
    N = len(A)
    return [[sum(A[i][k]*B[k][j] for k in range(N) if A[i][k] and B[k][j]) for j in range(N)] for i in range(N)]
def add(A, B, c=1): return [[A[i][j]+c*B[i][j] for j in range(len(A))] for i in range(len(A))]
def scal(c, A): return [[c*x for x in r] for r in A]

def mpar(A, n):
    ev = od = False
    for i in range(2*n):
        for j in range(2*n):
            if A[i][j]:
                if (i < n) == (j < n): ev = True
                else: od = True
    assert not (ev and od), "inhomogeneous"
    return 1 if od else 0

def sbr(A, B, n):
    pa, pb = mpar(A, n), mpar(B, n)
    return add(mul(A, B), mul(B, A), -(-1)**(pa*pb))

def strr(A, n):
    return sum(A[i][i] for i in range(n)) - sum(A[i][i] for i in range(n, 2*n))

def flat(A): return [x for r in A for x in r]

_coord_cache = {}
def _fq(x): return flint.fmpq(x.numerator, x.denominator)
def _coord_setup(basis):
    key = id(basis)
    if key in _coord_cache: return _coord_cache[key]
    rows = [flat(b) for b in basis]
    k, L = len(rows), len(rows[0])
    # greedy pivot columns giving an invertible k x k minor
    piv, cur = [], []
    import itertools
    for c in range(L):
        trial = piv + [c]
        Mt = flint.fmpq_mat(len(trial), k, [_fq(rows[j][cc]) for cc in trial for j in range(k)])
        if Mt.rank() == len(trial):
            piv = trial
            if len(piv) == k: break
    assert len(piv) == k
    M = flint.fmpq_mat(k, k, [_fq(rows[j][c]) for c in piv for j in range(k)])
    _coord_cache[key] = (piv, M.inv(), basis)
    return _coord_cache[key]

def solve_coords(basis, X):
    piv, Inv, _ = _coord_setup(basis)
    fx = flat(X)
    v = flint.fmpq_mat(len(piv), 1, [_fq(fx[c]) for c in piv])
    sol = Inv * v
    c = [F(int(sol[i, 0].p), int(sol[i, 0].q)) for i in range(len(piv))]
    Y = [[F(0)]*len(X) for _ in X]
    for ci, b in zip(c, basis):
        if ci: Y = add(Y, b, ci)
    assert Y == X, "not in span"
    return c

def matrix_of(op, basis):
    """op: matrix->matrix; returns list of columns coordinates -> matrix rows"""
    cols = [solve_coords(basis, op(b)) for b in basis]
    k = len(basis)
    return [[cols[j][i] for j in range(k)] for i in range(k)]

def charpoly_factored(M):
    if not M: return "1"
    k = len(M)
    den = 1
    for r in M:
        for x in r: den = den*x.denominator//__import__('math').gcd(den, x.denominator)
    Z = flint.fmpz_mat(k, k, [int(x*den) for r in M for x in r])
    cp = Z.charpoly()
    # roots of charpoly(den*M) are den*eigs
    c, fac = cp.factor()
    out = []
    for f, e in fac:
        if f.degree() == 1:
            a, b = f.coeffs()  # f = b x + a
            out.append(f"[{F(-int(a), int(b)*den)}]^{e}")
        else:
            out.append(f"({f}; scale {den})^{e}")
    return " ".join(out)

def tensor_op(A, B, pa_list, pb_list, opar_A):
    """operator A(x)B on tensor product with sign (-1)^{p(B-op) p(left vector)}:
       (A(x)B)(m(x)v) = (-1)^{p(B) p(m)} A m (x) B v.  opar_A unused (parity of B passed)"""
    raise NotImplementedError

# ---------------------------------------------------------------- p(n)
def pe_bases(n):
    G = GL(n); N = 2*n
    g, gp = [], []
    for i in range(n):
        for j in range(n):
            X = G.E(i, j); X[n+j][n+i] -= 1      # A - A^t-block: [[E_ij,0],[0,-E_ji]]
            g.append(X)
            Y = G.E(i, j); Y[n+j][n+i] += 1      # [[E_ij,0],[0,E_ji]]
            gp.append(Y)
    for i in range(n):
        for j in range(i, n):
            X = G.E(i, n+j)
            if i != j: X[j][n+i] += 1           # B symmetric
            g.append(X)
            if i != j:
                Y = G.E(i, n+j); Y[j][n+i] -= 1  # B antisymmetric
                gp.append(Y)
    for i in range(n):
        for j in range(i, n):
            if i != j:
                X = G.E(n+i, j); X[n+j][i] -= 1  # C antisymmetric
                g.append(X)
            Y = G.E(n+i, j)
            if i != j: Y[n+j][i] += 1            # C symmetric
            gp.append(Y)
    return g, gp

def check_subalg(basis, n):
    for a in basis:
        for b in basis:
            solve_coords(basis, sbr(a, b, n))

def dual_basis(g, gp, n):
    """X^i in span(gp) with str(X^i X_j) = delta_ij"""
    k = len(g)
    P = [[strr(mul(gp[a], g[j]), n) for a in range(k)] for j in range(k)]  # P[j][a]
    Pm = flint.fmpq_mat(k, k, [_fq(P[j][a]) for j in range(k) for a in range(k)])
    Inv = Pm.inv()
    dual = []
    for i in range(k):
        # sum_a c_a P[j][a] = delta_ij  -> c = Inv row? solve P c = e_i
        c = [F(int(Inv[a, i].p), int(Inv[a, i].q)) for a in range(k)]
        Xi = zeros(2*n)
        for ca, b in zip(c, gp):
            if ca: Xi = add(Xi, b, ca)
        dual.append(Xi)
    for i in range(k):
        for j in range(k):
            assert strr(mul(dual[i], g[j]), n) == (1 if i == j else 0)
    return dual

def kron_signed(Aops, Bops, lpar, sgnpar):
    pass

def build_tensor(MA, MB, parA_vecs, p_op_B):
    """matrix of v(x)w -> (-1)^{p_op_B * p(v)} (MA v)(x)(MB w); basis index (a,b)->a*dimB+b"""
    da, db = len(MA), len(MB)
    T = [[F(0)]*(da*db) for _ in range(da*db)]
    for a in range(da):
        for b in range(db):
            col = a*db + b
            s = -1 if (p_op_B and parA_vecs[a]) else 1
            for a2 in range(da):
                if MA[a2][a] == 0: continue
                for b2 in range(db):
                    if MB[b2][b] == 0: continue
                    T[a2*db+b2][col] += s*MA[a2][a]*MB[b2][b]
    return T

def madd(A, B, c=1): return [[A[i][j]+c*B[i][j] for j in range(len(A))] for i in range(len(A))]
def mmul(A, B):
    k = len(A); m = len(B[0])
    Bt = list(zip(*B))
    return [[sum(x*y for x, y in zip(A[i], Bt[j]) if x and y) for j in range(m)] for i in range(k)]

def pe_run(n):
    g, gp = pe_bases(n)
    check_subalg(g, n)
    G = g + gp
    parg = [mpar(x, n) for x in g]
    parG = [mpar(x, n) for x in G]
    dual = dual_basis(g, gp, n)
    adg = [matrix_of(lambda Y, X=X: sbr(X, Y, n), g) for X in g]           # ad on g
    adG_dual = [matrix_of(lambda Y, X=X: sbr(X, Y, n), G) for X in dual]   # ad of X^i on G
    adG_g = [matrix_of(lambda Y, X=X: sbr(X, Y, n), G) for X in g]         # ad of X_i on G
    Vrep_dual = [[[x for x in r] for r in X] for X in dual]                 # natural rep
    Vrep_g = g
    parV = [0]*n + [1]*n
    res = {}
    for name, N_dual, N_g, parN in (("V", Vrep_dual, Vrep_g, parV), ("gl", adG_dual, adG_g, parG)):
        dM, dN = len(g), len(N_dual[0])
        Om = [[F(0)]*(dM*dN) for _ in range(dM*dN)]
        for i in range(len(g)):
            Om = madd(Om, build_tensor(adg[i], N_dual[i], parg, parg[i]), 2)
        # invariance check: [Delta(y), Om] = 0 for y in g
        for i in range(len(g)):
            Dy = madd(build_tensor(adg[i], [[F(int(a == b)) for b in range(dN)] for a in range(dN)], parg, 0),
                      build_tensor([[F(int(a == b)) for b in range(dM)] for a in range(dM)], N_g[i], parg, parg[i]))
            assert mmul(Dy, Om) == mmul(Om, Dy), f"Omega not invariant on g(x){name}"
        # split by parity of the tensor vector
        tpar = [(parg[a] + parN[b]) % 2 for a in range(dM) for b in range(dN)]
        out = []
        for P in (0, 1):
            idx = [k for k in range(dM*dN) if tpar[k] == P]
            sub = [[Om[r][c] for c in idx] for r in idx]
            out.append(charpoly_factored(sub))
        res[name] = out
        print(f"p({n})  Omega on g(x){name}:  even part: {out[0]}\n            odd part: {out[1]}", flush=True)
        if name == "gl":
            # Omega^2 on g(x)g
            dg = len(g)
            idx = [a*dN + b for a in range(dM) for b in range(dg)]
            O2 = mmul(Om, Om)
            for P in (0, 1):
                ii = [k for k in idx if tpar[k] == P]
                sub = [[O2[r][c] for c in ii] for r in ii]
                print(f"p({n})  Omega^2 on g(x)g, parity {P}: {charpoly_factored(sub)}", flush=True)
            # Omega^2 on S^2(g): symmetrizer
            # S^2 = image of v(x)w + (-1)^{p v p w} w(x)v ; compute Omega^2 on g(x)g then restrict by projector
            Sproj = [[F(0)]*(dg*dg) for _ in range(dg*dg)]
            for a in range(dg):
                for b in range(dg):
                    Sproj[a*dg+b][a*dg+b] += F(1, 2)
                    Sproj[b*dg+a][a*dg+b] += F(1, 2)*(-1)**(parg[a]*parg[b])
            O2gg = [[O2[a1*dN+b1][a2*dN+b2] for a2 in range(dg) for b2 in range(dg)] for a1 in range(dg) for b1 in range(dg)]
            commute = mmul(O2gg, Sproj) == mmul(Sproj, O2gg)
            print(f"p({n})  Omega^2|g(x)g commutes with the flip: {commute}", flush=True)
    return res

if __name__ == "__main__":
    for n in [int(a) for a in sys.argv[1:]] or [2, 3]:
        pe_run(n)
