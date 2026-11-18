"""Fast driver (numpy over Z after scaling, flint for exact char. polys).

  pe n   : Serganova's fake Casimir Omega on p(n)(x)V and p(n)(x)gl(n|n)
  psq n  : the odd mixed Casimir D of psq(n) (odd form otr) on g(x)g, S^2 g, E^2 g, g
"""
import sys, math
from fractions import Fraction as F
import numpy as np
import flint
from fakecas import (GL, zeros, mul, add, mpar, sbr, strr, flat, solve_coords,
                     matrix_of, pe_bases, check_subalg, dual_basis, _fq)

def to_int(Ms):
    """list of Fraction matrices -> (common denominator, int numpy arrays)"""
    den = 1
    for M in Ms:
        for r in M:
            for x in r: den = den*x.denominator//math.gcd(den, x.denominator)
    return den, [np.array([[int(x*den) for x in r] for r in M], dtype=object) for M in Ms]

def kron_signed(A, B, parL, pB):
    """(A(x)B)(v(x)w) = (-1)^{pB p(v)} Av(x)Bw ; A acts on left factor"""
    K = np.kron(A, B)
    if pB:
        db = B.shape[0]
        sign = np.repeat(np.array([-1 if p else 1 for p in parL], dtype=object), db)
        K = K * sign[np.newaxis, :]
    return K

def cp(M, den=1):
    """factored char poly of integer matrix M/den"""
    k = M.shape[0]
    if k == 0: return "1"
    Z = flint.fmpz_mat(k, k, [int(x) for x in M.flatten()])
    c, fac = Z.charpoly().factor()
    out = []
    for f, e in fac:
        if f.degree() == 1:
            a, b = [int(t) for t in f.coeffs()]
            out.append(f"[{F(-a, b*den)}]^{e}")
        else:
            co = [int(t) for t in f.coeffs()]
            # rescale x -> den*x
            out.append(f"({'+'.join(f'{F(c_, 1)}*{den}^{i}' for i, c_ in enumerate(co))} in den*x)^{e}")
    return " ".join(out)

def cp_pretty(M, den=1):
    """char poly of M/den with x rescaled: returns factors in x"""
    k = M.shape[0]
    if k == 0: return "1"
    Z = flint.fmpz_mat(k, k, [int(x) for x in M.flatten()])
    c, fac = Z.charpoly().factor()
    out = []
    for f, e in fac:
        co = [int(t) for t in f.coeffs()]
        d = len(co) - 1
        # poly in y=den*x : sum co_i y^i = sum co_i den^i x^i ; make monic in x
        cx = [F(co[i]*den**i) for i in range(d+1)]
        lead = cx[-1]; cx = [c_/lead for c_ in cx]
        if d == 1:
            out.append(f"[{-cx[0]}]^{e}")
        else:
            out.append("(" + " + ".join(f"{c_}x^{i}" for i, c_ in enumerate(cx) if c_) + f")^{e}")
    return " ".join(out)

def restrict(M, S):
    """matrix of M on the column span of S (assumed invariant), exact"""
    k = S.shape[1]
    MS = M.dot(S)
    # pick pivot rows of S
    Sf = flint.fmpq_mat(S.shape[0], k, [int(x) for x in S.flatten()])
    piv = []
    for r in range(S.shape[0]):
        t = piv + [r]
        Mt = flint.fmpq_mat(len(t), k, [int(S[i, j]) for i in t for j in range(k)])
        if Mt.rank() == len(t):
            piv = t
            if len(piv) == k: break
    A = flint.fmpq_mat(k, k, [int(S[i, j]) for i in piv for j in range(k)])
    B = flint.fmpq_mat(k, k, [int(MS[i, j]) for i in piv for j in range(k)])
    R = A.solve(B)
    # verify invariance
    Rq = [[F(int(R[i, j].p), int(R[i, j].q)) for j in range(k)] for i in range(k)]
    chk = S.dot(np.array([[x for x in r] for r in Rq], dtype=object))
    assert (chk == MS).all(), "subspace not invariant"
    den = 1
    for r in Rq:
        for x in r: den = den*x.denominator//math.gcd(den, x.denominator)
    return den, np.array([[int(x*den) for x in r] for r in Rq], dtype=object)

def sym_bases(par):
    d = len(par)
    S, A = [], []
    for a in range(d):
        for b in range(a, d):
            s = (-1)**(par[a]*par[b])
            v = np.zeros(d*d, dtype=object); v[a*d+b] += 1; v[b*d+a] += s
            if any(v): S.append(v)
            w = np.zeros(d*d, dtype=object); w[a*d+b] += 1; w[b*d+a] -= s
            if any(w): A.append(w)
    return np.array(S, dtype=object).T, np.array(A, dtype=object).T

# ------------------------------------------------------------ pe
def pe(n):
    g, gp = pe_bases(n); check_subalg(g, n)
    G = g + gp
    parg = [mpar(x, n) for x in g]; parG = [mpar(x, n) for x in G]
    dual = dual_basis(g, gp, n)
    adg = [matrix_of(lambda Y, X=X: sbr(X, Y, n), g) for X in g]
    adGd = [] if __import__("os").environ.get("VONLY") else [matrix_of(lambda Y, X=X: sbr(X, Y, n), G) for X in dual]
    adGg = [] if __import__("os").environ.get("VONLY") else [matrix_of(lambda Y, X=X: sbr(X, Y, n), G) for X in g]
    print(f"p({n}): sdim g = ({parg.count(0)}|{parg.count(1)})", flush=True)
    import os
    cases = (("V", dual, g, [0]*n+[1]*n), ("gl(n|n)", adGd, adGg, parG))
    if os.environ.get("VONLY"): cases = cases[:1]
    for name, Nd, Ng, parN in cases:
        den, ints = to_int(adg + Nd + Ng)
        k = len(g)
        A = ints[:k]; B = ints[k:2*k]; C = ints[2*k:]
        dN = len(parN)
        Om = sum(kron_signed(A[i], B[i], parg, parg[i]) for i in range(k)) * 2   # scaled by den^2
        Om_den = den*den
        I_M = np.eye(k, dtype=object)*den; I_N = np.eye(dN, dtype=object)*den
        for i in range(k):
            Dy = kron_signed(A[i], I_N, parg, 0) + kron_signed(I_M, C[i], parg, parg[i])
            assert (Dy.dot(Om) == Om.dot(Dy)).all(), "not invariant"
        tpar = np.array([(parg[a]+parN[b]) % 2 for a in range(k) for b in range(dN)])
        for P in (0, 1):
            idx = np.where(tpar == P)[0]
            print(f"  Omega on g(x){name}, parity {P}: {cp_pretty(Om[np.ix_(idx, idx)], Om_den)}", flush=True)

# ------------------------------------------------------------ psq
def q_elem(A, B, n):
    M = zeros(2*n)
    for i in range(n):
        for j in range(n):
            M[i][j] = A[i][j]; M[n+i][n+j] = A[i][j]
            M[i][n+j] = B[i][j]; M[n+i][j] = B[i][j]
    return M

def psq_basis(n):
    basis = []
    E = lambda i, j: [[F(int(a == i and b == j)) for b in range(n)] for a in range(n)]
    Z = [[F(0)]*n for _ in range(n)]
    def h(k):
        M = E(k, k); M[k+1][k+1] -= 1; return M
    for i in range(n):
        for j in range(n):
            if i != j: basis.append(q_elem(E(i, j), Z, n))
    for k in range(n-1): basis.append(q_elem(h(k), Z, n))
    for i in range(n):
        for j in range(n):
            if i != j: basis.append(q_elem(Z, E(i, j), n))
    for k in range(n-1): basis.append(q_elem(Z, h(k), n))
    return basis

def proj_psq(M, n):
    """sq(n) element -> its class mod the identity (traceless A-block)"""
    t = F(sum(M[i][i] for i in range(n))) / n
    M = [r[:] for r in M]
    for i in range(2*n): M[i][i] -= t
    return M

def otr(M, n): return sum(M[i][n+i] for i in range(n))

def psq(n):
    g = psq_basis(n); k = len(g)
    par = [mpar(x, n) for x in g]
    br = lambda X, Y: proj_psq(sbr(X, Y, n), n)
    ad = [matrix_of(lambda Y, X=X: br(X, Y), g) for X in g]
    # odd form: Gram matrix
    Gm = [[otr(mul(g[i], g[j]), n) for j in range(k)] for i in range(k)]
    # invariance check of the form on psq (identity drops out since otr(1*Y)=tr B_Y=0)
    for x in range(k):
        for i in range(k):
            for j in range(k):
                lhs = otr(mul(br(g[x], g[i]), g[j]), n)
                rhs = otr(mul(g[i], br(g[x], g[j])), n)
                s = (-1)**(par[x]*par[i])
                assert lhs + s*rhs == 0, "form not invariant"
    Gq = flint.fmpq_mat(k, k, [_fq(Gm[i][j]) for i in range(k) for j in range(k)])
    Inv = Gq.inv()
    # dual basis x^i = sum_a Inv[a,i]?  want (x^i, x_j) = delta_ij : sum_a c_a G[a][j] = delta
    dualc = [[F(int(Inv[i, a].p), int(Inv[i, a].q)) for a in range(k)] for i in range(k)]
    # check: sum_a dualc[i][a] G[a][j] = delta
    for i in range(k):
        for j in range(k):
            assert sum(dualc[i][a]*Gm[a][j] for a in range(k)) == (1 if i == j else 0)
    adF = ad
    addual = []
    for i in range(k):
        M = [[sum(dualc[i][a]*adF[a][r][c] for a in range(k)) for c in range(k)] for r in range(k)]
        addual.append(M)
    den, ints = to_int(adF + addual)
    A = ints[:k]; B = ints[k:]
    pdual = [1-p for p in par]
    print(f"psq({n}): sdim g = ({par.count(0)}|{par.count(1)})", flush=True)
    # odd 'Casimir' on g: sum ad x_i ad x^i
    Cg = sum(A[i].dot(B[i]) for i in range(k))
    print(f"  sum ad x_i ad x^i on g is zero: {not Cg.any()}", flush=True)
    for variant in range(4):
        D = sum(kron_signed(A[i], B[i], par, pdual[i]) * ((-1)**(par[i]) if variant & 1 else 1) for i in range(k))
        if variant & 2:
            D = sum(kron_signed(B[i], A[i], par, par[i]) for i in range(k))
        ok = True
        I = np.eye(k, dtype=object)*den
        for y in range(k):
            Dy = kron_signed(A[y], I, par, 0) + kron_signed(I, A[y], par, par[y])
            if not (Dy.dot(D) == ((-1)**par[y])*D.dot(Dy)).all() and not (Dy.dot(D) == D.dot(Dy)).all():
                ok = False; break
        sc = (Dy.dot(D) == ((-1)**par[y])*D.dot(Dy)).all()
        print(f"  variant {variant}: invariant={ok} (supercommutes={sc}) zero={not D.any()}", flush=True)
        if ok and D.any(): break
    Dd = den*den
    S, Al = sym_bases(par)
    # does D map S^2 to E^2?
    DS = D.dot(S)
    D2 = D.dot(D)
    for nm, Sp in (("S^2", S), ("E^2", Al)):
        try:
            d2, R = restrict(D2, Sp)
            # parity split inside: compute parity of basis vectors
            print(f"  D^2 on {nm} (dim {Sp.shape[1]}): {cp_pretty(R, d2*Dd*Dd)}", flush=True)
        except AssertionError as e:
            print(f"  D^2 on {nm}: {e}", flush=True)
        try:
            restrict(D, Sp); print(f"  D preserves {nm}", flush=True)
        except AssertionError:
            print(f"  D does not preserve {nm}", flush=True)
    # D maps S2 into E2?
    try:
        k2 = Al.shape[1]
        restrict_into = flint.fmpq_mat(Al.shape[0], k2 + S.shape[1], [int(x) for x in np.hstack([Al, DS]).flatten()])
        print(f"  rank[E^2 | D S^2] = {restrict_into.rank()} vs dim E^2 = {k2}", flush=True)
    except Exception as e:
        print(e)
    full = D2
    tpar = np.array([(par[a]+par[b]) % 2 for a in range(k) for b in range(k)])
    for P in (0, 1):
        idx = np.where(tpar == P)[0]
        print(f"  D^2 on g(x)g parity {P}: {cp_pretty(full[np.ix_(idx, idx)], Dd*Dd)}", flush=True)

if __name__ == "__main__":
    what = sys.argv[1]
    for n in [int(a) for a in sys.argv[2:]]:
        (pe if what == "pe" else psq)(n)
