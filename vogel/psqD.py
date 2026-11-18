"""psq(n): the odd mixed Casimir D = sum (-1)^{p(x_i)} ad x_i (x) ad x^i on g(x)g
(x^i dual w.r.t. the odd form otr). Prints rank D on S^2->E^2, E^2->S^2, D^2, and the
odd 'Casimir' sum (-1)^{p(x_i)} ad x_i ad x^i on g."""
import sys
import numpy as np, flint
from fractions import Fraction as F
from fakecas import mul, mpar, sbr, matrix_of, _fq
from fakecas2 import psq_basis, proj_psq, otr, to_int, kron_signed, sym_bases

def rk(M):
    return flint.fmpz_mat(M.shape[0], M.shape[1], [int(x) for x in M.flatten()]).rank()

def run(n):
    g = psq_basis(n); k = len(g); par = [mpar(x, n) for x in g]
    br = lambda X, Y: proj_psq(sbr(X, Y, n), n)
    ad = [matrix_of(lambda Y, X=X: br(X, Y), g) for X in g]
    Gm = [[otr(mul(g[i], g[j]), n) for j in range(k)] for i in range(k)]
    Inv = flint.fmpq_mat(k, k, [_fq(Gm[i][j]) for i in range(k) for j in range(k)]).inv()
    dualc = [[F(int(Inv[i, a].p), int(Inv[i, a].q)) for a in range(k)] for i in range(k)]
    addual = [[[sum(dualc[i][a]*ad[a][r][c] for a in range(k)) for c in range(k)] for r in range(k)] for i in range(k)]
    den, ints = to_int(ad + addual); A = ints[:k]; B = ints[k:]
    sg = [(-1)**p for p in par]
    Cg = sum(sg[i]*A[i].dot(B[i]) for i in range(k))
    D = sum(sg[i]*kron_signed(A[i], B[i], par, 1-par[i]) for i in range(k))
    I = np.eye(k, dtype=object)
    for y in range(k):
        Dy = kron_signed(A[y], I, par, 0) + kron_signed(I, A[y], par, par[y])
        assert (Dy.dot(D) == sg[y]*D.dot(Dy)).all()
        assert (A[y].dot(Cg) == sg[y]*Cg.dot(A[y])).all() or (A[y].dot(Cg) == Cg.dot(A[y])).all()
    S, Al = sym_bases(par)
    D2 = D.dot(D)
    print(f"psq({n}) sdim ({par.count(0)}|{par.count(1)}): odd Casimir on g: rank {rk(Cg)}, square zero {not Cg.dot(Cg).any()};"
          f" D on g(x)g: rank {rk(D)}, D^2=0: {not D2.any()};  dim S^2={S.shape[1]}, E^2={Al.shape[1]};"
          f" rank D|S^2 = {rk(D.dot(S))}, rank D|E^2 = {rk(D.dot(Al))}", flush=True)
    # parity bookkeeping of ker/im on S^2 and E^2
    tp = np.array([(par[a]+par[b]) % 2 for a in range(k) for b in range(k)])
    def sd(Sp):
        ev = [j for j in range(Sp.shape[1]) if tp[np.nonzero(Sp[:, j])[0][0]] == 0]
        return len(ev), Sp.shape[1]-len(ev)
    print("   sdim S^2 =", sd(S), " sdim E^2 =", sd(Al))
    for nm, Sp in (("S2", S), ("E2", Al)):
        ev = [j for j in range(Sp.shape[1]) if tp[np.nonzero(Sp[:, j])[0][0]] == 0]
        od = [j for j in range(Sp.shape[1]) if j not in ev]
        print(f"   rank D on {nm}_even = {rk(D.dot(Sp[:, ev]))}, on {nm}_odd = {rk(D.dot(Sp[:, od]))}")

if __name__ == "__main__":
    for n in map(int, sys.argv[1:]): run(n)
