"""Binary osp(1|2)-invariant maps F_a x F_b -> F_c on O(1;N|1) over F_p (all bilinear maps).
Basis index 2r+e = t^{(r)} theta^e.  Generators K_theta, K_{t theta} (osp1.generators, s=-1).
Invariance: L^c_X T(phi,psi) = (-1)^{pi pX}[T(L^a_X phi,psi) + (-1)^{pX p(phi)} T(phi, L^b_X psi)].
Usage: python3 bin1.py p N          census of Theorem 7.4 for a,b,c in F_p (compare with binpred.py)
       python3 bin1.py p N theta    dimension of the K_theta-invariant bilinear maps of each parity (Lemma 7.2)"""
import sys
import os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import nullspace
from osp1 import setup, generators, rep

def sparse(M):
    return [[(k, v) for k, v in enumerate(row) if v] for row in M]

def cols(M):
    n = len(M); out = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if M[i][j]: out[j].append((i, M[i][j]))
    return out

def invariant_bilinear(S, gens_a, gens_b, gens_c, pi):
    p, n = S['p'], S['n']
    unk = [(k, i, j) for k in range(n) for i in range(n) for j in range(n) if (k - i - j - pi) % 2 == 0]
    pos = {u: m for m, u in enumerate(unk)}
    rows = []
    for (A, pX), (Bm, _), (C, _) in zip(gens_a, gens_b, gens_c):
        Cs = sparse(C); Acol = cols(A); Bcol = cols(Bm)
        s = -1 if (pi and pX) else 1
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if (k - i - j - pi - pX) % 2: continue   # equation component must have right parity
                    row = {}
                    for m, v in Cs[k]:
                        u = pos.get((m, i, j))
                        if u is not None: row[u] = (row.get(u, 0) + v) % p
                    for m, v in Acol[i]:            # T(A phi_i, psi_j) = sum_m A[m][i] T(phi_m,psi_j)
                        u = pos.get((k, m, j))
                        if u is not None: row[u] = (row.get(u, 0) - s*v) % p
                    s2 = s * (-1 if (pX and i % 2) else 1)
                    for m, v in Bcol[j]:
                        u = pos.get((k, i, m))
                        if u is not None: row[u] = (row.get(u, 0) - s2*v) % p
                    row = {c: v for c, v in row.items() if v}
                    if row:
                        r = [0]*len(unk)
                        for c, v in row.items(): r[c] = v
                        rows.append(r)
    if not rows:
        return [[1 if a == b else 0 for a in range(len(unk))] for b in range(len(unk))], unk
    return nullspace(rows, p), unk

if __name__ == "__main__":
    p, N = int(sys.argv[1]), int(sys.argv[2])
    S = setup(p, N); g = generators(S, -1)
    sel = [3, 4] if (len(sys.argv) < 4 or sys.argv[3] != 'theta') else [3]
    reps = {a: [rep(S, g, a)[i] for i in sel] for a in range(p)}
    if len(sel) == 1:
        for pi in (0, 1):
            ns, unk = invariant_bilinear(S, reps[0], reps[0], reps[0], pi)
            print(f"K_theta-invariant bilinear, parity {pi}: dim {len(ns)}  (expect 2P^2={2*S['P']**2})")
        sys.exit()
    for a in range(p):
        for b in range(p):
            for c in range(p):
                for pi in (0, 1):
                    ns, unk = invariant_bilinear(S, reps[a], reps[b], reps[c], pi)
                    if ns: print(f"p={p} N={N} a={a} b={b} c={c} {'odd' if pi else 'even'}: dim {len(ns)}", flush=True)
