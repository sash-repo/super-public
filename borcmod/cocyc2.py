"""H^1(osp(2|2); O(1;N|2)) with c(K_f) of the parity of f: the possible density multipliers
(Subsec. 6.2: the two cocycles f -> d_t f and f -> D_xi D_eta f).
Usage: python3 cocyc2.py p N      (on osp(2|2))
       python3 cocyc2.py p N k    (on the whole k(1;N|2))"""
import sys
import os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import nullspace, rank
from osp2 import Alg, osp22, structure
p, N = int(sys.argv[1]), int(sys.argv[2])
WHOLE = len(sys.argv) > 3 and sys.argv[3] == 'k'   # 'k': all K_f, f a basis monomial, i.e., the whole k(1;N|2)
A = Alg(p, N)
gens = [(f"b{k}", A.fn(k//4, (k%4)//2, k%2), A.par[k]) for k in range(4*p**N)] if WHOLE else osp22(A)
Ks, sc = structure(A, gens); n = A.n
# unknowns: c_i[k] for gen i, basis k with par k == par gen i
unk = [(i, k) for i in range(len(gens)) for k in range(n) if A.par[k] == gens[i][2]]
pos = {u: m for m, u in enumerate(unk)}
rows = []
for (i, j), a in sc.items():
    if j < i: continue
    pi, pj = gens[i][2], gens[j][2]
    # K_i(c_j) - (-1)^{pi pj} K_j(c_i) - sum a_h c_h = 0
    sg = -1 if (pi and pj) else 1
    for out in range(n):
        row = [0]*len(unk)
        for k in range(n):
            if (j, k) in pos and Ks[i][out][k]: row[pos[(j, k)]] = (row[pos[(j, k)]] + Ks[i][out][k]) % p
            if (i, k) in pos and Ks[j][out][k]: row[pos[(i, k)]] = (row[pos[(i, k)]] - sg*Ks[j][out][k]) % p
        for h, ah in enumerate(a):
            if ah and (h, out) in pos: row[pos[(h, out)]] = (row[pos[(h, out)]] - ah) % p
        if any(row): rows.append(row)
Z = nullspace(rows, p)
# the two multipliers of formula (actLamMu): c(K_f) = d_t f and c(K_f) = D_xi D_eta f
DD = A.mul(A.DXI, A.DETA)
for name, op in (("d_t f", A.Dt), ("D_xi D_eta f", DD)):
    v = [0]*len(unk)
    for i, (_, f, pf) in enumerate(gens):
        cf = A.apply(op, f)
        for k in range(n):
            if cf[k] % p: v[pos[(i, k)]] = cf[k] % p
    ok = all(sum(r[m]*v[m] for m in range(len(v))) % p == 0 for r in rows)
    print(f"c(K_f) = {name}: {'cocycle' if ok else 'NOT a cocycle'}")
# coboundaries: c_i = K_i(h), h even
cob = []
for k in range(n):
    if A.par[k]: continue
    v = [0]*len(unk)
    for i in range(len(gens)):
        for out in range(n):
            if Ks[i][out][k] and (i, out) in pos: v[pos[(i, out)]] = Ks[i][out][k] % p
    cob.append(v)
rB = rank(cob, p) if cob else 0
print(f"p={p} N={N}: dim Z^1={len(Z)}, dim B^1={rB}, dim H^1={len(Z)-rB}")
# representatives: show c on xieta, t for a basis of Z mod B
reps = []
cur = list(cob)
for z in Z:
    if rank(cur + [z], p) > rank(cur, p):
        cur.append(z); reps.append(z)
names = [g[0] for g in gens]
for z in reps:
    s = []
    for i in range(len(gens)):
        terms = [(k, z[pos[(i,k)]]) for k in range(n) if (i,k) in pos and z[pos[(i,k)]]]
        s.append(names[i] + ":" + ",".join(f"{c}*[{k//4},{(k%4)//2},{k%2}]" for k, c in terms))
    print("  ", "; ".join(s))
