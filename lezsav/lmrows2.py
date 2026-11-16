"""Where the residual sits at a minimum found by lm2.py: |Phi|^2 by the n_z of
the rows (coker vectors are spread over rows: weight by |Q|^2 of each row key)."""
import sys, os, pickle, numpy as np
sys.argv = [sys.argv[0], sys.argv[1]] + sys.argv[2:]
import lm2
d = pickle.load(open(sys.argv[2], "rb")); lam = d["lam"]
E = lm2.E
r = lm2.resid(lam[:, None])[:, 0]
ny = E.Ymat.shape[0]
# project back: residual vector in row space = Q^H-coords -> rows: Ymat^H r
rowv = E.Ymat.conj().T @ r[:ny]
from collections import defaultdict
D = defaultdict(float)
for k, n in E.yidx.items():
    D[k[2][0]] += abs(rowv[n]) ** 2
for i, k in enumerate(E.fkeys):
    D[k[2][0]] += abs(r[ny + i]) ** 2
tot = sum(D.values())
print("|Phi| = %.4f; share of |Phi|^2 by n_z:" % np.sqrt(tot))
print("  ".join("%d:%.3f" % (z, D[z] / tot) for z in sorted(D) if D[z] / tot > 1e-4))
lb = sorted(np.abs(lam)); print("|lam| max %.3g" % lb[-1])
# size of lam by block
nb = len(lam) // 56; lo = E.lo2 if hasattr(E, "lo2") else None
print("block norms:", " ".join("%.2f" % np.linalg.norm(lam[56 * b:56 * b + 56]) for b in range(nb)))
