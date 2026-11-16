import os, sys, random, pickle
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
import n4phi2 as P
from n4noeth import C
from n4rec import resid_full, qadd, qsmul
from n4noeth import mzero
from n4loop import step_of
P._init(sys.argv[1])
rng = random.Random(3)
lam = [C(rng.randint(-5, 5), rng.randint(-5, 5)) for _ in P._G["ker0"]]
co = P.coeffs_of_lam(lam)
u = mzero()
for j, c in enumerate(co):
    if c: u = qadd(u, qsmul(c, P._G["d2"][j]))
R = resid_full(qadd(P._G["q0"], u))
h = {}
for k in R:
    s = step_of(k)
    h.setdefault(s, {}); nz = k[2][0]; h[s][nz] = h[s].get(nz, 0) + 1
for s in sorted(h): print("step", s, sorted(h[s].items()))
# n_z of level-2 support of u
nzs = sorted({k[0] for i in (0,1) for j in (0,1) for k in u[i][j]})
print("u n_z support", nzs)
