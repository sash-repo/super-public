"""Statement 11.7.4, heading 3): the fiber c = infinity (hbar = 0) of Theorem 11.5.2.  The K_n
directions (n = 6, 8, 9, 10) reached by brackets of length <= 4 of Lambda_hbar (x) x^a, computed
over Q[c] and specialized to hbar = 0.  Run:  python3 run89c.py 4 3   (mode degree 4, |a| <= 3)."""
import sys
from gensym import *
Delta = int(sys.argv[1]); R = int(sys.argv[2]); N = 13
A = Alg(N); Rd = Red(Delta, N, A.V)
def classical(st, s):
    out = {}
    for k, P in st.items():
        co = P.coeffs(); d = s - len(k)
        assert all(co[i] == 0 for i in range(len(co)) if i > d), ("hbar pole", k)
        if 0 <= d < len(co) and co[d] != 0:
            out[k] = Fr(int(co[d].p), int(co[d].q))
    return out
pairs = {}
for b in range(-R, R + 1):
    for cc in range(b + 1, R + 1):
        e = A.br(Lm(b), Lm(cc)); pairs.setdefault(e[0], []).append(e)
trip = {}
for a in range(-R, R + 1):
    for D, lst in pairs.items():
        for e in lst:
            f = A.br(Lm(a), e); trip.setdefault(f[0], []).append(f)
els = [(e, 4) for e in trip.get(Delta, [])]
for a in range(-R, R + 1):
    for e in trip.get(Delta - (3 - a), []):
        els.append((A.br(Lm(a), e), 5))
for D1, l1 in pairs.items():
    for e1 in l1:
        for e2 in pairs.get(Delta - D1, []):
            els.append((A.br(e1, e2), 5))
print("elements", len(els), flush=True)
rows = []
for (D, st), s in els:
    cl = classical(st, s)
    r = Rd.reduce({k: conv(v) for k, v in cl.items()})
    rows.append({n: [Fr(int(x.coeffs()[0].p), int(x.coeffs()[0].q)) if x != 0 else Fr(0) for x in r[n]] for n in r})
for n in (6, 8, 9, 10):
    hi = [m for m in range(n + 1, N + 1) if Rd.K[m]]
    H = [[x for m in hi for x in r[m]] for r in rows]
    HL = [h + r[n] for h, r in zip(H, rows)]
    print(" classical K_%d (dim %d) reached:" % (n, len(Rd.K[n])), rank(HL) - rank(H), flush=True)
