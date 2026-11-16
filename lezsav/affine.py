"""The free rows of Phi (rows of step 1 that no level-4 unknown reaches) as
exact polynomials of degree <= 2 in lam:  f = c + L lam + Q(lam).
c = f(0); column j of L = (f(e_j) - f(-e_j))/2; Q(e_j,e_j) = (f(e_j)+f(-e_j))/2 - c.
Rows whose degree along a random line is 1 are affine (Q = 0 on a generic
line through 0 forces Q = 0); for them, is c + L lam = 0 consistent?"""
import os, sys, random, pickle, time
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
import multiprocessing as mp
from fractions import Fraction as Q
import n4phi2 as P
from n4noeth import C
import exactlin
TAG = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
def free(lam):
    ph, _ = P.phi_of_coeffs(P.coeffs_of_lam(lam))
    return {k: v for k, v in ph.items() if k[0] == "free"}
def unit(j, s):
    l = [C() for _ in P._G["ker0"]]; l[j] = C(s); return l
def col(j):
    return j, free(unit(j, 1)), free(unit(j, -1))
def line(t):
    rng = random.Random(23)
    v = [C(rng.randint(-5, 5), rng.randint(-5, 5)) for _ in P._G["ker0"]]
    return t, free([x * t for x in v])
if __name__ == "__main__":
    t0 = time.time()
    P._init(TAG)
    n = len(P._G["ker0"])
    c = free([C() for _ in range(n)])
    with mp.Pool(8, P._init, (TAG,)) as p:
        cols = {j: (a, b) for j, a, b in p.imap_unordered(col, range(n), chunksize=4)}
        V = dict(p.map(line, range(4)))
    print("evaluations done [%.0fs]" % (time.time() - t0), flush=True)
    keys = sorted(set(c) | set().union(*(set(a) | set(b) for a, b in cols.values()))
                  | set().union(*V.values()), key=str)
    L = {}; D = {}
    for j, (a, b) in cols.items():
        for k in set(a) | set(b):
            lv = (a.get(k, C()) - b.get(k, C())) * Q(1, 2)
            dv = (a.get(k, C()) + b.get(k, C())) * Q(1, 2) - c.get(k, C())
            if lv: L[(k, j)] = lv
            if dv: D[(k, j)] = dv
    # degree along the random line (t = 0..3; degree <= 2)
    deg = {}
    for k in keys:
        f = [V[t].get(k, C()) for t in range(4)]
        d1 = [f[i + 1] - f[i] for i in range(3)]; d2 = [d1[i + 1] - d1[i] for i in range(2)]
        d3 = d2[1] - d2[0]
        assert not d3, "degree > 2 along the line"
        deg[k] = 2 if d2[0] else (1 if d1[0] else 0)
    aff = [k for k in keys if deg[k] <= 1]
    print("free rows %d: affine %d, quadratic %d" % (len(keys), len(aff),
          len(keys) - len(aff)), flush=True)
    # consistency of c + L lam = 0 on the affine rows, over Q(i) in real form
    m = len(aff)
    rows = []
    for k in aff:
        r1 = [0] * (2 * n + 1); r2 = [0] * (2 * n + 1)
        for j in range(n):
            v = L.get((k, j))
            if v:
                r1[j] = v.re; r1[n + j] = -v.im
                r2[j] = v.im; r2[n + j] = v.re
        cv = c.get(k, C())
        r1[2 * n] = -cv.re; r2[2 * n] = -cv.im
        rows += [r1, r2]
    A = [r[:-1] for r in rows]
    rA, rAb = exactlin.rank(A), exactlin.rank(rows)
    print("affine subsystem: complex rank of L %d, of [L|c] %d -> %s"
          % (rA // 2, rAb // 2, "INCONSISTENT" if rAb > rA else "consistent"),
          flush=True)
    with open(os.path.join(HERE, "affine-%s.pkl" % TAG), "wb") as fh:
        pickle.dump(dict(c=c, L=L, D=D, keys=keys, deg=deg, aff=aff), fh)
    print("[%.0fs]" % (time.time() - t0))
