import sys, itertools
import sympy as sp
from vf import VF

def setup(m, n, k):
    V = VF(m, n+k)
    X = list(range(m)); TH = [m+i for i in range(n)]; XI = [m+n+j for j in range(k)]
    def mono(evars=(), ovars=()):
        e = [0]*m
        for v in evars: e[v] += 1
        o = tuple(sorted(v-m for v in ovars))
        assert len(set(o)) == len(o)
        return (tuple(e), o)
    subsets = [S for r in range(k+1) for S in itertools.combinations(XI, r)]
    # u_{-1}
    W = []
    for S in subsets:
        for v in X+TH:
            W.append({(mono((), S), v): 1})
    # u_0
    U0 = []
    for S in subsets:
        for a in X+TH:
            for v in X+TH:
                ev = (a,) if a in X else (); ov = tuple(S)+((a,) if a in TH else ())
                if len(set(ov)) < len(ov): continue
                U0.append({(mono(ev, ov), v): 1})
        for j in XI:
            U0.append({(mono((), S), j): 1})
    return V, W, U0

def coords(Y, basis):
    # basis elements are single-term fields with coeff 1 (up to monomial)
    idx = {next(iter(b)): i for i, b in enumerate(basis)}
    vec = [0]*len(basis)
    for key, c in Y.items():
        vec[idx[key]] += c
    return vec

def main(m, n, k):
    V, W, U0 = setup(m, n, k)
    N = len(W); pW = [V.par(w) for w in W]
    mats = []
    for D in U0:
        M = sp.zeros(N, N)
        for j, w in enumerate(W):
            c = coords(V.br(D, w), W)
            for i in range(N): M[i, j] = c[i]
        mats.append((M, V.par(D)))
    def strace(M): return sum(((-1)**pW[i])*M[i, i] for i in range(N))
    print("sdim W", pW.count(0), pW.count(1), " sdim u0", sum(1 for _, p in mats if p == 0), sum(1 for _, p in mats if p == 1))
    print("str of u0 elements nonzero:", [i for i, (M, p) in enumerate(mats) if strace(M) != 0])
    return V, W, U0, mats, pW

if __name__ == "__main__":
    main(*map(int, sys.argv[1:4]))
