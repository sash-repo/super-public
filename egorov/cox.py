"""Is Serganova's UW of sl(m|n) the Coxeter group C_g of its diagram in Table 1?

G = Coxeter group of the Table 1 diagram on the distinguished base (edges 3 along the
two A-chains, infinity at the odd node, 2 otherwise), realized faithfully by its Tits
representation (integer matrices, 2B(s,t) in {2,-1,0,-2}).  W = UW acts on the weight
sets of the typical Kac modules (uw.py).  The relations of the diagram hold in UW
(Lemma 1.4.1c of the text), so UW is a quotient of G; the Conjecture says G -> UW is injective.
Test: map the ball of radius L of G to W; injective on the ball <=> no relation of
length <= 2L beyond the Coxeter ones (for the modules used, hence for UW).
"""
import sys, itertools
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from uw import module_list, refl, unit

def cartan(m, n):
    N = m + n - 1; odd = m - 1
    M = [[0]*N for _ in range(N)]
    for i in range(N):
        M[i][i] = 2
        if i + 1 < N:
            v = -2 if odd in (i, i+1) else -1
            M[i][i+1] = M[i+1][i] = v
    return M

def tits_gens(M):
    N = len(M); gens = []
    for s in range(N):
        # sigma_s(e_t) = e_t - 2B(e_s,e_t) e_s ; matrix columns = images
        A = [[(1 if r == c else 0) for c in range(N)] for r in range(N)]
        for t in range(N):
            A[s][t] -= M[s][t]
        gens.append(tuple(map(tuple, A)))
    return gens

def mul(A, B):
    N = len(A)
    return tuple(tuple(sum(A[i][k]*B[k][j] for k in range(N)) for j in range(N)) for i in range(N))

def run(m, n, L, R):
    N = m + n
    mods = module_list(m, n, R=R)
    dist = [tuple(unit(i, N)[k] - unit(i+1, N)[k] for k in range(N)) for i in range(N-1)]
    P = [[refl(G, a, m, n) for G in mods] for a in dist]
    # index weights to make permutations into tuples
    idx = [ {g: i for i, g in enumerate(sorted(G))} for G in mods]
    srt = [sorted(G) for G in mods]
    Pt = [tuple(tuple(idx[j][P[s][j][g]] for g in srt[j]) for j in range(len(mods))) for s in range(N-1)]
    def act(s, t):  # t after s? we compose: new = gen_s o t
        return tuple(tuple(Pt[s][j][x] for x in t[j]) for j in range(len(mods)))
    T = tits_gens(cartan(m, n))
    I = tuple(tuple(1 if r == c else 0 for c in range(N-1)) for r in range(N-1))
    idt = tuple(tuple(range(len(s))) for s in srt)
    seen = {I: idt}; img = {idt: I}; front = [I]; clash = None
    for r in range(1, L+1):
        nf = []
        for X in front:
            for s in range(N-1):
                Y = mul(T[s], X)
                w = act(s, seen[X])
                if Y in seen:
                    assert seen[Y] == w, 'a Coxeter relation fails in UW'
                    continue
                if w in img and img[w] != Y:
                    clash = (r, Y); break
                seen[Y] = w; img[w] = Y; nf.append(Y)
            if clash: break
        if clash: break
        front = nf
        print(f"  sl({m}|{n}) radius {r}: ball {len(seen)}, images distinct", flush=True)
    return len(mods), len(seen), clash

if __name__ == '__main__':
    for (m, n, L, R) in [(1,2,30,3),(2,1,30,3),(3,1,9,3),(1,3,9,3),(2,2,0,0),(4,1,7,2),(1,4,7,2),(3,2,6,2),(2,3,6,2)]:
        if m == n: continue
        k, b, c = run(m, n, L, R)
        line = f"sl({m}|{n}): {k} modules, ball of radius {L} of G has {b} elements, " + \
               ("all images distinct" if c is None else f"CLASH at radius {c[0]}")
        print(line, flush=True)
