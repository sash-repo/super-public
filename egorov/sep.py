"""Which modules separate C_g?  Ball test (as in cox.py) with ONE module:
a typical Kac module K(k*lam0), k = 1, 2, 3, ..., and the symmetric powers S^k(V)."""
import sys, os, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from uw import kac_weights, typical, refl, unit, normalize
from cox import cartan, tits_gens, mul

def symw(k, m, n):  # weights of S^k(V), V = C^{m|n}: even coords any, odd coords 0/1
    G = set()
    for v in itertools.product(range(k+1), repeat=m+n):
        if sum(v) == k and all(x <= 1 for x in v[m:]):
            G.add(normalize(v, m, n))
    return frozenset(G)

def ball(m, n, mods, L):
    N = m+n
    dist = [tuple(unit(i,N)[k]-unit(i+1,N)[k] for k in range(N)) for i in range(N-1)]
    srt = [sorted(G) for G in mods]; idx = [{g:i for i,g in enumerate(s)} for s in srt]
    Pt = [tuple(tuple(idx[j][refl(G,a,m,n)[g]] for g in srt[j]) for j,G in enumerate(mods)) for a in dist]
    T = tits_gens(cartan(m,n)); I = tuple(tuple(int(r==c) for c in range(N-1)) for r in range(N-1))
    idt = tuple(tuple(range(len(s))) for s in srt)
    seen = {I: idt}; img = {idt: I}; front = [I]
    for r in range(1, L+1):
        nf = []
        for X in front:
            for s in range(N-1):
                Y = mul(T[s], X); w = tuple(tuple(Pt[s][j][x] for x in t) for j,t in enumerate(seen[X]))
                if Y in seen: continue
                if w in img and img[w] != Y: return r
                seen[Y] = w; img[w] = Y; nf.append(Y)
        front = nf
    return None

if __name__ == '__main__':
    cases = {(3,1):((2,1,0,0),8), (4,1):((3,2,1,0,0),6), (3,2):((2,1,0,1,0),5)}
    for (m,n),(l0,L) in cases.items():
        for k in (1,2,3,4):
            lam = tuple(k*x for x in l0)
            if not typical(lam,m,n): print(f"sl({m}|{n}) K({lam}) atypical"); continue
            c = ball(m,n,[kac_weights(lam,m,n)],L)
            print(f"sl({m}|{n}) K({lam}): " + (f"no clash up to radius {L}" if c is None else f"first clash at radius {c}"), flush=True)
        for k in (1,2,3,4,6):
            c = ball(m,n,[symw(k,m,n)],L)
            print(f"sl({m}|{n}) S^{k}V: " + (f"no clash up to radius {L}" if c is None else f"first clash at radius {c}"), flush=True)
