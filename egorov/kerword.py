"""The shortest element of C_g acting trivially on S^k V + Lambda^k V (sl(3|2))."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from sep import symw
import itertools
from uw import normalize
def lamw(k, m, n):
    G = set()
    for v in itertools.product(range(k+1), repeat=m+n):
        if sum(v) == k and all(x <= 1 for x in v[:m]): G.add(normalize(v, m, n))
    return frozenset(G)
from uw import refl, unit, kac_weights
from cox import cartan, tits_gens, mul
def find(m, n, mods, L):
    N=m+n; dist=[tuple(unit(i,N)[k]-unit(i+1,N)[k] for k in range(N)) for i in range(N-1)]
    srt=[sorted(G) for G in mods]; idx=[{g:i for i,g in enumerate(s)} for s in srt]
    Pt=[tuple(tuple(idx[j][refl(G,a,m,n)[g]] for g in srt[j]) for j,G in enumerate(mods)) for a in dist]
    T=tits_gens(cartan(m,n)); I=tuple(tuple(int(r==c) for c in range(N-1)) for r in range(N-1))
    idt=tuple(tuple(range(len(s))) for s in srt); seen={I:(idt,'')}; img={idt:(I,'')}; front=[I]
    for r in range(1,L+1):
        nf=[]
        for X in front:
            for s in range(N-1):
                Y=mul(T[s],X); w=tuple(tuple(Pt[s][j][x] for x in t) for j,t in enumerate(seen[X][0])); wd=str(s+1)+seen[X][1]
                if Y in seen: continue
                if w in img: return wd, img[w][1]
                seen[Y]=(w,wd); img[w]=(Y,wd); nf.append(Y)
        front=nf
sys.argv=[sys.argv[0]]
if __name__=='__main__':
    m,n=3,2
    a,b=find(m,n,[symw(4,m,n),lamw(4,m,n)],8)
    print("words with equal images on S^4V+Lambda^4V (generators 1,2 even, 3 odd, 4 even; leftmost acts last):",a,b)
    u = a + b[::-1]   # u = a * b^{-1}, nontrivial in C_g
    print("kernel element u = a b^{-1} =", u, "length", len(u))
    for k in range(1,9):
        mods=[symw(k,m,n),lamw(k,m,n)]
        N=m+n; dist=[tuple(unit(i,N)[kk]-unit(i+1,N)[kk] for kk in range(N)) for i in range(N-1)]
        P=[[refl(G,d,m,n) for G in mods] for d in dist]
        def apj(g,j):
            for c in reversed(u): g=P[int(c)-1][j][g]
            return g
        triv=all(apj(g,j)==g for j,G in enumerate(mods) for g in G)
        print(f"k={k}: u trivial on S^kV+Lambda^kV: {triv}")
    for lam in [(2,1,0,1,0),(4,2,0,2,0)]:
        G=kac_weights(lam,m,n); dist=[tuple(unit(i,5)[kk]-unit(i+1,5)[kk] for kk in range(5)) for i in range(4)]
        P=[refl(G,d,m,n) for d in dist]
        def ap(g):
            for c in reversed(u): g=P[int(c)-1][g]
            return g
        print(f"Kac K{lam}: u moves {sum(ap(g)!=g for g in G)} of {len(G)} weights")
