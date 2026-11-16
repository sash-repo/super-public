"""q : L -> End(L)/ad L, x -> (ad x)^2 mod ad L is F_2-linear (char 2:
(ad(x+y))^2 = (ad x)^2+(ad y)^2+ad[x,y]).  Its kernel K is Aut(L)-invariant.
Print invariants of K for each algebra."""
import sys, glob
from f2 import *
def sub_br(L, A, B):
    return span(L.br(a, b) for a in A for b in B)
def gen_sub(L, S):
    E = span(S); B = E.basis()
    while True:
        new = False
        for a in list(E.basis()):
            for b in list(E.basis()):
                if E.add(L.br(a, b)): new = True
        if not new: return E
def normalizer(L, S):
    # x with [x,S] ⊂ S: kernel of x -> ([x,s] mod S)_s
    E = span(S); n = L.n
    imgs = []
    for i in range(n):
        v = 0
        for k, s in enumerate(S):
            r = full_reduce(E, L.br(1 << i, s)); v |= r << (k * n)
        imgs.append(v)
    return kernel(imgs)
def run(f):
    L = Alg.load(f); n = L.n
    ads = [L.ad(1 << i) for i in range(n)]
    V = span(flat(A, n) for A in ads); assert len(V) == n
    res = [V.reduce(flat(mul(A, A), n))[0] for A in ads]
    K = kernel_mod(res, V)
    rq = n - len(K)
    KK = sub_br(L, K, K)
    G = gen_sub(L, K)
    Nm = normalizer(L, K)
    # derived series of K-generated
    ds = []
    D = G.basis()
    while D:
        D2 = sub_br(L, D, D).basis(); ds.append(len(D2))
        if len(D2) == len(D): break
        D = D2
    # [L,K]
    LK = sub_br(L, [1 << i for i in range(n)], K)
    print('%-12s rank q %d dimK %d [K,K] %d <K> %d N(K) %d [L,K] %d derived<K> %s'
          % (f.split('/')[-1], rq, len(K), len(KK), len(G), len(Nm), len(LK), ds), flush=True)
if __name__ == "__main__":
    for f in sys.argv[1:]: run(f)
