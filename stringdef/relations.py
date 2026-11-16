"""Super analog of Weinstein's Lemma 3.1: relations sum f(a,b) e_a * delta(e_b) = 0 in C^3_n,
and relations among delta(e_a)*delta(e_b) in C^4_n.  Cochains identified with chains via the
monomial inner product, delta = transpose of d.  usage: relations.py kind W"""
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from nsgonch import algebra, chains, dmat, normal, rank, F
from exactlin import nullspace

def setup(kind, W):
    B, br = algebra(kind, W)
    return B, br

def delta_of(B, br, b):
    """delta(e_b) as dict over 2-chains (sorted tuples), = column of d_2 at weight w_b."""
    w = B[b][1]
    M, a, c = dmat(B, br, 2, w)
    src = chains(B, 2, w); tgt = chains(B, 1, w)
    j = tgt.index((b,))
    from math import factorial
    from collections import Counter
    fac = lambda m: __import__('math').prod(factorial(c) for c in Counter(m).values())
    return {src[r]: F(M[r][j], fac(src[r])) for r in range(len(src)) if M[r][j]}

def mul(B, u, v):
    out = {}
    for m1, c1 in u.items():
        for m2, c2 in v.items():
            nm, s = normal(list(m1) + list(m2), B)
            if nm is None: continue
            out[nm] = out.get(nm, 0) + s * c1 * c2
    return {k: c for k, c in out.items() if c}

def study(kind, W, n):
    B, br = setup(kind, W)
    n = F(n)
    rows = []; labels = []
    basis = {m: i for i, m in enumerate(chains(B, 3, n))}
    for a in range(len(B)):
        for b in range(len(B)):
            if B[a][1] + B[b][1] != n: continue
            de = delta_of(B, br, b)
            if not de: continue
            v = mul(B, {(a,): 1}, de)
            rows.append([v.get(m, 0) for m in basis]); labels.append((B[a][0]+str(B[a][1]), 'd'+B[b][0]+str(B[b][1])))
    if not rows: return labels, 0, []
    r = rank(rows)
    rel = nullspace([list(c) for c in zip(*rows)]) if r < len(rows) else []
    return labels, r, rel

def study4(kind, W, n):
    B, br = setup(kind, W); n = F(n)
    basis = {m: i for i, m in enumerate(chains(B, 4, n))}
    rows = []
    for a in range(len(B)):
        for b in range(a, len(B)):
            if B[a][1] + B[b][1] != n: continue
            da, db = delta_of(B, br, a), delta_of(B, br, b)
            if not da or not db: continue
            v = mul(B, da, db)
            rows.append([v.get(m, 0) for m in basis])
    nz = [r for r in rows if any(r)]
    return len(rows), len(rows) - len(nz), (rank(nz) if nz else 0)

if __name__ == "__main__":
    kind, W = sys.argv[1], int(sys.argv[2])
    for n2 in range(2, 2 * W + 1):
        labels, r, rel = study(kind, W, F(n2, 2))
        if labels:
            print(kind, str(F(n2, 2)), "monomials", len(labels), "rank", r, "relations", len(labels) - r)
    print("second kind: products delta(e_a) delta(e_b) in C^4_n")
    for n2 in range(2, 2 * W + 1):
        tot, zero, r4 = study4(kind, W, F(n2, 2))
        if tot:
            print(kind, str(F(n2, 2)), "products", tot, "of them zero", zero, "rank", r4, "relations", tot - r4)
