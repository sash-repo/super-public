#!/usr/bin/env python3
"""fr(3;1), the Frank algebra, as the partial Cartan prolong
   (k(3)_{<=0}, Span(tp - p^(2)q, tq + pq^(2)))_{*,1}  inside k^(1)(3;1),
and its H^2(g_-;g) -- the Claim on H^2(fr(n)_-;fr(n)) for n=1
and the table of the subsection on fr(3;1)."""
from k3 import P, rref, nullspace
from run_k import g, keep, span_subalgebra, analyse

n = g.n
deg = [w[0] for w in g.wt]


def vec(d):
    v = [0] * n
    for m, c in d.items():
        v[keep.index(m)] = c % P
    return v


def comp(i):
    return [vec({keep[j]: 1}) for j in range(n) if deg[j] == i]


def build_fr():
    G = {i: comp(i) for i in (-2, -1, 0)}
    G[1] = [vec({(1, 1, 0): 1, (0, 2, 1): -1}),     # tp - p^(2)q
            vec({(1, 0, 1): 1, (0, 1, 2): 1})]      # tq + pq^(2)
    gm1 = comp(-1)
    for i in range(2, max(deg) + 1):
        cand = comp(i)
        prev = rref([r[:] for r in G[i - 1]], n)[0] if G[i - 1] else []
        piv = [next(t for t, x in enumerate(r) if x) for r in prev]

        def red(v):
            v = v[:]
            for t, pc in enumerate(piv):
                if v[pc]:
                    f = v[pc]
                    v = [(a - f * b) % P for a, b in zip(v, prev[t])]
            return v

        rows = []
        for y in gm1:
            imgs = [red(g.bracket(y, c)) for c in cand]
            for coord in range(n):
                r = [im[coord] for im in imgs]
                if any(r):
                    rows.append(r)
        sol = (nullspace(rows, len(cand)) if rows else
               [[1 if a == b else 0 for a in range(len(cand))]
                for b in range(len(cand))])
        G[i] = []
        for s in sol:
            v = [0] * n
            for co, c in zip(s, cand):
                if co:
                    v = [(a + co * b) % P for a, b in zip(v, c)]
            G[i].append(v)
    rows = [r for i in sorted(G) for r in G[i]]
    return span_subalgebra(g, rref(rows, n)[0])


if __name__ == "__main__":
    fr = build_fr()
    print("dim fr(3;1) =", fr.n, " Jacobi:", fr.check_jacobi()[0])
    analyse(fr, "fr(3;1)")
