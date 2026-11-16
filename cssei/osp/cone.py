"""Dimension (over the algebraic closure of F_2) of the cone of second-order-unobstructed
infinitesimal deformations: Groebner basis of the obstruction quadrics, Hilbert dimension via
the leading-monomial ideal (Krull dim of K[l]/in(I) by max independent set)."""
import pickle, itertools, sympy as sp
o = pickle.load(open('obstr.pkl', 'rb'))
for t in ('inner', 'outer'):
    keys = o[t]['keys']; Q = o[t]['Q']; h = max(max(k) for k in keys) + 1
    L = sp.symbols('l0:%d' % h)
    polys = [sp.Add(*[L[i] * L[j] for (i, j), x in zip(keys, q) if x]) for q in Q]
    G = sp.groebner(polys, *L, modulus=2, order='grevlex')
    lead = [sp.Poly(g, *L, modulus=2).monoms(order='grevlex')[0] for g in G.exprs]
    supp = [frozenset(i for i, e in enumerate(m) if e) for m in lead]
    # Krull dim = max |S| with S containing no support of a leading monomial
    best = 0
    for r in range(h, -1, -1):
        for S in itertools.combinations(range(h), r):
            Ss = set(S)
            if all(not s <= Ss for s in supp):
                best = r; break
        if best: break
    print(t, 'h =', h, 'groebner size', len(G.exprs), 'dim of the unobstructed cone =', best)
    # components: primary decomposition not available; print the basis
    for g in G.exprs: print('   ', g)
