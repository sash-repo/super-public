#!/usr/bin/env python3
r"""vle(7; 1) = F(vle(4; 1|3)): the prolong of g_{-1} = Pi(O(3;1)/K1),
g_0 = c(vect(3;1)) in vect(7;1).  Calibration: BGLLS (6.4) give
dim g_1..g_4 = 34, 22 (21), 7, 1 (-), dim vle^(1)(7;1) = 94, and the super
table sdim g_1..4 = 16|18, 10(9)|12, 4|3, 1(0)|0."""
import itertools
import sys
from f2vf import Space, find_weights
from prolong import prolong, simple_test, generators, ad_closure, restricted_closure, log

subsets = [A for r in (1, 2, 3) for A in itertools.combinations(range(3), r)]
idx = {A: i for i, A in enumerate(subsets)}           # coordinate x_A <-> Pi(xi^A)
par = [(len(A) + 1) % 2 for A in subsets]


def build(heights=None):
    m = len(subsets)
    sp = Space(heights or [1] * m, [1] * m, par)
    # g_{-1}
    neg = [{(0 << 5) | i} for i in range(m)]
    # g_0: vect(3;1) acting on O(3;1)/K, plus z
    g0 = []
    for B in [()] + subsets:
        for j in range(3):
            terms = set()
            for A in subsets:
                if j not in A:
                    continue
                R = tuple(sorted(set(A) - {j}))
                if set(B) & set(R):
                    continue
                img = tuple(sorted(set(B) | set(R)))
                if not img:
                    continue          # constants are killed
                # phi(e_A) = e_img  ->  field x_A d_img
                terms ^= {(sp.one[idx[A]] << 5) | idx[img]}
            if terms:
                g0.append(terms)
    z = {(sp.one[i] << 5) | i for i in range(m)}
    g0.append(z)
    return sp, neg, g0


if __name__ == '__main__':
    sp, neg, g0 = build()
    W = find_weights(sp, neg + g0)
    log('weight lattice rank', len(W[0]))
    sp = Space([1] * 7, [1] * 7, par, W)
    sp2, neg, g0 = build()
    G = prolong(sp, neg, g0)
    for d in G.degrees():
        log('deg %2d  dim %4d  sdim %s' % (d, G.dim(d), G.parity_dims(d)))
    log('total', G.dim(), G.parity_dims())
    gens = generators(G)
    J = ad_closure(sp, gens, G.basis(-1))
    log('J = ideal(g_-1):', J.dim(), J.parity_dims(), {d: J.dim(d) for d in J.degrees()})
    log(simple_test(sp, J))
    R = restricted_closure(sp, J)
    log('restricted closure of J:', R.dim(), R.parity_dims())
    RG = restricted_closure(sp, G)
    log('restricted closure of G:', RG.dim())
