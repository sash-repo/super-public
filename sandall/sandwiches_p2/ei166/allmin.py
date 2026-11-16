import sys,os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
#!/usr/bin/env python3
"""
The smallest codimension of a proper subalgebra: an exhaustive determination.

If M is a subalgebra of codimension c and 0 != f in U := M^perp, then M is a
totally isotropic subspace of the alternating form B_f(x,y) = f([x,y]) inside
ker f, whence dim M <= (dim L - 1) - rank(B_f|ker f)/2, i.e.
        rank(B_f restricted to ker f) <= 2(c - 1)   for every f in U \\ 0.
So for each c we only have to look at the (few) f satisfying that inequality and
enumerate the c-dimensional subspaces U of the dual made of such f.
"""
import glob, os, sys, itertools
from css import rref, in_span
from maxsub import restrict_rank, kernel
from thin import algebra_from_thin_table, load_table


def all_smallest_codim(L, cmax=5):
    n = L.n
    ranks = {f: restrict_rank(L, f) for f in range(1, 1 << n)}
    for c in range(1, cmax + 1):
        cand = [f for f in range(1, 1 << n) if ranks[f] <= 2 * (c - 1)]
        if len(cand) < 2 ** c - 1:
            continue
        candset = set(cand)
        found = []
        # enumerate c-dimensional subspaces contained in candset
        def rec(basis, start):
            if len(basis) == c:
                M = kernel(basis, n)
                if len(M) == n - c and L.is_subalgebra(M):
                    found.append((list(basis), M))
                return
            for i in range(start, len(cand)):
                f = cand[i]
                # all sums with the current span must be candidates too
                sums = [f]
                for e in span_elements(basis):
                    if e:
                        s = e ^ f
                        if s not in candset:
                            break
                        sums.append(s)
                else:
                    if not in_span_list(f, basis):
                        rec(basis + [f], i + 1)

        def span_elements(basis):
            els = [0]
            for b in basis:
                els += [e ^ b for e in els]
            return els

        def in_span_list(f, basis):
            return f in span_elements(basis)

        rec([], 0)
        if found:
            return found
    return None, 0, None


