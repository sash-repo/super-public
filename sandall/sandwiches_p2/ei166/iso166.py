#!/usr/bin/env python3
"""Is Eick's 16:6 isomorphic to its associated graded algebra?  Finer invariants:
the histogram of (rank ad_x, rank (ad_x)^2) over all x, sandwiches, and the solution
of the grading problem by linear algebra (the complements S of L1 in L0 that are subalgebras)."""
import sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from css import LieF2, rref, in_span, all_elements
from runall import build
from allmin import all_smallest_codim
from eick166 import graded, BASE

def rank_cols(cols):
    return len(rref([c for c in cols if c]))

def hist(L):
    H = collections.Counter()
    for u in range(1 << L.n):
        a = L.ad(u); a2 = L.ad2(u)
        H[(rank_cols(a), rank_cols(a2))] += 1
    return H

L = build(os.path.join(BASE, "Eick_{16,4}.txt"))
subs0 = all_smallest_codim(L, cmax=4)
seen = {}
for U, M in subs0:
    seen.setdefault(tuple(sorted(rref(M))), M)
M = list(seen.values())[0]
L0 = rref(M)
L1 = rref([u for u in all_elements(L0) if u and all(in_span(L.mul(u, 1 << j), L0) for j in range(L.n))])
LL, G, deg = graded(L, L0, L1)
hL, hG = hist(LL), hist(G)
print("hist L :", sorted(hL.items()), flush=True)
print("hist grL:", sorted(hG.items()), flush=True)
print("same histogram:", hL == hG, flush=True)
print("sandwiches L:", len(rref(LL.sandwiches())), " grL:", len(rref(G.sandwiches())), flush=True)
