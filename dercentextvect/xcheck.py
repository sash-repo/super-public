#!/usr/bin/env python3
"""Cross-check: H^2 computed block by block in the weight decomposition vs the
same H^2 computed with the grading not used at all.  Every case of the census
with dim L <= LIM is done both ways."""
import os
import sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dervect import Super, Echelon, svect, derived, poisson, contact
from centext import Cocycles, h2_ungraded

LIM = 60
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xcheck.log")

def one(tag, S, L, fh):
    E = Echelon([S.vec(X) for X in L], S.p)
    B = [S.unvec(r) for r in E.rows]
    if len(B) > LIM:
        return
    t = time.time()
    C = Cocycles(S, B)
    _, tot = C.h2()
    u = h2_ungraded(S, B)
    ok = "OK" if u == tot[0] + tot[1] else "MISMATCH"
    s = (f"{tag}: dimL={len(B)} graded={tot[0]}+{tot[1]}={tot[0]+tot[1]} "
         f"ungraded={u} {ok} [{time.time()-t:.0f}s]")
    print(s); sys.stdout.flush(); fh.write(s + "\n"); fh.flush()
    return ok == "OK"

def chain(tag, S, L, fh, depth=2):
    one(tag, S, L, fh)
    cur = L
    for k in range(1, depth + 1):
        nxt = derived(S, cur)
        if Echelon([S.vec(X) for X in nxt], S.p).dim == \
           Echelon([S.vec(X) for X in cur], S.p).dim:
            break
        one(f"{tag}^({k})", S, nxt, fh); cur = nxt

def main():
    with open(LOG, "w") as fh:
        for p in (3, 5):
            for (m, n, N) in [(1,0,[1]),(1,0,[2]),(2,0,[1,1]),(0,2,None),
                              (0,3,None),(1,1,[1]),(1,2,[1]),(1,1,[2])]:
                S = Super(p, m, n, N)
                one(f"p={p} vect({m}|{n};{N if N else '-'})", S, S.vect(), fh)
            for (m, n, N) in [(2,0,[1,1]),(3,0,[1,1,1]),(0,3,None),(0,4,None),
                              (1,1,[1]),(1,2,[1]),(2,1,[1,1]),(1,1,[2])]:
                S = Super(p, m, n, N)
                chain(f"p={p} svect({m}|{n};{N if N else '-'})", S, svect(S), fh)
            for (r, n) in [(1,0),(2,0),(1,1),(1,2),(0,3),(0,4)]:
                S = Super(p, 2*r, n, [1]*(2*r))
                L, _ = poisson(S, 'h', r)
                chain(f"p={p} h({2*r}|{n})", S, L, fh)
            for (r, n) in [(1,0),(0,2),(0,3),(1,1)]:
                S = Super(p, 1+2*r, n, [1]*(1+2*r), wt=[2]+[1]*(2*r+n))
                L, _ = contact(S, r)
                chain(f"p={p} k({1+2*r}|{n})", S, L, fh)
            for r in (1, 2):
                S = Super(p, r, r, [1]*r)
                L, _ = poisson(S, 'le', r)
                chain(f"p={p} le({r}|{r})", S, L, fh)

main()
