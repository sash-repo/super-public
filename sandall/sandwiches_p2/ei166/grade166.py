#!/usr/bin/env python3
"""Does Eick's 16:6 admit a Z-grading g_{-1}+g_0+g_1 inducing its W-filtration L > L0 > L1?
Equivalently (L1 being abelian): is there a complement S = {x + phi(x)} of L1 in L0 that is a
subalgebra, and an S-invariant complement V = {v + psi(v)} of L0 with [V,V]=0 and [V,L1] in S?
The conditions on phi are linear; for fixed phi, the conditions on psi except [V,V]=0 are linear;
[V,V]=0 is tested on every point of the resulting affine space."""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from css import rref, in_span, all_elements
from runall import build
from allmin import all_smallest_codim
from eick166 import graded, BASE

def solve_affine(rows, nvar):
    """rows: list of (coeff bitmask over nvar, rhs bit). Return (particular, kernel basis) or None."""
    piv = []
    for c, r in rows:
        for (pc, pr, pb) in piv:
            if (c >> pb) & 1: c ^= pc; r ^= pr
        if c:
            b = c.bit_length() - 1
            newp = []
            for (pc, pr, pbb) in piv:
                if (pc >> b) & 1: pc ^= c; pr ^= r
                newp.append((pc, pr, pbb))
            piv = newp + [(c, r, b)]
        elif r:
            return None
    part = 0
    pivbits = {pb for _, _, pb in piv}
    for pc, pr, pb in piv:
        if pr: part |= 1 << pb
    ker = []
    for f in range(nvar):
        if f in pivbits: continue
        v = 1 << f
        for pc, pr, pb in piv:
            if (pc >> f) & 1: v |= 1 << pb
        ker.append(v)
    return part, ker

def main():
    L = build(os.path.join(BASE, "Eick_{16,4}.txt"))
    seen = {}
    for U, M in all_smallest_codim(L, cmax=4):
        seen.setdefault(tuple(sorted(rref(M))), M)
    for k, M in enumerate(seen.values()):
        L0 = rref(M)
        L1 = rref([u for u in all_elements(L0) if u and all(in_span(L.mul(u, 1 << j), L0) for j in range(L.n))])
        LL, G, deg = graded(L, L0, L1)
        n = LL.n
        im = [i for i in range(n) if deg[i] == -1]; i0 = [i for i in range(n) if deg[i] == 0]; i1 = [i for i in range(n) if deg[i] == 1]
        br = lambda a, b: LL.mul(a, b)
        comp = lambda v, idx: [(v >> i) & 1 for i in idx]
        # phi: variables phi[a][c] = coefficient of e_{i1[c]} in phi(e_{i0[a]})
        nphi = len(i0) * len(i1)
        var = lambda a, c: a * len(i1) + c
        assert all(br(1 << x, 1 << y) == 0 for x in i1 for y in i1), "L1 not abelian"
        rows = []
        for a, b in itertools.combinations(range(len(i0)), 2):
            xa, xb = 1 << i0[a], 1 << i0[b]
            w = br(xa, xb)
            c_ab = comp(w, i0); l_ab = comp(w, i1)
            # condition, component c of L1:  l_ab[c] + ([xa, phi xb] + [phi xa, xb])_c + sum_d c_ab[d] phi[d][c] = 0
            for c in range(len(i1)):
                coef = 0
                for e in range(len(i1)):
                    if (br(xa, 1 << i1[e]) >> i1[c]) & 1: coef ^= 1 << var(b, e)
                    if (br(1 << i1[e], xb) >> i1[c]) & 1: coef ^= 1 << var(a, e)
                for d in range(len(i0)):
                    if c_ab[d]: coef ^= 1 << var(d, c)
                rows.append((coef, l_ab[c]))
        sol = solve_affine(rows, nphi)
        if sol is None:
            print("sub %d: no subalgebra complement S" % k); continue
        part, ker = sol
        print("sub %d: S-complements form an affine space of dim %d" % (k, len(ker)), flush=True)
        found = None
        for bits in range(1 << len(ker)):
            phi = part
            for t in range(len(ker)):
                if (bits >> t) & 1: phi ^= ker[t]
            def phiv(a):
                v = 0
                for c in range(len(i1)):
                    if (phi >> var(a, c)) & 1: v |= 1 << i1[c]
                return v
            S = [(1 << i0[a]) | phiv(a) for a in range(len(i0))]
            # decompose L0 = S + L1 : coordinates of y in L0 w.r.t. S-basis and L1
            def Spart(y):    # y in L0 -> (sum over a: coefficient)  using e_{i0} coordinates
                return [(y >> i0[a]) & 1 for a in range(len(i0))]
            def proj_L1_along_S(y):   # L1 component of y in L0 = S + L1
                s = 0
                for a, c in enumerate(Spart(y)):
                    if c: s ^= S[a]
                return y ^ s          # lies in L1
            # psi variables: psi(e_{im[v]}) = sum over L0-basis B0 (i0 + i1), var index
            B0 = i0 + i1
            npsi = len(im) * len(B0)
            pv = lambda v, b: v * len(B0) + b
            rows2 = []
            # [s, v + psi v] in V :  L0-component of [s,v] + [s, psi v] == psi( (L-L0)-component of [s, v] )
            for a in range(len(i0)):
                s = S[a]
                for v in range(len(im)):
                    w = br(s, 1 << im[v])
                    outc = comp(w, im)
                    for bidx in range(len(B0)):
                        coef = 0; rhs = (w >> B0[bidx]) & 1
                        for vv in range(len(im)):
                            for bb in range(len(B0)):
                                if ((br(s, 1 << B0[bb]) >> B0[bidx]) & 1): coef ^= 1 << pv(v, bb) if vv == v else 0
                        # [s, psi v]: psi v = sum_bb psi[v][bb] e_{B0[bb]}; its B0[bidx]-coefficient
                        coef = 0
                        for bb in range(len(B0)):
                            if (br(s, 1 << B0[bb]) >> B0[bidx]) & 1: coef ^= 1 << pv(v, bb)
                        for vv in range(len(im)):
                            if outc[vv]: coef ^= 1 << pv(vv, bidx)
                        rows2.append((coef, rhs))
            # [v + psi v, l] in S for l in L1: L1-component (along S) must vanish
            for v in range(len(im)):
                for l in i1:
                    w = br(1 << im[v], 1 << l)
                    # w lies in L0? ([g_{-1},g_1] in g_0 up to filtration: [L,L1] in L0)
                    base = proj_L1_along_S(w)
                    for c in range(len(i1)):
                        coef = 0; rhs = (base >> i1[c]) & 1
                        for bb in range(len(B0)):
                            t = proj_L1_along_S(br(1 << B0[bb], 1 << l))
                            if (t >> i1[c]) & 1: coef ^= 1 << pv(v, bb)
                        rows2.append((coef, rhs))
            sol2 = solve_affine(rows2, npsi)
            if sol2 is None: continue
            p2, k2 = sol2
            for bits2 in range(1 << len(k2)):
                psi = p2
                for t in range(len(k2)):
                    if (bits2 >> t) & 1: psi ^= k2[t]
                V = []
                for v in range(len(im)):
                    x = 1 << im[v]
                    for bb in range(len(B0)):
                        if (psi >> pv(v, bb)) & 1: x ^= 1 << B0[bb]
                    V.append(x)
                if all(br(x, y) == 0 for x in V for y in V):
                    found = (S, V); break
            if found: break
        print("sub %d: grading %s" % (k, "FOUND" if found else "none over F_2"), flush=True)
        if found:
            S, V = found
            B = V + S + [1 << x for x in i1]
            d = [-1]*len(V) + [0]*len(S) + [1]*len(i1)
            from eick166 import coords
            from css import LieF2
            ok = True
            for a in range(len(B)):
                for b in range(len(B)):
                    c = coords(br(B[a], B[b]), B)
                    for t in range(len(B)):
                        if (c >> t) & 1 and d[t] != d[a] + d[b]:
                            ok = False
            assert len(rref(B)) == 16
            print("verified: the bracket of L is homogeneous in the basis V+S+L1:", ok, flush=True)
            tab = [[coords(br(B[a], B[b]), B) for b in range(16)] for a in range(16)]
            open("eick166_graded_basis.txt", "w").write(repr([[[(tab[a][b] >> t) & 1 for t in range(16)] for b in range(16)] for a in range(16)]))
            return

main()
