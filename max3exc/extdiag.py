#!/usr/bin/env python3
"""Statement 9.4: the regular maximal subalgebras read off the extended Dynkin diagrams.
For every positive system P of osp_a(4|2), ag(2), ab(3) (given by generic
functionals; up to W(g_ev)), the simple roots Pi, the highest root theta and
its marks m_i; for each node with mark m_i>1, the fixed points of
exp(2 pi i ad h_i/m_i), h_i dual to Pi: roots with c_i = 0 mod m_i
(Borel--de Siebenthal deletion from the extended diagram Pi u {-theta}).
Which of them are maximal, and which W-orbit of the maximal-rank list."""
import random, sys
from fractions import Fraction as F
from regmax import RS, symmetric_maximal
from classify import GRAM, weyl_orbit

def solve(cols, v):
    # v = sum x_j cols[j], exact
    n = len(cols); r = len(v)
    M = [[F(cols[j][i]) for j in range(n)] + [F(v[i])] for i in range(r)]
    piv = []; row = 0
    for c in range(n):
        p = next((i for i in range(row, r) if M[i][c] != 0), None)
        if p is None: continue
        M[row], M[p] = M[p], M[row]
        M[row] = [x / M[row][c] for x in M[row]]
        for i in range(r):
            if i != row and M[i][c] != 0:
                M[i] = [a - M[i][c] * b for a, b in zip(M[i], M[row])]
        piv.append(c); row += 1
    for i in range(row, r): assert M[i][-1] == 0
    x = [F(0)] * n
    for i, c in enumerate(piv): x[c] = M[i][-1]
    return x

def run(name, trials=4000, seed=7):
    rs = RS(name); G = GRAM[name]
    rnk = len(rs.roots[0])
    maxes = symmetric_maximal(rs)
    orbs = []
    for S in maxes:
        if not any(S in o for o in orbs): orbs.append(weyl_orbit(rs, S, G))
    maxes = [min(o) for o in orbs]
    def label(S):
        for k, o in enumerate(orbs):
            if S in o: return k
        return None
    rnd = random.Random(seed)
    seenP = {}; results = {}
    for _ in range(trials):
        f = [rnd.randint(-10**6, 10**6) for _ in range(rnk)]
        val = [sum(a*b for a, b in zip(f, r)) for r in rs.roots]
        if 0 in val: continue
        P = sum(1 << a for a in range(rs.n) if val[a] > 0)
        orbP = None
        for key in seenP:
            if P in seenP[key][0]: orbP = key; break
        if orbP is not None: continue
        o = weyl_orbit(rs, P, G)
        pos = [a for a in range(rs.n) if P >> a & 1]
        prs = {rs.roots[a] for a in pos}
        simple = []
        for a in pos:
            r = rs.roots[a]
            dec = any(tuple(x - y for x, y in zip(r, rs.roots[b])) in prs for b in pos)
            if not dec: simple.append(a)
        assert len(simple) == rnk, (name, len(simple))
        cols = [rs.roots[a] for a in simple]
        coef = {a: solve(cols, rs.roots[a]) for a in range(rs.n)}
        for a in pos: assert all(c >= 0 and c.denominator == 1 for c in coef[a])
        ht = {a: sum(coef[a]) for a in pos}
        top = max(ht.values()); th = [a for a in pos if ht[a] == top]
        assert len(th) == 1; th = th[0]
        marks = [int(c) for c in coef[th]]
        dominates = all(all(coef[a][i] <= marks[i] for i in range(rnk)) for a in pos)
        pstr = "".join("o" if rs.par[a] else "e" for a in simple)
        # isotropic?  odd root r with (r,r)=0
        allr = set(rs.roots)
        def iso(a):   # odd root is non-isotropic iff its double is a root
            return tuple(2 * x for x in rs.roots[a]) not in allr
        types = "".join(("x" if iso(a) else "*") if rs.par[a] else "o" for a in simple)
        fixed = []
        for i, m in enumerate(marks):
            if m < 2: continue
            S = sum(1 << a for a in range(rs.n) if coef[a][i] % m == 0)
            fixed.append((i, m, rs.sdim(S), rs.closed(S), rs.is_max(S) if rs.closed(S) else False, label(S)))
        key = len(seenP)
        seenP[key] = (o, types, marks, th, dominates, fixed)
    print("==", name, " positive systems up to W(g_ev):", len(seenP))
    print("   maximal-rank maximal list (W-orbits):")
    for k, S in enumerate(maxes): print("     [%d] sdim %s" % (k, rs.sdim(S)))
    reach = set()
    for key, (o, types, marks, th, dom, fixed) in seenP.items():
        print("  Pi types %s  theta=%s marks %s  theta dominates: %s" % (types, rs.roots[th], marks, dom))
        for (i, m, sd, cl, mx, lb) in fixed:
            print("     delete node %d (mark %d): sdim %s closed %s maximal %s orbit %s" % (i, m, sd, cl, mx, lb))
            if mx: reach.add(lb)
    print("   orbits reached:", sorted(reach), "of", len(maxes))

if __name__ == "__main__":
    for nm in sys.argv[1:] or ["osp", "ag", "ab"]:
        run(nm)
