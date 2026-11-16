#!/usr/bin/env python3
"""Group the maximal-rank maximal subalgebras into W(g_ev)-orbits, name the
even part, and probe for maximal closed sets that are neither symmetric nor
parabolic (exhaustive for osp, random greedy maxima for ag, ab)."""
import sys, random
from fractions import Fraction as F
from regmax import RS, symmetric_maximal

GRAM = {
    "osp": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "ag": [[1, 0, 0], [0, F(2, 3), F(-1, 3)], [0, F(-1, 3), F(2, 3)]],
    "ab": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
}


def ip(G, a, b):
    return sum(G[i][j] * a[i] * b[j] for i in range(len(a)) for j in range(len(b)))


def weyl_orbit(rs, S, G):
    evr = [rs.roots[a] for a in range(rs.n) if rs.par[a] == 0]
    idx = {r: a for a, r in enumerate(rs.roots)}
    def refl(al, S):
        T = 0
        for a in range(rs.n):
            if S >> a & 1:
                b = rs.roots[a]
                c = 2 * ip(G, b, al) / ip(G, al, al)
                T |= 1 << idx[tuple(x - c * y for x, y in zip(b, al))]
        return T
    orb = {S}; todo = [S]
    while todo:
        X = todo.pop()
        for al in evr:
            Y = refl(al, X)
            if Y not in orb:
                orb.add(Y); todo.append(Y)
    return orb


def even_type(rs, S, G):
    """Cartan type of the even root subsystem in S (components by connectivity)."""
    E = [rs.roots[a] for a in range(rs.n) if S >> a & 1 and rs.par[a] == 0]
    comps = []
    left = set(E)
    while left:
        r = left.pop(); comp = {r}; todo = [r]
        while todo:
            x = todo.pop()
            for y in list(left):
                if ip(G, x, y) != 0:
                    left.remove(y); comp.add(y); todo.append(y)
        comps.append(comp)
    names = []
    for c in comps:
        n = len(c); lens = {ip(G, r, r) for r in c}
        if n == 2: names.append("A1")
        elif n == 6 and len(lens) == 1: names.append("A2")
        elif n == 12 and len(lens) == 2: names.append("G2")
        elif n == 12: names.append("A3")
        elif n == 8: names.append("B2")
        elif n == 18: names.append("B3")
        elif n == 20 : names.append("A4")
        elif n == 32: names.append("B4")
        else: names.append("?%d,%d" % (n, len(lens)))
    return "+".join(sorted(names))


def probe(rs, trials, seed=1):
    rnd = random.Random(seed); seen = {}
    for _ in range(trials):
        order = list(range(rs.n)); rnd.shuffle(order)
        S = 0
        for b in order:
            T = rs.closure(S | 1 << b)
            if T != rs.ALL:
                S = T
        sym = all((S >> a & 1) == (S >> rs.neg[a] & 1) for a in range(rs.n))
        par = all(S >> a & 1 or S >> rs.neg[a] & 1 for a in range(rs.n))
        kind = "symmetric" if sym else ("parabolic" if par else "OTHER")
        seen.setdefault(kind, set()).add(S)
    return {k: len(v) for k, v in seen.items()}


def exhaustive_other(rs):
    """all maximal closed sets, by brute force (small n only)."""
    res = {"symmetric": 0, "parabolic": 0, "OTHER": 0}
    for S in range(1, rs.ALL):
        if rs.closed(S) and rs.is_max(S):
            sym = all((S >> a & 1) == (S >> rs.neg[a] & 1) for a in range(rs.n))
            par = all(S >> a & 1 or S >> rs.neg[a] & 1 for a in range(rs.n))
            res["symmetric" if sym else ("parabolic" if par else "OTHER")] += 1
    return res


if __name__ == "__main__":
    for name in sys.argv[1:] or ["osp", "ag", "ab"]:
        rs = RS(name); G = GRAM[name]
        found = symmetric_maximal(rs)
        reps = []; done = set()
        for S in found:
            if S in done: continue
            orb = weyl_orbit(rs, S, G)
            done |= orb
            reps.append((S, len([T for T in orb if T in found]), len(orb)))
        print("==", name, ":", len(found), "symmetric maximal closed sets,",
              len(reps), "W(g_ev)-orbits")
        for S, k, o in reps:
            nodd = sum(1 for a in rs.members(S) if rs.par[a])
            print("   sdim", rs.sdim(S), " even roots:", even_type(rs, S, G),
                  " odd roots:", nodd, " orbit size", o)
        if name == "osp":
            print("   all maximal closed sets (exhaustive):", exhaustive_other(rs))
        else:
            print("   random greedy maximal closed sets:", probe(rs, 3000))
