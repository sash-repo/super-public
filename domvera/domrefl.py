"""Serganova's criterion of dominance ([S2], pp. 115-118) by odd reflections, exactly.

A root system is given by vectors in a basis with a diagonal Gram form; a base is a list of
roots. Odd reflection in an isotropic simple root a: B' = {-a} u {b+a if (a,b)!=0 else b},
the B-highest weight becomes lam-a if (lam,a)!=0 else lam ([S2], p. 116, Proposition).
lam is B-dominant iff for every even simple root g of g_0 (B0) and some base C, obtained
from B by odd reflections and containing g (or g/2, odd non-isotropic), the label
lam'(h_g) is in Z_+ ([S2], p. 118, Theorem).

Checks: [S2] p. 124 (osp_alpha(4|2), eps-basis, bases a)-d)) for several alpha, and
[S2] p. 122 (ag(2), four Dynkin diagrams) against Proposition 3.1 of the text.
"""
from fractions import Fraction as F
from itertools import product

def ip(G, u, v): return sum(G[i] * u[i] * v[i] for i in range(len(u)))
def add(u, v): return tuple(a + b for a, b in zip(u, v))
def sub(u, v): return tuple(a - b for a, b in zip(u, v))
def neg(u): return tuple(-a for a in u)
def sc(c, u): return tuple(c * a for a in u)

def reflect(G, B, lam, a):
    nb = []
    for b in B:
        if b == a: nb.append(neg(a))
        elif ip(G, a, b) != 0: nb.append(add(a, b))
        else: nb.append(b)
    return nb, (sub(lam, a) if ip(G, lam, a) != 0 else lam)

def orbit(G, B, lam):
    """all (base, weight) reachable by odd isotropic reflections"""
    seen = {}; todo = [(tuple(B), lam)]
    while todo:
        B, l = todo.pop()
        key = frozenset(B)
        if key in seen: continue
        seen[key] = (B, l)
        for a in B:
            if ip(G, a, a) == 0:
                nb, nl = reflect(G, list(B), l, a)
                todo.append((tuple(nb), nl))
    return list(seen.values())

def dominant(G, B, lam, even_simple):
    orb = orbit(G, B, lam)
    for g in even_simple:
        ok = None
        for C, l in orb:
            h = sc(F(1, 2), g)
            if g in C or (h in C and ip(G, h, h) != 0):
                lab = 2 * ip(G, l, g) / ip(G, g, g)
                if h in C and g not in C:   # odd non-isotropic simple: label on h_{2h}
                    pass
                ok = (lab.denominator == 1 and lab >= 0)
                break
        if ok is None: raise ValueError("no base contains %s" % (g,))
        if not ok: return False
    return True

# ---------- osp_alpha(4|2) = D(alpha), [S2] p. 124 ----------
def D_check(alpha, N=7, lo=-3):
    G = (F(1), F(alpha), F(-1 - alpha))      # (eps_i,eps_i); alpha = c2/c1
    e = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    E = lambda x, y, z: (F(x), F(y), F(z))
    bases = {
        'a': [E(1, 1, -1), E(1, -1, 1), E(-1, 1, 1)],
        'b': [E(2, 0, 0), E(0, 2, 0), E(-1, -1, 1)],
        'c': [E(2, 0, 0), E(0, 0, 2), E(-1, 1, -1)],
        'd': [E(0, 2, 0), E(0, 0, 2), E(1, -1, -1)]}
    even = [E(2, 0, 0), E(0, 2, 0), E(0, 0, 2)]
    a = F(alpha)
    Zp = lambda x: x.denominator == 1 and x >= 0
    N0 = lambda x: Zp(x) and x != 0
    def her(case, l1, l2, l3):
        if case == 'b':
            if not (Zp(l1) and Zp(l2) and Zp(l3)): return False
            if l3 == 0: return l1 == 0 and l2 == 0
            if l3 == 1: return l1 + a * l2 == -1 - a or l1 - a * l2 == a - 1
            return True
        if case == 'c':
            if not (Zp(l1) and Zp(l3) and Zp(l2)): return False
            if l2 == 0: return l1 == 0 and l3 == 0
            if l2 == 1: return l1 - (1 + a) * l3 == a or l1 + (1 + a) * l3 == -2 - a
            return True
        if case == 'd':
            if not (Zp(l2) and Zp(l3) and Zp(l1)): return False
            if l1 == 0: return l2 == 0 and l3 == 0
            if l1 == 1: return a * l2 - (1 + a) * l3 == -1 or a * l2 + (1 + a) * l3 == -2 * a - 1
            return True
    res = {}
    rng = [F(x) for x in range(lo, N)]
    for case, B in bases.items():
        bad = []; cnt = 0; ndom = 0
        for l in product(rng, repeat=3):
            dom = dominant(G, B, l, even)
            ndom += dom; cnt += 1
            if case != 'a' and dom != her(case, *l): bad.append((l, dom))
        res[case] = (cnt, ndom, bad)
    return res

if __name__ == "__main__":
    for al in [F(2), F(1, 3), F(1), F(-3), F(3, 2), F(-1, 2)]:
        r = D_check(al)
        for case, (cnt, nd, bad) in r.items():
            print("alpha=%s base %s: %d weights, %d dominant, %d disagree with [S2] p.124%s" % (
                al, case, cnt, nd, len(bad), (" e.g. " + str([(tuple(map(str, l)), d) for l, d in bad[:4]])) if bad else ""))

def D_shape(alpha, N=12):
    """the restatement in Sect. 4 of the text: base b, P = {0} u {lam3>=2} u ray (pt-1, qt-1, 1), |alpha|=p/q"""
    G = (F(1), F(alpha), F(-1 - alpha)); E = lambda x, y, z: (F(x), F(y), F(z))
    B = [E(2, 0, 0), E(0, 2, 0), E(-1, -1, 1)]; even = [E(2, 0, 0), E(0, 2, 0), E(0, 0, 2)]
    a = abs(F(alpha)); p, q = a.numerator, a.denominator
    bad = 0; n = 0
    for l in product([F(x) for x in range(-2, N)], repeat=3):
        l1, l2, l3 = l
        nat = all(x.denominator == 1 and x >= 0 for x in l)
        shape = nat and (l == (0, 0, 0) or l3 >= 2 or (l3 == 1 and (l1 + 1) * q == (l2 + 1) * p))
        bad += dominant(G, B, l, even) != shape; n += 1
    return n, bad

if __name__ == "__main__":
    for al in [F(2), F(1, 3), F(1), F(-3), F(3, 2), F(-1, 2), F(5, 7), F(-7, 4)]:
        print("shape check alpha=%s: %d weights, %d disagree" % ((al,) + D_shape(al)))
