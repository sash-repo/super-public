#!/usr/bin/env python3
"""For each W-orbit of maximal-rank maximal subalgebras: (1) the least order m of
an inner automorphism exp(2 pi i ad h) whose fixed points are exactly it;
(2) the simple ideals (odd part split by the even simple components);
(3) for the D(2,1;a) pieces, the parameter a and j(a)."""
import sys
from fractions import Fraction as F
from itertools import product
from regmax import RS, symmetric_maximal, roots_from_model
from classify import GRAM, weyl_orbit, even_type, ip
import exc3, d21a


def fixed_order(rs, S, mmax=4):
    r = len(rs.roots[0])
    for m in range(1, mmax + 1):
        for hv in product(range(2 * m), repeat=r):
            h = [F(x, m) for x in hv]
            T = 0
            for a, rt in enumerate(rs.roots):
                if sum(x * y for x, y in zip(h, rt)).denominator == 1:
                    T |= 1 << a
            if T == S:
                return m, h
    return None, None


def components(rs, S, G):
    """even simple components, and which odd roots each acts on nontrivially:
       even root e acts on odd root o iff e+o or o-e... (e+o root in S)."""
    E = [a for a in rs.members(S) if rs.par[a] == 0]
    O = [a for a in rs.members(S) if rs.par[a] == 1]
    idx = {rt: a for a, rt in enumerate(rs.roots)}
    # components of even roots
    comps = []; left = set(E)
    while left:
        x = left.pop(); c = {x}; todo = [x]
        while todo:
            y = todo.pop()
            for z in list(left):
                if ip(G, rs.roots[y], rs.roots[z]) != 0:
                    left.remove(z); c.add(z); todo.append(z)
        comps.append(sorted(c))
    out = []
    for c in comps:
        moved = set()
        for e in c:
            for o in O:
                if (e, o) in rs.prod:
                    moved.add(o)
        out.append((len(c), len(moved)))
    return out


def ab_coeffs(name, S, rs):
    """for a sub D(2,1;a): [x_w, x_-w] in terms of the three coroots."""
    if name == "ag":
        m = exc3.build_G3()
    else:
        m = exc3.build_F4()
    wt = [tuple(x) for x in m.WT]
    zero = tuple(0 for _ in wt[0])
    E = [rs.roots[a] for a in rs.members(S) if rs.par[a] == 0]
    O = [rs.roots[a] for a in rs.members(S) if rs.par[a] == 1]
    # three positive even roots (pairwise orthogonal) for the D(2,1;a)
    G = GRAM[name]
    pos = []
    for e in E:
        if any(x for x in e) and tuple(-x for x in e) not in pos and e not in pos:
            pos.append(e)
    # pick those acting on O
    acting = [e for e in pos if any(tuple(x + y for x, y in zip(e, o)) in O for o in O)]
    assert len(acting) == 3, acting
    # a highest odd weight w: w + e not in O for each acting e
    w = [o for o in O if all(tuple(x + y for x, y in zip(o, e)) not in O for e in acting)][0]
    mw = tuple(-x for x in w)
    iw, imw = wt.index(w), wt.index(mw)
    v = m.br(m.e(iw), m.e(imw))
    cart = [k for k in range(m.N) if wt[k] == zero]
    # coroots: h_e in cart-span with [h_e, x_f] = (2(f,e)/(e,e)) x_f -- solve linearly
    import lin
    # matrix of ad(cartan basis) eigenvalue on root vectors: lambda_k(root) = weight coordinate?
    # each cartan basis element acts on root vector k by a linear functional of its weight;
    # find it by bracketing.
    def eig(c, k):
        u = m.br(m.e(c), m.e(k)); return u[k]
    roots_idx = [k for k in range(m.N) if wt[k] != zero]
    coeffs = []
    # express v as functional on weights: f(rt) = sum v_c eig(c, rt)
    def f_of(vec, k):
        return sum(vec[c] * eig(c, k) for c in cart)
    # coroot functional of e: rt -> 2(rt,e)/(e,e). Find coefficients t_j s.t.
    # f_v = sum t_j coroot_j on all roots.
    rows = []
    for k in roots_idx:
        rt = wt[k]
        rows.append([2 * ip(G, rt, e) / ip(G, e, e) for e in acting] + [f_of(v, k)])
    t = solve(rows, 3)
    return acting, w, t


def solve(rows, n):
    from lin import rref
    M, piv = rref([list(r) for r in rows])
    assert n not in piv, "inconsistent"
    x = [F(0)] * n
    for i, p in enumerate(piv):
        x[p] = M[i][n]
    for r in rows:
        assert sum(a * b for a, b in zip(r[:n], x)) == r[n]
    return x


if __name__ == "__main__":
    for name in sys.argv[1:] or ["osp", "ag", "ab"]:
        rs = RS(name); G = GRAM[name]
        found = symmetric_maximal(rs); done = set()
        print("==", name)
        for S in found:
            if S in done: continue
            done |= weyl_orbit(rs, S, G)
            m, h = fixed_order(rs, S)
            print("  sdim", rs.sdim(S), even_type(rs, S, G),
                  " order", m, "h =", [str(x) for x in h] if h else None,
                  " components (even roots, odd roots moved):", components(rs, S, G))
            nodd = rs.sdim(S)[1]
            if nodd == 8 and even_type(rs, S, G).count("A1") >= 3:
                acting, w, t = ab_coeffs(name, S, rs)
                c = t
                a = c[1] / c[0]
                print("     D(2,1;a) piece: c =", [str(x) for x in c], "sum", sum(c),
                      " a = c2/c1 =", a, " j(a) =", (a*a+a+1)**3/(a*a*(a+1)**2))
