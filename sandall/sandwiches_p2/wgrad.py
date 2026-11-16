#!/usr/bin/env python3
"""For every maximal subalgebra of the smallest codimension, the associated
Weisfeiler grading: dims of g_i, and the isomorphism type of g_0."""
import glob, os, sys
from css import rref, in_span, all_elements
from maxsub import kernel
from minmax import smallest_codim
from thin import algebra_from_thin_table, load_table


def wgrading(L, L0):
    L0 = rref(L0)
    comp = [u for u in range(1, 1 << L.n) if not in_span(u, L0)]
    best = None
    for u in comp:                       # only the cosets matter; cheap enough
        M = rref(L0 + [u])
        changed = True
        while changed:
            changed = False
            for a in list(M):
                for b in L0:
                    w = L.mul(a, b)
                    if w and not in_span(w, M):
                        M = rref(M + [w]); changed = True
        if best is None or len(M) < len(best):
            best = M
    Lm = [L0, best]
    while len(Lm[-1]) < L.n:
        prev = Lm[-1]
        new = rref(prev + [L.mul(a, b) for a in Lm[1] for b in prev])
        if len(new) == len(prev):
            break
        Lm.append(new)
    Lp = [L0]
    while True:
        cur = Lp[-1]
        nxt = rref([u for u in all_elements(cur)
                    if u and all(in_span(L.mul(u, a), cur) for a in Lm[1])])
        if len(nxt) == len(cur):
            break
        Lp.append(nxt)
        if not nxt:
            break
    neg = [len(Lm[i + 1]) - len(Lm[i]) for i in range(len(Lm) - 1)]
    pos = [len(Lp[i]) - len(Lp[i + 1]) for i in range(len(Lp) - 1)] + [len(Lp[-1])]
    pos = [p for p in pos if p] if pos and pos[-1] == 0 else pos
    return neg, pos


def g0_type(L, L0, L1):
    """the quotient g_0 = L_0/L_1 as an abstract Lie algebra over F_2"""
    L0, L1 = rref(L0), rref(L1)
    reps = [u for u in L0 if not in_span(u, L1)]
    basis = []
    rows = list(L1)
    for u in L0:
        if not in_span(u, rows):
            basis.append(u); rows = rref(rows + [u])
    n = len(basis)
    red = [(rref(L1 + basis[:i + 1]), basis[i]) for i in range(n)]
    def coords(v):
        c = 0
        cur = v
        red2 = list(L1)
        # express v modulo L1 in terms of basis
        rows2 = [(b, 1 << i) for i, b in enumerate(basis)] + [(x, 0) for x in L1]
        out = []
        red3 = []
        for w, d in rows2:
            for w2, d2 in red3:
                if w ^ w2 < w:
                    w, d = w ^ w2, d ^ d2
            if w:
                red3.append((w, d)); red3.sort(key=lambda p: -p[0])
        w, d = cur, 0
        for w2, d2 in red3:
            if w ^ w2 < w:
                w, d = w ^ w2, d ^ d2
        assert w == 0
        return d
    from css import LieF2
    tab = [[coords(L.mul(a, b)) for b in basis] for a in basis]
    return LieF2(tab, "g_0")


def main(paths):
    for path in sorted(paths):
        name = os.path.basename(path)[:-4]
        L = algebra_from_thin_table(load_table(path), name)
        c, k, M = smallest_codim(L)
        print("=" * 70)
        print("%s: smallest codimension %d, %d maximal subalgebras of that codimension"
              % (name, c, k))
        neg, pos = wgrading(L, M)
        print("   W-grading dims: g_-i (i=1..) %s | g_i (i>=0) %s" % (neg, pos))
        S = rref(L.sandwiches()); sg = L.subalgebra_generated(S); nn = L.normaliser(sg)
        print("   dim s(L) = %d, codim n_L(s) = %d" % (len(sg), L.n - len(nn)))
        negn, posn = wgrading(L, nn)
        print("   W-grading from n_L(s): g_-i %s | g_i %s" % (negn, posn))


if __name__ == "__main__":
    main(sys.argv[1:] or sorted(glob.glob("css-bases/thin15/*.txt")))
