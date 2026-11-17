#!/usr/bin/env python3
"""
dermeta.py -- Problem (5) of the text (Subsec. 1.4) in the absence of even variables.

    What is the Lie algebra corresponding to ALL terms of eq. (13)
    and functorially depending on Lambda?

Here we answer it for  n = 0  even variables, i.e. we compute

    d_m(Lambda) := { D in Der_K(B) | D(Lambda) = 0 },
    B := Lambda(m) (x) Lambda,     Lambda supercommutative,

where Der is taken in the ASSOCIATIVE sense, D(bc) = D(b)c + bD(c) with NO
Koszul signs -- the only notion that makes sense for a parity-mixing map, and
the one Bavula uses in J. Algebra Appl. 8 (2009) 805--827 for Lambda = K
purely even.

Everything is exact (integer/rational linear algebra through exactlin.py,
which uses FLINT when available and a pure-Python reference otherwise).

Universality: it is enough to run Lambda = Lambda(N), the Grassmann algebra
on xi_1, ..., xi_N -- every finitely generated supercommutative superalgebra
is a quotient of a polynomial algebra tensored with such a Lambda, and the
even variables are excluded by hypothesis.

Run:  python3 dermeta.py     (faster with python-flint installed)
"""

import os
import sys
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "."))

from exactlin import backend, nullspace, rank            # noqa: E402


# ------------------------------------------------------------------ Grassmann

class Grass:
    """Grassmann algebra on m thetas (indices 0..m-1) and N xis (m..m+N-1).

    A basis monomial is a bitmask; an element is a dict {mask: coefficient}.
    """

    def __init__(self, m, N):
        self.m, self.N, self.n = m, N, m + N
        self.dim = 1 << self.n
        self.theta_mask = (1 << m) - 1
        self.xi_mask = ((1 << self.n) - 1) ^ self.theta_mask

    # --- basic monomial arithmetic ---------------------------------------

    def par(self, mask):
        return bin(mask).count("1") % 2

    def sign(self, A, B):
        """sign of theta_A * theta_B (disjoint masks), Koszul reordering."""
        s = 1
        for b in range(self.n):
            if not (B >> b) & 1:
                continue
            # count generators of A that must move past this one, i.e. > b
            s *= (-1) ** bin(A >> (b + 1)).count("1")
        return s

    def mul(self, u, v):
        out = {}
        for A, ua in u.items():
            for B, vb in v.items():
                if A & B:
                    continue
                c = self.sign(A, B) * ua * vb
                if c:
                    out[A | B] = out.get(A | B, 0) + c
        return {k: c for k, c in out.items() if c}

    def add(self, u, v, t=1):
        out = dict(u)
        for k, c in v.items():
            out[k] = out.get(k, 0) + t * c
        return {k: c for k, c in out.items() if c}

    def gen(self, i):
        return {1 << i: 1}

    def thetas(self):
        return [self.gen(i) for i in range(self.m)]

    def xis(self):
        return [self.gen(self.m + a) for a in range(self.N)]

    def basis(self, parity=None, inside=None):
        """basis masks, optionally of one parity and/or inside a submask."""
        out = []
        for k in range(self.dim):
            if parity is not None and self.par(k) != parity:
                continue
            if inside is not None and (k & ~inside):
                continue
            out.append(k)
        return out

    def vec(self, u):
        v = [0] * self.dim
        for k, c in u.items():
            v[k] += c
        return v


# ------------------------------------------------------------------ the system

def derivations(G, parity=None):
    """Basis of { D in Der(B) : D(xi_a) = 0 }, as tuples (h_0, ..., h_{m-1}).

    parity = 1 : h_j odd, i.e. D preserves parity ("even" derivations)
    parity = 0 : h_j even, i.e. D reverses parity ("odd" derivations)
    parity = None : no restriction.
    """
    m, dim = G.m, G.dim
    cols = []                        # (j, mask) for every unknown coefficient
    for j in range(m):
        for k in G.basis(parity=parity):
            cols.append((j, k))
    idx = {c: i for i, c in enumerate(cols)}

    rows = []
    th, xi = G.thetas(), G.xis()

    def emit(coeffs):
        """coeffs: list of (j, monomial-multiplier-left, right) contributions.

        Each contribution is (j, uL, uR) meaning  uL * h_j * uR.
        """
        block = [[0] * len(cols) for _ in range(dim)]
        for (j, uL, uR) in coeffs:
            for k in G.basis(parity=parity):
                term = G.mul(G.mul(uL, {k: 1}), uR)
                col = idx[(j, k)]
                for msk, c in term.items():
                    block[msk][col] += c
        for r in block:
            if any(r):
                rows.append(r)

    one = {0: 1}
    # relation theta_i theta_j + theta_j theta_i = 0
    for i in range(m):
        for j in range(i, m):
            emit([(i, one, th[j]), (j, th[i], one),
                  (j, one, th[i]), (i, th[j], one)])
    # relation theta_i xi_a + xi_a theta_i = 0   (D kills xi_a)
    for i in range(m):
        for a in range(G.N):
            emit([(i, one, xi[a]), (i, xi[a], one)])

    if not rows:
        rows = [[0] * len(cols)]
    ns = nullspace(rows)
    out = []
    for v in ns:
        hs = [dict() for _ in range(m)]
        for (j, k), val in zip(cols, v):
            if val:
                hs[j][k] = val
        out.append(tuple(hs))
    return out, cols


# ------------------------------------------------------------------ the models

def vect_even(G):
    """{ sum h_j del_j : h_j odd } -- the even part of vect(0|m) (x) Lambda."""
    out = []
    for j in range(G.m):
        for k in G.basis(parity=1):
            hs = [dict() for _ in range(G.m)]
            hs[j] = {k: 1}
            out.append(tuple(hs))
    return out


def inner(G, need_kills_xi=True):
    """{ ad_g : g odd } as tuples (h_j = g theta_j - theta_j g).

    If need_kills_xi, keep only those g with ad_g(xi_a) = 0 for all a.
    """
    th, xi = G.thetas(), G.xis()
    gens = []
    for k in G.basis(parity=1):
        g = {k: 1}
        if need_kills_xi:
            # ad_g(xi_a) = g xi_a - xi_a g  must vanish
            bad = False
            for a in range(G.N):
                if G.add(G.mul(g, xi[a]), G.mul(xi[a], g), -1):
                    bad = True
                    break
            if bad:
                continue
        hs = tuple(G.add(G.mul(g, th[j]), G.mul(th[j], g), -1)
                   for j in range(G.m))
        gens.append((g, hs))
    return gens


def flat(G, hs):
    v = []
    for h in hs:
        v.extend(G.vec(h))
    return v


def dim_span(G, family):
    rows = [flat(G, hs) for hs in family]
    if not rows:
        return 0
    return rank(rows)


def contained(G, family, host):
    """is span(family) contained in span(host)?"""
    H = [flat(G, hs) for hs in host]
    r0 = rank(H) if H else 0
    for hs in family:
        if rank(H + [flat(G, hs)]) != r0:
            return False
    return True


# ------------------------------------------------------------------ checks

def ann_of_odd_lambda(G):
    """{ g odd in B : g * lambda = 0 for every odd lambda in Lambda }."""
    xi = G.xis()
    out = []
    for k in G.basis(parity=1):
        g = {k: 1}
        if all(not G.mul(g, xi[a]) for a in range(G.N)):
            out.append(g)
    return out


def report(m, N):
    G = Grass(m, N)
    all_d, _ = derivations(G)
    ev_d, _ = derivations(G, parity=1)      # h_j odd  -> parity-preserving D
    od_d, _ = derivations(G, parity=0)      # h_j even -> parity-reversing D

    V = vect_even(G)
    I = inner(G)
    dimI = dim_span(G, [hs for (_, hs) in I])

    print("  m=%d N=%d :  dim d_m = %3d = %3d (parity-preserving) + %3d (reversing)"
          % (m, N, len(all_d), len(ev_d), len(od_d)))
    print("            vect_ev = %3d  %s ;  inner = %3d  %s"
          % (dim_span(G, V),
             "OK" if dim_span(G, V) == len(ev_d) else "MISMATCH",
             dimI,
             "OK" if dimI == len(od_d) else "MISMATCH"))

    ok = True
    ok &= len(all_d) == len(ev_d) + len(od_d)
    ok &= dim_span(G, V) == len(ev_d)
    ok &= contained(G, V, ev_d) and contained(G, ev_d, V)
    ok &= dimI == len(od_d)
    ok &= contained(G, [hs for (_, hs) in I], od_d)
    ok &= contained(G, od_d, [hs for (_, hs) in I])

    # predicted dimensions
    pred_ev = m * (1 << (G.n - 1))
    ann = ann_of_odd_lambda(G)
    pred_od = len(ann) - (1 if _central(G, ann) else 0)
    print("            predicted: %3d + %3d   %s"
          % (pred_ev, pred_od,
             "OK" if (pred_ev == len(ev_d) and pred_od == len(od_d)) else "MISMATCH"))
    ok &= pred_ev == len(ev_d) and pred_od == len(od_d)
    return ok


def _central(G, ann):
    """is there a nonzero g in span(ann) with ad_g = 0 on B?"""
    th = G.thetas()
    for g in ann:
        if all(not G.add(G.mul(g, t), G.mul(t, g), -1) for t in th):
            return True
    return False


def structure_checks(m, N):
    """ad_g^2 = 0, [ad_g, ad_g'] = 0, [D, ad_g] = ad_{D g},  S(a,b) = 0."""
    G = Grass(m, N)
    I = inner(G)
    th = G.thetas()
    bad = 0

    def apply(hs, u):
        """apply the derivation with D(theta_j) = h_j to a monomial u."""
        out = {}
        for mask, c in u.items():
            bits = [b for b in range(G.n) if (mask >> b) & 1]
            for pos, b in enumerate(bits):
                if b >= G.m:
                    continue                       # D kills the xis
                left = {0: 1}
                for bb in bits[:pos]:
                    left = G.mul(left, G.gen(bb))
                right = {0: 1}
                for bb in bits[pos + 1:]:
                    right = G.mul(right, G.gen(bb))
                term = G.mul(G.mul(left, hs[b]), right)
                for k, cc in term.items():
                    out[k] = out.get(k, 0) + c * cc
        return {k: c for k, c in out.items() if c}

    def comp(hs1, hs2):
        """the derivation D1 D2 - D2 D1, given by its values on the thetas."""
        return tuple(G.add(apply(hs1, hs2[j]), apply(hs2, hs1[j]), -1)
                     for j in range(G.m))

    for (g, hs) in I:
        # ad_g^2 = 0 on every basis monomial
        for k in range(G.dim):
            u = {k: 1}
            if apply(hs, apply(hs, u)):
                bad += 1
        # [ad_g, ad_g'] = 0
        for (g2, hs2) in I:
            if any(comp(hs, hs2)):
                bad += 1
    # [D, ad_g] = ad_{D(g)} for the even vector fields
    for hsV in vect_even(G):
        for (g, hs) in I:
            lhs = comp(hsV, hs)
            Dg = apply(hsV, g)
            rhs = tuple(G.add(G.mul(Dg, th[j]), G.mul(th[j], Dg), -1)
                        for j in range(G.m))
            if any(G.add(lhs[j], rhs[j], -1) for j in range(G.m)):
                bad += 1
    print("  m=%d N=%d : ad_g^2=0, [ad_g,ad_g']=0, [D,ad_g]=ad_{Dg} : %s"
          % (m, N, "all OK" if bad == 0 else "%d FAILURES" % bad))
    return bad == 0


def automorphism_test(m, N):
    """Is  theta_j |-> theta_j (1 + g)  an automorphism of B over Lambda?

    Tested on the defining relations, for every basis g of odd parity.
    """
    G = Grass(m, N)
    th, xi = G.thetas(), G.xis()
    ok_all, ok_ann, bad_nonann = [], [], []
    # the predicate the infinitesimal computation predicts:
    #   the shift  theta_j |-> theta_j g  must annihilate every odd lambda
    ann = set()
    for k in G.basis(parity=1):
        g = {k: 1}
        if all(not G.mul(G.mul(t, g), xi[a])
               for t in th for a in range(G.N)):
            ann.add(tuple(sorted(g.items())))
    for k in G.basis(parity=1):
        g = {k: 1}
        img = [G.mul(t, G.add({0: 1}, g)) for t in th]
        good = True
        for i in range(m):
            for j in range(m):
                if G.add(G.mul(img[i], img[j]), G.mul(img[j], img[i])):
                    good = False
            for a in range(G.N):
                if G.add(G.mul(img[i], xi[a]), G.mul(xi[a], img[i])):
                    good = False
        if good:
            ok_all.append(k)
            if tuple(sorted(g.items())) in ann:
                ok_ann.append(k)
        elif tuple(sorted(g.items())) in ann:
            bad_nonann.append(k)
    print("  m=%d N=%d : of %d odd basis g, %d give an automorphism; "
          "all of them annihilate Lambda_od : %s"
          % (m, N, len(G.basis(parity=1)), len(ok_all),
             "yes" if len(ok_all) == len(ok_ann) and not bad_nonann else "NO"))
    return len(ok_all) == len(ok_ann) and not bad_nonann


def volichenko_triviality(m, N):
    """For x = ad_g in the odd part of d_m: does x ensure nontriviality?

    S(a,b) = [[a,x],[b,x]] with a, b in the even part. Since the odd part is
    abelian, S vanishes identically and h_{ad_x} = (d_m)_ev.
    """
    G = Grass(m, N)
    I = inner(G)
    V = vect_even(G)
    th = G.thetas()

    def apply(hs, u):
        out = {}
        for mask, c in u.items():
            bits = [b for b in range(G.n) if (mask >> b) & 1]
            for pos, b in enumerate(bits):
                if b >= G.m:
                    continue
                left = {0: 1}
                for bb in bits[:pos]:
                    left = G.mul(left, G.gen(bb))
                right = {0: 1}
                for bb in bits[pos + 1:]:
                    right = G.mul(right, G.gen(bb))
                for k, cc in G.mul(G.mul(left, hs[b]), right).items():
                    out[k] = out.get(k, 0) + c * cc
        return {k: c for k, c in out.items() if c}

    def comp(h1, h2):
        return tuple(G.add(apply(h1, h2[j]), apply(h2, h1[j]), -1)
                     for j in range(G.m))

    worst = 0
    for (g, x) in I:
        for a in V:
            for b in V:
                ax, bx = comp(a, x), comp(b, x)
                if any(comp(ax, bx)):
                    worst += 1
    print("  m=%d N=%d : witnesses of nontriviality among %d x %d x %d triples: %d"
          % (m, N, len(I), len(V), len(V), worst))
    return worst == 0


def main():
    print("backend:", backend())
    print()
    print("1) the structure of d_m(Lambda) = { D in Der(B) : D(Lambda) = 0 }")
    ok = True
    for (m, N) in [(1, 0), (2, 0), (3, 0), (4, 0), (1, 1), (2, 1), (3, 1),
                   (1, 2), (2, 2), (3, 2), (2, 3)]:
        ok &= report(m, N)
    print()
    print("2) the odd part is abelian and homologic")
    for (m, N) in [(2, 0), (3, 0), (2, 1), (2, 2), (3, 2)]:
        ok &= structure_checks(m, N)
    print()
    print("3) theta_j |-> theta_j(1+g) is an automorphism iff g Lambda_od = 0")
    for (m, N) in [(1, 0), (2, 0), (3, 0), (1, 1), (2, 1), (2, 2), (3, 2), (2, 3)]:
        ok &= automorphism_test(m, N)
    print()
    print("4) every Volichenko algebra h_{ad_x} inside d_m is trivial")
    for (m, N) in [(2, 0), (3, 0), (2, 1), (2, 2)]:
        ok &= volichenko_triviality(m, N)
    print()
    print("ALL CHECKS PASSED" if ok else "*** SOMETHING FAILED ***")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
