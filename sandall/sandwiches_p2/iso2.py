#!/usr/bin/env python3
r"""
iso2.py --- an isomorphism search for restricted Lie algebras over F_2, seeded
by the root decomposition with respect to a maximal torus (`torus2.py`).

The question it was written for:

        is  Out ir(9; \One) = Der ir / ad ir  isomorphic to  ir(9; \One)?

Both are simple restricted Lie algebras of dimension 26 over F_2, and every
invariant computed so far agrees.  Brute force over triples of generators is
out of reach (2^26 elements, no small invariant class to seed with); the root
decomposition cuts the search down to size:

  * a maximal torus of either algebra is 2-dimensional and self-centralizing,
    so L = T + L_a + L_b + L_c with dim L_a = dim L_b = dim L_c = 8, a grading
    by the group F_2^2 = Hom(T, F_2);
  * an isomorphism carrying one maximal torus onto another is determined by a
    bijection of the three non-zero characters and by its values on a basis of
    L_a and of L_b, since [L_a, L_b] = L_c and [L_a, L_a] + [L_b, L_b] = T;
  * so the search is a depth-first assignment of images to those 16 vectors,
    each candidate list being (a subset of) a 256-element weight space, with
    PROPAGATION: as soon as phi is known on u and v it is known on [u, v] and
    on u^{[2]}, and every bracket among the assigned vectors must be
    consistent with it.  A contradiction prunes the branch at once.

Exhibiting one isomorphism is a proof; the script verifies the map it finds on
all 26 x 26 brackets and all 26 squares before reporting it.

Usage:  python3 iso2.py [--pairs] [--seconds N]
"""

import pickle
import random
import sys
import time

import lin2
import torus2
from lin2 import bits, echelon
from torus2 import Alg

random.seed(20260817)


# --------------------------------------------------------- the partial map

class Partial:
    """A partial linear map L1 -> L2, closed under propagation."""

    __slots__ = ('rec', 'A1', 'A2', 'img')

    def __init__(self, A1, A2):
        self.A1, self.A2 = A1, A2
        self.rec = []            # (pivot, src, img), decreasing pivots
        self.img = []            # echelon basis of the image so far

    def copy(self):
        p = Partial(self.A1, self.A2)
        p.rec = list(self.rec)
        p.img = list(self.img)
        return p

    def value(self, v):
        """(image, True) if phi(v) is already determined, else (rest, False)."""
        img = 0
        for p, s, i in self.rec:
            if v >> p & 1:
                v ^= s
                img ^= i
        return (img, True) if v == 0 else (v, False)

    def add(self, src, im):
        """Force phi(src) = im.  False on a contradiction."""
        for p, s, i in self.rec:
            if src >> p & 1:
                src ^= s
                im ^= i
        if src == 0:
            return im == 0                      # consistent?
        if im == 0:
            return False                        # would not be injective
        # injectivity: the new image must be outside the image span
        w = im
        for b in self.img:
            if w >> (b.bit_length() - 1) & 1:
                w ^= b
        if w == 0:
            return False
        self.img.append(w)
        self.img.sort(key=lambda u: -u.bit_length())
        self.rec.append((src.bit_length() - 1, src, im))
        self.rec.sort(key=lambda t: -t[0])
        return True

    def propagate(self):
        """Close under [phi u, phi v] = phi[u, v] and phi(u)^{[2]} = phi(u^{[2]}).

        Returns False as soon as an inconsistency shows up.
        """
        A1, A2 = self.A1, self.A2
        done = 0
        while True:
            cur = [(s, i) for _, s, i in self.rec]
            if len(cur) == done:
                break
            new = []
            for a in range(len(cur)):
                sa, ia = cur[a]
                if not self.add(A1.sq(sa), A2.sq(ia)):
                    return False
                for b in range(a + 1, len(cur)):
                    sb, ib = cur[b]
                    if not self.add(A1.br(sa, sb), A2.br(ia, ib)):
                        return False
            if len(self.rec) == len(cur):
                break
            done = len(cur)
        return True

    def dim(self):
        return len(self.rec)

    def matrix(self):
        """phi as a list of images of e_0, ..., e_{n-1} (only when total)."""
        out = []
        for k in range(self.A1.n):
            v, ok = self.value(1 << k)
            assert ok, 'the map is not total'
            out.append(v)
        return out


def verify(A1, A2, phi):
    """phi given as images of the basis vectors: a full check."""
    def ap(v):
        out = 0
        for i in bits(v):
            out ^= phi[i]
        return out
    if len(echelon(phi)) != A1.n:
        return False
    for i in range(A1.n):
        if ap(A1.SQ[i]) != A2.sq(phi[i]):
            return False
        for j in range(A1.n):
            if ap(A1.BR[i][j]) != A2.br(phi[i], phi[j]):
                return False
    return True


# ------------------------------------------------------------- the search

def inv_mat(M):
    """The inverse of an r x r matrix over F_2, or None if singular."""
    r = len(M)
    aug = [((sum((M[i][j] << j) for j in range(r))) | (1 << (r + i)))
           for i in range(r)]
    piv = {}
    for row in aug:
        v = row
        while v & ((1 << r) - 1):
            p = (v & ((1 << r) - 1)).bit_length() - 1
            if p in piv:
                v ^= piv[p]
            else:
                piv[p] = v
                break
        else:
            if v:
                return None
    if len(piv) != r:
        return None
    for p in sorted(piv):
        for q in sorted(piv):
            if q != p and piv[q] >> p & 1:
                piv[q] ^= piv[p]
    out = [[0] * r for _ in range(r)]
    for p, row in piv.items():
        for j in range(r):
            out[p][j] = (row >> (r + j)) & 1
    return out


def weight_image(M, chi, Mi=None):
    """The weight of phi(x) for x of weight chi:  chi' = M^{-1} chi."""
    r = len(M)
    if Mi is None:
        Mi = inv_mat(M)
    out = 0
    for k in range(r):
        s = 0
        for j in range(r):
            if Mi[k][j] and (chi >> j) & 1:
                s ^= 1
        if s:
            out |= 1 << k
    return out


def graded_search_M(A1, A2, T1, T2, M, budget=1e9, verbose=True):
    """Search for an isomorphism whose restriction to the torus T1 is the map
    phi(T1[j]) = sum_k M[j][k] T2[k].

    T1, T2 are r-element lists spanning the tori; a weight is an r-bit integer,
    bit j being the eigenvalue on T[j].
    """
    r = len(T1)
    assert len(T2) == r
    W1, W2 = torus2.weights(A1, T1), torus2.weights(A2, T2)
    Mi = inv_mat(M)
    if Mi is None:
        return None, 0
    chis = list(range(1, 1 << r))
    chimap = {chi: weight_image(M, chi, Mi) for chi in chis}
    for chi in chis:
        if len(W1.get(chi, [])) != len(W2.get(chimap[chi], [])):
            return None, 0
    base = Partial(A1, A2)
    for j in range(r):
        v = 0
        for k in range(r):
            if M[j][k]:
                v ^= T2[k]
        if not base.add(T1[j], v):
            return None, 0
    if not base.propagate():
        return None, 0

    # the source vectors to be assigned, and their candidate sets
    order = []
    for chi in chis + [0]:
        for v in W1.get(chi, []):
            order.append((v, chimap[chi] if chi else 0))
    cand = {}
    for chi in chis + [0]:
        cand[chi] = [v for v in span_nonzero(W2.get(chi, []))]

    nodes = [0]
    t0 = time.time()

    def dfs(pm, k):
        nodes[0] += 1
        if nodes[0] > budget or time.time() - t0 > 900:
            raise TimeoutError
        if pm.dim() == A1.n:
            return pm
        if k >= len(order):
            return None
        src, chi2 = order[k]
        got, known = pm.value(src)
        if known:
            return dfs(pm, k + 1)
        # invariants the image must share with the source
        r1 = A1.rank_ad(src)
        sq1, sqknown = pm.value(A1.sq(src))
        for h in cand[chi2]:
            if A2.rank_ad(h) != r1:
                continue
            if sqknown and A2.sq(h) != sq1:
                continue
            nx = pm.copy()
            if not nx.add(src, h):
                continue
            if not nx.propagate():
                continue
            out = dfs(nx, k + 1)
            if out is not None:
                return out
        return None

    try:
        res = dfs(base, 0)
    except TimeoutError:
        if verbose:
            print('     (timed out after %d nodes)' % nodes[0])
        return None, nodes[0]
    return res, nodes[0]


def graded_search(A1, A2, T1, T2, perm, budget=1e9, verbose=True):
    """As `graded_search_M`, with the torus map given by a bijection of the
    three non-zero characters."""
    return graded_search_M(A1, A2, T1, T2, torus_map(perm),
                           budget=budget, verbose=verbose)


def torus_map(perm):
    """The 2x2 matrix over F_2 carrying the character bijection `perm` of
    {1, 2, 3} = F_2^2 \\ 0 to a map of the tori (the transpose-inverse)."""
    # phi: T1 -> T2; chi' o phi = chi with perm(chi) = chi'.  In the bases
    # used, chi is the bit pattern; so the matrix of phi is the transpose of
    # the inverse of the matrix P with P(chi) = perm(chi).
    P = [[0, 0], [0, 0]]
    # perm is linear: determined by the images of 1 = (1,0) and 2 = (0,1)
    c1, c2 = perm[1], perm[2]
    P[0][0], P[0][1] = c1 & 1, (c1 >> 1) & 1
    P[1][0], P[1][1] = c2 & 1, (c2 >> 1) & 1
    # invert P over F_2 (2x2)
    det = (P[0][0] * P[1][1] + P[0][1] * P[1][0]) % 2
    assert det == 1, 'the character bijection is not linear'
    inv = [[P[1][1], P[0][1]], [P[1][0], P[0][0]]]
    # transpose
    return [[inv[0][0], inv[1][0]], [inv[0][1], inv[1][1]]]


def span_nonzero(basis):
    out = []
    for k in range(1, 1 << len(basis)):
        v = 0
        for i in bits(k):
            v ^= basis[i]
        out.append(v)
    return out


def linear_perms():
    """The six linear bijections of F_2^2 \\ 0, as dicts on {1, 2, 3}."""
    out = []
    for c1 in (1, 2, 3):
        for c2 in (1, 2, 3):
            if c2 == c1:
                continue
            out.append({1: c1, 2: c2, 3: c1 ^ c2})
    return out


def main():
    with open('algs.pkl', 'rb') as f:
        d = pickle.load(f)
    L1 = Alg(d['BR1'], d['SQ1'], 'ir(9;One)')
    L2 = Alg(d['BR2'], d['SQ2'], 'Out ir(9;One)')
    print('=== is Out ir(9;\\One) isomorphic to ir(9;\\One)? ===')
    t0 = time.time()
    T1 = torus2.maximal_torus(L1)
    T2 = torus2.maximal_torus(L2)
    print('  tori of dim %d and %d' % (len(T1), len(T2)))
    for perm in linear_perms():
        print('  trying the character bijection %s' % perm)
        phi, nodes = graded_search(L1, L2, T1, T2, perm)
        print('     %d nodes, %.1f s' % (nodes, time.time() - t0))
        if phi is not None:
            mat = phi.matrix()
            ok = verify(L1, L2, mat)
            print('  ISOMORPHISM FOUND, verified on all brackets: %s' % ok)
            print('  images of the basis:', mat)
            with open('iso-found.pkl', 'wb') as f:
                pickle.dump(mat, f)
            return
    print('  no isomorphism with these two tori')


if __name__ == '__main__':
    main()
