#!/usr/bin/env python3
r"""
iriso.py --- how many of the superizations of ir(9; \One) are isomorphic?

This settles the question of the graded superizations: the Theorem on the admissible parity functions
gives 16 admissible parity functions on x_1, ..., x_9, falling into four
families

   1)  p = 0                                  ir(9; \un)      sdim (26|0)
   2)  6 functions, x_1 even, 2 odd pairs     ir(5; \un|4)    sdim (14|12)
   3)  1 function, x_1 even, all pairs odd    ir(1; \un|8)    sdim (10|16)
   4)  8 functions, x_1 odd                   ir(4; \un|5)    sdim (14|12)

and the question is how many pairwise non-isomorphic Lie superalgebras the six
of family 2), resp. the eight of family 4), actually are.

Two superizations gr(p), gr(p') of the SAME Lie algebra ir are isomorphic as
Lie superalgebras precisely when some phi in Aut(ir) carries the Z/2-grading
gr(p) to gr(p').  So the script works from both ends:

  (A) UPPER bound on the number of classes -- explicit automorphisms.  Every
      permutation sigma of x_2, ..., x_9 fixing x_1 induces a substitution
      automorphism phi_sigma of O(9; \One), hence of vect(9; \One); phi_sigma
      is an automorphism of ir iff it maps ir into ir.  All 8! = 40320 of them
      are tested.  Since the parity of phi_sigma(v) with respect to p is the
      parity of v with respect to p . sigma, such a sigma gives an isomorphism
      gr(p . sigma) --> gr(p) of superalgebras (it preserves the squaring too,
      being induced by a substitution).  The orbits on the 16 admissible p are
      computed.

  (B) LOWER bound -- isomorphism invariants of the superalgebra, computed for
      each of the 16: the sdim, dim [g_od, g_od], dim [g_ev, g_ev], dim of the
      center of g_ev, the number of x in g_od with x o x = 0 (the homological
      elements: an invariant, the squaring being part of the structure), the
      number of x in g_ev with (ad x)^2 = 0, and whether g_od is an absolutely
      irreducible g_ev-module (Burnside).  Two superizations with different
      invariants are non-isomorphic ALSO as abstract (non-graded)
      superalgebras.

  (C) as a by-product, dim Der(ir(9; \One)): whether ir has outer derivations.

Everything after the construction of ir is done in COORDINATES with respect to
a fixed basis of the 26-dimensional space: the bracket is a table BR[i][j] of
26-bit masks and the squaring is the quadratic map polarizing to it,
  (x+y) o (x+y) = x o x + y o y + [x, y],
so that both are evaluated by XORing precomputed masks.

Usage:  python3 iriso.py
"""

import itertools
import sys
import time

import ir as M

NVAR = M.NVAR
PAIRS = M.PAIRS                                    # (2,9) (3,8) (4,7) (5,6)
DIM = 26


# ------------------------------------------------------------------- build ir

def build_ir():
    part = M.prolong(M.G0T)
    idl, uniq, _ = M.minimal_ideal(M.flatten(part))
    assert uniq and len(idl) == DIM, 'ir(9; \\One) not recovered'
    comps = {d: M.echelon([v for _, v in part[d] if M.in_span(v, idl)])
             for d in sorted(part)}
    # a graded basis, so that the degrees are readable off the coordinates
    basis = [v for d in sorted(comps) for _, v in comps[d]]
    assert len(M.echelon(basis)) == DIM
    return basis, comps


class Coords:
    """Coordinates with respect to a fixed basis of ir, and the tables."""

    def __init__(self, basis):
        self.basis = basis
        self.ech = M.echelon(basis)
        # express the echelon basis in terms of `basis`
        rec = []
        for j, u in enumerate(basis):
            v, cm = u, 1 << j
            for p, b, c in rec:
                if v >> p & 1:
                    v ^= b
                    cm ^= c
            if v:
                rec.append((v.bit_length() - 1, v, cm))
                rec.sort(key=lambda t: -t[0])
        self.rec = rec
        self.BR = [[self.co(M.bracket(basis[i], basis[j])) for j in range(DIM)]
                   for i in range(DIM)]
        self.SQ = [self.co(M.compose_square(basis[i])) for i in range(DIM)]
        # ad e_i as a 26-list of column masks
        self.AD = [[self.BR[i][j] for j in range(DIM)] for i in range(DIM)]

    def co(self, v):
        """coordinates of a vector field lying in ir, as a 26-bit mask."""
        cm = 0
        for p, b, c in self.rec:
            if v >> p & 1:
                v ^= b
                cm ^= c
        if v:
            raise ValueError('vector not in ir')
        return cm

    def br(self, x, y):
        out = 0
        for i in M.bits(x):
            row = self.BR[i]
            for j in M.bits(y):
                out ^= row[j]
        return out

    def sq(self, x):
        """x o x, by polarization: sum of the squares plus the brackets."""
        out = 0
        bs = list(M.bits(x))
        for k, i in enumerate(bs):
            out ^= self.SQ[i]
            row = self.BR[i]
            for j in bs[k + 1:]:
                out ^= row[j]
        return out

    def ad(self, x):
        """ad x as a list of 26 column masks."""
        cols = [0] * DIM
        for i in M.bits(x):
            row = self.BR[i]
            for j in range(DIM):
                cols[j] ^= row[j]
        return cols

    @staticmethod
    def applym(cols, y):
        out = 0
        for j in M.bits(y):
            out ^= cols[j]
        return out


def cech(vectors):
    """echelon form of a list of coordinate masks."""
    basis = []
    for v in vectors:
        for p, b in basis:
            if v >> p & 1:
                v ^= b
        if v:
            basis.append((v.bit_length() - 1, v))
            basis.sort(key=lambda pb: -pb[0])
    return basis


def cdim(vectors):
    return len(cech(vectors))


def cin_span(v, basis):
    for p, b in basis:
        if v >> p & 1:
            v ^= b
    return v == 0


def subspace(basis):
    """all 2^dim elements of the subspace spanned by `basis` (Gray order),
    yielded as (element, index of the bit toggled) -- index -1 for the first."""
    n = len(basis)
    v = 0
    yield v, -1
    for g in range(1, 1 << n):
        k = (g & -g).bit_length() - 1
        v ^= basis[k]
        yield v, k


# ------------------------------------------------- (A) substitution automorphisms

def apply_perm(v, perm):
    """phi_sigma(v): the substitution x_i -> x_{sigma(i)} on a vector field.

    perm[j] = image index of x_{j+1}, 0-based; a term x^mask d_j goes to
    x^{sigma(mask)} d_{sigma(j)}.  Over F_2 there are no signs, and the masks
    are squarefree, so the image mask is the permuted bit set.
    """
    out = 0
    for i in M.bits(v):
        mask, j = divmod(i, NVAR)
        nm = 0
        for b in M.bits(mask):
            nm |= 1 << perm[b]
        out ^= 1 << M.IDX(nm, perm[j])
    return out


def perm_automorphisms(ir):
    """All permutations of x_2..x_9 (x_1 fixed) inducing an automorphism of ir."""
    ech = M.echelon(ir)
    good = []
    for tail in itertools.permutations(range(1, NVAR)):
        perm = (0,) + tail
        if all(M.in_span(apply_perm(v, perm), ech) for v in ir):
            good.append(perm)
    return good


# --------------------------------------------------- the contact substitutions
#
# A bare permutation of x_2, ..., x_9 need NOT preserve the contact form
#   alpha = dx_1 + sum_{2<=i<=5} x_i dx_{11-i} :
# if it flips the two members of a pair, alpha changes by the exact form
# d(x_i x_{11-i}), and the flip has to be compensated by a shift of x_1, just
# as for the mirrors tau_S of Subsection 2.14.  So the substitutions to test
# are, for a permutation pi of x_2, ..., x_9 preserving the SET of the four
# hyperbolic pairs,
#
#   phi_pi:   x_1 |--> x_1 + Phi_T,   x_i |--> x_{pi(i)}   (i >= 2),
#   Phi_T = sum_{k in T} x_{a_k} x_{b_k},   T = sigma(the flipped pairs),
#
# sigma being the permutation of the four pairs induced by pi.  There are
# 4! * 2^4 = 384 of them, and phi_pi^{-1} is phi with pi^{-1} and the shift by
# Phi_F, F = the flipped set itself.  Note Phi_T^2 = 0, so phi_pi is a
# well-defined automorphism of the exterior algebra O(9; \One).

def pmul(a, b):
    """product of two polynomials, each a 512-bit set of squarefree monomials."""
    out = 0
    for i in M.bits(a):
        for j in M.bits(b):
            if not (i & j):
                out ^= 1 << (i | j)
    return out


def ppow(polys):
    out = 1                                     # the monomial 1 = mask 0
    for q in polys:
        out = pmul(out, q)
    return out


def pdiff(a, j):
    """d/dx_{j+1} of a polynomial over F_2 (no signs, squarefree monomials)."""
    out = 0
    for m in M.bits(a):
        if m >> j & 1:
            out ^= 1 << (m ^ (1 << j))
    return out


class Subst:
    """The substitution automorphism phi_pi of vect(9; \\One)."""

    def __init__(self, perm, shift_T, inv_perm, inv_shift_T):
        self.perm = perm
        self.img = self._images(perm, shift_T)
        inv = self._images(inv_perm, inv_shift_T)
        # c[j][k] = phi( d_j ( phi^{-1}(x_k) ) )
        self.c = [[self.apply_poly(pdiff(inv[k], j)) for k in range(NVAR)]
                  for j in range(NVAR)]

    @staticmethod
    def _images(perm, T):
        img = [1 << (1 << perm[j]) for j in range(NVAR)]     # x_j -> x_{perm(j)}
        phi_T = 0
        for k in T:
            a, b = PAIRS[k]
            phi_T ^= 1 << ((1 << (a - 1)) | (1 << (b - 1)))
        img[0] = (1 << (1 << perm[0])) ^ phi_T
        return img

    def apply_poly(self, a):
        out = 0
        for m in M.bits(a):
            out ^= ppow([self.img[b] for b in M.bits(m)])
        return out

    def apply_vf(self, v):
        """phi(D) = phi . D . phi^{-1} for a vector field D."""
        out = 0
        for i in M.bits(v):
            mask, j = divmod(i, NVAR)
            fm = self.apply_poly(1 << mask)
            for k in range(NVAR):
                cf = self.c[j][k]
                if cf:
                    for m in M.bits(pmul(fm, cf)):
                        out ^= 1 << M.IDX(m, k)
        return out


def contact_substitutions():
    """The 4! * 2^4 = 384 contact substitutions phi_pi, as Subst objects."""
    out = []
    for sigma in itertools.permutations(range(4)):
        for flips in itertools.product((0, 1), repeat=4):
            perm = [0] * NVAR
            for k in range(4):
                a, b = PAIRS[k]
                A, B = PAIRS[sigma[k]]
                if flips[k]:
                    A, B = B, A
                perm[a - 1] = A - 1
                perm[b - 1] = B - 1
            perm = tuple(perm)
            F = [k for k in range(4) if flips[k]]
            T = [sigma[k] for k in F]
            iperm = [0] * NVAR
            for j in range(NVAR):
                iperm[perm[j]] = j
            out.append((sigma, flips,
                        Subst(perm, T, tuple(iperm), F)))
    return out


def perm_on_pairs(perm):
    """The permutation of the four hyperbolic pairs induced by perm (or None),
    together with the tuple of flips (which pairs get their members swapped)."""
    idx = {}
    for k, (a, b) in enumerate(PAIRS):
        idx[a - 1] = idx[b - 1] = k
    img, flips = [], []
    for k, (a, b) in enumerate(PAIRS):
        ia, ib = perm[a - 1], perm[b - 1]
        if idx[ia] != idx[ib]:
            return None, None
        img.append(idx[ia])
        flips.append(1 if ia == PAIRS[idx[ia]][1] - 1 else 0)
    return tuple(img), tuple(flips)


# ------------------------------------------------------ the admissible parities

def admissible(ir):
    out = []
    for p in itertools.product((0, 1), repeat=NVAR):
        sp = M.split_by_parity(ir, p)
        if sp is None:
            continue
        ev, od = sp
        evb = M.echelon(ev)
        if all(M.in_span(M.compose_square(x), evb) for x in od):
            out.append((p, ev, od))
    return out


def family(p):
    """1..4 as in Theorem 2.3.4."""
    if not any(p):
        return 1
    if p[0] == 1:
        return 4
    npairs = sum(1 for (a, b) in PAIRS if p[a - 1] == 1)
    return 3 if npairs == 4 else 2


def label(p):
    """which pairs are odd (families 2, 3), resp. which member of each pair is
    odd (family 4)."""
    S = [k + 1 for k, (a, b) in enumerate(PAIRS) if p[a - 1] == 1]
    if p[0] == 0:
        return 'pairs ' + ''.join(map(str, S)) if S else '(none)'
    return 'S=' + (''.join(map(str, S)) or '0')


# ------------------------------------------------------------- (B) invariants

SWEEP_MAX = 16          # do not sweep a subspace of dim > 16 element by element


def invariants(C, ev, od):
    """Isomorphism invariants of the superalgebra (ev, od), in coordinates.

    The counting invariants are swept exhaustively; for p = 0, where g_ev is
    all of ir, the sweep would be over 2^26 elements and is skipped (that case
    is the Lie algebra ir itself, which is not one of the superizations whose
    isomorphism classes are at issue).  A skipped count is reported as -1.
    """
    ev = [C.co(v) for v in ev]
    od = [C.co(v) for v in od]
    oo = cdim([C.br(a, b) for a in od for b in od] + [C.sq(a) for a in od])
    ee = cdim([C.br(a, b) for a in ev for b in ev])
    if len(ev) <= SWEEP_MAX:
        zev = sum(1 for x, _ in subspace(ev)
                  if x and all(C.br(x, b) == 0 for b in ev))
        nil = 0
        for x, _ in subspace(ev):
            if not x:
                continue
            cols = C.ad(x)
            if all(C.applym(cols, cols[j]) == 0 for j in range(DIM)):
                nil += 1
    else:
        zev = nil = -1
    hom = (sum(1 for x, _ in subspace(od) if x and C.sq(x) == 0)
           if len(od) <= SWEEP_MAX else -1)
    irr = burnside(C, ev, od) if od else False
    ds = ds_census(C, ev, od) if od and len(od) <= SWEEP_MAX else {}
    sub = submodule_dims(C, ev, od) if od else ()
    return {'sdim': (len(ev), len(od)), 'oo': oo, 'ee': ee, 'z_ev': zev,
            'hom': hom, 'nil_ev': nil, 'od_irr': irr, 'ds': tuple(sorted(ds.items())),
            'sub': sub}


def meet_dim(A, B):
    """dim (span A) cap (span B) = dim A + dim B - dim (A + B)."""
    return cdim(A) + cdim(B) - cdim(list(A) + list(B))


def ds_census(C, ev, od):
    """The multiset of sdim of the Duflo--Serganova homology
    g_x = Ker(ad x)/Im(ad x) over the homological elements x (x odd, x o x = 0).

    An isomorphism of superalgebras carries homological elements to homological
    elements and their homologies to one another, so the multiset is an
    invariant.  Both Ker(ad x) and Im(ad x) are homogeneous (x being odd), so
    the sdim is obtained by intersecting each with g_ev and with g_od.
    """
    census = {}
    for x, _ in subspace(od):
        if not x or C.sq(x) != 0:
            continue
        cols = C.ad(x)
        im = [v for _, v in cech(cols)]
        ker, piv = [], []
        for j in range(DIM):
            img, src = cols[j], 1 << j
            for p, i0, s0 in piv:
                if img >> p & 1:
                    img ^= i0
                    src ^= s0
            if img:
                piv.append((img.bit_length() - 1, img, src))
                piv.sort(key=lambda t: -t[0])
            else:
                ker.append(src)
        e = meet_dim(ker, ev) - meet_dim(im, ev)
        o = meet_dim(ker, od) - meet_dim(im, od)
        census[(e, o)] = census.get((e, o), 0) + 1
    return census


def submodule_dims(C, ev, od):
    """The set of dims of the g_ev-submodules of g_od generated by a single
    element -- an invariant finer than absolute irreducibility.

    Done inside g_od: the action of the basis of g_ev is a list of matrices on
    the n-dimensional coordinate space of g_od.
    """
    n = len(od)
    rec = []
    for j, u in enumerate(od):
        v, cm = u, 1 << j
        for p, b, c in rec:
            if v >> p & 1:
                v ^= b
                cm ^= c
        if v:
            rec.append((v.bit_length() - 1, v, cm))
            rec.sort(key=lambda t: -t[0])

    def coord(v):
        cm = 0
        for p, b, c in rec:
            if v >> p & 1:
                v ^= b
                cm ^= c
        return None if v else cm

    mats = []
    for g in ev:
        col = [coord(C.br(g, u)) for u in od]
        if any(c is None for c in col):
            return ()                       # g_od is not a g_ev-submodule
        mats.append(col)

    def act(m, x):
        r = 0
        for j in M.bits(x):
            r ^= m[j]
        return r

    dims = set()
    for x in range(1, 1 << n):
        sub, front = cech([x]), [x]
        while front:
            new = []
            for v in front:
                for m in mats:
                    w = act(m, v)
                    if w and not cin_span(w, sub):
                        sub = cech([u for _, u in sub] + [w])
                        new.append(w)
            front = new
        dims.add(len(sub))
    return tuple(sorted(dims))


def burnside(C, ev, od):
    """Is `od` an absolutely irreducible ev-module?  Burnside: the associative
    algebra generated by the action is the whole of End(od)."""
    n = len(od)
    rec = []
    for j, u in enumerate(od):
        v, cm = u, 1 << j
        for p, b, c in rec:
            if v >> p & 1:
                v ^= b
                cm ^= c
        if v:
            rec.append((v.bit_length() - 1, v, cm))
            rec.sort(key=lambda t: -t[0])

    def coord(v):
        cm = 0
        for p, b, c in rec:
            if v >> p & 1:
                v ^= b
                cm ^= c
        return None if v else cm

    mats = []
    for g in ev:
        m = tuple(coord(C.br(g, u)) for u in od)
        if any(x is None for x in m):
            return False
        mats.append(m)

    def mul(a, b):
        return tuple(_xor(a, b[j]) for j in range(n))

    def _xor(a, mask):
        r = 0
        for k in M.bits(mask):
            r ^= a[k]
        return r

    ident = tuple(1 << j for j in range(n))
    flat = lambda m: sum(m[j] << (j * n) for j in range(n))     # noqa: E731
    alg = cech([flat(m) for m in [ident] + mats])
    frontier = [ident] + mats
    while frontier:
        newf = []
        for a in frontier:
            for b in mats:
                c = mul(a, b)
                if not cin_span(flat(c), alg):
                    alg = cech([v for _, v in alg] + [flat(c)])
                    newf.append(c)
        frontier = newf
    return len(alg) == n * n


# ------------------------------------------------------------ (C) derivations

def der_dim(C):
    r"""dim Der(ir) over F_2: solve D[e_i, e_j] = [D e_i, e_j] + [e_i, D e_j].

    Unknowns a_{j,i} (D e_j = sum_i a_{j,i} e_i), indexed by j * DIM + i.
    For each pair i < j the equation reads, coordinate t by coordinate t,
        sum_{k in [e_i,e_j]} a_{k,t}
      + sum_m a_{i,m} [e_m, e_j]_t + sum_m a_{j,m} [e_i, e_m]_t  =  0.
    """
    eqs = []
    for i in range(DIM):
        for j in range(i + 1, DIM):
            cols = [0] * (DIM * DIM)          # unknown index -> mask of t's
            for m in range(DIM):
                cols[i * DIM + m] ^= C.BR[m][j]
                cols[j * DIM + m] ^= C.BR[i][m]
            for k in M.bits(C.BR[i][j]):
                for t in range(DIM):
                    cols[k * DIM + t] ^= 1 << t
            for t in range(DIM):
                e = 0
                for u in range(DIM * DIM):
                    if cols[u] >> t & 1:
                        e |= 1 << u
                if e:
                    eqs.append(e)

    # a basis of the solution space, by back-substitution on the echelon form
    ech = cech(eqs)
    piv = {p for p, _ in ech}
    free = [u for u in range(DIM * DIM) if u not in piv]
    sols = []
    for u in free:
        sol = 1 << u
        for p, row in sorted(ech, key=lambda pb: pb[0]):
            if bin(row & sol).count('1') % 2:
                sol ^= 1 << p
        sols.append(sol)
    # verify: each solution really is a derivation
    for sol in sols:
        D = [0] * DIM
        for u in M.bits(sol):
            j, i = divmod(u, DIM)
            D[j] ^= 1 << i
        for i in range(DIM):
            for j in range(i + 1, DIM):
                lhs = 0
                for k in M.bits(C.BR[i][j]):
                    lhs ^= D[k]
                rhs = C.br(D[i], 1 << j) ^ C.br(1 << i, D[j])
                if lhs != rhs:
                    raise AssertionError('the derivation solver is wrong')
    assert len(sols) == DIM * DIM - len(ech)
    return len(sols)


# ------------------------------------------------------------------- reporting

def main():
    t0 = time.time()
    print('=== ir(9; \\One): the isomorphism classes of its superizations ===')
    basis, comps = build_ir()
    print('  ir(9; \\One) rebuilt: dim = %d, by degree %s'
          % (len(basis), M.gdims(comps)))
    C = Coords(basis)
    # sanity: the tables reproduce the vector-field bracket and squaring
    ok = all(C.co(M.bracket(basis[i], basis[j])) == C.BR[i][j]
             for i in range(DIM) for j in range(DIM))
    ok = ok and all(C.co(M.compose_square(basis[i])) == C.SQ[i]
                    for i in range(DIM))
    x = (1 << 3) | (1 << 7) | (1 << 20)
    ok = ok and C.sq(x) == C.co(M.compose_square(
        basis[3] ^ basis[7] ^ basis[20]))
    print('  the coordinate tables agree with the vector fields: %s' % ok)
    assert ok

    adm = admissible([v for v in basis])
    print('  admissible parity functions: %d' % len(adm))
    fam = {}
    for p, ev, od in adm:
        fam.setdefault(family(p), []).append(p)
    for f in sorted(fam):
        print('    family %d: %d function(s)' % (f, len(fam[f])))

    print()
    print('=== (A) substitution automorphisms of ir(9; \\One) ===')
    print('  testing all 8! = 40320 permutations of x_2, ..., x_9 ...')
    good = perm_automorphisms(basis)
    print('  automorphisms found: %d' % len(good))
    onpairs = {}
    for perm in good:
        sig, flips = perm_on_pairs(perm)
        onpairs.setdefault((sig, flips), []).append(perm)
    nopair = sum(1 for perm in good if perm_on_pairs(perm)[0] is None)
    print('  permutations not preserving the set of hyperbolic pairs: %d'
          % nopair)
    print('  action on the four hyperbolic pairs (sigma; flipped pairs):')
    for (sig, flips), ps in sorted(onpairs.items(),
                                   key=lambda kv: (kv[0][0], kv[0][1])):
        print('    sigma = %s   flips = %s   (%d permutation(s))'
              % (str(sig), str(flips), len(ps)))
    sigs = {sig for (sig, flips) in onpairs if sig is not None}
    print('  distinct permutations of the four pairs induced: %d of 24'
          % len(sigs))
    print('  the total flip x_i <-> x_{11-i} is induced: %s'
          % any(flips == (1, 1, 1, 1) for (sig, flips) in onpairs))

    print()
    print('  the 384 contact substitutions phi_pi (mirrors included) ...')
    subs = contact_substitutions()
    ech = M.echelon(basis)
    auts = []
    for sigma, flips, S in subs:
        img = [S.apply_vf(v) for v in basis]
        if all(M.in_span(w, ech) for w in img):
            # a self-check: phi must preserve the bracket
            ok = all(S.apply_vf(M.bracket(basis[i], basis[j]))
                     == M.bracket(img[i], img[j])
                     for i in range(0, DIM, 5) for j in range(0, DIM, 3))
            assert ok, 'phi is not a homomorphism -- the substitution is wrong'
            auts.append((sigma, flips, S, img))
    print('  contact substitutions preserving ir: %d of 384' % len(auts))
    bysig = {}
    for sigma, flips, S, img in auts:
        bysig.setdefault(sigma, []).append(flips)
    for sigma in sorted(bysig):
        print('    sigma = %s   flipped sets: %s'
              % (str(sigma), sorted(bysig[sigma])))
    print('  permutations of the four pairs realized: %d of 24' % len(bysig))
    tot = [f for _, f, _, _ in auts if f == (1, 1, 1, 1)]
    print('  the total mirror x_i <-> x_{11-i} (with x_1 -> x_1 + Phi) is an '
          'automorphism: %s' % bool(tot))

    # is the set of 48 a subgroup?  the 384 substitutions form the group
    # S_4 |x (Z/2)^4 (the shift being forced by alpha), so it is enough to see
    # that the underlying permutations of x_2, ..., x_9 compose inside the set
    perms = {S.perm for _, _, S, _ in auts}
    closed = True
    for a in perms:
        for b in perms:
            if tuple(a[b[j]] for j in range(NVAR)) not in perms:
                closed = False
    print('  the %d substitutions are closed under composition (a subgroup): %s'
          % (len(perms), closed))
    # the closed-form rule read off the census
    def rule(sigma, flips):
        inv0 = [k for k in range(4) if sigma[k] == 0][0]
        F = {0, inv0} - {None}
        F = {0, inv0} if inv0 != 0 else set()
        got = {k for k in range(4) if flips[k]}
        return got in (F, {0, 1, 2, 3} - F)
    ok_rule = (all(rule(sigma, flips) for sigma, flips, _, _ in auts)
               and len(auts) == 48)
    print('  RULE: phi is an automorphism iff the mirror set is '
          '{pair 1, sigma^{-1}(pair 1)} or its complement: %s' % ok_rule)

    print()
    print('  orbits on the 16 admissible parity functions')
    print('  (computed by comparing the SUBSPACES phi(ev_p), phi(od_p) with '
          'ev_q, od_q --')
    print('   no assumption that phi acts on the parity functions):')
    sp_of = {}
    for p, ev, od in adm:
        sp_of[p] = (M.echelon(ev), M.echelon(od))

    def image_of(p, S):
        ev, od = sp_of[p]
        return ([S.apply_vf(v) for _, v in ev], [S.apply_vf(v) for _, v in od])

    edges = {}
    for p, _, _ in adm:
        for sigma, flips, S, _ in auts:
            iev, iod = image_of(p, S)
            for q, _, _ in adm:
                qev, qod = sp_of[q]
                if (all(M.in_span(v, qev) for v in iev)
                        and all(M.in_span(v, qod) for v in iod)):
                    edges.setdefault(p, set()).add(q)
                    break
    admp = [p for p, _, _ in adm]
    seen, orbits = set(), []
    for p in admp:
        if p in seen:
            continue
        orb, front = set(), [p]
        while front:
            q = front.pop()
            if q in orb:
                continue
            orb.add(q)
            front.extend(edges.get(q, ()))
        seen |= orb
        orbits.append(sorted(orb))
    for orb in orbits:
        print('    family %d: orbit of size %-2d  {%s}'
              % (family(orb[0]), len(orb),
                 ', '.join(label(q) for q in orb)))
    per_family = {}
    for orb in orbits:
        per_family.setdefault(family(orb[0]), []).append(len(orb))
    print('  orbit sizes by family: %s'
          % {f: sorted(v) for f, v in sorted(per_family.items())})

    print()
    print('=== (B) isomorphism invariants of the 16 superizations ===')
    print('  %-3s %-10s %-8s %-7s %-7s %-9s %-10s %-9s %-12s %s'
          % ('fam', 'parities', 'sdim', '[od,od]', '[ev,ev]',
             'x o x = 0', '(ad x)^2=0', 'od irred', 'submod dims',
             'DS homology sdims'))
    inv = {}
    for p, ev, od in adm:
        d = invariants(C, ev, od)
        inv[p] = d
        ds = ', '.join('(%d|%d):%d' % (a, b, c) for (a, b), c in d['ds'])
        print('  %-3d %-10s %-8s %-7d %-7d %-9d %-10d %-9s %-12s %s'
              % (family(p), label(p), '(%d|%d)' % d['sdim'], d['oo'], d['ee'],
                 d['hom'], d['nil_ev'], d['od_irr'],
                 ','.join(map(str, d['sub'])) or '-', ds or '-'))
        sys.stdout.flush()

    print()
    sig_of = lambda p: tuple(sorted(inv[p].items()))          # noqa: E731
    for f in sorted(fam):
        print('  family %d: %d parity function(s), %d distinct invariant '
              'signature(s)'
              % (f, len(fam[f]), len({sig_of(p) for p in fam[f]})))
    f2 = {sig_of(p) for p in fam.get(2, [])}
    f4 = {sig_of(p) for p in fam.get(4, [])}
    print('  families 2) and 4) share an invariant signature: %s' % bool(f2 & f4))
    print('  (both are of sdim (14|12); a shared signature means the '
          'invariants do NOT separate them)')

    print()
    print('=== (C) derivations ===')
    dd = der_dim(C)
    print('  dim Der(ir(9; \\One)) = %d;  ad is injective (center 0), so '
          'dim Out(ir) = %d' % (dd, dd - DIM))

    print()
    print('  elapsed %.1f s' % (time.time() - t0))


if __name__ == '__main__':
    main()
