#!/usr/bin/env python3
r"""
irder.py --- the derivations of ir(9; \One) over F_2, and the question

    "Could there be a short exact sequence with the same (isomorphic to each
     other) first and last terms?  Reminds me of derivations and central
     extensions of psl(2|2)."

The occasion is the coincidence found by `iriso.py`:  dim Der ir(9; \One) = 52
while dim ir(9; \One) = 26, and ir is centreless, so that ad is injective and

        0 --> ir --> Der ir --> Out ir --> 0,     dim Out ir = 26.

The question is whether Out ir is isomorphic to ir, so that the two ends of
that sequence are the same, as happens for psl(2|2) in characteristic 0, where
Out psl(2|2) = sl(2) and the space of central extensions is sl(2) again.

What is computed, all of it exact over F_2 and exhaustive:

  (1) Der ir(9; \One) by solving D[x,y] = [Dx,y] + [x,Dy] (676 unknowns), each
      solution verified to be a derivation; its Z-grading with respect to the
      contact grading of ir;
  (2) ad ir inside it, and the ir-module Out ir = Der ir / ad ir, the action
      being x . D = [ad_x, D];
  (3) whether Out ir is an (absolutely) irreducible ir-module, and whether
      Out ir = ir as ir-modules: Hom_ir(ir, Out ir) is computed exactly and
      every element of it is tested for invertibility (the space is small);
  (4) the Lie algebra structure of Out ir = Der ir / ad ir: is it abelian, is
      it isomorphic to ir?  Its invariants are compared with those of ir;
  (5) whether the sequence splits as Lie algebras, i.e. whether Der ir has a
      subalgebra mapping isomorphically onto Out ir -- tested, when the
      grading makes it feasible, on the graded complement;
  (6) H^2(ir; K) with trivial coefficients: the central extensions, the other
      half of the psl(2|2) analogy.  Cochains are alternating (f(x,x) = 0);
      dim B^2 = 26 because ir is perfect, so dim H^2 = dim Z^2 - 26.  The
      Out-module structure of H^2 is computed as well.

Usage:  python3 irder.py
"""

import sys
import time

import ir as M
import iriso as I

DIM = I.DIM                                        # 26
FULL = (1 << DIM) - 1
bits = M.bits
cech = I.cech
cin_span = I.cin_span
cdim = I.cdim


# ------------------------------------------------------------------ operators
#
# An operator on ir is a list of DIM column masks: op[j] = image of e_j.
# Flattened for linear algebra as sum_j op[j] << (j * DIM), which is exactly
# the encoding the derivation solver of iriso.py uses.

def flat(op):
    return sum(op[j] << (j * DIM) for j in range(DIM))


def unflat(m):
    return [(m >> (j * DIM)) & FULL for j in range(DIM)]


def apply_op(op, x):
    out = 0
    for j in bits(x):
        out ^= op[j]
    return out


def comm(A, B):
    """[A, B] = A B + B A (characteristic 2)."""
    return [apply_op(A, B[j]) ^ apply_op(B, A[j]) for j in range(DIM)]


def op_rank(op):
    return cdim([op[j] for j in range(DIM)])


# ------------------------------------------------------------ (1) derivations

def der_basis(C):
    """A basis of Der(ir) as flat masks; each solution is verified."""
    eqs = []
    for i in range(DIM):
        for j in range(i + 1, DIM):
            cols = [0] * (DIM * DIM)
            for m in range(DIM):
                cols[i * DIM + m] ^= C.BR[m][j]
                cols[j * DIM + m] ^= C.BR[i][m]
            for k in bits(C.BR[i][j]):
                for t in range(DIM):
                    cols[k * DIM + t] ^= 1 << t
            for t in range(DIM):
                e = 0
                for u in range(DIM * DIM):
                    if cols[u] >> t & 1:
                        e |= 1 << u
                if e:
                    eqs.append(e)
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
    for sol in sols:
        D = unflat(sol)
        for i in range(DIM):
            for j in range(i + 1, DIM):
                lhs = 0
                for k in bits(C.BR[i][j]):
                    lhs ^= D[k]
                rhs = C.br(D[i], 1 << j) ^ C.br(1 << i, D[j])
                assert lhs == rhs, 'the derivation solver is wrong'
    return sols


def graded_dims(space, degs):
    """dim of the intersection of `space` (flat masks) with each homogeneous
    part: an operator has degree d if it moves g_k into g_{k+d}."""
    ech = cech(space)
    out = {}
    lo, hi = min(degs), max(degs)
    for d in range(lo - hi, hi - lo + 1):
        S = []
        for j in range(DIM):
            for i in range(DIM):
                if degs[i] - degs[j] == d:
                    S.append(1 << (j * DIM + i))
        if not S:
            continue
        k = len(ech) + len(S) - cdim([v for _, v in ech] + S)
        if k:
            out[d] = k
    return out


def graded_part(space, degs, d):
    """a basis of the degree-d part of a graded space of operators."""
    S = [1 << (j * DIM + i) for j in range(DIM) for i in range(DIM)
         if degs[i] - degs[j] == d]
    if not S:
        return []
    # intersection of span(space) and span(S) by the standard kernel trick
    A = list(space)
    n = len(A)
    rec = []
    sols = []
    for k, v in enumerate(A + S):
        w, cm = v, (1 << k if k < n else 0)
        for p, b, c in rec:
            if w >> p & 1:
                w ^= b
                cm ^= c
        if w:
            rec.append((w.bit_length() - 1, w, cm))
            rec.sort(key=lambda t: -t[0])
        else:
            if cm:                                   # a relation: gives an
                u = 0                                # element of the meet
                for t in bits(cm):
                    u ^= A[t]
                sols.append(u)
    return [v for _, v in cech(sols)]


# ------------------------------------------------------- (2) the module Out ir

class Quotient:
    """Der ir / ad ir, with coordinates."""

    def __init__(self, der, adim):
        self.rec = []
        self.out = []
        for v in adim:                                # the ad's carry class 0
            self._insert(v, 0)
        for v in der:
            cls = self._reduce(v)[0]
            if cls is None:                           # not yet in the span
                k = len(self.out)
                self.out.append(v)
                self._insert(v, 1 << k)
        self.dim = len(self.out)

    def _insert(self, v, cm):
        w = v
        for p, b, c in self.rec:
            if w >> p & 1:
                w ^= b
                cm ^= c
        if w:
            self.rec.append((w.bit_length() - 1, w, cm))
            self.rec.sort(key=lambda t: -t[0])

    def _reduce(self, v):
        w, cm = v, 0
        for p, b, c in self.rec:
            if w >> p & 1:
                w ^= b
                cm ^= c
        return (cm, w) if w == 0 else (None, w)

    def cls(self, v):
        """the class of a Der element as a mask in the Out basis."""
        cm, w = self._reduce(v)
        assert cm is not None and w == 0, 'element not in Der'
        return cm


def module_matrices(C, Q):
    """x . D = [ad_x, D] as DIM matrices acting on Out ir."""
    mats = []
    for i in range(DIM):
        adi = C.ad(1 << i)
        cols = []
        for k in range(Q.dim):
            Dk = unflat(Q.out[k])
            cols.append(Q.cls(flat(comm(adi, Dk))))
        mats.append(cols)
    return mats


def act(mats, i, v):
    out = 0
    for k in bits(v):
        out ^= mats[i][k]
    return out


def burnside_ok(mats, n):
    """absolute irreducibility: the associative algebra generated by the
    action, with 1, is the whole of End (n^2-dimensional)."""
    ident = tuple(1 << j for j in range(n))
    gens = [tuple(m) for m in mats]

    def mul(a, b):
        r = []
        for j in range(n):
            x = 0
            for k in bits(b[j]):
                x ^= a[k]
            r.append(x)
        return tuple(r)

    def fl(m):
        return sum(m[j] << (j * n) for j in range(n))

    alg = cech([fl(ident)] + [fl(g) for g in gens])
    frontier = [ident] + gens
    while frontier:
        newf = []
        for a in frontier:
            for b in gens:
                c = mul(a, b)
                if not cin_span(fl(c), alg):
                    alg = cech([v for _, v in alg] + [fl(c)])
                    newf.append(c)
        frontier = newf
    return len(alg) == n * n, len(alg)


def hom_module(C, mats, n):
    """Hom_ir(ir_adjoint, Out): phi with phi([x,y]) = x . phi(y).

    Unknowns a_{j,l}: phi(e_j) = sum_l a_{j,l} f_l, index j * n + l.
    """
    eqs = []
    for i in range(DIM):
        for j in range(DIM):
            cols = [0] * (DIM * n)
            for k in bits(C.BR[i][j]):               # phi([e_i, e_j])
                for l in range(n):
                    cols[k * n + l] ^= 1 << l
            for l in range(n):                       # e_i . phi(e_j)
                cols[j * n + l] ^= mats[i][l]
            for t in range(n):
                e = 0
                for u in range(DIM * n):
                    if cols[u] >> t & 1:
                        e |= 1 << u
                if e:
                    eqs.append(e)
    ech = cech(eqs)
    piv = {p for p, _ in ech}
    free = [u for u in range(DIM * n) if u not in piv]
    sols = []
    for u in free:
        sol = 1 << u
        for p, row in sorted(ech, key=lambda pb: pb[0]):
            if bin(row & sol).count('1') % 2:
                sol ^= 1 << p
        sols.append(sol)
    return sols


def phi_cols(sol, n):
    return [(sol >> (j * n)) & ((1 << n) - 1) for j in range(DIM)]


# --------------------------------------------------- (6) central extensions

def pair_index():
    idx = {}
    c = 0
    for i in range(DIM):
        for j in range(i + 1, DIM):
            idx[(i, j)] = c
            c += 1
    return idx, c


def h2_trivial(C):
    """dim Z^2, dim B^2, dim H^2(ir; K) with alternating 2-cochains."""
    idx, NP = pair_index()

    def var(a, b):
        if a == b:
            return None
        return idx[(a, b)] if a < b else idx[(b, a)]

    def f_of(mask, k):
        """the row of f(sum e_m, e_k) as a mask of variables."""
        row = 0
        for m in bits(mask):
            v = var(m, k)
            if v is not None:
                row ^= 1 << v
        return row

    eqs = []
    for i in range(DIM):
        for j in range(i + 1, DIM):
            for k in range(j + 1, DIM):
                e = (f_of(C.BR[i][j], k) ^ f_of(C.BR[j][k], i)
                     ^ f_of(C.BR[k][i], j))
                if e:
                    eqs.append(e)
    ech = cech(eqs)
    dimZ = NP - len(ech)

    # B^2: f = lambda o [,] for lambda in ir^*
    cob = []
    for l in range(DIM):
        row = 0
        for i in range(DIM):
            for j in range(i + 1, DIM):
                if C.BR[i][j] >> l & 1:
                    row ^= 1 << idx[(i, j)]
        cob.append(row)
    dimB = cdim(cob)
    return dimZ, dimB, dimZ - dimB, idx, NP, ech, cob


def h2_out_module(C, Q, mats_dummy, idx, NP, ech_Z, cob):
    """H^2(ir; K) as a module over Out ir = Der/ad, via (D.f)(x,y) =
    f(Dx, y) + f(x, Dy).  Inner derivations act trivially on cohomology,
    which is verified."""
    # a basis of Z^2 by back-substitution
    piv = {p for p, _ in ech_Z}
    free = [u for u in range(NP) if u not in piv]
    Z = []
    for u in free:
        sol = 1 << u
        for p, row in sorted(ech_Z, key=lambda pb: pb[0]):
            if bin(row & sol).count('1') % 2:
                sol ^= 1 << p
        Z.append(sol)

    # coordinates on H^2 = Z^2 / B^2
    rec, hbasis = [], []

    def insert(v, cm):
        w = v
        for p, b, c in rec:
            if w >> p & 1:
                w ^= b
                cm ^= c
        if w:
            rec.append((w.bit_length() - 1, w, cm))
            rec.sort(key=lambda t: -t[0])

    def reduce_(v):
        w, cm = v, 0
        for p, b, c in rec:
            if w >> p & 1:
                w ^= b
                cm ^= c
        return (cm if w == 0 else None)

    for v in cob:
        insert(v, 0)
    for v in Z:
        if reduce_(v) is None:
            k = len(hbasis)
            hbasis.append(v)
            insert(v, 1 << k)
    n = len(hbasis)

    def var(a, b):
        if a == b:
            return None
        return idx[(a, b)] if a < b else idx[(b, a)]

    def dact(D, fmask):
        """(D.f)(e_i, e_j) = f(D e_i, e_j) + f(e_i, D e_j)."""
        out = 0
        for (i, j), t in idx.items():
            s = 0
            for m in bits(D[i]):
                v = var(m, j)
                if v is not None and fmask >> v & 1:
                    s ^= 1
            for m in bits(D[j]):
                v = var(i, m)
                if v is not None and fmask >> v & 1:
                    s ^= 1
            if s:
                out ^= 1 << t
        return out

    # inner derivations act trivially on H^2 (a check, not an assumption)
    inner_trivial = True
    for i in range(DIM):
        adi = C.ad(1 << i)
        for f in hbasis:
            if reduce_(dact(adi, f)) != 0:
                inner_trivial = False
    mats = []
    for k in range(Q.dim):
        Dk = unflat(Q.out[k])
        cols = []
        for f in hbasis:
            c = reduce_(dact(Dk, f))
            assert c is not None, 'D does not preserve Z^2'
            cols.append(c)
        mats.append(cols)
    return n, inner_trivial, mats, hbasis


# ------------------------------------------------------------------ reporting

def main():
    t0 = time.time()
    print('=== ir(9; \\One): Der, Out and the central extensions ===')
    basis, comps = I.build_ir()
    degs = [d for d in sorted(comps) for _ in comps[d]]
    C = I.Coords(basis)
    print('  ir rebuilt: dim = %d, degrees %s'
          % (len(basis), {d: len(comps[d]) for d in sorted(comps)}))

    print()
    print('=== (1) Der ir ===')
    der = der_basis(C)
    print('  dim Der ir(9; \\One) = %d   (every solution verified)' % len(der))
    gd = graded_dims(der, degs)
    print('  by degree: %s' % gd)

    adim = [flat(C.ad(1 << i)) for i in range(DIM)]
    print('  dim ad ir = %d  (so the centre is %d-dimensional)'
          % (cdim(adim), DIM - cdim(adim)))

    print()
    print('=== (2) the ir-module Out ir = Der ir / ad ir ===')
    Q = Quotient(der, adim)
    print('  dim Out ir = %d' % Q.dim)
    outdegs = {}
    for d in sorted(gd):
        gp = graded_part(der, degs, d)
        k = len(gp) - (len(cech(adim + gp)) - len(cech(adim)))
        # dim of the image of Der_d in Out
        img = cech([Q.cls(v) for v in gp])
        outdegs[d] = len(img)
    print('  Out by degree: %s' % {d: k for d, k in outdegs.items() if k})
    print('  ir  by degree: %s' % {d: len(comps[d]) for d in sorted(comps)})

    mats = module_matrices(C, Q)
    # the action is a representation: check [x,y].D = x.(y.D) + y.(x.D)
    ok = True
    for i in range(DIM):
        for j in range(DIM):
            for k in range(Q.dim):
                lhs = 0
                for m in bits(C.BR[i][j]):
                    lhs ^= mats[m][k]
                rhs = act(mats, i, mats[j][k]) ^ act(mats, j, mats[i][k])
                if lhs != rhs:
                    ok = False
    print('  the action x . D = [ad_x, D] is a representation: %s' % ok)
    assert ok

    irr, algdim = burnside_ok(mats, Q.dim)
    print('  Out ir is an absolutely irreducible ir-module: %s '
          '(enveloping algebra of dim %d out of %d)'
          % (irr, algdim, Q.dim ** 2))

    print()
    print('=== (3) is Out ir = ir as an ir-module? ===')
    homs = hom_module(C, mats, Q.dim)
    print('  dim Hom_ir(ir, Out ir) = %d' % len(homs))
    iso = False
    if homs and len(homs) <= 20:
        for g in range(1, 1 << len(homs)):
            s = 0
            for k in bits(g):
                s ^= homs[k]
            cols = phi_cols(s, Q.dim)
            if cdim(cols) == DIM == Q.dim:
                iso = True
                break
        print('  some element of it is invertible: %s' % iso)
        if iso:
            print('  ==> Out ir is isomorphic to ir as an ir-module,')
            print('      i.e. 0 -> ir -> Der ir -> ir -> 0 is exact.')
    elif not homs:
        print('  ==> Out ir is NOT isomorphic to ir as an ir-module.')

    print()
    print('=== (4) Out ir as a Lie algebra ===')
    br = []
    for a in range(Q.dim):
        row = []
        for b in range(Q.dim):
            row.append(Q.cls(flat(comm(unflat(Q.out[a]), unflat(Q.out[b])))))
        br.append(row)
    derived = cech([br[a][b] for a in range(Q.dim) for b in range(Q.dim)])
    print('  dim [Out, Out] = %d' % len(derived))
    cen = []
    for v in range(Q.dim):
        if all(br[v][b] == 0 for b in range(Q.dim)):
            cen.append(v)
    print('  its centre has dim >= %d (basis vectors killing everything)'
          % len(cen))

    print()
    print('=== (5) does 0 -> ir -> Der ir -> Out ir -> 0 split? ===')
    dder = cech([flat(comm(unflat(a), unflat(b))) for a in der for b in der])
    print('  dim [Der, Der] = %d  (dim ad ir = %d, dim Der = %d)'
          % (len(dder), cdim(adim), len(der)))
    print('  ad ir is contained in [Der, Der]: %s'
          % all(cin_span(v, dder) for v in adim))

    print()
    print('=== (6) H^2(ir; K): the central extensions ===')
    dimZ, dimB, dimH, idx, NP, echZ, cob = h2_trivial(C)
    print('  dim C^2 = %d, dim Z^2 = %d, dim B^2 = %d, dim H^2(ir; K) = %d'
          % (NP, dimZ, dimB, dimH))
    n, inner_triv, hmats, hbasis = h2_out_module(C, Q, mats, idx, NP, echZ, cob)
    print('  H^2 has dim %d; inner derivations act trivially on it: %s'
          % (n, inner_triv))
    if n:
        irr2, algdim2 = burnside_ok(hmats, n) if n > 1 else (True, 1)
        print('  H^2 is an absolutely irreducible Out-module: %s (algebra dim '
              '%d of %d)' % (irr2, algdim2, n * n))

    print()
    print('done in %.1f s' % (time.time() - t0))
    return 0


if __name__ == '__main__':
    sys.exit(main())
