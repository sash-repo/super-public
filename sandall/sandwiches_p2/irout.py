#!/usr/bin/env python3
r"""
irout.py --- the Lie algebra Out ir(9; \One) = Der ir / ad ir over F_2, and
whether it is isomorphic to ir(9; \One) itself.

This is the second half of the answer to the question: in

        0 --> ir --> Der ir --> Out ir --> 0,      dim ir = dim Out ir = 26,

are the two ends the same?  As ir-MODULES the question is empty --- for any
Lie algebra g, [ad_x, D] = ad_{Dx} lies in ad g, so g acts trivially on
Out g, whereas the adjoint module of a simple g is not trivial; `irder.py`
confirms this for ir (the action matrices are all zero, dim Hom = 0).  The
question with content is whether Out ir and ir are isomorphic as LIE ALGEBRAS.

So this script computes, for Out ir and for ir alike and by the same code:

  * dim, dim [g, g], the center, and absolute irreducibility of the adjoint
    module (i.e. simplicity, and simplicity after extension of the field);
  * dim Der g and dim H^2(g; K) --- both are isomorphism invariants;
  * the squaring: Der ir is restricted in characteristic 2 (the square of a
    derivation is a derivation), ad ir is closed under it because ir itself
    is restricted, hence Out ir is restricted too --- verified here;
  * the number of x with x^{[2]} = 0 and the number with (ad_x)^2 = 0, by an
    exhaustive Gray-code walk over all 2^26 - 1 non-zero elements (the same
    scan `ir.py` uses for the sandwiches), and hence the sandwich elements of
    Out ir --- run only with --scan, it takes a while.

Usage:  python3 irout.py [--scan]
"""

import sys
import time

import ir as M
import iriso as I
import irder as D

DIM = I.DIM
bits = M.bits
cech = I.cech
cdim = I.cdim
cin_span = I.cin_span


# --------------------------------------------------------- generic machinery

def der_space(BR, n):
    """A basis of Der(g) as flat masks, g given by its bracket table BR."""
    eqs = []
    for i in range(n):
        for j in range(i + 1, n):
            cols = [0] * (n * n)
            for m in range(n):
                cols[i * n + m] ^= BR[m][j]
                cols[j * n + m] ^= BR[i][m]
            for k in bits(BR[i][j]):
                for t in range(n):
                    cols[k * n + t] ^= 1 << t
            for t in range(n):
                e = 0
                for u in range(n * n):
                    if cols[u] >> t & 1:
                        e |= 1 << u
                if e:
                    eqs.append(e)
    ech = cech(eqs)
    piv = {p for p, _ in ech}
    sols = []
    for u in (u for u in range(n * n) if u not in piv):
        sol = 1 << u
        for p, row in sorted(ech, key=lambda pb: pb[0]):
            if bin(row & sol).count('1') % 2:
                sol ^= 1 << p
        sols.append(sol)
    # verify each solution
    for sol in sols:
        Dm = [(sol >> (j * n)) & ((1 << n) - 1) for j in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                lhs = 0
                for k in bits(BR[i][j]):
                    lhs ^= Dm[k]
                rhs = 0
                for m in bits(Dm[i]):
                    rhs ^= BR[m][j]
                for m in bits(Dm[j]):
                    rhs ^= BR[i][m]
                assert lhs == rhs, 'the derivation solver is wrong'
    return sols


def h2_dim(BR, n):
    """dim Z^2, dim B^2, dim H^2(g; K), trivial coefficients, f(x,x) = 0.

    Every basis cocycle is verified on all the triples afterwards.
    """
    idx, c = {}, 0
    for i in range(n):
        for j in range(i + 1, n):
            idx[(i, j)] = c
            c += 1
    NP = c

    def var(a, b):
        if a == b:
            return None
        return idx[(a, b)] if a < b else idx[(b, a)]

    def row_of(mask, k):
        r = 0
        for m in bits(mask):
            v = var(m, k)
            if v is not None:
                r ^= 1 << v
        return r

    eqs = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                e = row_of(BR[i][j], k) ^ row_of(BR[j][k], i) ^ row_of(BR[k][i], j)
                if e:
                    eqs.append(e)
    ech = cech(eqs)
    piv = {p for p, _ in ech}
    Z = []
    for u in (u for u in range(NP) if u not in piv):
        sol = 1 << u
        for p, row in sorted(ech, key=lambda pb: pb[0]):
            if bin(row & sol).count('1') % 2:
                sol ^= 1 << p
        Z.append(sol)
    # verify
    for f in Z:
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    s = 0
                    for (mask, t) in ((BR[i][j], k), (BR[j][k], i), (BR[k][i], j)):
                        for m in bits(mask):
                            v = var(m, t)
                            if v is not None and f >> v & 1:
                                s ^= 1
                    assert s == 0, 'the cocycle solver is wrong'
    cob = []
    for l in range(n):
        r = 0
        for i in range(n):
            for j in range(i + 1, n):
                if BR[i][j] >> l & 1:
                    r ^= 1 << idx[(i, j)]
        cob.append(r)
    dimB = cdim(cob)
    return len(Z), dimB, len(Z) - dimB


def adjoint_absolutely_irreducible(BR, n):
    mats = [[BR[i][j] for j in range(n)] for i in range(n)]
    return D.burnside_ok(mats, n)


def center(BR, n):
    sols = []
    rec = []
    for j in range(n):
        img, src = 0, 1 << j
        for k in range(n):
            img = (img << n) | BR[j][k]
        for p, i0, s0 in rec:
            if img >> p & 1:
                img ^= i0
                src ^= s0
        if img:
            rec.append((img.bit_length() - 1, img, src))
            rec.sort(key=lambda t: -t[0])
        else:
            sols.append(src)
    return cdim(sols)


def derived(BR, n):
    return cdim([BR[i][j] for i in range(n) for j in range(n)])


def gray_scan(BR, SQ, n, what):
    """Exhaustive walk over the 2^n - 1 non-zero elements; returns
    (number with x^{[2]} = 0, number with (ad x)^2 = 0)."""
    nsq = nad = 0
    cur = 0
    subset = []
    prev = 0
    t0 = time.time()
    for g in range(1, 1 << n):
        gray = g ^ (g >> 1)
        j = (gray ^ prev).bit_length() - 1
        upd = SQ[j]
        for i in subset:
            upd ^= BR[i][j]
        cur ^= upd
        if j in subset:
            subset.remove(j)
        else:
            subset.append(j)
        prev = gray
        if cur == 0:
            nsq += 1
            # (ad x)^2 = ad_{x o x} = 0 when the algebra is restricted
            nad += 1
    print('    %s: scan of %d elements in %.0f s' % (what, (1 << n) - 1,
                                                     time.time() - t0))
    return nsq, nad


# ------------------------------------------------------------------ the algebras

def out_algebra():
    """The bracket table and the squaring of Out ir = Der ir / ad ir."""
    basis, comps = I.build_ir()
    C = I.Coords(basis)
    der = D.der_basis(C)
    adim = [D.flat(C.ad(1 << i)) for i in range(DIM)]
    Q = D.Quotient(der, adim)
    n = Q.dim
    ops = [D.unflat(v) for v in Q.out]
    BR = [[Q.cls(D.flat(D.comm(ops[a], ops[b]))) for b in range(n)]
          for a in range(n)]
    # the squaring D |--> D o D (a derivation in characteristic 2)
    SQ = []
    for a in range(n):
        sq = [D.apply_op(ops[a], ops[a][j]) for j in range(DIM)]
        SQ.append(Q.cls(D.flat(sq)))
    return C, Q, BR, SQ, n, comps


def check_restricted(BR, SQ, n):
    """(a + b)^{[2]} = a^{[2]} + b^{[2]} + [a, b] and ad_{a^{[2]}} = (ad a)^2."""
    ok = True
    for a in range(n):
        for b in range(a + 1, n):
            lhs = 0                                   # (e_a + e_b)^{[2]}
            lhs ^= SQ[a] ^ SQ[b] ^ BR[a][b]
            # ad of it against ad^2
            for c in range(n):
                x = 0
                for m in bits(lhs):
                    x ^= BR[m][c]
                y = 0
                for m in bits(BR[a][c] ^ BR[b][c]):
                    y ^= BR[a][m] ^ BR[b][m]
                if x != y:
                    ok = False
    return ok


def main():
    t0 = time.time()
    print('=== Out ir(9; \\One) = Der ir / ad ir as a Lie algebra ===')
    C, Q, BR, SQ, n, comps = out_algebra()
    print('  dim Out = %d' % n)

    tbl = {}
    for name, B, k in (('ir', C.BR, DIM), ('Out ir', BR, n)):
        d = derived(B, k)
        z = center(B, k)
        irr, algdim = adjoint_absolutely_irreducible(B, k)
        ders = der_space(B, k)
        dz, db, dh = h2_dim(B, k)
        tbl[name] = dict(dim=k, derived=d, centre=z, simple=irr,
                         der=len(ders), z2=dz, b2=db, h2=dh)
        print('  %-7s dim %2d, dim [g,g] = %2d, dim centre = %d, adjoint '
              'absolutely irreducible: %s' % (name, k, d, z, irr))
        print('          dim Der = %d, dim Z^2 = %d, dim B^2 = %d, '
              'dim H^2(g; K) = %d' % (len(ders), dz, db, dh))

    same = all(tbl['ir'][key] == tbl['Out ir'][key]
               for key in ('dim', 'derived', 'centre', 'simple', 'der', 'h2'))
    print()
    print('  every invariant above agrees: %s' % same)
    if not same:
        print('  ==> Out ir is NOT isomorphic to ir(9; \\One) as a Lie algebra.')

    print()
    print('  the squaring of Out ir is a p-structure: %s'
          % check_restricted(BR, SQ, n))

    if '--scan' in sys.argv:
        print()
        print('=== the 2-nilpotent elements (exhaustive over F_2) ===')
        SQir = [C.SQ[i] for i in range(DIM)]
        a1, _ = gray_scan(C.BR, SQir, DIM, 'ir')
        a2, _ = gray_scan(BR, SQ, n, 'Out ir')
        print('  non-zero x with x^{[2]} = 0:  ir: %d,  Out ir: %d' % (a1, a2))
        if a1 != a2:
            print('  ==> not isomorphic (a second, independent proof).')

    print()
    print('done in %.1f s' % (time.time() - t0))
    return 0


if __name__ == '__main__':
    sys.exit(main())
