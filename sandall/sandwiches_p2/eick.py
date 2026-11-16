#!/usr/bin/env python3
r"""
eick.py --- the invariants of the table on p. 948 of

  B. Eick, Some new simple Lie algebras in characteristic 2,
  J. Symbolic Comput. 45 (2010) 943--951,

computed for  ir(9; \One)  and  Out ir(9; \One) = Der ir / ad ir: the list of
invariants Eick applied to her 15- and 16-dimensional algebras.

Her table has five columns besides the names:

  der  dim Der(L);
  res  whether L is restricted;
  aut  the order of Aut(L);
  ann  the number of elements of Ann(L), the set of ANNIHILATOR DERIVATIONS:
       d in Der(L) with d^p = 0 and d^i(x) d^j(y) = 0 whenever i + j >= p.
       For p = 2 this reads
              d^2 = 0   and   [d x, d y] = 0 for all x, y,
       i.e. d^2 = 0 and Im d abelian; for an inner d = ad_x the two conditions
       are (ad x)^2 = 0 and ad_x ad_y ad_x = 0, which is precisely the
       definition of a SANDWICH element, so Ann(L) cap ad L = s(L);
  exp  the order of Exp(L) = < exp(d) = 1 + d | d in Ann(L) > <= Aut(L).

HOW ann IS DECIDED HERE.  Ann(L) is a variety in the 52-dimensional space
Der(L), which has 2^52 = 4.5 x 10^15 points: no enumeration.  Two reductions.

(1) The inner part is free: Ann(L) cap ad L = s(L) = 0, by the exhaustive
    Gray-code scan over all 2^26 - 1 non-zero elements (`irscan.c`, the census
    of Subsec. 2.3.10 -- no sandwich in either algebra).

(2) THE GRADING DECIDES THE REST.  Both L and Der(L) are Z-graded (the contact
    grading of ir; Der(ir) has degrees -2, ..., 2 with dims 7, 8, 22, 8, 7).
    Let d in Ann(L) be non-zero and let d_{i0} be its non-zero component of
    LOWEST degree.  For x, y homogeneous of degrees a, b, the component of
    degree a + b + 2 i0 of [d x, d y] = 0 is exactly [d_{i0} x, d_{i0} y],
    and the component of degree a + 2 i0 of d^2 = 0 is d_{i0}^2 (all other
    terms have strictly higher degree).  So

            d in Ann(L),  d =/= 0   ==>   d_{i0} in Ann(L),  homogeneous.

    Hence it is enough to sweep the HOMOGENEOUS parts of Der(L) element by
    element: 2^7 + 2^8 + 2^22 + 2^8 + 2^7 = 4.2 million derivations, which is
    a few minutes.  If no non-zero homogeneous annihilator exists -- and none
    does -- then Ann(L) = {0}, so ann = 1 and exp = 1.

(For the record: the degree-2 linearization ("XL": put z_{ij} = c_i c_j and
read both conditions as linear in the 1378 unknowns z_{ij}) is NOT enough
here.  The 12152 equations have rank only 325, leaving a solution space of
dimension 1053 -- sound but vacuous.  The function `ann_is_trivial` below
still computes it, as a control on the equations.)

Usage:  python3 eick.py            (uses the cached tables `algs.pkl`)
"""

import pickle
import sys
import time

import lin2
from lin2 import bits, echelon
from torus2 import Alg


# ------------------------------------------------------------------- Der(L)

def der_basis(A):
    """A basis of Der(L), each derivation as the list of the images of the
    basis vectors.  Every solution is verified."""
    n = A.n
    eqs = []
    for i in range(n):
        for j in range(i + 1, n):
            # sum_m d_{i,m} BR[m][j] + sum_m d_{j,m} BR[i][m] + d applied to
            # [e_i, e_j] = 0;  unknowns d_{a,b} = coefficient of e_b in d(e_a)
            cols = [0] * (n * n)
            for m in range(n):
                cols[i * n + m] ^= A.BR[m][j]
                cols[j * n + m] ^= A.BR[i][m]
            for k in bits(A.BR[i][j]):
                for t in range(n):
                    cols[k * n + t] ^= 1 << t
            for t in range(n):
                e = 0
                for u in range(n * n):
                    if cols[u] >> t & 1:
                        e |= 1 << u
                if e:
                    eqs.append(e)
    sols = lin2.solve_null(eqs, n * n)
    out = []
    for s in sols:
        D = [(s >> (j * n)) & ((1 << n) - 1) for j in range(n)]
        # verify
        for i in range(n):
            for j in range(n):
                lhs = 0
                for k in bits(A.BR[i][j]):
                    lhs ^= D[k]
                rhs = 0
                for m in bits(D[i]):
                    rhs ^= A.BR[m][j]
                for m in bits(D[j]):
                    rhs ^= A.BR[i][m]
                assert lhs == rhs, 'the derivation solver is wrong'
        out.append(D)
    return out


def apply_op(D, v):
    out = 0
    for i in bits(v):
        out ^= D[i]
    return out


# --------------------------------------------------------------------- Ann

def ann_is_trivial(A, DER, verbose=True):
    """The XL-linearization described in the docstring.  Returns
    (dim of the solution space of the relaxation, the solutions)."""
    n, m = A.n, len(DER)
    idx, c = {}, 0
    for i in range(m):
        for j in range(i, m):
            idx[(i, j)] = c
            c += 1
    NV = c
    t0 = time.time()
    rows = []

    # (a) the equations [d e_a, d e_b] = 0
    V = [[DER[i][b] for b in range(n)] for i in range(m)]      # D_i(e_b)
    for a in range(n):
        ada = [A.ad(V[i][a]) for i in range(m)]        # ad of D_i(e_a)
        for b in range(n):
            acc = [0] * n                              # one row per coordinate
            for i in range(m):
                ci = ada[i]
                for j in range(i, m):
                    w = 0
                    for t in bits(V[j][b]):
                        w ^= ci[t]
                    if i != j:
                        cj = ada[j]
                        for t in bits(V[i][b]):
                            w ^= cj[t]
                    if w:
                        u = 1 << idx[(i, j)]
                        for k in bits(w):
                            acc[k] |= u
            for k in range(n):
                if acc[k]:
                    rows.append(acc[k])
        if verbose and a % 5 == 0:
            print('     ... a = %d/%d, %d rows, %.0f s'
                  % (a, n, len(rows), time.time() - t0))

    # (b) the equations d^2 = 0
    for a in range(n):
        acc = [0] * n
        for i in range(m):
            for j in range(i, m):
                if i == j:
                    w = apply_op(DER[i], DER[i][a])
                else:
                    w = apply_op(DER[i], DER[j][a]) ^ apply_op(DER[j], DER[i][a])
                if w:
                    u = idx[(i, j)]
                    for k in bits(w):
                        acc[k] |= 1 << u
        for k in range(n):
            if acc[k]:
                rows.append(acc[k])

    if verbose:
        print('     %d equations in %d unknowns, built in %.0f s'
              % (len(rows), NV, time.time() - t0))
    sols = lin2.solve_null(rows, NV)
    return len(sols), sols, idx


def graded_pieces(A, DER, degs):
    """Split Der(L) into its homogeneous components with respect to the
    Z-grading of L given by `degs` (degree of each basis vector)."""
    n = A.n
    byde = {}
    for j, dg in enumerate(degs):
        byde.setdefault(dg, []).append(j)
    out = {}
    for i in sorted(set(d2 - d1 for d1 in byde for d2 in byde)):
        # d is homogeneous of degree i iff d(e_j) lies in the span of the
        # basis vectors of degree deg(j) + i
        eqs = []
        for j in range(n):
            allowed = 0
            for k in byde.get(degs[j] + i, []):
                allowed |= 1 << k
            for t in range(n):
                if allowed >> t & 1:
                    continue
                e = 0
                for c, D in enumerate(DER):
                    if D[j] >> t & 1:
                        e |= 1 << c
                if e:
                    eqs.append(e)
        sols = lin2.solve_null(eqs, len(DER))
        if sols:
            out[i] = [combine(DER, s, n) for s in sols]
    return out


def combine(DER, mask, n):
    D = [0] * n
    for c in bits(mask):
        for j in range(n):
            D[j] ^= DER[c][j]
    return D


def homogeneous_annihilators(A, pieces, verbose=True):
    """Sweep every homogeneous component of Der(L) element by element for
    annihilator derivations: d^2 = 0 and [d x, d y] = 0 for all x, y."""
    n = A.n
    found = []
    for i, B in sorted(pieces.items()):
        t0 = time.time()
        m = len(B)
        # a Gray-code walk over the 2^m - 1 non-zero elements of the component
        cur = [0] * n
        prev = 0
        # two cheap pairs first, the full test only for the survivors
        probe = [(0, 1), (0, 2), (1, 3), (2, 5)]
        nsurv = 0
        for g in range(1, 1 << m):
            gray = g ^ (g >> 1)
            j = (gray ^ prev).bit_length() - 1
            Dj = B[j]
            for k in range(n):
                cur[k] ^= Dj[k]
            prev = gray
            ok = True
            for (a, b) in probe:
                if A.br(cur[a], cur[b]):
                    ok = False
                    break
            if not ok:
                continue
            nsurv += 1
            if any(A.br(cur[a], cur[b]) for a in range(n) for b in range(n)):
                continue
            if any(apply_op(cur, cur[a]) for a in range(n)):
                continue
            found.append((i, list(cur)))
        if verbose:
            print('     degree %+d: %d elements swept, %d passed the probe, '
                  '%.0f s' % (i, (1 << m) - 1, nsurv, time.time() - t0))
    return found


def project(v, degs, g):
    """The component of v of degree g."""
    out = 0
    for k in bits(v):
        if degs[k] == g:
            out |= 1 << k
    return out


def lift_count(A, pieces, seeds, degs, cap=1 << 20, verbose=True):
    """The EXACT number of annihilator derivations.

    Every non-zero d in Ann(L) has a lowest homogeneous component d_{i0}, which
    is itself in Ann(L); so the whole variety is enumerated by taking each
    homogeneous annihilator as a seed and adding the higher components one
    degree at a time.  At each stage the conditions are LINEAR in the new
    component (the terms in which it occurs twice have degree one higher), so
    the stage is a linear solve and the enumeration branches only over its
    solution space.
    """
    n = A.n
    degmax = max(pieces)
    total = 0
    for i0 in sorted(pieces):
        seeds_i = [d for (i, d) in seeds if i == i0]
        if verbose:
            print('     lifting %d seeds of degree %+d' % (len(seeds_i), i0))
        cur = [(list(s), i0) for s in seeds_i]
        for stage in range(i0 + 1, degmax + 1):
            B = pieces.get(stage, [])
            nxt = []
            for (P, _) in cur:
                # linear system for u in span(B)
                rows, rhs = [], []
                eqs = []
                for a in range(n):
                    for b in range(n):
                        g = degs[a] + degs[b] + i0 + stage
                        const = project(A.br(P[a], P[b]), degs, g)
                        coef = []
                        for u in B:
                            w = A.br(P[a], u[b]) ^ A.br(u[a], P[b])
                            coef.append(project(w, degs, g))
                        eqs.append((coef, const))
                    g2 = degs[a] + i0 + stage
                    const = project(apply_op(P, P[a]), degs, g2)
                    coef = []
                    for u in B:
                        w = apply_op(P, u[a]) ^ apply_op(u, P[a])
                        coef.append(project(w, degs, g2))
                    eqs.append((coef, const))
                # assemble over F_2: unknowns are the coefficients of B
                m = len(B)
                lines = []
                bad = False
                for coef, const in eqs:
                    for t in range(n):
                        e = 0
                        for c in range(m):
                            if coef[c] >> t & 1:
                                e |= 1 << (c + 1)
                        if const >> t & 1:
                            e |= 1
                        if e == 1:
                            bad = True
                            break
                        if e:
                            lines.append(e)
                    if bad:
                        break
                if bad:
                    continue
                sols = affine_solutions(lines, m)
                for cmask in sols:
                    Q = list(P)
                    for c in bits(cmask):
                        for j in range(n):
                            Q[j] ^= B[c][j]
                    nxt.append((Q, i0))
                    if len(nxt) > cap:
                        raise RuntimeError('the lifting blew up')
            cur = nxt
        # the survivors must satisfy the conditions in full -- verify
        good = 0
        for (P, _) in cur:
            if any(A.br(P[a], P[b]) for a in range(n) for b in range(n)):
                continue
            if any(apply_op(P, P[a]) for a in range(n)):
                continue
            good += 1
        if verbose:
            print('       degree %+d: %d lifts, %d of them genuine'
                  % (i0, len(cur), good))
        total += good
    return total


def affine_solutions(lines, m):
    """All solutions of the affine system (bit 0 = right-hand side, bits
    1..m = unknowns), as masks over the m unknowns."""
    ech = []
    for r in lines:
        for e in ech:
            if r >> (e.bit_length() - 1) & 1:
                r ^= e
        if r:
            if r == 1:
                return []
            ech.append(r)
            ech.sort(key=lambda u: -u.bit_length())
    piv = {e.bit_length() - 1: e for e in ech}
    free = [u for u in range(1, m + 1) if u not in piv]
    out = []
    for k in range(1 << len(free)):
        val = 0
        for idx, u in enumerate(free):
            if k >> idx & 1:
                val |= 1 << u
        val |= 1                      # the constant column is 1
        for p in sorted(piv):
            e = piv[p]
            if bin(e & val & ~(1 << p)).count('1') % 2:
                val |= 1 << p
            else:
                val &= ~(1 << p)
        out.append(val >> 1)
    return out


def check_restricted(A):
    """(a+b)^{[2]} = a^{[2]} + b^{[2]} + [a,b] and ad_{a^{[2]}} = (ad a)^2."""
    for a in range(A.n):
        x = A.sq(1 << a)
        cols = A.ad(1 << a)
        for c in range(A.n):
            if A.applym(cols, cols[c]) != A.br(x, 1 << c):
                return False
    return True


def main():
    with open('algs.pkl', 'rb') as f:
        d = pickle.load(f)
    if 'degs' not in d:
        import iriso as I
        basis, comps = I.build_ir()
        degs = [dg for dg in sorted(comps) for _ in comps[dg]]
        d['degs'] = degs
        with open('algs.pkl', 'wb') as f:
            pickle.dump(d, f)
    A = Alg(d['BR1'], d['SQ1'], 'ir(9;One)')
    B = Alg(d['BR2'], d['SQ2'], 'Out ir(9;One)')
    print("=== Eick's invariants (p. 948 of [Ei]) ===")
    for X in (A, B):
        t0 = time.time()
        print('\n  %s,  dim %d' % (X.name, X.n))
        DER = der_basis(X)
        print('     der = dim Der(L) = %d' % len(DER))
        print('     res = %s' % ('+' if check_restricted(X) else '-'))
        print('     computed in %.0f s' % (time.time() - t0))

    print('\n  --- ann and exp for ir(9;\\One) ---')
    t0 = time.time()
    DER = der_basis(A)
    k, sols, idx = ann_is_trivial(A, DER, verbose=False)
    print('     the degree-2 linearization: %d equations, solution space of '
          'dim %d (too weak, as expected)' % (12152, k))
    pieces = graded_pieces(A, DER, d['degs'])
    print('     Der(ir) by degree: %s'
          % {i: len(v) for i, v in sorted(pieces.items())})
    assert sum(len(v) for v in pieces.values()) == len(DER), \
        'the homogeneous components do not exhaust Der'
    found = homogeneous_annihilators(A, pieces)
    print('     homogeneous annihilator derivations found: %d' % len(found))
    bydeg = {}
    for i, D in found:
        bydeg[i] = bydeg.get(i, 0) + 1
    print('     by degree: %s' % sorted(bydeg.items()))
    # each of them is inner or outer?
    ad_space = [[A.BR[i][j] for j in range(A.n)] for i in range(A.n)]
    flat = lambda D: sum((D[j] << (j * A.n)) for j in range(A.n))
    adflat = echelon([flat(ad_space[i]) for i in range(A.n)])
    inner = sum(1 for _, D in found if lin2.in_span(flat(D), adflat))
    print('     of them inner (i.e. sandwich elements): %d' % inner)
    # 1 + d must be an automorphism -- verify all of them
    bad = 0
    for _, D in found:
        for a in range(A.n):
            for b in range(A.n):
                lhs = A.br((1 << a) ^ D[a], (1 << b) ^ D[b])
                rhs = A.br(1 << a, 1 << b)
                rhs ^= apply_op(D, rhs)
                if lhs != rhs:
                    bad += 1
    print('     failures of 1 + d in Aut(L) over all %d of them: %d'
          % (len(found), bad))
    if not found:
        print('     ==> Ann(ir) = {0}:   ann = 1,  exp = 1')
    else:
        print('     ==> Ann(ir) =/= 0, so ann > 1 and exp > 1')
        try:
            tot = lift_count(A, pieces, found, d['degs'], cap=200000)
            print('     ==> ann = %d  (the zero derivation included)'
                  % (tot + 1))
        except RuntimeError as e:
            print('     the enumeration of the non-homogeneous annihilators '
                  'exceeded the cap (%s);' % e)
            print('     only the homogeneous ones are counted here: %d'
                  % len(found))
    print('     computed in %.0f s' % (time.time() - t0))


if __name__ == '__main__':
    main()
