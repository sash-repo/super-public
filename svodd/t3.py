#!/usr/bin/env python3
"""
enum.py -- T_3: the COMPLETE enumeration of the first-layer classes of a germ.

T_2 (orb.py) reduces a given vector and separates two given vectors; it never
enumerates, so the list of classes had to come from a structure theorem proved
by hand.  T_3 enumerates, and exactly, by the graded stratification the sweep
already provides.

The scheme.  The layer is graded, Nt_f = (+)_e Nt^(e), and the stabiliser
algebra is non-negatively graded, s_f = (+)_{d>=0} s_d with s_d Nt^(e) < Nt^(e+d).
Normalise a vector degree by degree.  Suppose the components of v of degree
< d are already normalised (and those of degree >= d are still zero).  Put

    h = { M in rho(s_f) : (M v)_i = 0 whenever deg i < d },

the Lie algebra of the subgroup that fixes what has been normalised -- a
subalgebra, because rho(s_f) preserves the filtration.  For a candidate value
c in Nt^(d) of the next component,

    ( M (v+c) )_d  =  (M v)_d  +  M_0 c ,        M in h,

M_0 being the degree-0 graded component of M.  So the action on the degree-d
component is AFFINE, with translation part

    T = { (M v)_d : M in h }        (a subspace of Nt^(d))

and linear part the image L = { M_0 : M in h } of h in gl(Nt^(d)).  Hence
normalising the degree-d component is one linear-group orbit problem on the
quotient U = Nt^(d)/T, of dimension at most that of one graded piece.  When
the group is transitive on U minus 0 -- certified, not sampled: the image
algebra is shown to contain sl(U), or sp(U) for an invariant symplectic form --
the problem has exactly two answers, "in T" and "the representative", and the
enumeration branches in two.  Recursing over d produces every class exactly
once, with an explicit representative.

Nothing here is random.  The leaves are then checked pairwise by T_2's
certificates, and each stratum by T_2's reduction on random points.
"""

from fractions import Fraction as F
from math import gcd

import stab
import orb
import sv


# ------------------------------------------------------------ linear algebra

def rref(rows):
    """reduced row echelon form (list of rows), pivots cleared above and below"""
    rows = [list(r) for r in rows if any(r)]
    if not rows:
        return []
    n = len(rows[0])
    r = 0
    for c in range(n):
        piv = next((i for i in range(r, len(rows)) if rows[i][c]), None)
        if piv is None:
            continue
        rows[r], rows[piv] = rows[piv], rows[r]
        pv = rows[r][c]
        rows[r] = [x / pv for x in rows[r]]
        for i in range(len(rows)):
            if i != r and rows[i][c]:
                fct = rows[i][c]
                rows[i] = [a - fct * b for a, b in zip(rows[i], rows[r])]
        r += 1
        if r == len(rows):
            break
    import param
    if param.RECORD:
        param.record_rank(("rref", r))
    return [row for row in rows[:r] if any(row)]


def pivots_of(R):
    out = []
    for row in R:
        p = next((i for i, x in enumerate(row) if x), None)
        if p is not None:
            out.append(p)
    return out


def span_rank(mats):
    return stab.rank_rows([sum(M, []) for M in mats]) if mats else 0


def span_contains(big, small):
    rb = [sum(M, []) for M in big]
    rs = [sum(M, []) for M in small]
    if not rs or all(not any(r) for r in rs):
        return True
    if not rb:
        return False
    return stab.rank_rows(rb) == stab.rank_rows(rb + rs)


def sl_basis(k):
    out = []
    for i in range(k):
        for j in range(k):
            if i != j:
                M = [[F(0)] * k for _ in range(k)]
                M[i][j] = F(1)
                out.append(M)
    for i in range(k - 1):
        M = [[F(0)] * k for _ in range(k)]
        M[i][i], M[i + 1][i + 1] = F(1), F(-1)
        out.append(M)
    return out


def invariant_skew_forms(mats, k):
    """basis of { J skew : M^T J + J M = 0 for all M in mats }"""
    keys = [(i, j) for i in range(k) for j in range(k) if i < j]
    pos = {kk: t for t, kk in enumerate(keys)}
    rows = []
    for M in mats:
        for a in range(k):
            for b in range(k):
                eq = [F(0)] * len(keys)
                for c in range(k):
                    if c != b:
                        eq[pos[(min(c, b), max(c, b))]] += \
                            (F(1) if c < b else F(-1)) * M[c][a]
                    if a != c:
                        eq[pos[(min(a, c), max(a, c))]] += \
                            (F(1) if a < c else F(-1)) * M[c][b]
                if any(eq):
                    rows.append(eq)
    if not rows:
        sols = [[F(1) if a == b else F(0) for b in range(len(keys))]
                for a in range(len(keys))]
    else:
        cols = [{i: rows[i][j] for i in range(len(rows)) if rows[i][j]}
                for j in range(len(keys))]
        sols = stab.nullspace(cols, set(range(len(rows))))
    out = []
    for s in sols:
        J = [[F(0)] * k for _ in range(k)]
        for t, (i, j) in enumerate(keys):
            J[i][j], J[j][i] = s[t], -s[t]
        out.append(J)
    return out


def det(M):
    k = len(M)
    A = [row[:] for row in M]
    d = F(1)
    for c in range(k):
        piv = next((r for r in range(c, k) if A[r][c]), None)
        if piv is None:
            return F(0)
        if piv != c:
            A[c], A[piv] = A[piv], A[c]
            d = -d
        d *= A[c][c]
        inv = F(1) / A[c][c]
        for r in range(c + 1, k):
            if A[r][c]:
                fac = A[r][c] * inv
                A[r] = [A[r][t] - fac * A[c][t] for t in range(k)]
    return d


def sp_basis(J):
    """basis of sp(J) = { M : M^T J + J M = 0 }"""
    k = len(J)
    keys = [(i, j) for i in range(k) for j in range(k)]
    pos = {kk: t for t, kk in enumerate(keys)}
    rows = []
    for a in range(k):
        for b in range(k):
            eq = [F(0)] * len(keys)
            for c in range(k):
                eq[pos[(c, a)]] += J[c][b]
                eq[pos[(c, b)]] += J[a][c]
            if any(eq):
                rows.append(eq)
    cols = [{i: rows[i][j] for i in range(len(rows)) if rows[i][j]}
            for j in range(len(keys))]
    sols = stab.nullspace(cols, set(range(len(rows))))
    out = []
    for s in sols:
        M = [[F(0)] * k for _ in range(k)]
        for t, (i, j) in enumerate(keys):
            M[i][j] = s[t]
        out.append(M)
    return out


def transitive_certificate(mats, k):
    """
    Is the connected group generated by `mats` transitive on U minus 0,
    dim U = k?  A matrix algebra containing sl(U), or sp(U) for a nondegenerate
    invariant skew form, is transitive on the nonzero vectors of the
    tautological module; that is the certificate returned.
    """
    if k == 0:
        return True, "the space is zero"
    if k == 1:
        nz = any(M[0][0] for M in mats)
        return nz, ("a nonzero weight on a line" if nz
                    else "the algebra acts by zero on a line")
    if span_contains(mats, sl_basis(k)):
        return True, f"the image contains sl({k})"
    for J in invariant_skew_forms(mats, k):
        if det(J) != 0 and span_contains(mats, sp_basis(J)):
            return True, f"the image contains sp({k}) of an invariant form"
    return False, f"neither sl({k}) nor sp({k}) inside the image"


# --------------------------------------------- orbits of a small linear group

def mat_inv(P):
    k = len(P)
    A = [row[:] + [F(1) if i == j else F(0) for j in range(k)]
         for i, row in enumerate(P)]
    r = 0
    for c in range(k):
        piv = next((i for i in range(r, k) if A[i][c]), None)
        if piv is None:
            raise ValueError("singular")
        A[r], A[piv] = A[piv], A[r]
        pv = A[r][c]
        A[r] = [x / pv for x in A[r]]
        for i in range(k):
            if i != r and A[i][c]:
                fct = A[i][c]
                A[i] = [a - fct * b for a, b in zip(A[i], A[r])]
        r += 1
    return [row[k:] for row in A]


def matmul(A, B):
    n, m, l = len(A), len(B[0]), len(B)
    return [[sum(A[i][t] * B[t][j] for t in range(l)) for j in range(m)]
            for i in range(n)]


def split_certificate(mats, k, depth):
    """
    Certificate (d): U = W_1 (+) W_2 with both summands invariant AND the image
    containing each summand's algebra separately -- that is, the map into
    gl(W_1) x gl(W_2) is onto.  Then a vector is a pair and the two components
    are normalised independently, so the orbits are the products of the orbits
    of the summands (the pair (0,0) excluded).  Returns (reps, why) or None.
    """
    W1 = invariant_subspace(mats, k)
    if not W1:
        return None
    w = len(W1)
    for j in range(k):
        e = [F(1) if i == j else F(0) for i in range(k)]
        S = orb.submodule(mats, e)
        W2 = S.basis()
        if not (0 < len(W2) < k) or len(W2) + w != k:
            continue
        if stab.rank_rows([list(b) for b in W1] + [list(b) for b in W2]) != k:
            continue
        cols = [list(b) for b in W1] + [list(b) for b in W2]
        P = [[cols[b][i] for b in range(k)] for i in range(k)]
        try:
            Pi = mat_inv(P)
        except ValueError:
            continue
        conj = [matmul(Pi, matmul(M, P)) for M in mats]
        offd = any(M[i][j] for M in conj
                   for i in range(k) for j in range(k)
                   if (i < w) != (j < w))
        if offd:
            continue
        # each summand's algebra must be inside the image on its own
        A1 = [[[M[i][j] for j in range(w)] for i in range(w)] for M in conj]
        A2 = [[[M[i][j] for j in range(w, k)] for i in range(w, k)]
              for M in conj]
        emb1, emb2 = [], []
        for A in A1:
            M = [[F(0)] * k for _ in range(k)]
            for i in range(w):
                for j in range(w):
                    M[i][j] = A[i][j]
            emb1.append(M)
        for B in A2:
            M = [[F(0)] * k for _ in range(k)]
            for i in range(k - w):
                for j in range(k - w):
                    M[w + i][w + j] = B[i][j]
            emb2.append(M)
        if not (span_contains(conj, emb1) and span_contains(conj, emb2)):
            continue
        r1, c1 = orbit_reps(A1, w, depth + 1)
        r2, c2 = orbit_reps(A2, k - w, depth + 1)
        if r1 is None or r2 is None:
            continue
        out = []

        def tr(full):
            return [sum(P[i][j] * full[j] for j in range(k)) for i in range(k)]

        for a in [None] + list(r1):
            for b in [None] + list(r2):
                if a is None and b is None:
                    continue
                va = list(a[0]) if a else [F(0)] * w
                vb = list(b[0]) if b else [F(0)] * (k - w)
                dirs = ([tr(list(u) + [F(0)] * (k - w)) for u in a[2]] if a
                        else [])
                dirs += ([tr([F(0)] * w + list(u)) for u in b[2]] if b else [])
                out.append((tr(va + vb),
                            (a[1] if a else 0) + (b[1] if b else 0), dirs))
        return out, (f"the space splits as {w} + {k - w} into invariant "
                     f"summands whose algebras both lie in the image; on them, "
                     f"{c1}; and {c2}")
    return None


def invariant_subspace(mats, k):
    """a proper nonzero invariant subspace, as a list of basis vectors, or None"""
    best = None
    for j in range(k):
        e = [F(1) if i == j else F(0) for i in range(k)]
        S = orb.submodule(mats, e)
        if 0 < S.dim() < k and (best is None or S.dim() < len(best)):
            best = S.basis()
    return best


def invariant_subspaces(mats, k):
    """every cyclic proper invariant subspace, smallest first, deduplicated"""
    out, seen = [], set()
    for j in range(k):
        e = [F(1) if i == j else F(0) for i in range(k)]
        S = orb.submodule(mats, e)
        if not (0 < S.dim() < k):
            continue
        B = S.basis()
        key = tuple(tuple(r) for r in rref([list(b) for b in B]))
        if key in seen:
            continue
        seen.add(key)
        out.append([list(b) for b in B])
    out.sort(key=len)
    return out


def adapted(mats, k, W):
    """
    A basis of U adapted to the subspace W.  Returns (P, conj, w) with P the
    change of basis (columns = the new basis, W first) and conj the matrices
    in it -- block upper triangular -- or None if W is not invariant.
    """
    cols = [list(b) for b in W]
    w = len(cols)
    for j in range(k):
        e = [F(1) if i == j else F(0) for i in range(k)]
        if stab.rank_rows(cols + [e]) > len(cols):
            cols.append(e)
    if len(cols) != k:
        return None
    P = [[cols[j][i] for j in range(k)] for i in range(k)]
    try:
        conj = [matmul(mat_inv(P), matmul(M, P)) for M in mats]
    except ValueError:
        return None
    if any(M[i][j] for M in conj for i in range(w, k) for j in range(w)):
        return None
    return P, conj, w


def fibre_certificate(mats, k, depth):
    """
    Certificate (e), the FIBRE certificate -- the relative form of (a)--(c),
    and the one that closes the nodes (c) leaves open.

    Let W be invariant.  Certificate (c) asks that the whole block Hom(U/W,W)
    lie in the image, which makes each fibre of U --> U/W a single orbit.  When
    it does not, the fibres must be classified rather than assumed transitive,
    and that is what is done here.  Write L for the image.  For a vector v with
    nonzero class vbar in U/W, put

        s   = { M in L : M vbar = 0 }        the stabiliser of vbar, a subalgebra
        T   = s.v  (contained in W)          the translations

    On the affine space v + W the group Stab(vbar) acts AFFINELY, by
    u |-> M u + Mv, and the fibre is classified by affine.affine_orbit_reps,
    which proves what it returns and assumes nothing about T.  Two things were
    assumed here before, and both are false in general:

      * that T = W makes the fibre one orbit.  The tangent to the orbit through
        u is { Mv + Mu }, which varies with u and is merely contained in
        T + s.u.  Already with dim W = 1 and one generator M acting by 1 with
        Mv = 1 the translations fill W, and yet -1 is a fixed point: two
        orbits, not one;
      * that T is s-stable, which is what the reduction to the residual space
        W/T needs.  In corank 1|5 it is not.

    The affine module replaces both by its certificates (A1)-(A3).  A family of
    classes arises exactly as before, when the stabiliser acts by ZERO on a
    residual space of dimension d > 0, and that conclusion is still
    unconditional rather than one about the connected group only: Stab(vbar) is
    an algebraic group whose identity component acts there with zero
    derivative, hence trivially, so the whole of it acts through its finite
    component group, and a d-dimensional space modulo a finite group is still
    d-dimensional.

    Returns (reps, certificate) or None.
    """
    import affine
    for W in invariant_subspaces(mats, k):
        ad = adapted(mats, k, W)
        if ad is None:
            continue
        P, conj, w = ad
        span = basis_of_span(conj)
        subW = [[[M[i][j] for j in range(w)] for i in range(w)] for M in span]
        subQ = [[[M[i][j] for j in range(w, k)] for i in range(w, k)]
                for M in span]

        # the orbits inside W, and the orbits of the quotient
        rw, cw = orbit_reps(subW, w, depth + 1)
        if rw is None:
            continue
        rq, cq = orbit_reps(subQ, k - w, depth + 1)
        if rq is None:
            continue

        out, notes, failed = [], [], False

        def tr(full):
            return [sum(P[i][j] * full[j] for j in range(k)) for i in range(k)]

        for r, fam, dr in rw:                 # the classes lying inside W
            out.append((tr(list(r) + [F(0)] * (k - w)), fam,
                        [tr(list(u) + [F(0)] * (k - w)) for u in dr]))

        def fibre_for(member):
            """
            The classes of the fibre over the quotient class `member`, as
            (y, ff, fd) in the coordinates of W.  Ground-field agnostic: it is
            called with a numeric member, and with the GENERIC member of a
            family over K(mu) -- the parametric certificate of param.py.
            """
            v = [F(0)] * w + list(member)     # a lift of the quotient rep
            # s = { M in span : the quotient part of M v vanishes }
            cols, eqs = [], set(range(k - w))
            for M in span:
                img = orb.apply_mat(M, v)
                cols.append({i: img[w + i] for i in range(k - w) if img[w + i]})
            sols = stab.nullspace(cols, eqs)
            s = [orb.mat_comb(span, c) for c in sols]
            s = [M for M in s if not orb.mat_zero(M)]
            import param
            if param.RECORD:
                # dim s(mu), the stabiliser of the member inside the span: an
                # INTRINSIC number, and the one that decides the fibre
                param.record_rank(("dim_s", len(s)))
            # On the fibre v + W the stabiliser acts AFFINELY, with linear part
            # M|_W and translations (Mv)|_W.  This is the affine problem, and it
            # is solved as one -- not by assuming the translations fill W (they
            # need not), nor that they span an invariant subspace (they need
            # not either).
            gens = [([[M[i][j] for j in range(w)] for i in range(w)],
                     [orb.apply_mat(M, v)[i] for i in range(w)]) for M in s]
            return affine.affine_orbit_reps(gens, w, depth + 1) + (v,)

        for r, fam, dr in rq:                 # one fibre over each quotient class
            member, node = list(r), None
            # over a family whose stabiliser moves, the fibre is classified over
            # the GENERIC member, over K(mu): the parametric certificate
            if fam and not affine.uniform_family(
                    [[[M[i][j] for j in range(w, k)] for i in range(w, k)]
                     for M in span], lambda z: list(z), list(r), list(dr)):
                import param
                gm = param.generic_member(list(r), list(dr)) \
                    if param.active() else None
                if gm is None:
                    failed = True
                    break
                member = gm[0]
                node = param.note_node("fibre", {
                    "k": k, "w": w, "fam": fam, "r": list(r),
                    "dirs": list(dr), "params": list(param.LAST_PARAMS),
                    "run": (lambda vals, _r=list(r), _dr=list(dr):
                            fibre_for([_r[i] + sum(v * z[i] for v, z
                                                   in zip(vals, _dr))
                                       for i in range(len(_r))])[:2])})
            rf, cf, v = fibre_for(member)
            if node is not None:
                import param
                param.close_node(node, (rf, cf))
            if rf is None:
                failed = True
                break
            up = [tr([F(0)] * w + list(u)) for u in dr]   # family of U/W
            for y, ff, fd in rf:
                full = list(y) + list(v[w:])
                out.append((tr(full), fam + ff,
                            up + [tr(list(z) + [F(0)] * (k - w)) for z in fd]))
            notes.append(f"over one class the fibre of dimension {w} is an "
                         f"affine problem in which {cf}")
        if failed:
            continue
        return out, ("the fibre certificate over an invariant subspace of "
                     f"dimension {w} of {k}: inside it, {cw}; on the quotient, "
                     f"{cq}; " + "; ".join(notes[:3]))
    return None


def orbit_reps(mats, k, depth=0):
    """
    Representatives of every orbit of the connected group generated by `mats`
    on U minus 0, dim U = k, each with a proof.  Returns (reps, certificate)
    or (None, reason); a rep is a pair (vector, family) with family = 0 for a
    single orbit and family = t > 0 for a t-parameter family of orbits
    (the scalar multiples of the vector, pairwise inequivalent).  The
    certificates used are

      (a) the image contains sl(U) or sp(U)  ==>  U minus 0 is one orbit;
      (b) the image acts by ZERO on U  ==>  every line is fixed pointwise, so
          U minus 0 is a family of orbits of dimension dim U -- a modulus;
      (c) an invariant subspace W with the whole block Hom(U/W, W) inside the
          image  ==>  the orbits outside W are the preimages of the orbits of
          U/W minus 0, and inside W they are the orbits of W;
      (d) a splitting into invariant summands whose algebras both lie in the
          image  ==>  the two components are normalised independently;
      (e) the FIBRE certificate -- an invariant subspace W with Hom(U/W,W) NOT
          inside the image, the fibres being classified instead of assumed
          transitive.  See fibre_certificate; it subsumes (c).
    """
    if k == 0:
        return [], "the space is zero"
    ok, why = transitive_certificate(mats, k)
    if ok:
        return [([F(1) if i == 0 else F(0) for i in range(k)], 0, [])], why
    if all(orb.mat_zero(M) for M in mats):
        # every point is its own orbit: the family runs along the whole space
        ident = [[F(1) if i == j else F(0) for i in range(k)]
                 for j in range(k)]
        return ([(ident[0], k, ident)],
                f"the algebra acts by zero on a space of dimension {k}: "
                f"every point is its own orbit")
    if depth > 6:
        return None, "the recursion did not terminate"
    dec = split_certificate(mats, k, depth)
    if dec is not None:
        return dec
    W = invariant_subspace(mats, k)
    if not W:
        return None, f"no invariant subspace and {why}"
    w = len(W)
    # a basis of U adapted to W
    cols = [list(b) for b in W]
    for j in range(k):
        e = [F(1) if i == j else F(0) for i in range(k)]
        if stab.rank_rows(cols + [e]) > len(cols):
            cols.append(e)
    P = [[cols[j][i] for j in range(k)] for i in range(k)]     # columns = basis
    Pi = mat_inv(P)
    conj = [matmul(Pi, matmul(M, P)) for M in mats]
    if any(conj[a][i][j] for a in range(len(conj))
           for i in range(w, k) for j in range(w)):
        return None, "the subspace found is not invariant after all"
    # the block Hom(U/W, W) must be inside the image
    block = []
    for i in range(w):
        for j in range(w, k):
            M = [[F(0)] * k for _ in range(k)]
            M[i][j] = F(1)
            block.append(M)
    if not span_contains(conj, block):
        fib = fibre_certificate(mats, k, depth)
        if fib is not None:
            return fib
        return None, ("an invariant subspace of dimension "
                      f"{w} of {k}, but Hom(U/W,W) is not inside the image")
    subW = [[[M[i][j] for j in range(w)] for i in range(w)] for M in conj]
    subQ = [[[M[i][j] for j in range(w, k)] for i in range(w, k)] for M in conj]
    rw, cw = orbit_reps(subW, w, depth + 1)
    if rw is None:
        fib = fibre_certificate(mats, k, depth)
        if fib is not None:
            return fib
        return None, f"inside the invariant subspace: {cw}"
    rq, cq = orbit_reps(subQ, k - w, depth + 1)
    if rq is None:
        fib = fibre_certificate(mats, k, depth)
        if fib is not None:
            return fib
        return None, f"on the quotient by the invariant subspace: {cq}"
    out = []

    def push(vec, fam, dirs, pad):
        """transport a rep (and its family directions) into the basis of U"""
        def tr(u):
            full = ([F(0)] * w + list(u)) if pad else (list(u) + [F(0)] * (k - w))
            return [sum(P[i][j] * full[j] for j in range(k)) for i in range(k)]
        out.append((tr(vec), fam, [tr(u) for u in dirs]))

    for r, fam, dr in rw:
        push(r, fam, dr, False)
    for r, fam, dr in rq:
        push(r, fam, dr, True)
    return out, (f"an invariant subspace of dimension {w} with Hom(U/W,W) "
                 f"inside the image; on it, {cw}; on the quotient, {cq}")


# ---------------------------------------------------------------- the layer

def quasi_weights(f, p, q):
    """
    The quasi-homogeneous weights of the germ f: positive integers w(x_i)=1
    scaled up, w(xi_j), and a weighted degree d with every monomial of f of
    weight d.  Returns (wx, wxi) or None if no such weights exist.

    Why this is needed at all.  T_3 grades the layer by the degree of the
    monomials representing it, and needs the stabiliser to preserve the
    filtration that grading defines.  When I_f is homogeneous this is automatic.
    When it is not -- and it is not, as soon as f mixes weights, e.g.
    f = x^5 + x.xi_1.xi_2 + x^4.xi_3.xi_4 -- reduction modulo (m I_f) can lower
    the degree of a monomial, a monomial basis is then not a graded basis, and
    the precondition fails.  The cure is not to abandon the grading but to take
    the right one: for the weights in which f IS quasi-homogeneous, I_f is
    homogeneous again and the filtration is restored.
    """
    mons = list(f)
    if not mons:
        return None
    # unknowns: w(xi_0..xi_{q-1}), d;  w(x_i) = 1
    # equation per monomial: sum_{j in xi-set} w_j - d = -sum(x-exponents)
    rows = []
    for m in mons:
        row = [F(0)] * (q + 1)
        for j in m[1]:
            row[j] = row[j] + F(1)
        row[q] = F(-1)
        rows.append(row + [F(-sum(m[0]))])
    # Gaussian elimination on the augmented system
    r, piv = 0, {}
    for c in range(q + 1):
        s = next((i for i in range(r, len(rows)) if rows[i][c]), None)
        if s is None:
            continue
        rows[r], rows[s] = rows[s], rows[r]
        pv = rows[r][c]
        rows[r] = [x / pv for x in rows[r]]
        for i in range(len(rows)):
            if i != r and rows[i][c]:
                fc = rows[i][c]
                rows[i] = [a - fc * b for a, b in zip(rows[i], rows[r])]
        piv[c] = r
        r += 1
    for i in range(r, len(rows)):
        if rows[i][q + 1] and not any(rows[i][:q + 1]):
            return None                          # inconsistent: no such weights
    free = [c for c in range(q + 1) if c not in piv]

    def solve_with(vals, homog=False):
        """the solution whose free unknowns take the given values"""
        sol = [None] * (q + 1)
        for c, t in zip(free, vals):
            sol[c] = t
        for c in sorted(piv, reverse=True):
            i = piv[c]
            rhs = F(0) if homog else rows[i][q + 1]
            sol[c] = rhs - sum(rows[i][j] * sol[j]
                               for j in range(c + 1, q + 1))
        return sol

    # Only a POSITIVE solution is a grading in which the layer is bounded
    # below, and putting the free unknowns at 1 need not give one: for
    # f = x^5 + x.xi_1.xi_2 + x^4.xi_3.xi_4 it gives w(xi_3) = 0.  Nor will any
    # one uniform value do, since the two pairs of odd indeterminates must be
    # split at different heights.  What is wanted is the BALANCED solution --
    # the one of least norm, which splits each pair evenly (here 2,2 and
    # 1/2,1/2) -- and it is obtained from the normal equations below.
    # The target of the minimisation is the all-ones vector, not the origin:
    # an odd indeterminate that does not occur in f at all -- xi_3, xi_4 in
    # III_2(n) = x^n + x.xi_1.xi_2 -- is unconstrained, and minimising the norm
    # would give it the weight 0, which is not a grading.  Pulling towards the
    # weight of x instead gives it 1 and leaves every constrained pair balanced.
    nf = len(free)
    p0 = solve_with([F(0)] * nf)
    ker = [solve_with([F(1) if a == b else F(0) for a in range(nf)], True)
           for b in range(nf)]
    G = [[sum(ker[a][j] * ker[b][j] for j in range(q)) for b in range(nf)]
         for a in range(nf)]
    rhs = [-sum(ker[a][j] * (p0[j] - F(1)) for j in range(q))
           for a in range(nf)]
    t = [F(0)] * nf
    aug = [G[a][:] + [rhs[a]] for a in range(nf)]
    rr, pv2 = 0, {}
    for c in range(nf):
        s = next((i for i in range(rr, nf) if aug[i][c]), None)
        if s is None:
            continue
        aug[rr], aug[s] = aug[s], aug[rr]
        pvv = aug[rr][c]
        aug[rr] = [x / pvv for x in aug[rr]]
        for i in range(nf):
            if i != rr and aug[i][c]:
                fc = aug[i][c]
                aug[i] = [a - fc * b for a, b in zip(aug[i], aug[rr])]
        pv2[c] = rr
        rr += 1
    for c in sorted(pv2, reverse=True):
        t[c] = aug[pv2[c]][nf] - sum(aug[pv2[c]][j] * t[j]
                                     for j in range(c + 1, nf))
    sol = solve_with(t)
    if not (all(w > 0 for w in sol[:q]) and sol[q] > 0):
        sol = None
        for tt in (F(1, 2), F(1), F(1, 4), F(3, 2), F(2)):
            c = solve_with([tt] * nf)
            if all(w > 0 for w in c[:q]) and c[q] > 0:
                sol = c
                break
        if sol is None:
            return None
    wxi, d = sol[:q], sol[q]
    den = 1
    for w in list(wxi) + [F(1), d]:
        den = den * w.denominator // gcd(den, w.denominator)
    wx = [den] * p
    wxi = [int(w * den) for w in wxi]
    # verify: every monomial of f really has one and the same weight
    def wt(m):
        return sum(e * den for e in m[0]) + sum(wxi[j] for j in m[1])
    if len({wt(m) for m in mons}) != 1:
        return None
    return wx, wxi


class Layer:
    def __init__(self, res):
        self.mats = res["mats"]
        self.basis = res["basis"]
        self.n = len(self.basis)
        self.degs = [stab.mdeg(b) for b in self.basis]
        self.weights = None
        if not self._filt_ok(self.degs):
            # the naive degree does not work: try the quasi-homogeneous one
            w = quasi_weights(res["f"], res["p"], res["q"])
            if w is not None:
                wx, wxi = w
                wdeg = [sum(e * wx[i] for i, e in enumerate(b[0]))
                        + sum(wxi[j] for j in b[1]) for b in self.basis]
                if self._filt_ok(wdeg):
                    self.degs = wdeg
                    self.weights = w
        self.bydeg = {}
        for i, d in enumerate(self.degs):
            self.bydeg.setdefault(d, []).append(i)
        self.degrees = sorted(self.bydeg)
        # a basis of the span of rho(s_f)
        self.span = basis_of_span(self.mats)
        # T_3 needs only that rho(s_f) preserve the filtration by degree --
        # true because the stabiliser consists of vector fields vanishing at
        # the origin -- and never that it be graded: the linear part of the
        # affine action at each degree is the induced map on the associated
        # graded, which is the block diagonal proj0(M) whether or not M is
        # homogeneous.  The check below is that filtration property.
        self.filtration_ok = self._filt_ok(self.degs)

    def _filt_ok(self, degs):
        """does rho(s_f) preserve the filtration defined by these degrees?"""
        return all(M[i][j] == 0 for M in self.mats for i in range(self.n)
                   for j in range(self.n) if degs[i] < degs[j])

    def proj0(self, M):
        """the degree-0 graded component of the matrix M"""
        return [[M[i][j] if self.degs[i] == self.degs[j] else F(0)
                 for j in range(self.n)] for i in range(self.n)]

    def show(self, v):
        out = []
        for i in range(self.n):
            if v[i]:
                nm = sv.show_mon(self.basis[i], "x", "xi")
                out.append(nm if v[i] == 1 else f"{v[i]}.{nm}")
        return " + ".join(out) if out else "0"

    def stab_subalgebra(self, v, upto):
        """a basis of { M in span : (M v)_i = 0 whenever deg i < upto }"""
        tgt = [i for i in range(self.n) if self.degs[i] < upto]
        if not tgt:
            return list(self.span)
        rows = [[orb.apply_mat(M, v)[i] for i in tgt] for M in self.span]
        # unknowns = the coefficients of the generators, equations = the
        # coordinates of degree < upto that must be left alone
        cols = [{j: rows[a][j] for j in range(len(tgt)) if rows[a][j]}
                for a in range(len(rows))]
        sols = stab.nullspace(cols, set(range(len(tgt))))
        out = []
        for s in sols:
            M = orb.mat_comb(self.span, s)
            if not orb.mat_zero(M):
                out.append(M)
        return out


def basis_of_span(mats):
    keep, rows = [], []
    for M in mats:
        r = sum(M, [])
        if stab.rank_rows(rows + [r]) > len(rows):
            rows.append(r)
            keep.append(M)
    return keep


# --------------------------------------------------------- the enumeration

def classify(res, maxleaves=400, affine_mode=True):
    """
    The complete list of classes.  Returns (Layer, classes, certified, notes)
    with classes a list of dicts {v, dim, log}.

    At each degree d the stabiliser of the lower degrees acts on the piece
    Nt^(d) AFFINELY -- by u |-> proj0(M)u + (Mv)|_d.  With affine_mode the
    orbits of that action are computed by affine.affine_orbit_reps, which
    proves what it returns; with affine_mode=False the linear shortcut is used
    instead: quotient the piece by the translation subspace T and classify the
    induced linear action, which needs T invariant (it is not, always) and
    takes every fibre for one orbit (it is not, always).  The flag is kept so
    that the two can be compared; see the docstring of affine.py.
    """
    if affine_mode:
        return classify_affine(res, maxleaves)
    L = Layer(res)
    certified = L.filtration_ok
    notes = [] if L.filtration_ok else ["rho(s_f) does not preserve the "
                                        "filtration by degree"]

    leaves = []
    stack = [([F(0)] * L.n, 0, [], 0, [])]
    while stack:
        v, di, log, fam, dirs = stack.pop()
        if di >= len(L.degrees):
            leaves.append({"v": v, "log": log, "family": fam, "dirs": dirs})
            continue
        d = L.degrees[di]
        idx = L.bydeg[d]
        k0 = len(idx)
        h = L.stab_subalgebra(v, d)
        # the translations
        T = rref([[orb.apply_mat(M, v)[i] for i in idx] for M in h])
        # the linear part, on the piece
        L0 = [[[L.proj0(M)[i][j] for j in idx] for i in idx] for M in h]
        # T must be stable under L0 for the quotient to make sense
        if T and not stable_under(T, L0):
            certified = False
            notes.append(f"degree {d}: the translation subspace is not stable")
        Q, lift, red = quotient(T, k0)
        qmats = [induced(M, Q, red) for M in L0]
        k = len(Q)
        if k == 0:
            stack.append((v, di + 1, log, fam, dirs))
            continue
        reps, why = orbit_reps(qmats, k)
        if reps is None:
            certified = False
            notes.append(f"degree {d}: {why} (dim U = {k})")
            reps = [([F(1) if i == 0 else F(0) for i in range(k)], 0, [])]

        def into_layer(r):
            """the piece-vector r, lifted to a vector of the whole layer"""
            u = [F(0)] * L.n
            for a, i in enumerate(idx):
                u[i] = sum(r[b] * lift[b][a] for b in range(k))
            return u

        # the component lies in T -- normalise it to zero -- or it is one of
        # the representatives of the orbits of U minus 0
        stack.append((v, di + 1, log + [(d, 0, k, why)], fam, dirs))
        for t, (r, ff, fd) in enumerate(reps):
            u = into_layer(r)
            w = [v[i] + u[i] for i in range(L.n)]
            stack.append((w, di + 1, log + [(d, t + 1, k, why)], fam + ff,
                          dirs + [into_layer(z) for z in fd]))
        if len(stack) + len(leaves) > maxleaves:
            certified = False
            notes.append("the enumeration exceeded maxleaves")
            break

    out, seen = [], set()
    for lf in leaves:
        key = tuple(lf["v"])
        if key in seen:
            continue
        seen.add(key)
        out.append({"v": lf["v"], "dim": orb.orbit_dim(L.mats, lf["v"]),
                    "log": lf["log"], "family": lf["family"],
                    "dirs": lf["dirs"]})
    out.sort(key=lambda r: (-r["dim"], tuple(r["v"])))
    return L, out, certified, notes


def classify_affine(res, maxleaves=400):
    """
    The same enumeration, degree by degree, but each degree solved as the
    affine problem it is.  Nothing is assumed about the translation subspace:
    affine.affine_orbit_reps returns the orbits of the affine action on the
    whole piece -- the orbit through the origin included, an affine space
    having no origin -- each with a certificate, or nothing at all.
    """
    import affine
    import param
    import os
    # the ground field K(mu_1..mu_80) for family nodes; NOPARAM=1 empties the
    # pool, so every family node declines exactly as it did before the
    # parametric certificate existed -- kept for the comparison of the two
    param.begin(0 if os.environ.get("NOPARAM")
                else int(os.environ.get("PARAMPOOL", "80")))
    L = Layer(res)
    certified = L.filtration_ok
    notes = [] if L.filtration_ok else ["rho(s_f) does not preserve the "
                                        "filtration by degree"]
    leaves = []
    stack = [([F(0)] * L.n, 0, [], 0, [])]
    while stack:
        v, di, log, fam, dirs = stack.pop()
        if di >= len(L.degrees):
            leaves.append({"v": v, "log": log, "family": fam, "dirs": dirs})
            continue
        d = L.degrees[di]
        idx = L.bydeg[d]
        k0 = len(idx)
        h = L.stab_subalgebra(v, d)
        gens = [([[L.proj0(M)[i][j] for j in idx] for i in idx],
                 [orb.apply_mat(M, v)[i] for i in idx]) for M in h]
        reps, why = affine.affine_orbit_reps(gens, k0)
        if reps is None:
            certified = False
            notes.append(f"degree {d}: {why} (dim of the piece = {k0})")
            reps = [([F(0)] * k0, 0, [])]

        def into_layer(r):
            u = [F(0)] * L.n
            for a, i in enumerate(idx):
                u[i] = r[a]
            return u

        for t, (r, ff, fd) in enumerate(reps):
            u = into_layer(r)
            w = [v[i] + u[i] for i in range(L.n)]
            stack.append((w, di + 1, log + [(d, t, k0, why)], fam + ff,
                          dirs + [into_layer(z) for z in fd]))
        if len(stack) + len(leaves) > maxleaves:
            certified = False
            notes.append("the enumeration exceeded maxleaves")
            break

    out, seen = [], set()
    for lf in leaves:
        key = tuple(lf["v"])
        if key in seen:
            continue
        seen.add(key)
        out.append({"v": lf["v"], "dim": orb.orbit_dim(L.mats, lf["v"]),
                    "log": lf["log"], "family": lf["family"],
                    "dirs": lf["dirs"]})
    out.sort(key=lambda r: (-r["dim"], tuple(r["v"])))
    if param.NODES:
        # the parametric certificate was used: the answer is proved for every
        # member of every family outside the zero set of what was inverted
        ex = param.exceptional()
        notes.append(f"parametric: {len(param.NODES)} family node(s) "
                     f"classified over the generic member; the answer holds "
                     f"off the {len(ex)} hypersurface(s) "
                     + ", ".join(str(g) for g in ex[:12]))
    return L, out, certified, notes


def stable_under(T, mats):
    rk = len(T)
    rows = [list(r) for r in T]
    for M in mats:
        for r in T:
            u = [sum(M[i][j] * r[j] for j in range(len(r)))
                 for i in range(len(r))]
            if stab.rank_rows(rows + [u]) != rk:
                return False
    return True


def quotient(T, k):
    """
    T in rref.  Returns (free, lift, red):
      free : the free coordinates, a basis of U = k-space / T
      lift : lift[j] a vector of k-space representing the j-th basis vector
      red  : red(vec) -> the coordinates of vec mod T in that basis
    """
    piv = pivots_of(T)
    free = [i for i in range(k) if i not in piv]

    def red(vec):
        w = list(vec)
        for row in T:
            p = next(i for i, x in enumerate(row) if x)
            if w[p]:
                c = w[p] / row[p]
                w = [w[i] - c * row[i] for i in range(k)]
        return [w[i] for i in free]

    lift = []
    for j in free:
        e = [F(1) if i == j else F(0) for i in range(k)]
        lift.append(e)
    return free, lift, red


def induced(M, Q, red):
    """the matrix induced by M on the quotient, in the basis of lifts"""
    k = len(Q)
    n = len(M)
    out = [[F(0)] * k for _ in range(k)]
    for j in range(k):
        e = [F(1) if i == Q[j] else F(0) for i in range(n)]
        img = [sum(M[i][t] * e[t] for t in range(n)) for i in range(n)]
        col = red(img)
        for i in range(k):
            out[i][j] = col[i]
    return out
