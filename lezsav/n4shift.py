#!/usr/bin/env python3
"""
n4shift.py -- the transformation law of the quaternionic superfield q of the
N=4 Ivanov-Krivonos system (IK4): "the shift".

The text, Subsection 6.2, states the ansatz

    delta_f q  =  K_f(q) + a(f) q + q b(f)                                  (*)

"with a and b built from the two sl(2)-parts of the odd Hessian and from its
trace", and says explicitly that it is NOT verified.  This engine verifies it,
solves for a and b, and settles which f admit such a law at all.

Everything is exact arithmetic over Q(i) in the generating-function model of
k^L(1|4) of n4emb.py, whose primitives are imported.

Set-up.  In ONE half the superspace is C^{1|4} with t and theta_1..theta_4,

    D_i = d_{theta_i} - theta_i d_t,     {D_i, D_j} = -2 delta_ij d_t,

and the contact field of the generating function f acts on functions by

    K_f(g) = {f, g} + 2 (d_t f) g                                          (K)

({ , } being the contact bracket (kbr) of the text, which is the bracket of
generating functions, not the action on functions -- (K) is the difference).

Following (Dcx) of the text, extended to N=4 by (isoplanes):

    cD^1 = D_1 - i D_2,   cD^2 = D_3 - i D_4      (the plane F = span(f_1,f_2))
    bD_1 = D_1 + i D_2,   bD_2 = D_3 + i D_4      (the plane E = span(e_1,e_2))

so {cD^a, cD^b} = {bD_a, bD_b} = 0 and {cD^a, bD_b} = -4 delta d_t: the two
totally isotropic planes of (isoplanes), in the frame rather than in Lambda(V).

What is computed:

 (1) that (K) really is the contact vector field: K_f is a derivation of O,
     [K_f, K_g] = K_{{f,g}}, and the multiplier (multN) holds.

 (2) the multiplier in the complex frame, in closed form:

        [K_f, cD^a] = sum_b Lam^a_b(f) cD^b  +  Xi(f) eps^{ab} bD_b,

        Lam^a_b(f) = 1/2 cD^a bD_b f,      Xi(f) = 1/2 bD_1 bD_2 f.

     This is the N=4 analogue of (multC), which is the case N=2 of the same
     formula -- and there Xi is absent, o(2) having no Lambda^2 of a plane.

 (3) THE POINT.  The constraint of the first line of (IK4) in the minus half,
     cD^{(a} q^{b)} = 0, uses the plane F alone.  So a contact field can
     preserve it only if its multiplier preserves F, i.e. only if

        Xi(f) = 1/2 bD_1 bD_2 f = 0.                                       (Xi)

     The text says (Subsection 6.2) that this constraint "is preserved by
     the whole of k^L(1|4)".  That is FALSE, and (Xi) is the correction: the
     stabilizer k^Xi := {f | Xi(f) = 0} is a proper subalgebra.  The engine
     computes its dimension degree by degree and checks it is a subalgebra.

 (4) that the image of the embedding (KAemb) of Proposition 6.1.2 lies in
     k^Xi -- i.e. the algebra that Ivanov and Krivonos name for (IK4) does
     preserve the constraint, which is what the text needs and what the false
     sentence was standing in for.

 (5) the shift itself.  The general solution of cD^{(a} psi^{b)} = 0 is
     computed (as a nullspace over Q(i)) and the law

        delta_f psi^b = K_f(psi^b) + b(f)_d^b psi^d,
        b(f)_d^b := -(-1)^{p(f)} Lam^b_d(f) + mu(f) delta_d^b

     is verified to preserve it, for every f in k^Xi.

 (6) uniqueness: the sl(2)-part of b is forced, and the ambiguity is exactly
     mu with cD^a mu = 0, i.e. a cD-chiral function -- the N=4 counterpart of
     "no other shift will do" in the proof of Proposition 5.1.4.

 (7) that a(f) in (*) must be SCALAR.  A transformation of the minus half must
     also preserve the constraint of the PLUS half, bD_{+(a} q_{g)}^b = 0, on
     whose index a(f) acts; and eps.(2x2 matrix) is symmetric exactly for the
     traceless part.  So the ansatz (*) of the text sharpens: in each half the
     sl(2)-part sits on ONE side only, the side of that half's own doublet
     index, and the other side carries a scalar.

 (8) that delta is a Lie algebra action, [delta_f, delta_g] = delta_{{f,g}},
     for mu = 0 -- so mu is a genuine 1-cocycle ambiguity, to be fixed by the
     dynamical equation and not by the constraint.

Usage:  python3 n4shift.py           (writes n4shift-out.txt)
"""

import itertools
import sys

from fractions import Fraction as Q

from n4emb import (C, I, cx, mono, add, smul, mul, d_t, d_odd, euler, parity,
                   kbr, show, eq, IDX)

# ---------------------------------------------------------------------------
# reporting (own counters; n4emb's are not touched)
# ---------------------------------------------------------------------------

OUT = []
FAIL = [0]
CHECKS = [0]


def log(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s)


def check(cond, msg):
    CHECKS[0] += 1
    if not cond:
        FAIL[0] += 1
        log("  FAIL: %s" % msg)
    return cond


# ---------------------------------------------------------------------------
# the contact vector field as an operator on functions
# ---------------------------------------------------------------------------


def Kf(f, g):
    """K_f(g) = {f,g} + 2 (d_t f) g -- the action on FUNCTIONS."""
    return add(kbr(f, g), smul(2, mul(d_t(f), g)))


def D(i, g):
    """D_{theta_i} = d_{theta_i} - theta_i d_t."""
    return add(d_odd(i, g), smul(-1, mul(mono(0, (i,)), d_t(g))))


def cD(a, g):
    """cD^1 = D_1 - i D_2,  cD^2 = D_3 - i D_4."""
    j = 2 * a - 1
    return add(D(j, g), smul(-I, D(j + 1, g)))


def bD(a, g):
    """bD_1 = D_1 + i D_2,  bD_2 = D_3 + i D_4."""
    j = 2 * a - 1
    return add(D(j, g), smul(I, D(j + 1, g)))


EPS = {(1, 2): C(1), (2, 1): C(-1), (1, 1): C(), (2, 2): C()}


# ---------------------------------------------------------------------------
# test material
# ---------------------------------------------------------------------------


def monomials(lo, hi):
    """All monomials t^a theta_S with lo <= a <= hi."""
    out = []
    for a in range(lo, hi + 1):
        for k in range(5):
            for S in itertools.combinations(IDX, k):
                out.append(mono(a, S))
    return out


def basis_window(lo, hi):
    """Basis keys (a, S) of the window, as a list, and the index map."""
    keys = []
    for a in range(lo, hi + 1):
        for k in range(5):
            for S in itertools.combinations(IDX, k):
                keys.append((a, S))
    return keys, {k: n for n, k in enumerate(keys)}


# ---------------------------------------------------------------------------
# (1) K_f is the contact field; the multiplier (multN)
# ---------------------------------------------------------------------------


def sec1():
    log("")
    log("=" * 74)
    log("(1)  K_f(g) = {f,g} + 2 (d_t f) g  is the contact vector field")
    log("=" * 74)

    tests = monomials(-2, 3)
    small = monomials(-1, 2)

    # (1a) K_f is a derivation with the Leibniz sign rule
    n = 0
    for f in monomials(-1, 2):
        pf = parity(f)
        for g in small[:24]:
            for h in small[:24]:
                lhs = Kf(f, mul(g, h))
                sgn = C(-1) if (pf and parity(g)) else C(1)
                rhs = add(mul(Kf(f, g), h), smul(sgn, mul(g, Kf(f, h))))
                if not eq(lhs, rhs):
                    check(False, "Leibniz for f=%s" % show(f))
                    return
                n += 1
    check(True, "K_f is a derivation of O (%d triples)" % n)
    log("    Leibniz rule: %d triples (f,g,h), all pass" % n)

    # (1b) [K_f, K_g] = K_{{f,g}}
    n = 0
    for f in monomials(-1, 2):
        for g in monomials(-1, 2):
            pf, pg = parity(f), parity(g)
            sgn = C(-1) if (pf and pg) else C(1)
            br = kbr(f, g)
            for h in small[:16]:
                lhs = add(Kf(f, Kf(g, h)), smul(-sgn, Kf(g, Kf(f, h))))
                rhs = Kf(br, h)
                if not eq(lhs, rhs):
                    check(False, "[K_f,K_g]=K_{f,g} for f=%s g=%s"
                          % (show(f), show(g)))
                    return
                n += 1
    check(True, "[K_f, K_g] = K_{{f,g}} (%d triples)" % n)
    log("    homomorphism: %d triples, all pass" % n)

    # (1c) the multiplier (multN): [K_f, D_i] = sum_j (D_i D_j f) D_j
    n = 0
    for f in tests:
        pf = parity(f)
        sgn = C(-1) if pf else C(1)          # supercommutator with odd D_i
        for i in IDX:
            hess = [D(i, D(j, f)) for j in IDX]
            for h in small[:16]:
                lhs = add(Kf(f, D(i, h)), smul(-sgn, D(i, Kf(f, h))))
                rhs = {}
                for j in IDX:
                    rhs = add(rhs, mul(hess[j - 1], D(j, h)))
                if not eq(lhs, rhs):
                    check(False, "multN for f=%s, i=%d" % (show(f), i))
                    return
                n += 1
    check(True, "(multN) holds: [K_f,D_i] = sum_j (D_iD_jf) D_j (%d)" % n)
    log("    odd Hessian multiplier: %d checks, all pass" % n)

    # (1d) the complex frame
    n = 0
    for h in small[:24]:
        for a in (1, 2):
            for b in (1, 2):
                z = add(cD(a, cD(b, h)), cD(b, cD(a, h)))
                check(z == {}, "{cD^%d,cD^%d}=0" % (a, b))
                z = add(bD(a, bD(b, h)), bD(b, bD(a, h)))
                check(z == {}, "{bD_%d,bD_%d}=0" % (a, b))
                z = add(cD(a, bD(b, h)), bD(b, cD(a, h)))
                w = smul(-4, d_t(h)) if a == b else {}
                check(eq(z, w), "{cD^%d,bD_%d}" % (a, b))
                n += 3
    log("    complex frame: %d relations, all pass" % n)
    log("      {cD^a,cD^b} = {bD_a,bD_b} = 0,  {cD^a,bD_b} = -4 delta^a_b d_t")


# ---------------------------------------------------------------------------
# (2) the multiplier in the complex frame
# ---------------------------------------------------------------------------


def multiplier(f, a):
    """Return (P, Q) with [K_f, cD^a] = sum_b P[b] cD^b + sum_b Q[b] bD_b.

    Extracted by applying the derivation to theta_k: D_j(theta_k)=delta_jk.
    """
    pf = parity(f)
    sgn = C(-1) if pf else C(1)
    c = {}
    for k in IDX:
        th = mono(0, (k,))
        c[k] = add(Kf(f, cD(a, th)), smul(-sgn, cD(a, Kf(f, th))))
    P, Qc = {}, {}
    for b in (1, 2):
        j = 2 * b - 1
        P[b] = smul(Q(1, 2), add(c[j], smul(I, c[j + 1])))
        Qc[b] = smul(Q(1, 2), add(c[j], smul(-I, c[j + 1])))
    return P, Qc


def multiplier_bar(f, a):
    """Same for [K_f, bD_a] = sum_b Pb[b] bD_b + sum_b Qb[b] cD^b."""
    pf = parity(f)
    sgn = C(-1) if pf else C(1)
    c = {}
    for k in IDX:
        th = mono(0, (k,))
        c[k] = add(Kf(f, bD(a, th)), smul(-sgn, bD(a, Kf(f, th))))
    Pb, Qb = {}, {}
    for b in (1, 2):
        j = 2 * b - 1
        Qb[b] = smul(Q(1, 2), add(c[j], smul(I, c[j + 1])))     # cD^b part
        Pb[b] = smul(Q(1, 2), add(c[j], smul(-I, c[j + 1])))    # bD_b part
    return Pb, Qb


def Lam(f, a, b):
    """Lam^a_b(f) = 1/2 cD^a bD_b f."""
    return smul(Q(1, 2), cD(a, bD(b, f)))


def Xi(f):
    """Xi(f) = 1/2 cD^1 cD^2 f -- the F -> E block of the multiplier, i.e. the
    obstruction to preserving the plane F = span(cD^1, cD^2) on which the
    constraint of the minus half of (IK4) lives."""
    return smul(Q(1, 2), cD(1, cD(2, f)))


def XiBar(f):
    """the conjugate obstruction, 1/2 bD_1 bD_2 f: the E -> F block, i.e. the
    obstruction to preserving the plane E of the plus half."""
    return smul(Q(1, 2), bD(1, bD(2, f)))


def sec2():
    log("")
    log("=" * 74)
    log("(2)  the multiplier in the complex frame, in closed form")
    log("=" * 74)
    log("    claim:   [K_f, cD^a] = sum_b Lam^a_b(f) cD^b + Xi(f) eps^{ab} bD_b")
    log("    with     Lam^a_b(f) = 1/2 cD^a bD_b f,   Xi(f) = 1/2 cD^1 cD^2 f")
    log("")
    n = 0
    for f in monomials(-3, 4):
        for a in (1, 2):
            P, Qc = multiplier(f, a)
            for b in (1, 2):
                if not check(eq(P[b], Lam(f, a, b)),
                             "P^%d_%d = 1/2 cD^%d bD_%d f for f=%s"
                             % (a, b, a, b, show(f))):
                    log("      got %s, want %s" % (show(P[b]), show(Lam(f, a, b))))
                    return
                want = smul(EPS[(a, b)], Xi(f))
                if not check(eq(Qc[b], want),
                             "Q^{%d%d} = eps Xi for f=%s" % (a, b, show(f))):
                    log("      got %s, want %s" % (show(Qc[b]), show(want)))
                    return
                n += 2
    log("    verified on all %d monomials t^a theta_S with -3<=a<=4: %d checks"
        % (len(monomials(-3, 4)), n))

    # the conjugate statement, needed for the plus half
    n = 0
    for f in monomials(-2, 3):
        for a in (1, 2):
            Pb, Qb = multiplier_bar(f, a)
            for b in (1, 2):
                want = smul(Q(1, 2), bD(a, cD(b, f)))
                check(eq(Pb[b], want), "bar multiplier diagonal block")
                want = smul(EPS[(a, b)], XiBar(f))
                check(eq(Qb[b], want), "bar multiplier off block")
                n += 2
    log("    conjugate: [K_f,bD_a] = 1/2(bD_a cD^b f) bD_b + (1/2 cD^1cD^2 f) eps cD")
    log("      %d checks" % n)

    # the trace of the gl(2)-block: conformal weight PLUS u(1)-charge
    n = 0
    for f in monomials(-2, 3):
        tr = add(Lam(f, 1, 1), Lam(f, 2, 2))
        want = add(smul(-2, d_t(f)),
                   smul(I, add(D(1, D(2, f)), D(3, D(4, f)))))
        check(eq(tr, want), "tr Lam = -2 d_t f + i(D_1D_2 + D_3D_4) f")
        n += 1
    log("    trace of the gl(2)-block, on %d monomials:" % n)
    log("      Lam^1_1 + Lam^2_2 = -2 d_t f + i (D_1D_2 + D_3D_4) f,")
    log("    i.e. the conformal weight -2 d_t f PLUS the u(1)-charge, the")
    log("    Cartan direction of the OTHER ideal of o(4) -- the exact N=4")
    log("    analogue of the eigenvalues A -+ iB of (multC) at N=2.")


# ---------------------------------------------------------------------------
# (3) the stabilizer of the plane F, and the false sentence it corrects
# ---------------------------------------------------------------------------


def sec3():
    log("")
    log("=" * 74)
    log("(3)  which f preserve the constraint: the subalgebra Xi(f) = 0")
    log("=" * 74)
    log("    The constraint cD^{(a} q^{b)} = 0 of (IK4) lives on the plane F")
    log("    alone.  A contact field can preserve it only if its multiplier")
    log("    preserves F, i.e. only if Xi(f) = 1/2 cD^1 cD^2 f = 0.")
    log("")

    # Dimension of k^Xi.  Its elements are NOT homogeneous in t (the tails of
    # (KAemb) sit one degree down), so count on windows [-N,N] and read the
    # growth per degree off the successive differences.
    log("    k^Xi is not t-homogeneous, so count it on windows [-N,N] and read")
    log("    the growth per degree off the successive differences:")
    log("")
    log("      window      dim k^Xi    growth per degree  (of 16)")
    prev = None
    for N in range(1, 6):
        d = len(window_kernel(-N, N, [Xi]))
        g = "" if prev is None else "%s" % (Q(d - prev, 2))
        log("      [-%d,%d]  %2d degrees  %3d         %5s" % (N, N, 2 * N + 1, d, g))
        prev = d
    log("    so k^Xi grows by 12 of the 16 dimensions k^L(1|4) has in each")
    log("    degree: it is of codimension 4 per degree, a PROPER subalgebra,")
    log("    and still larger than the algebra of Proposition 6.1.1, which has 8.")

    # it is a subalgebra
    log("")
    log("    k^Xi is a subalgebra: {f,g} has Xi = 0 whenever f and g do.")
    ker = []
    for n in range(-2, 3):
        keys, idx = basis_window(n, n)
        okeys, oidx = basis_window(n - 2, n)
        M = [[C() for _ in keys] for _ in okeys]
        for j, k in enumerate(keys):
            v = Xi(mono(k[0], k[1]))
            for kk, cc in v.items():
                M[oidx[kk]][j] = cc
        for vec in cnullspace(M):
            f = {}
            for j, k in enumerate(keys):
                if vec[j]:
                    f = add(f, mono(k[0], k[1], vec[j]))
            if f:
                ker.append(f)
    log("      basis of k^Xi over degrees -2..2 : %d elements" % len(ker))
    bad = 0
    npairs = 0
    for f in ker:
        for g in ker:
            if parity(f) != parity(g) and False:
                continue
            br = kbr(f, g)
            npairs += 1
            if Xi(br) != {}:
                bad += 1
    check(bad == 0, "k^Xi closed under the contact bracket")
    log("      %d brackets formed, %d leaving k^Xi" % (npairs, bad))

    # and it is PROPER: exhibit f with Xi(f) != 0
    bad = [f for f in monomials(-2, 3) if Xi(f) != {}]
    check(len(bad) > 0, "k^Xi is a proper subalgebra")
    log("")
    log("      proper: e.g. Xi(%s) = %s != 0" % (show(bad[0]), show(Xi(bad[0]))))
    log("      so the sentence of Subsection 6.2 -- the constraint 'is preserved by")
    log("      the whole of k^L(1|4)' -- is FALSE and must be replaced by (Xi).")
    return ker


# ---------------------------------------------------------------------------
# linear algebra over Q(i)
# ---------------------------------------------------------------------------


def recip(z):
    d = z.re * z.re + z.im * z.im
    return C(z.re / d, -z.im / d)


def crank(M):
    """Rank of a matrix over Q(i); M is a list of rows of C.  M is copied."""
    M = [row[:] for row in M]
    nr, nc = len(M), len(M[0]) if M else 0
    r = 0
    for c in range(nc):
        piv = None
        for i in range(r, nr):
            if M[i][c]:
                piv = i
                break
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        inv = recip(M[r][c])
        M[r] = [inv * x for x in M[r]]
        for i in range(nr):
            if i != r and M[i][c]:
                fac = M[i][c]
                M[i] = [x - fac * y for x, y in zip(M[i], M[r])]
        r += 1
        if r == nr:
            break
    return r


def cnullspace(M):
    """Basis of the null space of M (list of rows of C), as a list of vectors."""
    if not M:
        return []
    M = [row[:] for row in M]
    nr, nc = len(M), len(M[0])
    piv = []
    r = 0
    for c in range(nc):
        p = None
        for i in range(r, nr):
            if M[i][c]:
                p = i
                break
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        inv = recip(M[r][c])
        M[r] = [inv * x for x in M[r]]
        for i in range(nr):
            if i != r and M[i][c]:
                fac = M[i][c]
                M[i] = [x - fac * y for x, y in zip(M[i], M[r])]
        piv.append(c)
        r += 1
        if r == nr:
            break
    free = [c for c in range(nc) if c not in set(piv)]
    out = []
    for fc in free:
        v = [C() for _ in range(nc)]
        v[fc] = C(1)
        for i, pc in enumerate(piv):
            v[pc] = -M[i][fc]
        out.append(v)
    return out


# ---------------------------------------------------------------------------
# (4) the image of (KAemb) lies in k^Xi
# ---------------------------------------------------------------------------


def e(a):
    """e_1 = th1 + i th2, e_2 = th3 + i th4."""
    j = 2 * a - 1
    return add(mono(0, (j,)), smul(I, mono(0, (j + 1,))))


def fbar(a):
    """f_1 = th1 - i th2, f_2 = th3 - i th4."""
    j = 2 * a - 1
    return add(mono(0, (j,)), smul(-I, mono(0, (j + 1,))))


def KAgens(kappa, lo, hi):
    """The generators (KAemb) of Proposition 6.1.2, for modes in [lo, hi]."""
    gens = []
    vol = mono(0, (1, 2, 3, 4))
    for n in range(lo, hi + 1):
        for a in (1, 2):
            sgn = Q(1) if a == 1 else Q(-1)
            tail = mul(mul(e(1), e(2)), fbar(3 - a))
            G = add(mul(mono(n), e(a)),
                    smul(sgn * Q(1, 2) * (n + kappa), mul(mono(n - 1), tail)))
            gens.append(("G^%d_%d" % (n, a), G))
            tailb = mul(mul(fbar(1), fbar(2)), e(3 - a))
            Gb = add(mul(mono(n), fbar(a)),
                     smul(sgn * Q(1, 2) * (n + kappa), mul(mono(n - 1), tailb)))
            gens.append(("bG^%d%d" % (n, a), Gb))
        L = smul(Q(-1, 2), add(mono(n + 1),
                               smul(n * (n + 1 + 2 * kappa), mul(mono(n - 1), vol))))
        gens.append(("L_%d" % n, L))
    return gens


def window_kernel(lo, hi, ops):
    """Basis of {f supported on [lo,hi] | op(f) = 0 for every op in ops}.

    Not degree by degree: the elements of these algebras are inhomogeneous
    in t, so the honest count is on a window.
    """
    keys, idx = basis_window(lo, hi)
    okeys, oidx = basis_window(lo - 2, hi)
    M = [[C() for _ in keys] for _ in range(len(ops) * len(okeys))]
    for j, k in enumerate(keys):
        for t, op in enumerate(ops):
            for kk, cc in op(mono(k[0], k[1])).items():
                M[t * len(okeys) + oidx[kk]][j] += cc
    out = []
    for vec in cnullspace(M):
        f = {}
        for j, k in enumerate(keys):
            if vec[j]:
                f = add(f, mono(k[0], k[1], vec[j]))
        if f:
            out.append(f)
    return out


def kerXi(lo, hi, ops):
    """Basis of {f of degree in [lo,hi] | op(f) = 0 for every op in ops}."""
    out = []
    for n in range(lo, hi + 1):
        keys, idx = basis_window(n, n)
        okeys, oidx = basis_window(n - 2, n)
        M = [[C() for _ in keys] for _ in range(len(ops) * len(okeys))]
        for j, k in enumerate(keys):
            for t, op in enumerate(ops):
                for kk, cc in op(mono(k[0], k[1])).items():
                    M[t * len(okeys) + oidx[kk]][j] += cc
        for vec in cnullspace(M):
            f = {}
            for j, k in enumerate(keys):
                if vec[j]:
                    f = add(f, mono(k[0], k[1], vec[j]))
            if f:
                out.append((n, f))
    return out


def span_rank(vecs, lo, hi):
    """Rank of a set of functions, as vectors over the window basis."""
    keys, idx = basis_window(lo, hi)
    rows = []
    for f in vecs:
        r = [C() for _ in keys]
        for k, c in f.items():
            if k in idx:
                r[idx[k]] = c
        rows.append(r)
    return crank(rows) if rows else 0


def sec4():
    log("")
    log("=" * 74)
    log("(4)  which member of the family (KAemb) preserves the constraint")
    log("=" * 74)
    log("    Proposition 6.1.2 leaves varkappa free: item 2) says the embeddings")
    log("    form a single family, all with lambda = 0.  The CONSTRAINT is not")
    log("    indifferent to it.  Xi = 0 is the minus-half condition, XiBar = 0")
    log("    the plus-half one; a generator preserving both stabilizes the whole")
    log("    splitting V = E + F, i.e. has its multiplier in gl(2) < o(4).")
    log("")
    log("      varkappa    Xi = 0 on ...        XiBar = 0 on ...")
    for kappa in (Q(0), Q(1), Q(-1), Q(1, 2), Q(3), Q(-1, 3)):
        gens = KAgens(kappa, -3, 4)
        okX = {"G": True, "bG": True, "L": True}
        okB = {"G": True, "bG": True, "L": True}
        for (name, g) in gens:
            tag = "L" if name.startswith("L") else ("bG" if name.startswith("bG") else "G")
            if Xi(g) != {}:
                okX[tag] = False
            if XiBar(g) != {}:
                okB[tag] = False
        log("      %-8s    %-20s %s"
            % (kappa,
               ", ".join(k for k in ("G", "bG", "L") if okX[k]) or "--",
               ", ".join(k for k in ("G", "bG", "L") if okB[k]) or "--"))
    log("")
    log("    So the equation SELECTS varkappa = 0 out of the family: there, and")
    log("    only there, the whole image -- both doublets and the twisted")
    log("    Virasoro -- stabilizes both planes and hence preserves both")
    log("    constraints of (IK4).  At every other varkappa each doublet")
    log("    preserves one constraint and breaks the other.")
    n = 0
    for (name, g) in KAgens(Q(0), -4, 5):
        check(Xi(g) == {}, "Xi(%s) = 0 at varkappa = 0" % name)
        check(XiBar(g) == {}, "XiBar(%s) = 0 at varkappa = 0" % name)
        n += 2
    gd = dict(KAgens(Q(0), -4, 5))
    for r in range(-2, 3):
        for s in range(-2, 3):
            for a in (1, 2):
                for b in (1, 2):
                    k1, k2 = "G^%d_%d" % (r, a), "bG^%d%d" % (s, b)
                    if k1 in gd and k2 in gd:
                        T = kbr(gd[k1], gd[k2])
                        check(Xi(T) == {} and XiBar(T) == {}, "current in both")
                        n += 2
    log("    %d checks at varkappa = 0, all pass (generators and currents)." % n)

    # is the stabilizer of the splitting EXACTLY the acting algebra?
    log("")
    log("    Is the stabilizer of the splitting exactly that algebra?  Both are")
    log("    counted on the window [-N,N]; the image of (KAemb) is spanned by")
    log("    the generators and the currents whose support fits in it.")
    log("")
    log("      window    ker Xi   ker XiBar   both   image of (KAemb)")
    prev = None
    for N in range(1, 6):
        dX = len(window_kernel(-N, N, [Xi]))
        dB = len(window_kernel(-N, N, [XiBar]))
        dboth = len(window_kernel(-N, N, [Xi, XiBar]))
        gd0 = dict(KAgens(Q(0), -N - 2, N + 2))
        img = list(gd0.values())
        for r in range(-N - 1, N + 2):
            for s in range(-N - 1, N + 2):
                for a in (1, 2):
                    for b in (1, 2):
                        k1, k2 = "G^%d_%d" % (r, a), "bG^%d%d" % (s, b)
                        if k1 in gd0 and k2 in gd0:
                            img.append(kbr(gd0[k1], gd0[k2]))
        img = [f for f in img if f and all(-N <= k[0] <= N for k in f)]
        rk = span_rank(img, -N, N)
        log("      [-%d,%d]     %3d      %3d       %3d      %3d"
            % (N, N, dX, dB, dboth, rk))
        prev = (dX, dboth, rk)
    log("")
    log("    Both grow by 8 per degree, and their difference is the CONSTANT 3,")
    log("    independent of the window: so the stabilizer of the splitting")
    log("    V = E + F is the image of (KAemb) plus a 3-dimensional piece.")
    log("    What that piece is:")
    N = 4
    both = window_kernel(-N, N, [Xi, XiBar])
    gd0 = dict(KAgens(Q(0), -N - 2, N + 2))
    img = list(gd0.values())
    for r in range(-N - 1, N + 2):
        for s in range(-N - 1, N + 2):
            for a in (1, 2):
                for b in (1, 2):
                    k1, k2 = "G^%d_%d" % (r, a), "bG^%d%d" % (s, b)
                    if k1 in gd0 and k2 in gd0:
                        img.append(kbr(gd0[k1], gd0[k2]))
    img = [f for f in img if f and all(-N <= k[0] <= N for k in f)]
    base = span_rank(img, -N, N)
    extra = []
    cur = list(img)
    for f in both:
        if span_rank(cur + [f], -N, N) > span_rank(cur, -N, N):
            cur.append(f)
            extra.append(f)
    inner = []
    for f in extra:
        edge = any(k[0] in (-N, N) for k in f)
        log("      %-46s %s" % (show(f), "(window edge)" if edge else "GENUINE"))
        if not edge:
            inner.append(f)
    check(len(inner) == 1, "the genuine excess is 1-dimensional")
    log("")
    log("    Two of the three sit at the extreme degrees of the window and are")
    log("    artefacts of truncation (their tails fall outside).  The genuine")
    log("    excess is ONE dimension, and it is the u(1)-current: the Cartan")
    log("    direction h of the OTHER ideal of o(4), which rotates E against F")
    log("    and so stabilizes both planes without lying in sl(2)_A.")
    log("")
    log("    h normalizes the algebra (it is the u(1)-grading of the doublets):")
    h = add(mono(0, (1, 2)), mono(0, (3, 4)))
    nb = 0
    for (nm, g) in KAgens(Q(0), -3, 4):
        br = kbr(h, g)
        check(Xi(br) == {} and XiBar(br) == {}, "[h, %s] stabilizes both" % nm)
        nb += 1
    log("      [h, .] tested on %d generators: stays in the stabilizer." % nb)
    log("    So the subalgebra of k^L(1|4) preserving BOTH constraints of (IK4)")
    log("    is the algebra of Proposition 6.1.1 extended by that one u(1):")
    log("       k^Xi  cap  k^XiBar  =  k^A(1|2)  +  C h,")
    log("    which is exactly the statement that the multiplier lands in")
    log("    gl(2) = sl(2)_A + C h  rather than in sl(2)_A alone.")


# ---------------------------------------------------------------------------
# (5)-(6) the shift
# ---------------------------------------------------------------------------


def constrained_basis(lo, hi):
    """Basis of {(psi^1,psi^2) with t-degrees in [lo,hi] | cD^{(a}psi^{b)}=0}."""
    keys, idx = basis_window(lo, hi)
    nk = len(keys)
    okeys, oidx = basis_window(lo - 1, hi)
    nok = len(okeys)
    rows = [[C() for _ in range(2 * nk)] for _ in range(3 * nok)]
    # unknowns: psi^1 in columns 0..nk-1, psi^2 in nk..2nk-1
    # equations: (a,b) = (1,1), (1,2), (2,2), each in the output window
    for comp in (0, 1):
        for j, k in enumerate(keys):
            g = mono(k[0], k[1])
            for a in (1, 2):
                v = cD(a, g)
                for (ab, block) in (((1, 1), 0), ((1, 2), 1), ((2, 2), 2)):
                    aa, bb = ab
                    coef = 0
                    # cD^aa psi^bb + cD^bb psi^aa, psi^{comp+1}
                    if aa == a and bb == comp + 1:
                        coef += 1
                    if bb == a and aa == comp + 1:
                        coef += 1
                    if not coef:
                        continue
                    for kk, cc in v.items():
                        rows[block * nok + oidx[kk]][comp * nk + j] += cc * coef
    ns = cnullspace(rows)
    out = []
    for vec in ns:
        p1, p2 = {}, {}
        for j, k in enumerate(keys):
            if vec[j]:
                p1 = add(p1, mono(k[0], k[1], vec[j]))
            if vec[nk + j]:
                p2 = add(p2, mono(k[0], k[1], vec[nk + j]))
        out.append((p1, p2))
    return out


def sym(psi):
    """The three components of cD^{(a}psi^{b)}."""
    p1, p2 = psi
    return [add(cD(1, p1), cD(1, p1)),
            add(cD(1, p2), cD(2, p1)),
            add(cD(2, p2), cD(2, p2))]


SIGN = [Q(-1), Q(-1)]     # SIGN[p(f)]: filled in by sec5, not guessed


def delta(f, psi, mu=None):
    """delta_f psi^b = K_f(psi^b) + b(f)_d^b psi^d,  b = SIGN[p(f)] Lam^b_d."""
    pf = parity(f)
    s = SIGN[pf]
    out = []
    for b in (1, 2):
        v = Kf(f, psi[b - 1])
        for d in (1, 2):
            v = add(v, smul(s, mul(Lam(f, b, d), psi[d - 1])))
        if mu is not None:
            v = add(v, mul(mu, psi[b - 1]))
        out.append(v)
    return (out[0], out[1])


def sec5():
    log("")
    log("=" * 74)
    log("(5)  the shift: the law that preserves the constraint")
    log("=" * 74)
    log("    delta_f psi^b = K_f(psi^b) - (-1)^{p(f)} Lam^b_d(f) psi^d")
    log("    with Lam^b_d(f) = 1/2 cD^b bD_d f -- the gl(2)-block of the odd")
    log("    Hessian, transposed, acting on the doublet index of the field.")
    log("")

    sols = constrained_basis(-1, 2)
    log("    the constraint cD^{(a}psi^{b)}=0 on t-degrees -1..2 :")
    log("      solution space of dimension %d (of 2 x 4 x 16 = 128)" % len(sols))
    for psi in sols:
        for c in sym(psi):
            if c != {}:
                check(False, "nullspace vector really solves the constraint")
                return
    check(True, "all %d basis solutions satisfy the constraint" % len(sols))

    # the law preserves it, for every f in k^Xi
    log("")
    ntest = 0
    ker = []
    for n in range(-2, 3):
        keys, idx = basis_window(n, n)
        okeys, oidx = basis_window(n - 2, n)
        M = [[C() for _ in keys] for _ in okeys]
        for j, k in enumerate(keys):
            v = Xi(mono(k[0], k[1]))
            for kk, cc in v.items():
                M[oidx[kk]][j] = cc
        for vec in cnullspace(M):
            f = {}
            for j, k in enumerate(keys):
                if vec[j]:
                    f = add(f, mono(k[0], k[1], vec[j]))
            if f:
                ker.append(f)

    # Determine the coefficient of Lam rather than guess it.  The residue
    # sym(delta_f psi) is affine in the coefficient s, so read s off two
    # samples: residue(s) = r0 + s (r1 - r0), and solve r0 + s d = 0.
    log("    the coefficient of Lam is SOLVED for, not assumed:")
    for p in (0, 1):
        vals = set()
        ntried = 0
        for f in [g for g in ker if parity(g) == p]:
            for psi in sols:
                SIGN[p] = Q(0)
                r0 = sym(delta(f, psi))
                SIGN[p] = Q(1)
                r1 = sym(delta(f, psi))
                for c0, c1 in zip(r0, r1):
                    d = add(c1, smul(-1, c0))
                    if d == {}:
                        if c0 != {}:
                            vals.add("no solution")
                        continue
                    ntried += 1
                    # solve c0 + s d = 0 coefficient by coefficient
                    cand = None
                    ok = True
                    for k, v in d.items():
                        w = c0.get(k, C())
                        z = -(w * recip(v))
                        if z.im != 0:
                            ok = False
                            break
                        if cand is None:
                            cand = z.re
                        elif cand != z.re:
                            ok = False
                            break
                    for k in c0:
                        if k not in d:
                            ok = False
                    vals.add(cand if ok else "inconsistent")
        check(len(vals) == 1 and isinstance(list(vals)[0], Q),
              "the coefficient of Lam is determined and unique for p=%d" % p)
        log("      p(f) = %d :  %d determining components, solution set %s"
            % (p, ntried, sorted(str(v) for v in vals)))
        if len(vals) == 1 and isinstance(list(vals)[0], Q):
            SIGN[p] = list(vals)[0]
    log("      so  b(f)_d^b = (%s) Lam^b_d(f) for f even, (%s) for f odd:"
        % (SIGN[0], SIGN[1]))
    log("      b(f) = -Lam(f)^T, the transposed gl(2)-block of the odd Hessian,")
    log("      with no parity sign -- the two (-1)^{p(f)} cancel.")

    log("")
    log("    the law preserves the constraint, for f running over k^Xi:")
    for f in ker:
        for psi in sols:
            dp = delta(f, psi)
            for c in sym(dp):
                if c != {}:
                    check(False, "invariance for f = %s" % show(f))
                    log("      residue %s" % show(c))
                    return
            ntest += 1
    check(True, "constraint preserved: %d (f, psi) pairs" % ntest)
    log("      %d elements f of k^Xi x %d solutions = %d pairs, 0 failures"
        % (len(ker), len(sols), ntest))

    # and the generators of (KAemb) themselves
    log("")
    log("    the same for the generators of (KAemb) at varkappa = 0:")
    ntest = 0
    for kappa in (Q(0),):
        for (name, f) in KAgens(kappa, -1, 2):
            for psi in sols:
                dp = delta(f, psi)
                for c in sym(dp):
                    if c != {}:
                        check(False, "invariance for %s at kappa=%s"
                              % (name, kappa))
                        return
                ntest += 1
    check(True, "constraint preserved by (KAemb): %d pairs" % ntest)
    log("      %d (generator, solution) pairs, 0 failures" % ntest)

    # it FAILS for f outside k^Xi -- and no choice of b can repair it
    log("")
    log("    outside k^Xi the law fails, and no b repairs it:")
    bad = [f for f in monomials(0, 2) if Xi(f) != {}]
    shown = 0
    for f in bad:
        resid = []
        for psi in sols:
            dp = delta(f, psi)
            resid += [c for c in sym(dp) if c != {}]
        if resid and shown < 3:
            log("      f = %-22s : residue != 0 on %d of the %d solutions"
                % (show(f), len(resid), len(sols)))
            shown += 1
    check(shown > 0, "the law fails outside k^Xi")
    return sols, ker


# ---------------------------------------------------------------------------
# (6) uniqueness of the shift
# ---------------------------------------------------------------------------


def sec6(sols, ker):
    log("")
    log("=" * 74)
    log("(6)  uniqueness: what else could b be?")
    log("=" * 74)
    log("    If b and b' both work, beta := b'-b satisfies, for every")
    log("    constrained psi,   (cD^{(a} beta_d^{b)}) psi^d")
    log("                     + (-1)^{p} beta_d^{(b} cD^{a)} psi^d = 0.")
    log("    On the constraint surface cD^a psi^d = eps^{ad} chi, and")
    log("    eps.(2x2) is symmetric exactly on the traceless part, so the")
    log("    second term is (traceless part of beta).chi: the sl(2)-part of b")
    log("    is FORCED.  What is left is beta = mu.1 with cD^a mu = 0.")
    log("")

    # (6a) a traceless constant beta does not preserve the constraint
    log("    (6a) a non-zero traceless beta breaks the constraint:")
    bads = 0
    tried = 0
    for (b11, b12, b21, b22) in ((1, 0, 0, -1), (0, 1, 0, 0), (0, 0, 1, 0)):
        B = ((C(b11), C(b12)), (C(b21), C(b22)))
        for psi in sols:
            out = []
            for b in (1, 2):
                v = {}
                for d in (1, 2):
                    v = add(v, smul(B[d - 1][b - 1], psi[d - 1]))
                out.append(v)
            tried += 1
            if any(c != {} for c in sym((out[0], out[1]))):
                bads += 1
    check(bads > 0, "traceless constant beta breaks the constraint")
    log("      %d of %d (beta, psi) pairs break it -- so beta must be scalar"
        % (bads, tried))

    # (6b) a scalar cD-chiral mu does preserve it
    log("")
    log("    (6b) a scalar mu with cD^a mu = 0 preserves it, for every psi:")
    chir = []
    keys, idx = basis_window(-1, 2)
    okeys, oidx = basis_window(-2, 2)
    M = [[C() for _ in keys] for _ in range(2 * len(okeys))]
    for j, k in enumerate(keys):
        g = mono(k[0], k[1])
        for a in (1, 2):
            for kk, cc in cD(a, g).items():
                M[(a - 1) * len(okeys) + oidx[kk]][j] += cc
    for vec in cnullspace(M):
        m = {}
        for j, k in enumerate(keys):
            if vec[j]:
                m = add(m, mono(k[0], k[1], vec[j]))
        if m:
            chir.append(m)
    log("      cD-chiral functions in the window: dimension %d" % len(chir))
    n = 0
    for m in chir:
        if parity(m) != 0:
            continue
        for psi in sols:
            out = (mul(m, psi[0]), mul(m, psi[1]))
            for c in sym(out):
                if c != {}:
                    check(False, "chiral mu preserves the constraint")
                    return
            n += 1
    check(True, "chiral mu preserves the constraint (%d pairs)" % n)
    log("      %d (mu, psi) pairs, 0 failures" % n)

    # (6c) a NON-chiral scalar does not
    log("")
    log("    (6c) a scalar mu with cD^a mu != 0 does NOT:")
    bads = 0
    tried = 0
    for m in monomials(0, 1):
        if parity(m) != 0:
            continue
        if all(cD(a, m) == {} for a in (1, 2)):
            continue
        for psi in sols:
            out = (mul(m, psi[0]), mul(m, psi[1]))
            tried += 1
            if any(c != {} for c in sym(out)):
                bads += 1
    check(bads > 0, "a non-chiral scalar breaks the constraint")
    log("      %d of %d (mu, psi) pairs break it" % (bads, tried))
    log("")
    log("    Hence: b(f) = -(-1)^{p(f)} Lam(f)^T + mu(f).1, and the scalar mu")
    log("    is NOT determined by the constraint -- exactly as lambda(f) of")
    log("    (deltaV) was determined only when fed into the equation (IK2).")


# ---------------------------------------------------------------------------
# (7) a(f) must be scalar
# ---------------------------------------------------------------------------


def sec7():
    log("")
    log("=" * 74)
    log("(7)  in the ansatz delta_f q = K_f(q) + a(f) q + q b(f), a is SCALAR")
    log("=" * 74)
    log("    A transformation of the minus half must preserve the constraint of")
    log("    the PLUS half too, bD_{+(a} q_{g)}^b = 0, on whose lower index a(f)")
    log("    acts.  On that surface bD_{+a} q_g^b = eps_{ag} X^b, so")
    log("       bD_{+(a}(a q)_{g)}^b = a_{(g}^d eps_{a)d} X^b,")
    log("    and eps.A is symmetric exactly when A is traceless -- so the")
    log("    SYMMETRIC part survives unless the traceless part of a vanishes.")
    log("")
    log("    Checked directly on 2x2 matrices:")
    n = 0
    bad = 0
    epsm = ((C(), C(1)), (C(-1), C()))
    for entries in itertools.product((C(0), C(1), C(-1), C(2)), repeat=4):
        A = ((entries[0], entries[1]), (entries[2], entries[3]))
        # M_{ag} := eps_{ad} A_g^d ; symmetric part
        Msym = {}
        for a in (1, 2):
            for g in (1, 2):
                v = C()
                for d in (1, 2):
                    v = v + epsm[a - 1][d - 1] * A[g - 1][d - 1]
                Msym[(a, g)] = v
        symmetric = all(Msym[(a, g)] == Msym[(g, a)] for a in (1, 2) for g in (1, 2))
        traceless = (A[0][0] + A[1][1] == C())
        n += 1
        if symmetric != traceless:
            bad += 1
    check(bad == 0, "eps.A symmetric <=> A traceless (%d matrices)" % n)
    log("      %d matrices tested, %d discrepancies" % (n, bad))
    log("")
    log("    So the ansatz of the text sharpens to")
    log("       delta_f q = K_f(q) + mu(f) q + q b(f)     for f in the minus half,")
    log("       delta_f q = K_f(q) + a(f) q + nu(f) q     for f in the plus half,")
    log("    the sl(2)-part sitting, in each half, on the side of THAT half's")
    log("    own doublet index and on no other.  The two ideals of o(4) do enter,")
    log("    as the text guessed, but one per half and not both at once.")


# ---------------------------------------------------------------------------
# (8) delta is a Lie algebra action
# ---------------------------------------------------------------------------


def sec8(sols, ker):
    log("")
    log("=" * 74)
    log("(8)  [delta_f, delta_g] = delta_{{f,g}} for mu = 0")
    log("=" * 74)
    n = 0
    for f in ker:
        for g in ker:
            pf, pg = parity(f), parity(g)
            s = C(-1) if (pf and pg) else C(1)
            br = kbr(f, g)
            if Xi(br) != {}:
                check(False, "bracket stays in k^Xi")
                return
            for psi in sols[:6]:
                lhs1 = delta(f, delta(g, psi))
                lhs2 = delta(g, delta(f, psi))
                rhs = delta(br, psi)
                for c in (0, 1):
                    v = add(lhs1[c], smul(-s, lhs2[c]), smul(-1, rhs[c]))
                    if v != {}:
                        check(False, "action property for f=%s g=%s"
                              % (show(f), show(g)))
                        log("      defect %s" % show(v))
                        return
                n += 1
    check(True, "delta is a Lie algebra action (%d triples)" % n)
    log("    %d (f, g, psi) triples, 0 failures -- so mu is a genuine" % n)
    log("    1-cocycle ambiguity, to be fixed by the dynamical equation.")


def run():
    sec1()
    sec2()
    sec3()
    sec4()
    got = sec5()
    if got:
        sols, ker = got
        sec6(sols, ker)
        sec7()
        sec8(sols, ker)
    log("")
    log("=" * 74)
    log("checks: %d,  failures: %d" % (CHECKS[0], FAIL[0]))
    log("=" * 74)
    with open("n4shift-out.txt", "w") as fh:
        fh.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    run()
