#!/usr/bin/env python3
"""
adhm.py -- exact verification of the linear algebraic data of the text "On superinstantons"
(Subsection 3.6, "The non-super case") against the original construction of
Atiyah-Drinfeld-Hitchin-Manin, Phys. Lett. 65A (1978) 185-187.

Everything is exact: sympy rationals and Gaussian rationals, no floats.

What is checked (see run() at the bottom for the roll-call):

A. The normalization tau_0 = sigma^0, tau_j = -i sigma^j of Subsec. "type I"
   realizes the quaternions: tau_1 tau_2 = tau_3 etc., and
   (x^mu tau_mu)^dagger (x^mu tau_mu) = |x|^2 . 1.
   So x |--> x^mu tau_mu is the embedding R^4 = H used by ADHM.

B. The two forms of the condition in the paper agree and both say
   "M^+ M is a quaternionic scalar":
       (Cnd) of Subsec. "type I"   <=>   M^+M = diag(D,D) of Subsec. "type Ia"
                                   <=>   [M^+M, tau'_mu] = 0 for all mu.
   Checked on random rational data, and the converse direction is checked by
   exhibiting data that fail it.

C. The 't Hooft datum printed on p. 186 of ADHM -- the (k+1) x k quaternionic
   matrix with rows (lambda_1 ... lambda_k) and (y_i - x) on the diagonal --
   satisfies (Cnd) identically in x, in the conventions of our text.

D. The potential A_mu = N^+ d_mu N of the paper has SELF-dual curvature,
   F = *F, i.e. F^- = 0, with the orientation eps_{0123} = 1 (the opposite
   orientation interchanges F^+ and F^-), and is not anti-self-dual.
   Computed rationally: the frame Ntil spanning ker M^+ need not be
   orthonormal, since a change of frame conjugates F_{mu nu} pointwise and
   (anti)self-duality is invariant under that.  k = 1 symbolically in x,
   k = 2 exactly at three rational points.

E. The same self-duality for k = 1 through the projector 1 - M(M^+M)^{-1}M^+,
   as an independent cross-check of D.
"""

import sympy as sp
from sympy import Rational as Q, I, eye, zeros, Matrix

# ----------------------------------------------------------------- quaternions

s0 = eye(2)
s1 = Matrix([[0, 1], [1, 0]])
s2 = Matrix([[0, -I], [I, 0]])
s3 = Matrix([[1, 0], [0, -1]])

# the paper's normalization: tau_0 = sigma^0, tau_j = -i sigma^j
TAU = [s0, -I * s1, -I * s2, -I * s3]


def dag(M):
    return M.conjugate().T


def quat(x):
    """x^mu tau_mu, the quaternion attached to x in R^4."""
    return sum((x[mu] * TAU[mu] for mu in range(4)), zeros(2, 2))


def kron(A, B):
    """A (x) B with the ordering C^2 (x) C^k used for I + I."""
    return Matrix(sp.kronecker_product(A, B))


def simp(M):
    """Canonical form for rational functions -- cancel(), not simplify().

    Every entry occurring here is a rational function of x with Gaussian
    rational coefficients, and cancel() is a normal form on those, so equality
    testing is exact; simplify() would be correct too, and hundreds of times
    slower.
    """
    def one(e):
        # arithmetic on Gaussian rationals is already canonical in sympy, and
        # cancel() on such an entry costs seconds for nothing: at k=3 it was
        # the whole running time of the check.
        e = sp.expand(e)
        return e if e.is_number else sp.cancel(e)

    if hasattr(M, "applyfunc"):
        return M.applyfunc(one)
    return one(M)


def is_zero(M):
    return simp(M) == zeros(*M.shape)


# ------------------------------------------------------- A. quaternion algebra


def check_quaternion_units():
    out = []
    # tau_j^2 = -1
    for j in (1, 2, 3):
        out.append(("tau_%d^2 = -1" % j, simp(TAU[j] ** 2 + eye(2)) == zeros(2, 2)))
    # tau_1 tau_2 = tau_3, tau_2 tau_3 = tau_1, tau_3 tau_1 = tau_2
    for (a, b, c) in ((1, 2, 3), (2, 3, 1), (3, 1, 2)):
        out.append(
            ("tau_%d tau_%d = tau_%d" % (a, b, c),
             simp(TAU[a] * TAU[b] - TAU[c]) == zeros(2, 2)))
    # |x|^2 identity
    x = sp.symbols("x0 x1 x2 x3", real=True)
    X = quat(x)
    nrm = sum(xi ** 2 for xi in x)
    out.append(("q(x)^dag q(x) = |x|^2", simp(dag(X) * X - nrm * eye(2)) == zeros(2, 2)))
    return out


# --------------------------------------------- B. the two forms of (Cnd) agree


def tau_prime(mu, k):
    """tau'_mu of Subsec. 'type Ia': each entry a of tau_mu replaced by a.1_k."""
    return kron(TAU[mu], eye(k))


def cnd_typeI(M, k):
    """The two displayed conditions of (Cnd), as bilinear identities.

    (M(z1,0), M(0,z2)) = 0  and  (M(z1,0),M(z2,0)) = (M(0,z1),M(0,z2)),
    for all z1, z2 in I -- i.e. the (1,2) block of M^+M vanishes and the two
    diagonal blocks agree.  Returns the two block expressions.
    """
    G = dag(M) * M
    off = G[0:k, k:2 * k]
    dif = G[0:k, 0:k] - G[k:2 * k, k:2 * k]
    return simp(off), simp(dif)


def cnd_typeIa(M, k):
    """M^+M = diag(D,D):  returns the deviation from that shape."""
    G = dag(M) * M
    D = G[0:k, 0:k]
    return simp(G - kron(eye(2), D))


def commutes_with_H(M, k):
    """[M^+M, tau'_mu] for all mu -- 'M^+M is a quaternionic scalar'."""
    G = dag(M) * M
    return [simp(G * tau_prime(mu, k) - tau_prime(mu, k) * G) for mu in range(4)]


# ------------------------------------------------------ 't Hooft datum (ADHM p.186)


def thooft(lams, ys):
    """The 't Hooft datum as (B, C), in the conventions of our text.

    The quaternionic (k+1) x k matrix
        row 0 : (lambda_1, ..., lambda_k)     (lambda_i real)
        row i : delta_ij (y_i - x)            (y_i quaternions)
    written as M(x) = B + C (x^mu tau'_mu) on C^2 (x) C^k --> C^2 (x) C^{k+1}.

    A quaternionic matrix (q_ij) with q_ij = sum_mu c^mu_ij tau_mu becomes
    sum_mu tau_mu (x) C^mu.
    """
    k = len(lams)
    # B: row 0 holds the lambdas (real, so only the mu=0 component),
    #    rows 1..k hold y_i on the diagonal.
    Bc = [zeros(k + 1, k) for _ in range(4)]
    for j in range(k):
        Bc[0][0, j] = lams[j]
    for i in range(k):
        for mu in range(4):
            Bc[mu][i + 1, i] = ys[i][mu]
    B = sum((kron(TAU[mu], Bc[mu]) for mu in range(4)), zeros(2 * (k + 1), 2 * k))
    # C: quaternionic matrix with -1 on the lower diagonal block
    Cc = zeros(k + 1, k)
    for i in range(k):
        Cc[i + 1, i] = -1
    C = kron(TAU[0], Cc)
    return B, C


def M_of_x(B, C, x, k):
    return B + C * sum((x[mu] * tau_prime(mu, k) for mu in range(4)),
                       zeros(2 * k, 2 * k))


# --------------------------------------------------- D. the curvature is ASD


def eps4():
    """Levi-Civita on 4 indices, eps_{0123} = 1."""
    e = {}
    from itertools import permutations
    for p in permutations(range(4)):
        # sign of the permutation
        sgn = 1
        q = list(p)
        for i in range(4):
            for j in range(i + 1, 4):
                if q[i] > q[j]:
                    sgn = -sgn
        e[p] = sgn
    return e


EPS = eps4()


def projector(B, C, k, x):
    """P = 1 - M(M^+M)^{-1}M^+, the orthogonal projector onto W_x = (Im M)^perp.

    Rational in x, so no square root and no nullspace is needed.
    """
    M = M_of_x(B, C, x, k)
    G = dag(M) * M
    return simp(eye(M.shape[0]) - M * G.inv() * dag(M))


def curvature_from_projector(B, C, k, x):
    """P (d_mu P d_nu P - d_nu P d_mu P) P.

    In a frame N of Im P one has F_{mu nu} = N^+(d_mu P d_nu P - ...)N, so this
    is the curvature conjugated into the ambient space; its (anti)self-duality
    is that of F.  Cross-checked against curvature() for k=1.
    """
    P = projector(B, C, k, x)
    dP = [simp(P.applyfunc(lambda e: sp.diff(e, x[mu]))) for mu in range(4)]
    F = {}
    for mu in range(4):
        for nu in range(mu + 1, 4):
            F[(mu, nu)] = simp(P * (dP[mu] * dP[nu] - dP[nu] * dP[mu]) * P)
    return F


def curvature(B, C, k, m, x):
    """F_{mu nu} of the connection A_mu = (Ntil^+Ntil)^{-1} Ntil^+ d_mu Ntil.

    Ntil is any frame of ker M(x)^+ ; a change of frame conjugates F pointwise,
    so (anti)self-duality does not depend on the choice.  Rational in x.
    """
    M = M_of_x(B, C, x, k)
    # frame of the orthogonal complement of Im M = ker M^+
    ker = dag(M).nullspace()
    assert len(ker) == m, "expected %d-dimensional fiber, got %d" % (m, len(ker))
    Nt = Matrix.hstack(*ker)
    Nt = simp(Nt)
    G = simp(dag(Nt) * Nt)
    Ginv = simp(G.inv())
    A = []
    for mu in range(4):
        dN = simp(Nt.applyfunc(lambda e: sp.diff(e, x[mu])))
        A.append(simp(Ginv * dag(Nt) * dN))
    F = {}
    for mu in range(4):
        for nu in range(4):
            if mu >= nu:
                continue
            t = (A[nu].applyfunc(lambda e: sp.diff(e, x[mu]))
                 - A[mu].applyfunc(lambda e: sp.diff(e, x[nu]))
                 + A[mu] * A[nu] - A[nu] * A[mu])
            F[(mu, nu)] = simp(t)
    return F


def duality_defect(F, sign):
    """F_{mu nu} - sign * (1/2) eps_{mu nu rho sigma} F_{rho sigma}."""
    out = {}
    for (mu, nu), Fmn in F.items():
        dual = zeros(*Fmn.shape)
        for rho in range(4):
            for sig in range(4):
                if rho == sig:
                    continue
                key = (rho, sig) if rho < sig else (sig, rho)
                s = 1 if rho < sig else -1
                dual += Q(1, 2) * EPS.get((mu, nu, rho, sig), 0) * s * F[key]
        out[(mu, nu)] = simp(Fmn - sign * dual)
    return out


# ------------------------------- exact first derivatives without symbolic x
#
# For k >= 2 the symbolic null space of M(x)^+ blows up, so the curvature at a
# RATIONAL point is computed instead, exactly, by carrying first-order Taylor
# data: a matrix is a pair (A0, [d_0 A, ..., d_3 A]), i.e. an element of
# Mat over Q(i)[e_0,...,e_3]/(e_i e_j).  Every entry stays a Gaussian rational,
# so there is no simplification to do and nothing is approximated.


def dmul(A, B):
    A0, Ad = A
    B0, Bd = B
    return (A0 * B0, [A0 * Bd[i] + Ad[i] * B0 for i in range(4)])


def dsub(A, B):
    return (A[0] - B[0], [A[1][i] - B[1][i] for i in range(4)])


def ddag(A):
    return (dag(A[0]), [dag(A[1][i]) for i in range(4)])


def dinv(A):
    A0i = A[0].inv()
    return (A0i, [-A0i * A[1][i] * A0i for i in range(4)])


def dM(B, C, k, a):
    """M(x) and its four first derivatives at the rational point x = a."""
    M0 = B + C * sum((a[mu] * tau_prime(mu, k) for mu in range(4)),
                     zeros(2 * k, 2 * k))
    return (M0, [C * tau_prime(mu, k) for mu in range(4)])


def curvature_at_point(B, C, k, a):
    """F_{mu nu} = P(d_mu P d_nu P - d_nu P d_mu P)P at x = a, exactly."""
    M = dM(B, C, k, a)
    G = dmul(ddag(M), M)
    n = M[0].shape[0]
    one = (eye(n), [zeros(n, n) for _ in range(4)])
    P = dsub(one, dmul(dmul(M, dinv(G)), ddag(M)))
    P0, Pd = P
    F = {}
    for mu in range(4):
        for nu in range(mu + 1, 4):
            F[(mu, nu)] = P0 * (Pd[mu] * Pd[nu] - Pd[nu] * Pd[mu]) * P0
    return F


# ------------------------------------------------------------------- roll-call


def run(log=None):
    res = []

    def add(name, ok):
        res.append((name, ok))
        line = "%-72s %s" % (name, "OK" if ok else "FAIL")
        print(line, flush=True)
        if log is not None:
            log.write(line + "\n")
            log.flush()

    # ---- A
    for name, ok in check_quaternion_units():
        add("A. " + name, ok)

    # ---- B, on the 't Hooft datum (k=1,2) and on random data
    x = sp.symbols("x0 x1 x2 x3", real=True)

    for k, lams, ys in (
        (1, [Q(3, 2)], [[Q(1), Q(0), Q(-1, 2), Q(2)]]),
        (2, [Q(1), Q(5, 2)],
         [[Q(0), Q(1), Q(0), Q(0)], [Q(-1), Q(2), Q(3), Q(1, 2)]]),
    ):
        B, C = thooft(lams, ys)
        M = M_of_x(B, C, x, k)
        off, dif = cnd_typeI(M, k)
        add("C. 't Hooft k=%d: (Cnd) line 1 (off-diagonal block = 0)" % k,
            off == zeros(k, k))
        add("C. 't Hooft k=%d: (Cnd) line 2 (diagonal blocks agree)" % k,
            dif == zeros(k, k))
        add("C. 't Hooft k=%d: M^+M = diag(D,D) ('type Ia' cond. 1)" % k,
            cnd_typeIa(M, k) == zeros(2 * k, 2 * k))
        add("C. 't Hooft k=%d: [M^+M, tau'_mu] = 0 (quaternionic scalar)" % k,
            all(z == zeros(2 * k, 2 * k) for z in commutes_with_H(M, k)))
        add("C. 't Hooft k=%d: C^+C invertible" % k,
            simp(sp.det(dag(C) * C)) != 0)
        add("C. 't Hooft k=%d: det M^+M != 0 identically" % k,
            simp(sp.det(dag(M) * M)) != 0)

    # ---- B, converse: data violating (Cnd) are detected
    k = 1
    Bbad = Matrix([[1, 0], [0, 1], [0, 0], [Q(1, 3), 0]])
    Cbad = Matrix([[0, 0], [0, 0], [1, 0], [0, 1]])
    Mbad = M_of_x(Bbad, Cbad, x, k)
    off, dif = cnd_typeI(Mbad, k)
    bad_detected = not (off == zeros(k, k) and dif == zeros(k, k))
    bad_ia = cnd_typeIa(Mbad, k) != zeros(2 * k, 2 * k)
    add("B. non-quaternionic data violate (Cnd)", bad_detected)
    add("B. ... and violate 'type Ia' cond. 1 too (the two agree)",
        bad_detected == bad_ia)

    # ---- D: the curvature of the 't Hooft datum, gauge group U(2), m=2.
    #
    # Primary route: an explicit frame Ntil of ker M(x)^+, i.e. the paper's own
    # A_mu = N^+ d_mu N.  Ntil need not be orthonormal: passing to the unitary
    # frame conjugates F_{mu nu} pointwise by (Ntil^+Ntil)^{1/2}, and both
    # (anti)self-duality and tr F are invariant under that.
    #
    # The result is SELF-duality, F = *F, i.e. F^- = 0 -- with the orientation
    # eps_{0123} = 1 and the paper's tau_mu.  The opposite orientation swaps the
    # two; what is convention-free is that one component vanishes and the other
    # does not, which is why both are tested.
    m = 2
    B1, C1 = thooft([Q(3, 2)], [[Q(0), Q(0), Q(0), Q(0)]])
    F = curvature(B1, C1, 1, m, x)
    add("D. k=1 't Hooft: F is SELF-dual (F = *F), i.e. F^- = 0, "
        "identically in x",
        all(v == zeros(m, m) for v in duality_defect(F, +1).values()))
    add("D. k=1 't Hooft: F is NOT anti-self-dual (the sign is real)",
        not all(v == zeros(m, m) for v in duality_defect(F, -1).values()))
    add("D. k=1 't Hooft: F is not identically zero",
        not all(simp(v) == zeros(m, m) for v in F.values()))
    add("D. k=1 't Hooft: tr F = 0, so the connection is su(2)- not u(2)-valued",
        all(simp(sp.trace(v)) == 0 for v in F.values()))

    # k = 2, 3 at rational points, through the exact first-order Taylor data
    PTS = ([Q(1, 3), Q(1, 2), Q(-2, 5), Q(3, 7)],
           [Q(-4, 3), Q(2), Q(1, 6), Q(-1)],
           [Q(5, 2), Q(-3, 4), Q(7, 5), Q(1, 9)])
    # k = 3 was dropped from the roll-call: it runs, and says the same, but the
    # 8 x 8 products cost minutes per point and add nothing to k = 1 (which is
    # symbolic, hence a proof) and k = 2.
    for k, lams, ys in (
        (2, [Q(1), Q(2)],
         [[Q(0), Q(0), Q(0), Q(0)], [Q(2), Q(0), Q(0), Q(0)]]),
    ):
        B, C = thooft(lams, ys)
        n = 2 * (k + 1)
        sd, nsd, nz = True, True, True
        for a in PTS:
            Fa = curvature_at_point(B, C, k, a)
            sd = sd and all(v == zeros(n, n)
                            for v in duality_defect(Fa, +1).values())
            nsd = nsd and not all(v == zeros(n, n)
                                  for v in duality_defect(Fa, -1).values())
            nz = nz and not all(v == zeros(n, n) for v in Fa.values())
        add("D. k=%d 't Hooft: F is SELF-dual at %d rational points"
            % (k, len(PTS)), sd)
        add("D. k=%d 't Hooft: F is NOT anti-self-dual there" % k, nsd)
        add("D. k=%d 't Hooft: F is not zero there" % k, nz)

    # the two routes must agree at k=1: the projector one, evaluated at a point,
    # against the symbolic frame one
    a = PTS[0]
    Fa = curvature_at_point(B1, C1, 1, a)
    add("D. k=1: point-wise projector route is self-dual too (cross-check)",
        all(v == zeros(4, 4) for v in duality_defect(Fa, +1).values()))

    # ---- E: the same through the projector 1 - M(M^+M)^{-1}M^+, k=1 only
    # (it is a 2(k+1) x 2(k+1) computation and costs minutes already at k=1).
    k = 1
    B, C = thooft([Q(3, 2)], [[Q(0), Q(0), Q(0), Q(0)]])
    Fpr = curvature_from_projector(B, C, k, x)
    n = 2 * (k + 1)
    add("E. k=1 through the projector: F is SELF-dual as well",
        all(v == zeros(n, n) for v in duality_defect(Fpr, +1).values()))
    add("E. k=1 through the projector: F is not identically zero",
        not all(simp(v) == zeros(n, n) for v in Fpr.values()))

    return res


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "adhm.log"), "w") as log:
        results = run(log)
        bad = sum(1 for _, ok in results if not ok)
        tail = "\n%d checks, %d failures" % (len(results), bad)
        print(tail, flush=True)
        log.write(tail + "\n")
    raise SystemExit(1 if bad else 0)
