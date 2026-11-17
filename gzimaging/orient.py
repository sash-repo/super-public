"""Orientation: what the clinic's group really is, and what opens when it is used.

One mathematical claim (due to H. Andersen) about the imaging text is checked here
from scratch.

  Subsec. 2.2 names the group the clinic imposes as E(3).  Patients are repositioned,
  never mirrored, so the group is SE(3); on the boundary surface Sigma that is SO(2)
  and not O(2).  Statement 3.12 counts O(2)-invariants.  Under SO(2) an
  ORIENTATION-ODD (pseudo-scalar) sector opens that 3.12 does not see, and the
  answer of the metric lift, formula (prop39), lives in it -- both the bracket and
  the Reeb term change sign under a reflection of Sigma.

What is computed, and nothing is asserted that the run does not check:

  Part 1.  The twist tau(A,B) = 1/2 tr(A eps B), eps = g^{-1}omega the rotation by
           pi/2.  Its three faces -- 1/2 tr(A eps B) = Im(conj(alpha) beta)
           = 1/4 (k1-k2)_A (k1-k2)_B sin 2 dphi -- antisymmetry, vanishing on an
           umbilic, and that omega ALONE cannot form it (Prop. B.3 item 2).

  Part 2.  The census, in both sectors, at weight 0 and weight 2.  The even half is
           the calibration: it must return Statement 3.12's own 15 = 10 + 5 and its
           2 at weight 0.  The odd half is the new count.

  Part 3.  The parity of (prop39) itself: evaluated on explicit polynomial tensor
           fields, and compared with its own reflection.

  Part 4.  Statement D.3 evaluated at the smoothing scales a study would actually
           use, and the resolution that is paid for them.

Conventions are those of `metriclift.py`, from which the index machinery is taken:
Sigma a surface, g a metric, A and B covariant symmetric 2-tensor fields, the weight
of an expression counting a covariant derivative 1 and the Gauss curvature K as 2.
"""

import os
import sys
from fractions import Fraction as Fr
from itertools import product, permutations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exactlin import nullspace, rank                        # noqa: E402

import sympy as sp                                          # noqa: E402

from metriclift import _sym_positions, DIM, J as SO2, REFL  # noqa: E402

LOG = None
CHECKS = [0, 0]


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    if LOG:
        LOG.write(s + "\n")
        LOG.flush()


def check(name, ok):
    CHECKS[0] += 1
    if not ok:
        CHECKS[1] += 1
    say("  [%s] %s" % ("ok" if ok else "FAILED", name))


# ===================================================================== part 1
# The twist.

def rot():
    """eps = g^{-1} omega in an orthonormal frame: the rotation by pi/2."""
    return sp.Matrix([[0, -1], [1, 0]])


def traceless(M):
    return M - sp.Rational(1, 2) * M.trace() * sp.eye(2)


def alpha_of(A):
    """alpha = 1/2 (A_11 - A_22) - i A_12, the paper's (hopf)."""
    return sp.Rational(1, 2) * (A[0, 0] - A[1, 1]) - sp.I * A[0, 1]


def tau(A, B):
    return sp.Rational(1, 2) * (A * rot() * B).trace()


def sym_from(kappa1, kappa2, phi):
    """The symmetric 2-tensor with principal curvatures (kappa1, kappa2) whose first
    principal direction makes the angle phi with the frame."""
    R = sp.Matrix([[sp.cos(phi), -sp.sin(phi)], [sp.sin(phi), sp.cos(phi)]])
    return R * sp.diag(kappa1, kappa2) * R.T


def part1():
    say("\n=== 1. The twist tau(A,B) = 1/2 tr(A eps B)")

    p, q, r, s, a, b = sp.symbols("p q r s a b", real=True)
    A = sp.Matrix([[a + p, q], [q, a - p]])
    B = sp.Matrix([[b + r, s], [s, b - r]])

    t = sp.expand(tau(A, B))
    say("  on A = a.1 + (p,q), B = b.1 + (r,s):  tau =", t)
    check("tau is independent of the traces a, b -- it sees the traceless parts only",
          sp.simplify(sp.diff(t, a)) == 0 and sp.simplify(sp.diff(t, b)) == 0)
    check("tau = q r - p s", sp.simplify(t - (q * r - p * s)) == 0)

    al, be = alpha_of(A), alpha_of(B)
    check("tau = Im(conj(alpha) beta)",
          sp.simplify(t - sp.im(sp.expand(sp.conjugate(al) * be))) == 0)
    check("Re(conj(alpha) beta) = 1/2 <Atl, Btl>",
          sp.simplify(sp.re(sp.expand(sp.conjugate(al) * be))
                      - sp.Rational(1, 2)
                      * sum(traceless(A)[i, j] * traceless(B)[i, j]
                            for i in range(2) for j in range(2))) == 0)

    check("tau is antisymmetric", sp.simplify(sp.expand(tau(B, A)) + t) == 0)
    check("tau(A,A) = 0", sp.simplify(sp.expand(tau(A, A))) == 0)

    # umbilic: A a multiple of g
    check("tau vanishes when either argument is umbilic",
          sp.simplify(sp.expand(tau(a * sp.eye(2), B))) == 0)

    # the principal-curvature form
    k1, k2, l1, l2, ph, ps = sp.symbols("k1 k2 l1 l2 varphi psi", real=True)
    AA, BB = sym_from(k1, k2, ph), sym_from(l1, l2, ps)
    lhs = sp.simplify(sp.expand_trig(sp.expand(tau(AA, BB))))
    rhs = sp.Rational(1, 4) * (k1 - k2) * (l1 - l2) * sp.sin(2 * (ph - ps))
    check("tau = 1/4 (k1-k2)_A (k1-k2)_B sin 2(phi_A - phi_B)",
          sp.simplify(sp.expand_trig(sp.expand(lhs - rhs))) == 0)

    # what omega alone cannot do (Prop. B.3 item 2): the three symplectic pairings
    # are symmetric, so no antisymmetric pointwise invariant of a pair exists there.
    def sympl(X, Y):
        return X.trace() * Y.trace() - (X * Y).trace()
    check("the pairing <.,.> of Prop. B.3 is symmetric",
          sp.simplify(sp.expand(sympl(A, B) - sympl(B, A))) == 0)
    say("  so the generators of Prop. B.3 item 2) are all symmetric and tau is not")
    say("  among them: tau needs g and omega together, i.e. eps = g^{-1}omega.")

    # and it is genuinely a pseudo-scalar: a reflection flips it
    M = sp.Matrix([[1, 0], [0, -1]])
    check("tau is ODD under a reflection of the frame",
          sp.simplify(sp.expand(tau(M.T * A * M, M.T * B * M) + t)) == 0)
    t0 = sp.Symbol("t0", real=True)
    rot_both = tau(sym_from(k1, k2, ph + t0), sym_from(l1, l2, ps + t0))
    ok, n = True, 0
    for i in range(7):
        for j in range(5):
            sub = {k1: 3, k2: -1, l1: 2, l2: 5,
                   ph: sp.Rational(i, 7), ps: sp.Rational(j, 5) + 1,
                   t0: sp.Rational(2 * i - j, 11)}
            n += 1
            ok = ok and sp.simplify(rot_both.subs(sub) - rhs.subs(sub)) == 0
    check("tau is EVEN under a rotation of the frame (%d exact pairs)" % n, ok and n == 35)


# ===================================================================== part 2
# The census in both sectors.
#
# `metriclift.invariant_space` builds the O(2)-invariant bilinear forms on the jet
# spaces by imposing so(2)-invariance and then killing every component that a
# reflection would flip.  The orientation-ODD sector is the same computation with
# the last condition reversed: keep exactly the components a reflection flips.

def invariant_space_pm(a, b, parity):
    """parity = +1: reflection-invariant (true scalars, the O(2) count of 3.12).
       parity = -1: reflection-anti-invariant (pseudo-scalars, new).
    Returns (basis, index list)."""
    n = 4 + a + b
    idx = _sym_positions(n)
    pos = {t: i for i, t in enumerate(idx)}
    rows = []

    def add_sym(k, l):
        for t in idx:
            if t[k] <= t[l]:
                continue
            u = list(t)
            u[k], u[l] = u[l], u[k]
            r = [0] * len(idx)
            r[pos[t]] += 1
            r[pos[tuple(u)]] -= 1
            rows.append(r)

    Aid, Ader = (0, 1), tuple(range(2, 2 + a))
    Bid, Bder = (2 + a, 3 + a), tuple(range(4 + a, 4 + a + b))
    add_sym(*Aid)
    add_sym(*Bid)
    for grp in (Ader, Bder):
        for i in range(len(grp) - 1):
            add_sym(grp[i], grp[i + 1])

    # so(2)-invariance -- the same in both sectors
    for t in idx:
        r = [0] * len(idx)
        for k in range(n):
            for mm in range(DIM):
                if SO2[mm][t[k]]:
                    u = list(t)
                    u[k] = mm
                    r[pos[tuple(u)]] += SO2[mm][t[k]]
        if any(r):
            rows.append(r)

    # the reflection condition, in whichever sector
    for t in idx:
        sign = 1
        for k in range(n):
            sign *= REFL[t[k]][t[k]]
        if sign != parity:
            r = [0] * len(idx)
            r[pos[t]] = 1
            rows.append(r)

    if not rows:
        return [[1 if i == j else 0 for i in range(len(idx))] for j in range(len(idx))], idx
    return nullspace(rows), idx


def swap_split(a, b, parity):
    """dim of the +1 and -1 eigenspaces of A <-> B inside the (a,b)=(1,1) block."""
    assert (a, b) == (1, 1)
    n = 6
    idx = _sym_positions(n)
    pos = {t: i for i, t in enumerate(idx)}
    basis, _ = invariant_space_pm(1, 1, parity)

    def swap_vec(v):
        w = [Fr(0)] * len(idx)
        for t in idx:
            u = (t[3], t[4], t[5], t[0], t[1], t[2])
            w[pos[u]] = Fr(v[pos[t]])
        return w

    sym, anti = [], []
    for v in basis:
        v = [Fr(z) for z in v]
        sv = swap_vec(v)
        sym.append([x + y for x, y in zip(v, sv)])
        anti.append([x - y for x, y in zip(v, sv)])
    return len(basis), rank(sym), rank(anti)


def census(parity, label):
    say("\n  --- the %s sector" % label)
    d = {}
    for (a, b) in [(2, 0), (1, 1), (0, 2)]:
        basis, _ = invariant_space_pm(a, b, parity)
        d[(a, b)] = len(basis)
        say("    (nabla^%d A).(nabla^%d B):  dim = %d" % (a, b, len(basis)))
    check("the two outer blocks agree by A <-> B", d[(2, 0)] == d[(0, 2)])

    tot11, ds, da = swap_split(1, 1, parity)
    say("    inside (nabla A).(nabla B):  symmetric %d, antisymmetric %d" % (ds, da))
    check("the (1,1) block splits %d = %d + %d" % (tot11, ds, da), ds + da == tot11)

    # weight 0, and with it the curvature terms K * (weight-0 invariant)
    basis00, _ = invariant_space_pm(0, 0, parity)
    n00 = len(basis00)
    # the A <-> B split of the weight-0 block
    idx0 = _sym_positions(4)
    pos0 = {t: i for i, t in enumerate(idx0)}
    s0, a0 = [], []
    for v in basis00:
        v = [Fr(z) for z in v]
        w = [Fr(0)] * len(idx0)
        for t in idx0:
            w[pos0[(t[2], t[3], t[0], t[1])]] = v[pos0[t]]
        s0.append([x + y for x, y in zip(v, w)])
        a0.append([x - y for x, y in zip(v, w)])
    ds0, da0 = rank(s0), rank(a0)
    say("    weight 0:  dim = %d = %d symmetric + %d antisymmetric" % (n00, ds0, da0))
    check("the weight-0 block splits %d = %d + %d" % (n00, ds0, da0), ds0 + da0 == n00)

    # the curvature terms are K times the weight-0 invariants, with the same split
    sym2 = d[(2, 0)] + ds + ds0
    anti2 = d[(2, 0)] + da + da0
    say("    curvature terms K * (weight-0): %d, with the same split %d + %d"
        % (n00, ds0, da0))
    say("    WEIGHT 2 TOTAL  %d = %d symmetric + %d antisymmetric"
        % (sym2 + anti2, sym2, anti2))
    return dict(w0=(n00, ds0, da0), w2=(sym2 + anti2, sym2, anti2),
                blocks=(d[(2, 0)], tot11, ds, da))


def part2():
    say("\n=== 2. The census of bilinear scalars, by sector")

    ev = census(+1, "EVEN (orientation-preserving AND reflections: the O(2) count of 3.12)")
    check("the even weight-0 count is 2, both symmetric", ev["w0"] == (2, 2, 0))
    check("the even weight-2 count is 15 = 10 + 5 -- Statement 3.12 reproduced",
          ev["w2"] == (15, 10, 5))
    check("the even blocks are 4 + 4 of nabla^2, and 5 of nabla.nabla splitting 4 + 1",
          ev["blocks"] == (4, 5, 4, 1))

    od = census(-1, "ODD (pseudo-scalars: what SO(2) adds and O(2) does not see)")

    say("\n  The odd sector in the shape H. Andersen gives it:")
    b20, t11, ds, da = od["blocks"]
    say("    (nabla^2 A).B  %d,   A.(nabla^2 B)  %d,   nabla A . nabla B  %d,"
        "   K tr(A eps B)  %d" % (b20, b20, t11, od["w0"][0]))
    check("her 3 + 3 + 5 + 1 = 12 at weight 2",
          (b20, b20, t11, od["w0"][0]) == (3, 3, 5, 1) and od["w2"][0] == 12)
    check("of the 12, eight are antisymmetric", od["w2"][2] == 8)
    check("at weight 0 the odd sector is one-dimensional and antisymmetric -- the twist",
          od["w0"] == (1, 0, 1))

    tot_even = ev["w2"][0] + ev["w0"][0]
    tot_odd = od["w2"][0] + od["w0"][0]
    ae, ao = ev["w2"][2] + ev["w0"][2], od["w2"][2] + od["w0"][2]
    say("\n  Weight at most 2, the two sectors together:")
    say("    even %d (antisymmetric %d) + odd %d (antisymmetric %d)"
        % (tot_even, ae, tot_odd, ao))
    say("    SO(2) total %d, of which %d antisymmetric" % (tot_even + tot_odd, ae + ao))
    check("the paper's 17 with 5 antisymmetric becomes 30 with 14",
          (tot_even, ae) == (17, 5) and (tot_even + tot_odd, ae + ao) == (30, 14))
    say("    so 5 survive if antisymmetric becomes 5 + 9 = 14, the 9 being")
    say("    8 of weight 2 and the twist, of weight 0.")

    # the twist, as a vector, must span the odd weight-0 space
    basis00, idx0 = invariant_space_pm(0, 0, -1)
    pos0 = {t: i for i, t in enumerate(idx0)}
    # tau = 1/2 A_ij eps_jk B_ki -- as a coefficient tensor c[i,j,k,l] A_ij B_kl
    # tau = 1/2 A_ij eps_jk B_kl delta_li, so c[i,j,k,l] = 1/2 eps_jk delta_li,
    # symmetrized over the two tensor pairs, the arguments being symmetric.
    eps = [[0, -1], [1, 0]]
    v = [Fr(0)] * len(idx0)
    for i, jj, k, l in product(range(2), repeat=4):
        if eps[jj][k] and l == i:
            v[pos0[(i, jj, k, l)]] += Fr(eps[jj][k], 2)
    for grp in ((0, 1), (2, 3)):
        w = [Fr(0)] * len(idx0)
        for t in idx0:
            u = list(t)
            u[grp[0]], u[grp[1]] = t[grp[1]], t[grp[0]]
            w[pos0[t]] = (v[pos0[t]] + v[pos0[tuple(u)]]) / 2
        v = w
    M = [[Fr(z) for z in r] for r in basis00]
    check("the twist is the odd weight-0 invariant (it spans that line)",
          rank(M) == 1 and rank(M + [v]) == 1)


# ===================================================================== part 3
# The parity of the metric lift's answer.

def part3():
    say("\n=== 3. The parity of (prop39) under a reflection of Sigma")
    x, y, th = sp.symbols("x y vartheta", real=True)

    def field(cs):
        """a symmetric 2-tensor field with affine entries, coefficients cs."""
        c = iter(cs)
        e = lambda: next(c) + next(c) * x + next(c) * y          # noqa: E731
        a11, a12, a22 = e(), e(), e()
        return sp.Matrix([[a11, a12], [a12, a22]])

    A = field([1, 2, -1, 3, -2, 1, -1, 1, 2])
    B = field([2, -1, 3, 1, 1, -2, 4, -3, 1])

    v = sp.Matrix([sp.cos(th), sp.sin(th)])
    Jv = sp.Matrix([-sp.sin(th), sp.cos(th)])

    def kap(T):
        return sp.expand_trig(sp.expand((v.T * T * v)[0, 0]))

    def dtheta(f):
        return sp.diff(f, th)

    def along(f, w):
        return sp.expand(w[0] * sp.diff(f, x) + w[1] * sp.diff(f, y))

    def F(A, B):
        kA, kB = kap(A), kap(B)
        brk = dtheta(kA) * along(kB, v) - along(kA, v) * dtheta(kB)
        reeb = -along(kA, Jv) * kB + kA * along(kB, Jv)
        return sp.expand(brk + reeb)

    f = F(A, B)
    check("(prop39) is not identically zero on this pair", sp.simplify(f) != 0)
    check("(prop39) is antisymmetric in (A,B)", sp.simplify(sp.expand(F(B, A) + f)) == 0)

    # the reflection sigma: (x,y) -> (x,-y), and on directions vartheta -> -vartheta
    M = sp.Matrix([[1, 0], [0, -1]])

    def pull(T):
        Ts = T.subs({y: -y}, simultaneous=True)
        return sp.expand(M.T * Ts * M)

    lhs = sp.expand_trig(sp.expand(F(pull(A), pull(B))))
    rhs = sp.expand_trig(sp.expand(-f.subs({y: -y, th: -th}, simultaneous=True)))
    check("F(sigma*A, sigma*B)(x,y,vartheta) = - F(A,B)(x,-y,-vartheta): it is ODD",
          sp.simplify(sp.expand_trig(sp.expand(lhs - rhs))) == 0)

    # the two halves separately, which is what her reason names
    def brk_only(A, B):
        kA, kB = kap(A), kap(B)
        return sp.expand(dtheta(kA) * along(kB, v) - along(kA, v) * dtheta(kB))

    def reeb_only(A, B):
        kA, kB = kap(A), kap(B)
        return sp.expand(-along(kA, Jv) * kB + kA * along(kB, Jv))

    for nm, G in (("the bracket", brk_only), ("the Reeb term", reeb_only)):
        g0 = sp.expand_trig(sp.expand(G(A, B)))
        g1 = sp.expand_trig(sp.expand(G(pull(A), pull(B))))
        g2 = sp.expand_trig(sp.expand(-g0.subs({y: -y, th: -th}, simultaneous=True)))
        check("%s alone is odd under the reflection" % nm,
              sp.simplify(sp.expand_trig(sp.expand(g1 - g2))) == 0)

    # and a control: a term with two horizontal derivatives is EVEN
    def two_horiz(A, B):
        return sp.expand(along(kap(A), v) * along(kap(B), v))
    g0 = sp.expand_trig(sp.expand(two_horiz(A, B)))
    g1 = sp.expand_trig(sp.expand(two_horiz(pull(A), pull(B))))
    g2 = sp.expand_trig(sp.expand(g0.subs({y: -y, th: -th}, simultaneous=True)))
    check("control: (nabla_v kA)(nabla_v kB) is EVEN under the same reflection",
          sp.simplify(sp.expand_trig(sp.expand(g1 - g2))) == 0)

    # the mean over the directions of (prop39) is zero -- the paper's own reason
    m = sp.integrate(f, (th, 0, 2 * sp.pi))
    check("the mean of (prop39) over the directions is zero", sp.simplify(m) == 0)


# ===================================================================== part 4
# Statement D.3 at the scales a study would use, and the resolution it costs.

def Ck(k):
    """C_k = M_k sqrt(2^k/(2k-1)!!), M_k = max |He_k(u)| exp(-u^2/2)."""
    u = sp.Symbol("u", real=True)
    He = sp.hermite_poly(k, u / sp.sqrt(2)) * 2 ** sp.Rational(-k, 2) if k else sp.Integer(1)
    f = sp.lambdify(u, sp.Abs(He) * sp.exp(-u ** 2 / 2), "math")
    best = 0.0
    n = 400000
    for i in range(n + 1):
        t = -8.0 + 16.0 * i / n
        best = max(best, f(t))
    dfact = 1
    for j in range(1, 2 * k, 2):
        dfact *= j
    return best * (2.0 ** k / dfact) ** 0.5


def part4():
    say("\n=== 4. Statement D.3 at other smoothing scales, and the resolution")
    C = [Ck(k) for k in range(5)]
    say("  C_k for 0 <= k <= 4: " + ", ".join("%.3f" % c for c in C))
    check("the prefactors of Statement D.3 are reproduced",
          all(abs(C[k] - v) < 5e-3
              for k, v in enumerate([1.0, 0.858, 1.155, 1.008, 1.171])))

    say("  penalty of an order-3 descriptor against the curvatures, SNR(2)/SNR(3):")
    rows = []
    for s in (1.0, 2.0, 3.0):
        for L in (2.0, 3.0, 5.0):
            r = s / (L * L + s * s) ** 0.5
            pen = (C[2] * r ** 2) / (C[3] * r ** 3)
            rows.append((s, L, pen))
            say("    s = %.1f mm, L = %.1f mm:  %.2f" % (s, L, pen))
    want = {(2.0, 3.0): 2.1, (2.0, 5.0): 3.1, (3.0, 3.0): 1.6, (3.0, 5.0): 2.2,
            (1.0, 2.0): 2.6, (1.0, 5.0): 5.8}
    for (s, L, pen) in rows:
        if (s, L) in want:
            check("s = %.0f, L = %.0f gives %.1f" % (s, L, want[(s, L)]),
                  abs(pen - want[(s, L)]) < 0.05)

    say("  the rows, in the shape of the table (tab:snr), for k = 3:")
    for s in (2.0, 3.0):
        for L in (3.0, 5.0):
            r = s / (L * L + s * s) ** 0.5
            zero = C[3] * r ** 3
            two = (C[2] * r ** 2) / (C[3] * r ** 3)
            say("    %.1f & %.1f & %.2f & 3 & 1/%.1f & 1/%.1f"
                % (s, L, s / L, 1 / zero, two))

    # the resolution: when do two Gaussian features of scale s merge?
    say("  the resolution paid for a smoothing s: two equal features at distance d")
    say("  merge when the smoothed profile stops being bimodal.")
    d, s = sp.symbols("d s", positive=True)
    xx = sp.Symbol("xx", real=True)
    prof = sp.exp(-(xx - d / 2) ** 2 / (2 * s ** 2)) + sp.exp(-(xx + d / 2) ** 2 / (2 * s ** 2))
    second = sp.simplify(sp.diff(prof, xx, 2).subs(xx, 0))
    sol = sp.solve(sp.simplify(second / sp.exp(-d ** 2 / (8 * s ** 2))), d)
    say("    the center turns from a minimum into a maximum at d =", sol)
    check("two features merge below d = 2s", any(sp.simplify(z - 2 * s) == 0 for z in sol))
    say("    so at s = 3 mm two features closer than 6 mm merge; at s = 2 mm, 4 mm.")

    # the bilinear-product point: relative errors add in quadrature
    say("  the signature is (order 2) x (order 3) in the voxels; for a product")
    say("  the relative errors add in quadrature, so with e2 <= e3 the total is")
    say("  between e3 and sqrt(2) e3 -- the k = 3 row is the bound, to within 1.42.")
    for e2, e3 in ((0.1, 0.4), (0.2, 0.5), (0.3, 0.3)):
        tot = (e2 * e2 + e3 * e3) ** 0.5
        check("e2 = %.1f, e3 = %.1f: total %.3f is in [e3, sqrt2 e3]" % (e2, e3, tot),
              e3 - 1e-12 <= tot <= 2 ** 0.5 * e3 + 1e-12)


def main():
    global LOG
    LOG = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "orient.log"), "w")
    say("Orientation, the twist, and the odd sector.")
    say("Nothing below is asserted that the run does not check.")
    part1()
    part2()
    part3()
    part4()
    say("\n%d checks, %d failed" % (CHECKS[0], CHECKS[1]))
    LOG.close()
    return 1 if CHECKS[1] else 0


if __name__ == "__main__":
    sys.exit(main())
