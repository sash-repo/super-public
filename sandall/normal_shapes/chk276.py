#!/usr/bin/env python3
"""Machine check of the p = 2 statements of the text.

For every normal shape of table (3.3) available over F_2 the script computes,
at the level of OPERATORS on O(m;u) (so that the summand d_i^2, which is not a
vector field when u_i > 1, is not lost), the square f^{[2]} of every basis
generating function f, and compares it with the closed formulas of the paper
specialized to p = 2:

    (u_i u_~i)^[2] = u_i u_~i                                    (pstrH 1)
    (u_i u_j)^[2]  = 0        for i,j both <= k or both > k      (pstrH 2)
    (u_j u_~i)^[2] = Phi_ij ubar_i ubar_{~j} u_j u_~i            (O4p), a = 1
    (u_j)^[2]      = sum_i alpha_ij u_i^(2^{u_i})                (pu2)
    (u_~i)^[2]     = sum_j alpha_ij u_~j^(2^{u_~j})              (pu2)

Run:  python3 chk276.py
"""
import sys, os, time, itertools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sk2 import (Ring, Ham, omega0, omega1, omega2, shapes, G_path,
                 genfun_basis, prepare_solver, solve_prepared, op_of_field,
                 op_of_operator_square, admissible_paths, mat_det)
import nalt2
from nalt2 import act, bracket, d2, gram


def Phi(ring, k, alpha, i, j):
    """Phi_ij of eq. (O4p) at p = 2: the sum over admissible paths, no signs."""
    alg = ring.big
    out = 0
    for P in admissible_paths(alpha, k, i, j):
        w = 1
        for s in range(len(P) - 1):
            w &= alpha[P[s]][P[s + 1]]
        if not w:
            continue
        t = alg.one
        for s in P[1:-1]:
            t = alg.mul(t, alg.mul(ring.ubar(s), ring.ubar(s + k)))
        out ^= t
    return out


def predict(ring, k, alpha, f_desc):
    """predicted f^{[2]} for f described as ('lin', i) or ('quad', i, j)."""
    alg = ring.big
    if f_desc[0] == 'lin':
        i = f_desc[1]
        out = 0
        if i < k:                                    # f = u_i, a "first half"
            for r in range(k):
                if alpha[r][i]:
                    out ^= ring.utop(r)
        else:                                        # f = u_~i
            for c in range(k):
                if alpha[i - k][c]:
                    out ^= ring.utop(c + k)
        return out
    _, a, b = f_desc                                 # f = u_a u_b, a < b
    if b == a + k:                                   # the torus
        return alg.mul(ring.gen(a), ring.gen(b))
    if (a < k) == (b < k):                           # same half
        return 0
    # f = u_j u_~i with j = the small index in the first half
    if a < k:
        j, i = a, b - k
    else:
        j, i = b, a - k
    ph = Phi(ring, k, alpha, i, j)
    if not ph:
        return 0
    t = alg.mul(ring.ubar(i), ring.ubar(j + k))
    t = alg.mul(t, alg.mul(ring.gen(j), ring.gen(i + k)))
    return alg.mul(ph, t)


def check_shape(k, alpha, name, u=None, quiet=False):
    m = 2 * k
    ring = Ring(m, u if u else (1,) * m)
    alg = ring.big
    om = omega1(ring, k, alpha)
    assert not d2(alg, om), "not closed"
    ham = Ham(ring, om)
    basis = genfun_basis(ring)
    piv = prepare_solver(ring, ham, basis)
    small_mons = [1 << t for t in ring.small]
    notinner, mismatches, nonzero = [], [], []
    for f in small_mons:
        D = ham.field(f)
        if not ham.preserves_small(D):
            notinner.append((f, "H_f leaves W(m;u)"))
            continue
        sq = op_of_operator_square(ring, D)
        g = solve_prepared(ring, piv, basis, sq)
        if g is None:
            notinner.append((f, "square not inner"))
            continue
        if g:
            nonzero.append((f, g))
        # compare with the prediction, where the paper has a formula
        a = alg.mons[[t for t in range(alg.dim) if (f >> t) & 1][0]]
        supp = [i for i in range(m) if a[i]]
        deg = sum(a)
        pred = None
        if deg == 1 and len(supp) == 1:
            pred = predict(ring, k, alpha, ('lin', supp[0]))
        elif deg == 2 and len(supp) == 2:
            pred = predict(ring, k, alpha, ('quad', supp[0], supp[1]))
        if pred is not None:
            # both sides are defined up to a constant (H_c = 0)
            if (g ^ pred) & ~alg.one:
                mismatches.append((f, g, pred))
    if not quiet:
        print("  %-14s u=%-16s basis %3d   non-inner %d   nonzero [2] %2d   "
              "formula mismatches %d"
              % (name, str(ring.u), len(small_mons), len(notinner),
                 len(nonzero), len(mismatches)))
        for f, why in notinner[:6]:
            print("        non-inner: %-16s (%s)" % (ring.show(f), why))
        for f, g, pred in mismatches[:6]:
            print("        MISMATCH %-12s computed %-22s predicted %s"
                  % (ring.show(f), ring.show(g), ring.show(pred)))
    return len(notinner), len(mismatches), len(nonzero)


def main():
    t0 = time.time()
    print("=" * 78)
    print("A) the 2-structure of h_omega(2k;u) at p = 2, u = 1")
    print("=" * 78)
    tot_bad = 0
    for k in (1, 2, 3):
        for name, alpha in shapes(k):
            ni, mm, nz = check_shape(k, alpha, name)
            tot_bad += ni + mm
        # the undeformed algebra, for comparison
        ni, mm, nz = check_shape(k, [[0] * k for _ in range(k)], "omega_0")
        tot_bad += ni + mm
    print("\n  total (non-inner + mismatches) for u = 1:", tot_bad)

    print("\n" + "=" * 78)
    print("B) shearing vectors u != 1")
    print("=" * 78)
    for u in [(2, 2), (2, 1, 1, 2), (1, 2, 2, 1), (2, 2, 2, 2)]:
        k = len(u) // 2
        for name, alpha in [("omega_0", [[0] * k for _ in range(k)])] + shapes(k):
            check_shape(k, alpha, name, u)
    print("\n[%.1f s]" % (time.time() - t0))


if __name__ == "__main__":
    main()
