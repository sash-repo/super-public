#!/usr/bin/env python3
"""Email 276: the p = 2 case of the Poisson brackets and of the p-structures.

Run:  python3 run276.py [what]
where `what` is one of  brackets | pstr | second | nonalt | all.
"""
import sys, os, itertools, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sk2 import (Ring, Ham, omega0, omega1, omega2, omega_nonalt, shapes,
                 G_path, mat_inv, mat_det, genfun_basis, field_matrix,
                 solve_field, J0, Ccyc, Jkr)
import nalt2
from nalt2 import act, bracket, lie, d2, gram


def hdr(s):
    print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78)


# ------------------------------------------------------------------ brackets

def check_brackets(k, u=None, verbose=True):
    """for every shape: closedness, non-degeneracy, G = G_path, and the
    Poisson bracket on all pairs of monomials; plus [H_a,H_b] = H_{a,b}."""
    m = 2 * k
    ring = Ring(m, u if u else (1,) * m)
    alg = ring.big
    for name, alpha in shapes(k):
        om = omega1(ring, k, alpha)
        closed = not d2(alg, om)
        ham = Ham(ring, om)                    # asserts closed + non-degenerate
        # the block G = (1+S)^{-1} read off the full inverse Gram matrix
        Gfull = ham.Ginv
        G = [[Gfull[i + k][j] for j in range(k)] for i in range(k)]
        Gp = G_path(ring, k, alpha)
        agree = (G == Gp)
        # Poisson bracket on all pairs of monomials of O(m;u)
        mons = [1 << t for t in ring.small]
        bad_br = bad_jac = 0
        for a in mons:
            Ha = ham.field(a)
            if not ham.preserves_small(Ha):
                bad_br += 1
            for b in mons:
                lhs = act(alg, Ha, b)
                rhs = 0
                for i in range(k):
                    for j in range(k):
                        if not G[i][j]:
                            continue
                        t = alg.mul(alg.partial(i + k, a), alg.partial(j, b))
                        t ^= alg.mul(alg.partial(j, a), alg.partial(i + k, b))
                        rhs ^= alg.mul(G[i][j], t)
                if lhs != rhs:
                    bad_br += 1
        # [H_a,H_b] = H_{a,b} on a sample
        for a in mons:
            Ha = ham.field(a)
            for b in mons:
                Hb = ham.field(b)
                if bracket(alg, Ha, Hb) != ham.field(act(alg, Ha, b)):
                    bad_jac += 1
        if verbose:
            print("k=%d u=%s %-14s closed=%s nondeg=%s G=G_path:%s "
                  "bracket errors:%d  [H,H]=H errors:%d"
                  % (k, ring.u, name, closed, True, agree, bad_br, bad_jac))
    return True


# --------------------------------------------------------------- 2-structure

def pstructure(k, u=None, alpha=None, name="", show=None):
    """compute f^{[2]} for every basis generating function f."""
    m = 2 * k
    ring = Ring(m, u if u else (1,) * m)
    alg = ring.big
    om = omega1(ring, k, alpha) if alpha else omega0(ring, k)
    ham = Ham(ring, om)
    basis = genfun_basis(ring)
    cols = field_matrix(ham, basis)
    res = {}
    fails = []
    for f in basis:
        D = ham.field(f)
        D2 = ham.square(D)
        if any(not ring.in_small(c) for c in D2):
            fails.append((f, "square leaves W(m;u)"))
            continue
        g = solve_field(cols, basis, D2, ring)
        if g is None:
            fails.append((f, "not inner"))
        else:
            res[f] = g
    return ring, ham, basis, res, fails


def report_pstr(k, u=None):
    m = 2 * k
    for name, alpha in [("omega_0", None)] + shapes(k):
        ring, ham, basis, res, fails = pstructure(k, u, alpha, name)
        alg = ring.big
        nz = {f: g for f, g in res.items() if g}
        print("\n  %-14s  computed %d/%d,  non-zero squares: %d,  failures: %d"
              % (name, len(res), len(basis), len(nz), len(fails)))
        for f, why in fails:
            print("      FAIL  %-18s %s" % (ring.show(f), why))
        # print the interesting ones
        items = sorted(nz.items(), key=lambda kv: len(ring.show(kv[0])))
        for f, g in items[:24]:
            extra = "" if ring.in_small(g) else "   <-- outside O(m;u)"
            print("      (%s)^[2] = %s%s" % (ring.show(f), ring.show(g), extra))


# ------------------------------------------------------------- the 2nd type

def check_second(k, j, u=None):
    m = 2 * k
    ring = Ring(m, u if u else (1,) * m)
    alg = ring.big
    om = omega2(ring, k, j)
    closed = not d2(alg, om)
    G = gram(alg, om)
    nd = alg.const_term(mat_det(alg, G)) == 1
    print("  k=%d j=%d u=%s: closed=%s nondeg=%s" % (k, j, ring.u, closed, nd))
    if not nd:
        return
    ham = Ham(ring, om)
    basis = genfun_basis(ring)
    cols = field_matrix(ham, basis)
    notinner = []
    for f in basis:
        D = ham.field(f)
        if not ham.preserves_small(D):
            continue
        D2 = ham.square(D)
        g = solve_field(cols, basis, D2, ring)
        if g is None:
            notinner.append(f)
    print("    generating functions whose (ad_f)^2 is not inner: %d %s"
          % (len(notinner), [ring.show(f) for f in notinner][:8]))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    if what in ("brackets", "all"):
        hdr("1) the bracket of the 1st-type forms at p = 2")
        check_brackets(1)
        check_brackets(2)
        check_brackets(2, (2, 1, 1, 2))
        check_brackets(3)
    if what in ("pstr", "all"):
        hdr("2) the 2-structure")
        report_pstr(1)
        report_pstr(2)
    if what in ("second", "all"):
        hdr("3) the 2nd-type forms at p = 2")
        for k in (1, 2):
            for j in range(2 * k):
                check_second(k, j)
    print("\n[%.1f s]" % (time.time() - t0))
