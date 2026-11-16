"""Where along the family svect^L_lambda(1|2) is the derivative a coboundary?

family.py settles ONE point of the family, lambda_0 = 1/3:
there d mu / d lambda is not a coboundary, so H^2(g;g) != 0.  Vaintrob's
theorem on the moduli of g-curves (Lecture Notes in Physics 375 (1991)
263--268) excludes exactly one algebra of the same list, and excludes it only
for alpha NOT in Q.  So the question is whether our non-rigidity has the same
dividing line -- rational against irrational -- or holds all along the family.

This module answers it by re-running the exact coboundary test of `family.py`
at a list of rational lambda_0, integers among them.  The symbolic part
(the basis over Q(lambda), the structure constants, Jacobi identically in
lambda) is done once; only the specialization is repeated.

Two guards, because a basis chosen over Q(lambda) can degenerate when lambda
is specialized:

  1) the dimension of Ker div_{lambda_0} in each (degree, parity) is recomputed
     at the numerical lambda_0 and compared with the symbolic count;
  2) super Jacobi is re-checked for the SPECIALIZED structure constants.

And the two controls of `family.py` are re-run at every point: a verdict "not
a coboundary" is worthless unless the test can still say "is a coboundary".

    ../venv/bin/python lamscan.py [N] [core]
"""

from fractions import Fraction as Q
import sys, os, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sympy as sp
import family as F
from family import (LAM, V, div, basis, vectors, struct, numeric,
                    cocycle_check, controls, is_coboundary, jacobi_check)

# the points: integers, and rationals with small and with larger denominator
LAMBDAS = [Q(0), Q(1), Q(2), Q(-1), Q(1, 2), Q(1, 3), Q(2, 3), Q(3, 2),
           Q(5, 7), Q(22, 7)]


def kernel_dim_at(d, p, lam0):
    """dim Ker div_{lam0} on the (d,p) part, computed at the number lam0."""
    keys = V.homog_keys(d, (p,))
    if not keys:
        return 0
    rows = {}
    for j, k in enumerate(keys):
        for t, c in div({k: sp.Integer(1)}, sp.Rational(lam0)).items():
            rows.setdefault(t, {})[j] = c
    if not rows:
        return len(keys)
    M = sp.Matrix([[r.get(j, 0) for j in range(len(keys))]
                   for r in rows.values()])
    return len(keys) - M.rank()


def singular_at(mu, dmu, lam0):
    """True if any structure constant, or its derivative, has a pole at lam0."""
    for src in (mu, dmu):
        for cell in src.values():
            for c in cell.values():
                v = sp.simplify(c.subs(LAM, sp.Rational(lam0)))
                if not v.is_finite:
                    return True
    return False


def main(N=3, core=1):
    t0 = time.time()
    print("svect^L_lambda(1|2): the coboundary test along the family")
    print("  window N=%d, core |deg|<=%d" % (N, core))

    vec, mu = struct(N)
    nmov = sum(1 for cell in mu.values() for c in cell.values()
               if c.free_symbols)
    print("  %d basis vectors over Q(lambda); %d of %d structure constants "
          "move with lambda" % (len(vec), nmov,
                                sum(len(c) for c in mu.values())))

    bad, tested = jacobi_check(vec, mu, N)
    print("  super Jacobi identically in lambda: %s (%d triples)"
          % ("OK" if bad == 0 else "FAILS in %d" % bad, tested))
    if bad:
        raise SystemExit("the family is not a family; nothing below means anything")

    # the symbolic derivative, once
    dmu = {ij: {k: sp.diff(c, LAM) for k, c in cell.items()}
           for ij, cell in mu.items()}

    # the symbolic dimension of each block, for guard 1
    symdim = {}
    for (b, p, d, i) in vec:
        symdim[(d, p)] = symdim.get((d, p), 0) + 1

    print()
    print("  lambda_0   dims  Jacobi  cocycle  ctrl1 ctrl2   delta(phi)=dmu ?")
    print("  " + "-" * 66)
    out = []
    for lam0 in LAMBDAS:
        # guard 1: does the Q(lambda)-basis stay a basis at lambda_0?
        dims_ok = all(kernel_dim_at(d, p, lam0) == n
                      for (d, p), n in symdim.items())

        # guard 0: is the FRAME regular there?  A basis chosen over Q(lambda)
        # may have poles at particular lambda_0 (it does at lambda_0 = 0),
        # and then the structure constants specialize to nothing.  That is a
        # defect of the frame, not a statement about the algebra.
        if singular_at(mu, dmu, lam0):
            print("  %8s   %-4s  %s" % (lam0, "OK" if dims_ok else "DROP",
                                        "frame has a pole here -- skipped"))
            out.append((lam0, dims_ok, None, None, None, None, None))
            continue

        mu0 = numeric(mu, lam0)
        cd = {}
        for ij, cell in dmu.items():
            c = {}
            for k, v in cell.items():
                w = sp.simplify(v.subs(LAM, sp.Rational(lam0)))
                if w != 0:
                    c[k] = Q(str(sp.Rational(w)))
            if c:
                cd[ij] = c

        # guard 2: Jacobi for the specialized constants
        jb, _ = jacobi_check(vec, {ij: {k: sp.Rational(v) for k, v in cell.items()}
                                   for ij, cell in mu0.items()}, N)
        cb, _ = cocycle_check(vec, mu0, cd, N)
        c1, c2 = controls(vec, mu0, core)
        ok, sizes = is_coboundary(vec, mu0, cd, core)

        verdict = ("YES" if ok else "NO")
        if not (c1 and c2):
            verdict = "VOID"
        print("  %8s   %-4s  %-6s  %-7s  %-5s %-5s   %s"
              % (lam0, "OK" if dims_ok else "DROP",
                 "OK" if jb == 0 else "BAD",
                 "OK" if cb == 0 else "BAD",
                 "yes" if c1 else "NO", "yes" if c2 else "NO", verdict))
        out.append((lam0, dims_ok, jb == 0, cb == 0, c1, c2, ok))

    print()
    print("  sizes of the last system: %s" % sizes)
    nonco = [str(l) for (l, dm, jb, cb, c1, c2, ok) in out
             if ok is False and dm and jb and cb and c1 and c2]
    co = [str(l) for (l, dm, jb, cb, c1, c2, ok) in out
          if ok is True and dm and jb and cb and c1 and c2]
    skipped = [str(l) for (l, dm, jb, cb, c1, c2, ok) in out if ok is None]
    if skipped:
        print("  frame singular, not tested: lambda_0 = %s" % ", ".join(skipped))
    print("  NOT a coboundary (so H^2 != 0) at lambda_0 = %s" % ", ".join(nonco))
    print("  IS  a coboundary at lambda_0 = %s" % (", ".join(co) or "(none)"))
    print("  [%.1fs]" % (time.time() - t0))


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    core = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    main(N=N, core=core)
