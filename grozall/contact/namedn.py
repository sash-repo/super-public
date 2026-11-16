"""The named operators, checked directly (no solving) in any dimension.

  1) the bracket family on the densities
         {f,g}_{lam,mu} = varpi^{ab}(e_a f)(e_b g) + mu (Rf) g - lam f (Rg)
     is k(2n+1)-invariant F_lam (x) F_mu -> F_{lam+mu+1}, and at lam = mu = -1 it is
     the contact bracket;
  2) the three arrows of the contact complex in the S^k-fibers:
         d: F_0 -> T(1,0)  and  delta: T(1,n) -> F_{n+1} = Vol,   both of W = 1,
     exist for every n, whereas the SECOND-order arrow T(1,0) -> T(1,1) exists for
     n = 1 only.
"""

import sys
from fractions import Fraction as Fr

from contactn import Contact, Module, padd, pscal, pmul, pzero, fiber_monomials
from genn import rand_field

FAIL = []


def check(name, cond):
    print(("  ok   " if cond else "  FAIL ") + name)
    if not cond:
        FAIL.append(name)


def bracket_op(alg, lam, mu, F, G):
    """The W=2 family, on densities written as plain polynomials."""
    out = pzero()
    for a in range(1, alg.dim + 1):
        for b in range(1, alg.dim + 1):
            v = alg.varpi_up(a, b)
            if v:
                out = padd(out, pscal(v, pmul(alg.e(a, F), alg.e(b, G))))
    out = padd(out, pscal(mu, pmul(alg.R(F), G)), pscal(-lam, pmul(F, alg.R(G))))
    return out


def main():
    for n in (1, 2, 3):
        alg = Contact(n)
        print("k(%d):" % (2 * n + 1))
        z0 = (0,) * alg.dim
        Fs = [rand_field(alg, 0, 3, 11 + i) for i in range(2)]
        Gs = [rand_field(alg, 0, 3, 55 + i) for i in range(2)]
        pol = lambda X: {e: v for (e, mu) in [(k[0], k[1])] for k, v in [(k, X[k])]
                         } if False else None

        def as_poly(X):
            return {e: v for (e, mu), v in X.items()}

        def as_dens(p):
            return {(e, z0): v for e, v in p.items()}

        gens = alg.k_piece(0) + alg.k_piece(1) + alg.k_piece(2)
        for (lam, mu) in [(Fr(0), Fr(0)), (Fr(2, 3), Fr(-1, 5)), (Fr(-1), Fr(-1)),
                          (Fr(-3, 2), Fr(5, 4))]:
            M1, M2 = Module(alg, 0, lam), Module(alg, 0, mu)
            M3 = Module(alg, 0, lam + mu + 1)
            ok = True
            for f in gens:
                for F in Fs:
                    for G in Gs:
                        pF, pG = as_poly(F), as_poly(G)
                        lhs = M3.act(f, as_dens(bracket_op(alg, lam, mu, pF, pG)))
                        r = bracket_op(alg, lam, mu, as_poly(M1.act(f, F)), pG)
                        r = padd(r, bracket_op(alg, lam, mu, pF,
                                               as_poly(M2.act(f, G))))
                        if lhs != as_dens(r):
                            ok = False
            check("the bracket family is invariant at (%s,%s)" % (lam, mu), ok)
        # at lam = mu = -1 it is the contact bracket
        ok = True
        for F in Fs:
            for G in Gs:
                pF, pG = as_poly(F), as_poly(G)
                if bracket_op(alg, Fr(-1), Fr(-1), pF, pG) != alg.kbracket(pF, pG):
                    ok = False
        check("at lambda = mu = -1 it is the contact bracket", ok)

        # the contact complex in the S^k fibers
        import linn
        d1 = linn.ULin(n, 0, 1, 1).solve(Fr(0))
        check("d: F_0 -> T(1,0) exists (W=1)", d1 == 1)
        d2 = linn.ULin(n, 1, 0, 1).solve(Fr(n))
        check("delta: T(1,%d) -> Vol = F_%d exists (W=1)" % (n, n + 1), d2 == 1)
        mid = [linn.ULin(n, 1, 1, 2).solve(Fr(a, 2)) for a in range(-4, 7)]
        check("the second-order arrow T(1,l) -> T(1,l+1): %s"
              % ("exists" if any(mid) else "does NOT exist"),
              (any(mid) if n == 1 else not any(mid)))
        print()
    print("%d checks failed" % len(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
