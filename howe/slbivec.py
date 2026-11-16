"""
slbivec.py -- how much of clause 3 of the Theorem of [L2] survives?

[L2] = D. Leites, Funkts. Anal. Prilozh. 16:1 (1982) 76-77, whose Theorem says
that Y_+ (multiplication by the odd canonical 2-form omega_1), Y_- (interior
multiplication by the conjugate bivector) and E = [Y_+, Y_-] define on Omega and
Omega^{(*)} the SL(n)-invariant structures of sl(1|1)-modules.

Three computations, all exact over Q:

  (A)  CROSS-CHECK, with the independent engine omcheck.py (constant coefficients,
       L_0 = pe(n)), that Y_- = sum_i d/d'xi_i d/d'x_i commutes with the gl(n)
       part of pe(n) and with nothing else.  slcheck.py found this with a
       completely separate implementation; two engines must agree.

  (B)  The full space of candidates.  An odd, form-degree (-2), constant-
       coefficient interior multiplication is exactly

            iota_C = sum_{i,j} c_{ij} d/d'xi_i d/d'x_j,       C in gl(n),

       an n^2-dimensional space (the operators d/d'x d/d'x and d/d'xi d/d'xi are
       even, hence of the wrong parity for [Y_+, Y_-] to be even).  We compute
       the subspace of those iota_C that commute with gl(n), with pe(n), with
       sle(n) and with le(n).  If the last two are 0, no choice of bivector
       makes clause 3 literally true.

  (C)  The structure of Omega(const) over the sl(1|1) spanned by Y_+, Y_-, E:
       dim Ker E degree by degree, and whether E is semisimple.  This is what
       the graph of [L3, Subsec. 2.4 b)] draws, and it explains why that graph
       has three composition factors exactly in degree n.

Usage:  python3 slbivec.py [nmax] [imax]
"""

import sys
from fractions import Fraction as F

import omcheck as O
import slcheck as S
from le import D as VF_of


# ------------------------------------------------------- Y_- on Omega(const)
def dpx(n, v, i):
    """d/d'x_i on Omega(const) (odd derivation, exterior part first)."""
    out = {}
    for (A, B), c in v.items():
        if i not in A:
            continue
        pos = A.index(i)
        O.add(out, (A[:pos] + A[pos + 1:], B), c * (-1) ** pos)
    return out


def dpxi(n, v, i):
    """d/d'xi_i on Omega(const) (even derivation, no sign)."""
    out = {}
    for (A, B), c in v.items():
        if B[i] == 0:
            continue
        B2 = list(B)
        B2[i] -= 1
        O.add(out, (A, tuple(B2)), c * B[i])
    return out


def Yminus_const(n, v, C=None):
    """iota_C = sum_{i,j} C[i][j] d/d'xi_i d/d'x_j ; default C = identity."""
    out = {}
    for i in range(n):
        for j in range(n):
            c = (1 if i == j else 0) if C is None else C[i][j]
            if not c:
                continue
            for m, v2 in dpxi(n, dpx(n, v, j), i).items():
                O.add(out, m, c * v2)
    return out


def Yplus_const(n, v):
    return O.omega_mul(n, v)


def Econst(n, v):
    out = Yplus_const(n, Yminus_const(n, v))
    for m, c in Yminus_const(n, Yplus_const(n, v)).items():
        O.add(out, m, c)
    return out


# --------------------------------------------------------------------- (A)
def cross_check(n, imax):
    print("(A) cross-check on Omega(const) with the pe(n) engine of omcheck.py")
    _, allg = O.gens_pe(n)
    mons = []
    for i in range(imax + 1):
        mons += O.basis(n, i)
    ok, bad = [], []
    for g in allg:
        fails = None
        for m in mons:
            v = {m: F(1)}
            # [L_D, Y_-] = L_D Y_- - (-1)^{p(D)} Y_- L_D
            r = g.act(Yminus_const(n, v))
            for mm, c in Yminus_const(n, g.act(v)).items():
                O.add(r, mm, -((-1) ** g.pD) * c)
            if r:
                fails = m
                break
        name = "%s(%d,%d)" % (g.kind, g.i + 1, g.j + 1)
        (ok if fails is None else bad).append(name)
    print("    commutes with     : %s" % (" ".join(ok) if ok else "nothing"))
    print("    does NOT commute  : %s" % (" ".join(bad) if bad else "-"))
    print("    => Y_- is invariant exactly under the gl(n) part: %s"
          % all(x.startswith("gl") for x in ok))
    # and Y_+ must commute with everything
    bad2 = []
    for g in allg:
        for m in mons:
            v = {m: F(1)}
            r = g.act(Yplus_const(n, v))
            for mm, c in Yplus_const(n, g.act(v)).items():
                O.add(r, mm, -((-1) ** g.pD) * c)
            if r:
                bad2.append("%s(%d,%d)" % (g.kind, g.i + 1, g.j + 1))
                break
    print("    [L_D, Y_+] != 0 for: %s" % (" ".join(bad2) if bad2 else "- (Y_+ is pe(n)-invariant)"))
    print()


# --------------------------------------------------------------------- (B)
def invariant_bivectors(n, cmax, fmax, dmax):
    """the subspace of {iota_C} commuting with a given set of vector fields."""
    print("(B) which interior multiplications iota_C, C in gl(n), are invariant?")
    mons = S.monomials(n, cmax, fmax)
    tests = [{m: F(1)} for m in mons]
    basisC = [(i, j) for i in range(n) for j in range(n)]

    def iota(idx):
        i, j = basisC[idx]

        def op(nn, e):
            return S.d_form(nn, "'xi", i, S.d_form(nn, "'x", j, e))
        return op

    ops = [iota(k) for k in range(len(basisC))]

    def rows_for(vfs):
        rows = []
        for v in vfs:
            for e in tests:
                cols = []
                for op in ops:
                    r = op(n, S.lie(n, v, e))
                    S.addall(r, S.lie(n, v, op(n, e)), -((-1) ** (v.par * 1)))
                    # [L_D, iota] = L_D iota - (-1)^{p} iota L_D  (same up to sign)
                    cols.append(r)
                targets = set()
                for r in cols:
                    targets |= set(r.keys())
                for t in sorted(targets):
                    rows.append([r.get(t, F(0)) for r in cols])
        return rows

    fs = S.gen_functions(n, dmax)
    groups = {
        "gl(n)          ": [f for f in fs if _is_gl(n, f)],
        "pe(n) = le(n)_0": [f for f in fs if _deg(f) == 2],
        "sle(n)         ": [f for f in fs if not S.laplace(n, f)],
        "le(n)          ": fs,
    }
    for name, gg in groups.items():
        vfs = [VF_of(f, n) for f in gg]
        ns = O.nullspace(rows_for(vfs), len(basisC))
        print("    invariant under %s : dim = %d%s"
              % (name, len(ns), "   " + _describe(n, ns, basisC) if ns else ""))
    print()


def _deg(f):
    (a, S_), _ = next(iter(f.items()))
    return sum(a) + len(S_)


def _is_gl(n, f):
    (a, S_), _ = next(iter(f.items()))
    return sum(a) == 1 and len(S_) == 1


def _describe(n, ns, basisC):
    out = []
    for v in ns:
        terms = [("%s*" % c if c != 1 else "") + "E%d%d" % (i + 1, j + 1)
                 for (i, j), c in zip(basisC, v) if c]
        out.append(" + ".join(terms))
    return "spanned by  " + ";  ".join(out)


# --------------------------------------------------------------------- (C)
def structure(n, imax):
    print("(C) E = [Y_+, Y_-] on Omega(const):  eigenvalues and Ker E")
    print("    %3s %6s %8s %s" % ("i", "dim", "dim KerE", "eigenvalues of E"))
    formula = True
    for i in range(imax + 1):
        bas = O.basis(n, i)
        idx = {m: k for k, m in enumerate(bas)}
        d = len(bas)
        if d == 0:
            continue
        M = []
        for m in bas:
            M.append(O.to_vec(Econst(n, {m: F(1)}), idx))
        diag = all(M[k][c] == 0 for k in range(d) for c in range(d) if c != k)
        ev = sorted({M[k][k] for k in range(d)}) if diag else None
        # the closed formula:  E('x^A 'xi^B) = (n - |A| + |B|) . 'x^A 'xi^B
        for k, (A, B) in enumerate(bas):
            if M[k][k] != n - len(A) + sum(B) or not diag:
                formula = False
        ker = d - O.rank(M)
        print("    %3d %6d %8d %s" % (i, d, ker,
              (" ".join(str(x) for x in ev) if diag
               else "(not diagonal in the monomial basis)")))
    print("    E is diagonal with E('x^A 'xi^B) = (n - |A| + |B|) . 'x^A 'xi^B : %s"
          % formula)

    # Ker E is the line C.'x_1...'x_n and it is killed by Y_+ and by Y_-,
    # i.e. it is a TRIVIAL sl(1|1)-summand: with E semisimple this is exactly
    # complete reducibility of Omega(const) over sl(1|1).
    top = {(tuple(range(n)), (0,) * n): F(1)}
    print("    Y_+ and Y_- kill 'x_1...'x_n (trivial summand)              : %s"
          % (not Yplus_const(n, top) and not Yminus_const(n, top)))

    # ... but that line is NOT a pe(n)-submodule, so the sl(1|1)-splitting is
    # finer than the pe(n)-structure.
    _, allg = O.gens_pe(n)
    tm = next(iter(top))
    movers = ["%s(%d,%d)" % (g.kind, g.i + 1, g.j + 1)
              for g in allg if set(g.act(top)) - {tm}]   # leaves the line?
    print("    C.'x_1...'x_n is pe(n)-stable                               : %s%s"
          % (not movers, ("   (moved by " + " ".join(movers[:4]) + ")") if movers else ""))
    print()


if __name__ == "__main__":
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    imax = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    for n in range(2, nmax + 1):
        print("=" * 74)
        print("n = %d" % n)
        print("=" * 74)
        cross_check(n, imax)
        invariant_bivectors(n, 2, 3, 3)
        structure(n, imax)
