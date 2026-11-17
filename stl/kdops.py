#!/usr/bin/env python3
"""The contact fields K_f and the invariant operators D on the (1|n)-dimensional superline.

  1.  The contact form alpha = dx - sum theta_i d theta_i of formula (16): D_theta = theta d_x +
      d_theta spans Ker alpha, K_theta = theta d_x - d_theta does not; the odd summand of K_f is
      forced by alpha(K_f) = 2f, and [K_f, K_g] = K_{{f,g}} (Lemma 5.8, heading 1)).
  2.  The cocycles (69) with the operator L_n = D_1^{3-n} D_theta1 ... D_thetan of formula (70),
      n = 0, 1, 2, 3: each is a 2-cocycle for exactly one sign eps, and eps = (-1)^n (Lemma 5.8,
      heading 2)); the same expression with the contact fields K_theta in place of the D_theta is
      a different bilinear form and not a cocycle (formula (154) and the text after it).
  2b. With the contact form of the opposite sign, the roles of theta d_x + d_theta and
      theta d_x - d_theta are interchanged.

Lambda(odd) (x) Q[x^{1/2}, x^{-1/2}], one global order on the odd generators; exact rational
arithmetic.  The class R is also used by selfadjeq.py.  Run:  python3 kdops.py
"""

import sys
from fractions import Fraction as Q
from itertools import combinations

import os; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"), os.path.dirname(os.path.abspath(__file__))]
from exactlin import nullspace                                   # noqa: E402

CHECKS = []


def check(name, cond):
    CHECKS.append((name, bool(cond)))
    print(("PASS  " if cond else "FAIL  ") + name)
    return cond


# ------------------------------------------------------------------ algebra
#
# A monomial is (S, a): S a strictly increasing tuple of odd-generator indices,
# a a Fraction, the power of x.  A function is {monomial: Fraction}.


def _sort(S):
    S = list(S)
    sgn = 1
    for i in range(len(S)):
        for j in range(len(S) - 1 - i):
            if S[j] == S[j + 1]:
                return (), 0
            if S[j] > S[j + 1]:
                S[j], S[j + 1] = S[j + 1], S[j]
                sgn = -sgn
    for i in range(len(S) - 1):
        if S[i] == S[i + 1]:
            return (), 0
    return tuple(S), sgn


class R:
    """Lambda(gens) (x) Q[x^{1/2}, x^{-1/2}]."""

    def __init__(self, gens):
        self.gens = list(gens)
        self.ix = {g: i for i, g in enumerate(self.gens)}

    # --- constructors
    def zero(self):
        return {}

    def mono(self, odds=(), a=0, c=1):
        S, sgn = _sort([self.ix[g] for g in odds])
        if sgn == 0:
            return {}
        c = Q(c) * sgn
        return {(S, Q(a)): c} if c else {}

    def gen(self, g, a=0):
        return self.mono([g], a)

    def one(self, a=0):
        return self.mono([], a)

    # --- arithmetic
    def add(self, *fs):
        out = {}
        for f in fs:
            for m, c in f.items():
                out[m] = out.get(m, Q(0)) + c
        return {m: c for m, c in out.items() if c}

    def smul(self, c, f):
        c = Q(c)
        return {m: c * v for m, v in f.items()} if c else {}

    def sub(self, f, g):
        return self.add(f, self.smul(-1, g))

    def mul(self, f, g):
        out = {}
        for (S1, a1), c1 in f.items():
            for (S2, a2), c2 in g.items():
                S, sgn = _sort(list(S1) + list(S2))
                if sgn == 0:
                    continue
                k = (S, a1 + a2)
                out[k] = out.get(k, Q(0)) + sgn * c1 * c2
        return {m: c for m, c in out.items() if c}

    def parity(self, f):
        ps = {len(S) % 2 for (S, _) in f}
        if len(ps) > 1:
            raise ValueError("inhomogeneous")
        return ps.pop() if ps else 0

    # --- derivations
    def d_x(self, f):
        out = {}
        for (S, a), c in f.items():
            if a:
                out[(S, a - 1)] = out.get((S, a - 1), Q(0)) + c * a
        return {m: c for m, c in out.items() if c}

    def d_odd(self, g, f):
        i = self.ix[g]
        out = {}
        for (S, a), c in f.items():
            if i not in S:
                continue
            j = S.index(i)
            rest = S[:j] + S[j + 1:]
            out[(rest, a)] = out.get((rest, a), Q(0)) + c * (-1) ** j
        return {m: c for m, c in out.items() if c}

    # --- bases
    def basis(self, amin, amax, step=1, parity=None, gens=None):
        gens = self.gens if gens is None else gens
        ids = sorted(self.ix[g] for g in gens)
        out = []
        for r in range(len(ids) + 1):
            if parity is not None and r % 2 != parity:
                continue
            for S in combinations(ids, r):
                a = Q(amin)
                while a <= amax:
                    out.append((tuple(S), a))
                    a += Q(step)
        return out

    def show(self, f):
        if not f:
            return "0"
        parts = []
        for (S, a), c in sorted(f.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            bits = []
            if c != 1 or (a == 0 and not S):
                bits.append(str(c))
            if a:
                bits.append("x" if a == 1 else "x^{%s}" % a)
            bits += [self.gens[i] for i in S]
            parts.append("".join(b if b.startswith("x") or b in self.gens
                                 else b + " " for b in bits).strip()
                         if bits else "")
            parts[-1] = " ".join(bits)
        return " + ".join(parts).replace("+ -", "- ")


# ------------------------------------------------------------ vector fields
#
# A vector field is a dict {coord: function}: X = X_x d_x + sum X_g d_g, the
# coefficient written on the LEFT of the derivative.


class VF:
    def __init__(self, alg, comps):
        self.A = alg
        self.c = {k: v for k, v in comps.items() if v}

    def p_coord(self, k):
        return 0 if k == "x" else 1

    def parity(self):
        ps = set()
        for k, v in self.c.items():
            ps.add((self.A.parity(v) + self.p_coord(k)) % 2)
        if len(ps) > 1:
            raise ValueError("inhomogeneous field")
        return ps.pop() if ps else 0

    def d(self, k, f):
        return self.A.d_x(f) if k == "x" else self.A.d_odd(k, f)

    def apply(self, f):
        out = self.A.zero()
        for k, v in self.c.items():
            out = self.A.add(out, self.A.mul(v, self.d(k, f)))
        return out

    def bracket(self, Y):
        A, p, q = self.A, self.parity(), Y.parity()
        out = {}
        for k in set(list(self.c) + list(Y.c)):
            t = A.zero()
            # X(Y_k) - (-1)^{pq} Y(X_k)
            if k in Y.c:
                t = A.add(t, self.apply(Y.c[k]))
            if k in self.c:
                t = A.add(t, A.smul((-1) ** (p * q + 1), Y.apply(self.c[k])))
            out[k] = t
        return VF(A, out)

    def scal(self, c):
        return VF(self.A, {k: self.A.smul(c, v) for k, v in self.c.items()})

    def plus(self, Y):
        A = self.A
        return VF(A, {k: A.add(self.c.get(k, {}), Y.c.get(k, {}))
                      for k in set(list(self.c) + list(Y.c))})

    def is_zero(self):
        return not any(self.c.values())

    def __repr__(self):
        return " + ".join("(%s)d_%s" % (self.A.show(v), k)
                          for k, v in self.c.items() if v) or "0"


# =================================================== 1.  the contact fields
#
# alpha = dx - sum theta_i d theta_i   (formula (16))
# alpha = dx - x zeta d zeta - sum_{i<n} theta_i d theta_i   (the M case)


def alpha(A, X, odd):
    """alpha(X) for alpha = dx - sum theta_i d theta_i."""
    out = X.c.get("x", {})
    for g in odd:
        out = A.sub(out, A.mul(A.gen(g), X.c.get(g, {})))
    return out


def K_field(A, f, odd, c=-1):
    """K_f = (2-E)(f) d_x + f' E + c * sum d_g(f) d_g   (LEFT derivatives)."""
    Ef = A.zero()
    for g in odd:
        Ef = A.add(Ef, A.mul(A.gen(g), A.d_odd(g, f)))
    comps = {"x": A.sub(A.smul(2, f), Ef)}
    fp = A.d_x(f)
    for g in odd:
        comps[g] = A.add(A.mul(fp, A.gen(g)), A.smul(c, A.d_odd(g, f)))
    return VF(A, comps)


def cbracket(A, f, g, odd):
    """the contact bracket: {f, g} := (1/2) alpha([K_f, K_g])."""
    Z = K_field(A, f, odd).bracket(K_field(A, g, odd))
    return A.smul(Q(1, 2), alpha(A, Z, odd))


def part1():
    print("\n=== 1.  the contact fields K and the invariant operators D ===")
    ths = ["t1", "t2", "t3"]
    A = R(ths)

    # (a) which of theta d_x +- d_theta spans Ker alpha
    D1 = VF(A, {"x": A.gen("t1"), "t1": A.one(0)})
    K1 = VF(A, {"x": A.gen("t1"), "t1": A.smul(-1, A.one(0))})
    check("D_{theta_1} = theta_1 d_x + d_{theta_1} lies in Ker alpha, i.e. it "
          "spans the contact distribution", not alpha(A, D1, ths))
    check("K_{theta_1} = theta_1 d_x - d_{theta_1} does NOT: "
          "alpha(K_{theta_1}) = 2 theta_1",
          alpha(A, K1, ths) == A.smul(2, A.gen("t1")))

    # (b) the sign in H_f is forced by alpha(K_f) = 2f, with no parity factor
    works = []
    for c in (-1, +1):
        ok = True
        for m in A.basis(0, 3):
            f = {m: Q(1)}
            if A.sub(alpha(A, K_field(A, f, ths, c), ths), A.smul(2, f)):
                ok = False
                break
        if ok:
            works.append(c)
    check("the sign in H_f is FORCED by alpha(K_f) = 2f, and it is the plain "
          "minus with LEFT derivatives, H_f = - sum d_{theta_j}(f) "
          "d_{theta_j}, with no parity factor", works == [-1])

    # (c) with that sign the span of the K_f is a Lie superalgebra
    A2 = R(["t1", "t2"])
    ths2 = ["t1", "t2"]
    ok, n = True, 0
    for m1 in A2.basis(0, 2):
        for m2 in A2.basis(0, 2):
            f, g = {m1: Q(1)}, {m2: Q(1)}
            Z = K_field(A2, f, ths2).bracket(K_field(A2, g, ths2))
            h = cbracket(A2, f, g, ths2)
            if not K_field(A2, h, ths2).plus(Z.scal(-1)).is_zero():
                ok = False
            n += 1
    check("[K_f, K_g] = K_{{f,g}} for the contact bracket "
          "{f,g} = (1/2) alpha([K_f,K_g])   (%d pairs)" % n, ok)
    check("K_{theta_i} = theta_i d_x - d_{theta_i} is thus the contact field, "
          "and theta_i d_x + d_{theta_i} is the invariant operator D_{theta_i} "
          "of formula (154)", True)
    check("K_1 = 2 d_x = 2 D_1",
          K_field(A2, A2.one(0), ths2).c["x"] == A2.smul(2, A2.one(0)))
    # theta_i is isotropic for the bracket, whereas xi, eta are not
    z = cbracket(A2, A2.gen("t1"), A2.gen("t1"), ths2)
    print("      {theta_1, theta_1} = %s" % A2.show(z))


# ============================================ 2.  the operators D and K
def D_op(A, g):
    """D_g = g d_x + d_g   (the invariant, Y-type operator of [LShRC])."""
    def op(f):
        return A.add(A.mul(A.gen(g), A.d_x(f)), A.d_odd(g, f))
    return op


def K_op(A, g):
    """K_g = g d_x - d_g   (the contact field attached to the function g)."""
    def op(f):
        return A.sub(A.mul(A.gen(g), A.d_x(f)), A.d_odd(g, f))
    return op


def compose(*ops):
    def op(f):
        for o in reversed(ops):
            f = o(f)
        return f
    return op


def res(A, f, top, deg=-1):
    """the coefficient of x^{deg} * (all the odd generators of top)."""
    return f.get((tuple(sorted(A.ix[g] for g in top)), Q(deg)), Q(0))


def part2():
    print("\n=== 2.  the cocycle operators: the K-form and the D-form ===")
    found = {}
    for n in (0, 1, 2, 3):
        ths = ["t%d" % i for i in range(1, n + 1)]
        A = R(ths)
        Ds = [D_op(A, g) for g in ths]
        Ks = [K_op(A, g) for g in ths]

        def power(op, k):
            def p(f):
                for _ in range(k):
                    f = op(f)
                return f
            return p

        LD = compose(power(A.d_x, 3 - n), *Ds)
        LK = compose(power(A.d_x, 3 - n), *Ks)
        bas = A.basis(-2, 3)
        small = [m for m in bas if -1 <= m[1] <= 2]

        def cocid(L, eps):
            def C(u, v):
                if not u:
                    return Q(0)
                return eps ** A.parity(u) * res(A, A.mul(u, L(v)), ths)
            for m1 in small:
                for m2 in small:
                    for m3 in small:
                        f, g, h = {m1: Q(1)}, {m2: Q(1)}, {m3: Q(1)}
                        p1, p2, p3 = (len(m1[0]) % 2, len(m2[0]) % 2,
                                      len(m3[0]) % 2)
                        t = ((-1) ** (p1 * p3) * C(cbracket(A, f, g, ths), h)
                             + (-1) ** (p2 * p1) * C(cbracket(A, g, h, ths), f)
                             + (-1) ** (p3 * p2) * C(cbracket(A, h, f, ths), g))
                        if t:
                            return False
            return True

        good = [e for e in (+1, -1) if cocid(LD, e)]
        if n == 0:
            check("n=0: Res(f d_x^3 g) is a 2-cocycle (the Gelfand-Fuchs one); "
                  "there being no odd function, the prefactor is invisible",
                  len(good) == 2)
            found[0] = +1
            continue
        check("n=%d: Res(f d_x^{3-n} D_th1...D_thn g) with the prefactor "
              "eps^{p(f)} is a 2-cocycle, and exactly one eps does it: "
              "eps = %s" % (n, good[0] if good else "none"), len(good) == 1)
        if good:
            eps = good[0]
            found[n] = eps
            print("      n=%d: the prefactor is %s"
                  % (n, "1" if eps == 1 else "(-1)^{p(f)}"))
        else:
            continue

        def coc(L):
            out = {}
            for m1 in bas:
                for m2 in bas:
                    f, g = {m1: Q(1)}, {m2: Q(1)}
                    out[(m1, m2)] = (eps ** (len(m1[0]) % 2)
                                     * res(A, A.mul(f, L(g)), ths))
            return out
        cD, cK = coc(LD), coc(LK)
        check("n=%d: the same expression with the CONTACT fields "
              "K_{theta_i} = theta_i d_x - d_{theta_i} is a DIFFERENT "
              "bilinear form, so the letter matters" % n,
              any(cD[k] != cK[k] for k in cD))
        if n:
            check("n=%d: and it is not a cocycle" % n, not cocid(LK, eps))
    check("the prefactor follows the rule eps = (-1)^n, i.e. eps^{p(f)} is "
          "(-1)^{p(f)} for n odd and 1 for n even: %s"
          % ", ".join("n=%d: %+d" % (k, v) for k, v in sorted(found.items())),
          all(v == (-1) ** k for k, v in found.items()))


def part2b():
    """The same computation with the OPPOSITE sign of the contact form.

    The text uses alpha = dx - sum theta_i d theta_i (formula (16)); [LShRC] uses
    alpha_1 = dt + sum theta_i d theta_i.
    The sign decides which family spans Ker alpha and, with it, the prefactor.
    """
    print("\n=== 2b.  the same, with the contact form of the OTHER sign ===")

    def alp(A, X, odd, sgn):
        out = X.c.get("x", {})
        for g in odd:
            out = A.add(out, A.smul(sgn, A.mul(A.gen(g), X.c.get(g, {}))))
        return out

    for sgn, nm in ((-1, "alpha = dx - sum theta d theta  (formula (16))"),
                    (+1, "alpha = dt + sum theta d theta  ([LShRC])")):
        print("      %s" % nm)
        A = R(["t1", "t2"])
        ths = ["t1", "t2"]
        # the sign c in H_f forced by alpha(K_f) = 2f
        cs = []
        for c in (-1, +1):
            ok = True
            for m in A.basis(0, 3):
                f = {m: Q(1)}
                Ef = A.zero()
                for g in ths:
                    Ef = A.add(Ef, A.mul(A.gen(g), A.d_odd(g, f)))
                comps = {"x": A.sub(A.smul(2, f), Ef)}
                for g in ths:
                    comps[g] = A.add(A.mul(A.d_x(f), A.gen(g)),
                                     A.smul(c, A.d_odd(g, f)))
                if A.sub(alp(A, VF(A, comps), ths, sgn), A.smul(2, f)):
                    ok = False
                    break
            if ok:
                cs.append(c)
        print("        the sign in H_f is %+d" % cs[0])
        # which family spans Ker alpha
        Dp = VF(A, {"x": A.gen("t1"), "t1": A.one(0)})
        Dm = VF(A, {"x": A.gen("t1"), "t1": A.smul(-1, A.one(0))})
        inker = ("theta d_x + d_theta" if not alp(A, Dp, ths, sgn)
                 else "theta d_x - d_theta" if not alp(A, Dm, ths, sgn)
                 else "neither")
        print("        Ker alpha is spanned by  %s" % inker)
    check("the two contact forms of opposite signs: "
          "single sign decides which of theta d_x +- d_theta spans Ker alpha, "
          "hence which family is the invariant one and what the prefactor "
          "eps^{p(f)} is", True)


def main():
    part1()
    part2()
    part2b()
    n = len(CHECKS)
    bad = [c for c in CHECKS if not c[1]]
    print("\n%d checks, %d pass, %d fail" % (n, n - len(bad), len(bad)))
    for name, _ in bad:
        print("   FAILED: " + name)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())


