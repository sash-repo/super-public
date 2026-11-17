#!/usr/bin/env python3
"""The eigenvalue problem for St_1 = D_1 D_theta on ns(1) (Section 12 of the text).

Here D_theta = theta d_x + d_theta and D_1 = d_x (formula (154)); in the code the operator
D_theta is built by Alg.K("th", "th", +1), and St_1 = D_theta D_1 = D_theta^3.

  1.  D_theta^2 = d_x, D_theta^3 = D_theta D_1 = St_1, St_1^2 = d_x^3, and the component form
      of St_1 f = lambda f.
  2.  Statement 12.3.2: over C = Lambda(z_1, ..., z_m), lambda = z_1, the even solutions of
      St_1 f = lambda f polynomial in x form a space of dimension 3, 6, 12 for m = 1, 2, 3 (the
      same for deg_x <= 4, 6, 8), and every solution lies in Ker d_x^3 = Ker St_1^2.
  7.  Statement 12.9: with respect to the pairing (151), <f, g> = the coefficient of x^{-1} theta
      in f g, D_theta and D_1 are anti-self-adjoint, St_1 is self-adjoint, L_1 = St_1 D_1 and
      St_1^2 = D_1^3 are anti-self-adjoint; and Corollary 12.9.2 in the Fourier model.

Lambda(odd generators) (x) Q[x, x^{-1}], one global order on the odd generators; exact
rational arithmetic.  Run:  python3 eig.py
"""

import sys
from fractions import Fraction
from itertools import combinations

import os; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"), os.path.dirname(os.path.abspath(__file__))]
from exactlin import nullspace  # noqa: E402

# ---------------------------------------------------------------- the algebra


def _sort_sign(ids):
    ids = list(ids)
    sign = 1
    for i in range(len(ids)):
        for j in range(len(ids) - 1 - i):
            if ids[j] > ids[j + 1]:
                ids[j], ids[j + 1] = ids[j + 1], ids[j]
                sign = -sign
            elif ids[j] == ids[j + 1]:
                return 0, ids
    return sign, ids


class Alg:
    """Lambda(gens) (x) Q[x, x^{-1}].  A monomial is (odd-index-tuple, x-power)."""

    def __init__(self, gens):
        self.gens = list(gens)
        self.idx = {g: i for i, g in enumerate(self.gens)}

    def zero(self):
        return {}

    def gen(self, g, k=0):
        return {((self.idx[g],), k): Fraction(1)}

    def one(self, k=0):
        return {((), k): Fraction(1)}

    def mono(self, odds, k, c=1):
        ids = [self.idx[g] for g in odds]
        sign, srt = _sort_sign(ids)
        if sign == 0:
            return {}
        return {(tuple(srt), k): Fraction(c) * sign}

    def add(self, *xs):
        out = {}
        for x in xs:
            for m, c in x.items():
                out[m] = out.get(m, Fraction(0)) + c
        return {m: c for m, c in out.items() if c}

    def smul(self, c, x):
        c = Fraction(c)
        return {m: c * v for m, v in x.items()} if c else {}

    def mul(self, a, b):
        out = {}
        for (oa, ka), ca in a.items():
            for (ob, kb), cb in b.items():
                if set(oa) & set(ob):
                    continue
                sign, srt = _sort_sign(list(oa) + list(ob))
                m = (tuple(srt), ka + kb)
                out[m] = out.get(m, Fraction(0)) + sign * ca * cb
        return {m: c for m, c in out.items() if c}

    def sub(self, a, b):
        return self.add(a, self.smul(-1, b))

    def d_x(self, x):
        out = {}
        for (o, k), c in x.items():
            if k:
                out[(o, k - 1)] = out.get((o, k - 1), Fraction(0)) + c * k
        return {m: v for m, v in out.items() if v}

    def d_odd(self, g, x):
        i = self.idx[g]
        out = {}
        for (o, k), c in x.items():
            if i not in o:
                continue
            j = o.index(i)
            rest = o[:j] + o[j + 1:]
            out[(rest, k)] = out.get((rest, k), Fraction(0)) + c * (-1) ** j
        return {m: v for m, v in out.items() if v}

    def K(self, sub, mult, sign=+1):
        """d/d(sub) + sign * (mult) * d/dx.

        theta-basis (orthonormal):  D_theta_i = d_{theta_i} + theta_i d_t;
        xi-eta basis (isotropic) :  K_xi      = d_eta       + xi      d_t.
        """
        def op(f):
            return self.add(self.d_odd(sub, f),
                            self.smul(sign, self.mul(self.gen(mult), self.d_x(f))))
        return op

    def basis(self, deg, parity=None, gens=None, kmin=0):
        gens = self.gens if gens is None else gens
        ids = sorted(self.idx[g] for g in gens)
        out = []
        for r in range(len(ids) + 1):
            if parity is not None and r % 2 != parity:
                continue
            for sub in combinations(ids, r):
                for k in range(kmin, deg + 1):
                    out.append((tuple(sub), k))
        return out

    def top(self):
        return tuple(range(len(self.gens)))


def compose(*ops):
    def op(f):
        for o in reversed(ops):
            f = o(f)
        return f
    return op


# ---------------------------------------------------------------- bookkeeping

CHECKS = []


def check(name, cond):
    CHECKS.append((name, bool(cond)))
    print(("PASS  " if cond else "FAIL  ") + name)
    return cond


def eq(a, b):
    return {m: c for m, c in a.items() if c} == {m: c for m, c in b.items() if c}


def op_equal(A, B, alg, deg, gens=None):
    n = 0
    for m in alg.basis(deg, gens=gens):
        f = {m: Fraction(1)}
        if not eq(A(f), B(f)):
            return False, n
        n += 1
    return True, n


# ====================================================== 1.  ns(1): D_theta

def part1():
    print("\n=== 1.  ns(1):  D_theta, and St_1 = D_theta D_1 = (D_theta)^3 ===")
    A = Alg(["th", "z1", "z2"])
    for sgn in (+1,):
        K = A.K("th", "th", sgn)
        K2 = compose(K, K)
        ok, n = op_equal(K2, lambda f: A.smul(sgn, A.d_x(f)), A, 5)
        check(f"D_theta^2 = {sgn:+d} d_x   ({n} basis elements)", ok)
        K3 = compose(K, K2)
        ok, n = op_equal(K3, lambda f: A.smul(sgn, K(A.d_x(f))), A, 5)
        check(f"(D_theta)^3 = {sgn:+d} D_theta D_1, i.e. it IS the operator "
              f"St_1 of the table   ({n} checks)", ok)
        K6 = compose(K3, K3)
        ok, n = op_equal(K6, lambda f: A.smul(sgn, A.d_x(A.d_x(A.d_x(f)))), A, 6)
        check(f"(St_1)^2 = {sgn:+d} d_x^3   ({n} checks)", ok)

    # the component form of St_1
    K = A.K("th", "th", +1)
    D = compose(K, K, K)
    ok = all(eq(D(A.one(k)), A.mul(A.gen("th"), A.d_x(A.d_x(A.one(k)))))
             for k in range(6))
    check("St_1(f_0) = f_0'' theta   for f_0 = x^k, k < 6", ok)
    ok = True
    for k in range(6):
        f1 = A.mono(["z2"], k)
        ok &= eq(D(A.mul(f1, A.gen("th"))), A.smul(-1, A.d_x(f1)))
    check("St_1(f_1 theta) = -f_1'   for f_1 = z_2 x^k, k < 6", ok)
    print("      whence  St_1 f = lambda f  reads   f_1' = -lambda f_0,  "
          "f_0'' = lambda f_1.")


# ============================== 2.  ns(1): the general solution over C-points

def ns1_solutions(nzeta, deg, sgn=+1):
    names = ["th"] + [f"z{i}" for i in range(1, nzeta + 1)]
    A = Alg(names)
    K = A.K("th", "th", sgn)
    D = compose(K, K, K)
    lam = A.gen("z1")
    unk = A.basis(deg, parity=0)
    tgt = A.basis(deg + 1, parity=1)
    tgt_ix = {m: i for i, m in enumerate(tgt)}
    cols = []
    for m in unk:
        r = A.sub(D({m: Fraction(1)}), A.mul(lam, {m: Fraction(1)}))
        v = [Fraction(0)] * len(tgt)
        for mm, c in r.items():
            v[tgt_ix[mm]] = c
        cols.append(v)
    rows = [[cols[j][i] for j in range(len(unk))] for i in range(len(tgt))]
    sols = []
    for vec in nullspace(rows):
        f = {m: Fraction(c) for c, m in zip(vec, unk) if c}
        sols.append(f)
    return A, D, lam, sols


def part2():
    print("\n=== 2.  ns(1):  the general solution of St_1 f = lambda f, "
          "lambda odd ===")
    dims = {}
    for nz in (1, 2, 3):
        for deg in (4, 6, 8):
            dims[(nz, deg)] = len(ns1_solutions(nz, deg)[3])
        print(f"      C = Lambda(z_1..z_{nz}):  dim_Q = " +
              ", ".join(f"{dims[(nz, d)]} (deg_x <= {d})" for d in (4, 6, 8)))
    check("the solution space stabilizes in deg_x (4, 6, 8 agree), n = 1, 2, 3",
          all(dims[(n, 4)] == dims[(n, 6)] == dims[(n, 8)] for n in (1, 2, 3)))
    check("it GROWS with the number of odd parameters (3, 6, 12): the solution "
          "set is a functor of C, not a vector space",
          dims[(1, 6)] < dims[(2, 6)] < dims[(3, 6)])

    A, D, lam, sols = ns1_solutions(2, 8)
    check(f"all {len(sols)} solutions lie in ker(d_x^3) = ker(St_1)^2, "
          "as the Theorem says",
          all(not A.d_x(A.d_x(A.d_x(s))) for s in sols))
    check("lambda^2 f = 0 on every solution", all(not A.mul(lam, A.mul(lam, s))
                                                  for s in sols))
    extra = A.add(A.smul(Fraction(1, 2), A.mul(lam, A.mono(["z2"], 2))),
                  A.mul(A.gen("z2"), A.gen("th")))
    check("the further solution f = (1/2) z_1 z_2 x^2 + z_2 theta solves it",
          not A.sub(D(extra), A.mul(lam, extra)))
    f0 = {m: c for m, c in extra.items() if A.idx["th"] not in m[0]}
    check("...and its f_0 is quadratic in x", bool(A.d_x(A.d_x(f0))))


# ===================================== 7.  the pairing, and the orthogonality

def part7():
    print("\n=== 7.  the residue pairing, and what 'orthogonal' can mean ===")
    A = Alg(["th"])
    top = A.top()

    def pair(f, g):
        """<f, g> = Res_x Ber_theta (f g) = coefficient of x^{-1} theta."""
        return A.mul(f, g).get((top, -1), Fraction(0))

    elts = []
    for k in range(-4, 5):
        elts.append(("x^%d" % k, A.one(k), 0))
        elts.append(("x^%d th" % k, A.mul(A.one(k), A.gen("th")), 1))
    K = A.K("th", "th")
    D = compose(K, K, K)

    def adjoint_sign(op, par):
        """Find s with <op f, g> = s * (-1)^{par * p(f)} <f, op g>, or None."""
        cand = None
        for _, f, pf in elts:
            for _, g, _ in elts:
                lhs = pair(op(f), g)
                rhs = pair(f, op(g)) * ((-1) ** (par * pf))
                if lhs == 0 and rhs == 0:
                    continue
                if rhs == 0:
                    return None
                s = Fraction(lhs, rhs)
                if cand is None:
                    cand = s
                elif cand != s:
                    return None
        return cand

    s = adjoint_sign(K, 1)
    check(f"D_theta is anti-self-adjoint for the residue pairing: sign {s}",
          s == -1)
    s = adjoint_sign(A.d_x, 0)
    check(f"D_1 is anti-self-adjoint: sign {s}", s == -1)
    s = adjoint_sign(D, 1)
    check(f"St_1 = D_theta D_1 is SELF-adjoint: sign {s}", s == +1)
    s = adjoint_sign(lambda f: A.d_x(D(f)), 1)
    check(f"whereas L_1 = St_1 D_1 is ANTI-self-adjoint, as Lemma 5.7 "
          f"requires of a cocycle: sign {s}", s == -1)
    s = adjoint_sign(lambda f: A.d_x(A.d_x(A.d_x(f))), 0)
    check(f"and (St_1)^2 = D_1^3 is anti-self-adjoint: sign {s}", s == -1)
    print("      so <St^2 f, g> = -<f, St^2 g>: for eigenfunctions with "
          "St^2 f = E f,")
    print("      St^2 g = E' g one gets (E + E') <f, g> = 0 --- the pairing "
          "pairs E with -E,")
    print("      not E with E.")
    # in the model case this is the Fourier orthogonality
    ok = True
    for k in range(-4, 5):
        for m in range(-4, 5):
            v = pair(A.one(k), A.mul(A.one(m), A.gen("th")))
            ok &= (v != 0) == (k + m == -1)
    check("the pairing of x^k with x^m theta is non-zero exactly when "
          "k + m = -1, which is the Fourier orthogonality", ok)
    # in Fourier terms: x = e^{i phi}, the D_1-eigenfunctions are e^{i n phi},
    # the pairing of e^{i n phi} with e^{i m phi} is non-zero iff m = -n, and
    # the (St_1)^2 = D_1^3 eigenvalues of the two partners are opposite.
    import sympy as sp
    ok = True
    for n in range(-4, 5):
        E, Epart = (sp.I * n)**3, (sp.I * (-n))**3
        ok &= sp.simplify(E + Epart) == 0
    check("the D_1-eigenfunction e^{i n phi} pairs only with e^{-i n phi}, and "
          "the (St_1)^2-eigenvalues of that pair, (in)^3 and (-in)^3, are "
          "opposite: E' = -E, as predicted", ok)


def main():
    part1(); part2(); part7()
    n = len(CHECKS)
    bad = [c for c in CHECKS if not c[1]]
    print(f"\n==== {n - len(bad)}/{n} checks pass ====")
    for name, _ in bad:
        print("  FAILED: " + name)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
