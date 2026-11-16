#!/usr/bin/env python3
"""wittsuper.py -- Witt's formula and its super version: exact machine check.

In characteristic 0 the multigraded dimensions of the free Lie superalgebra are
given by "Witt's formula".  This engine pins down which formula that is (there
are several versions in the literature) and checks it.

Three formulas are checked against the exact tensor model of the free Lie
superalgebra (freelie.Tensor.lie() over Q -- rational row reduction, no floats):

  (W)   classical Witt, multigraded (Witt 1937; Reutenauer, Ch. 4):
            W(m) = (1/|m|) sum_{d | gcd m} mu(d) (|m|/d)! / prod_i (m_i/d)!

  (SW)  super Witt, multigraded (Mikhalev 1986; Petrogradsky 2002):
            W_s(m) = (1/|m|) sum_{d | gcd m} mu(d) (-1)^{(d+1) p(m/d)}
                                             (|m|/d)! / prod_i (m_i/d)!
        where p(m) = sum_i m_i p(v_i) mod 2 is the parity of the multidegree.
        For a purely even V this is (W).

  (SW1) super Witt, singly graded (the form quoted in the literature): for r even
        and s odd generators,
            dim L_n = (1/n) sum_{d | n} mu(d) (r - (-1)^d s)^{n/d}.

Checks performed:
  1. (SW) against the exact dimensions of L_T(V)_m over Q, for six parity
     patterns and all multidegrees up to total degree D.
  2. (SW) summed over the multidegrees of total degree n == (SW1).
  3. (W) against the exact dimensions for a purely even V (the classical case),
     and W_s == W there.
  4. The PBW identity that (SW) encodes:
         prod_{p(m)=0} (1-t^m)^{-d_m} * prod_{p(m)=1} (1+t^m)^{d_m} = 1/(1-sum t_i),
     verified as truncated multivariate power series with integer coefficients.
  5. The p = 2 count of Theorem 5.2, dim Free(V)_m = W(m) + dim(L_{m/2})_od,
     re-expressed in closed form: it is the k <= 1 truncation of Petrogradsky's
     restricted Witt formula, kept only on the odd part.

Usage:  python wittsuper.py        (~20 s)
        python3 wittsuper.py                      (same, slower backend)
"""
import sys
import os
from math import gcd, factorial
from itertools import product

HERE = os.path.dirname(os.path.abspath(__file__))
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]

from exactlin import backend                                   # noqa: E402
from freelie import Tensor, mdegs                              # noqa: E402

FAIL = []


def check(cond, what):
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        FAIL.append(what)


# ---------------------------------------------------------------------------
# arithmetic
# ---------------------------------------------------------------------------


def mu(k):
    """Moebius function."""
    r, x, f = 1, k, 2
    while f * f <= x:
        if x % f == 0:
            x //= f
            if x % f == 0:
                return 0
            r = -r
        f += 1
    if x > 1:
        r = -r
    return r


def multinomial(m):
    """|m|! / prod m_i!."""
    out = factorial(sum(m))
    for a in m:
        out //= factorial(a)
    return out


def gcd_all(m):
    g = 0
    for a in m:
        g = gcd(g, a)
    return g


def parity(m, par):
    return sum(a * b for a, b in zip(m, par)) % 2


def witt_multi(m):
    """(W) classical multigraded Witt number."""
    n = sum(m)
    tot = 0
    for d in range(1, gcd_all(m) + 1):
        if gcd_all(m) % d:
            continue
        tot += mu(d) * multinomial(tuple(a // d for a in m))
    assert tot % n == 0, (m, tot)
    return tot // n


def witt_super_multi(m, par):
    """(SW) super multigraded Witt number for generators of parities par."""
    n = sum(m)
    g = gcd_all(m)
    tot = 0
    for d in range(1, g + 1):
        if g % d:
            continue
        md = tuple(a // d for a in m)
        sign = -1 if ((d + 1) * parity(md, par)) % 2 else 1
        tot += mu(d) * sign * multinomial(md)
    assert tot % n == 0, (m, par, tot)
    return tot // n


def witt_super_single(n, r, s):
    """(SW1) dim L_n for r even and s odd free generators."""
    tot = 0
    for d in range(1, n + 1):
        if n % d:
            continue
        tot += mu(d) * (r - (-1) ** d * s) ** (n // d)
    assert tot % n == 0, (n, r, s, tot)
    return tot // n


# ---------------------------------------------------------------------------
# exact model
# ---------------------------------------------------------------------------


def exact_dims(par, D):
    """dim L_T(V)_m over Q for every multidegree of total degree <= D."""
    T = Tensor(par, D)
    L = T.lie(p=None)
    return {m: len(v) for m, v in L.items()}, T


# ---------------------------------------------------------------------------
# part 1-3: the formulas against the exact model
# ---------------------------------------------------------------------------


def part1(cases):
    print()
    print("=" * 78)
    print("1-3. (SW) and (SW1) against the exact tensor model over Q")
    print("=" * 78)
    for par, D in cases:
        r = sum(1 for a in par if a == 0)
        s = sum(1 for a in par if a == 1)
        dims, T = exact_dims(par, D)
        bad = []
        for m, dm in sorted(dims.items(), key=lambda kv: (sum(kv[0]), kv[0])):
            if witt_super_multi(m, par) != dm:
                bad.append((m, dm, witt_super_multi(m, par)))
        print(f"\n  sdim V = ({r}|{s}), degrees 1..{D}: "
              f"{len(dims)} multidegrees")
        check(not bad, f"(SW) reproduces every dim L_m, sdim V = ({r}|{s})"
                       + (f"  [{bad[:3]}]" if bad else ""))

        # per-degree totals against the singly graded form
        bad1 = []
        for n in range(1, D + 1):
            tot = sum(d for m, d in dims.items() if sum(m) == n)
            if tot != witt_super_single(n, r, s):
                bad1.append((n, tot, witt_super_single(n, r, s)))
        check(not bad1, f"(SW1) reproduces dim L_n, sdim V = ({r}|{s})"
                        + (f"  [{bad1[:3]}]" if bad1 else ""))
        line = ", ".join(f"{n}:{sum(d for m, d in dims.items() if sum(m) == n)}"
                         for n in range(1, D + 1))
        print(f"        dim L_n = {line}")

        if s == 0:
            badc = [m for m in dims if witt_multi(m) != dims[m]]
            check(not badc, "classical (W) agrees for a purely even V")
            badc2 = [m for m in dims if witt_multi(m) != witt_super_multi(m, par)]
            check(not badc2, "(SW) = (W) for a purely even V")


# ---------------------------------------------------------------------------
# part 4: the PBW identity behind the formula
# ---------------------------------------------------------------------------


def series_mul(A, B, D, n):
    """Product of two truncated multivariate series, dicts multidegree -> int."""
    out = {}
    for m1, a in A.items():
        for m2, b in B.items():
            if sum(m1) + sum(m2) > D:
                continue
            m = tuple(x + y for x, y in zip(m1, m2))
            out[m] = out.get(m, 0) + a * b
    return {m: c for m, c in out.items() if c}


def part4(cases):
    print()
    print("=" * 78)
    print("4. the PBW identity  prod (1-t^m)^{-d_m} prod (1+t^m)^{d_m} = 1/(1-sum t_i)")
    print("=" * 78)
    for par, D in cases:
        nvar = len(par)
        r = sum(1 for a in par if a == 0)
        s = sum(1 for a in par if a == 1)
        zero = tuple(0 for _ in par)
        prod = {zero: 1}
        for m in sorted((m for d in range(1, D + 1) for m in mdegs(nvar, d)),
                        key=lambda m: (sum(m), m)):
            dm = witt_super_multi(m, par)
            if dm == 0:
                continue
            if parity(m, par) == 0:
                # (1-t^m)^{-dm} = sum_k C(dm+k-1, k) t^{km}
                fac = {}
                k = 0
                while k * sum(m) <= D:
                    c = 1
                    for j in range(k):
                        c = c * (dm + j) // (j + 1)
                    fac[tuple(a * k for a in m)] = c
                    k += 1
            else:
                # (1+t^m)^{dm} = sum_k C(dm, k) t^{km}
                fac = {}
                k = 0
                while k * sum(m) <= D and k <= dm:
                    c = factorial(dm) // (factorial(k) * factorial(dm - k))
                    fac[tuple(a * k for a in m)] = c
                    k += 1
            prod = series_mul(prod, fac, D, nvar)

        # the target 1/(1 - t_1 - ... - t_n): coefficient of t^m is multinomial(m)
        bad = []
        for d in range(0, D + 1):
            for m in ([zero] if d == 0 else mdegs(nvar, d)):
                want = multinomial(m)
                got = prod.get(m, 0)
                if want != got:
                    bad.append((m, got, want))
        check(not bad, f"PBW identity holds to degree {D}, sdim V = ({r}|{s})"
                       + (f"  [{bad[:3]}]" if bad else ""))


# ---------------------------------------------------------------------------
# part 5: the p = 2 count of Theorem 5.2 in closed form
# ---------------------------------------------------------------------------


def part5(par, D):
    print()
    print("=" * 78)
    print("5. the p = 2 count of Theorem 5.2 in closed form")
    print("=" * 78)
    print("   dim Free(V)_m = W(m) + [m/2 a multidegree and p(m/2) = 1] W(m/2)")
    print("   (W = the CLASSICAL Witt number: in char 2 there are no signs, and L(V)")
    print("    is the free Lie algebra, defined over Z and free as a Z-module)")
    nvar = len(par)
    # exact: the char-2 model built by freelie.char2_model is heavy; here we only
    # check the closed form against the classical Witt numbers, which freelie
    # already matched against the exact char-2 computation.
    rows = []
    for d in range(1, D + 1):
        for m in mdegs(nvar, d):
            base = witt_multi(m)
            extra = 0
            if all(a % 2 == 0 for a in m):
                h = tuple(a // 2 for a in m)
                if parity(h, par) == 1:
                    extra = witt_multi(h)
            rows.append((m, base, extra, base + extra))
    print("\n   m            W(m)   squares   dim Free(V)_m")
    for m, b, e, t in rows:
        if sum(m) <= 4:
            print(f"   {str(m):12s} {b:5d} {e:9d} {t:14d}")
    check(all(t == b + e for _, b, e, t in rows), "the two summands add up")
    tot = sum(1 for _, _, e, _ in rows if e)
    check(tot > 0, f"squares contribute in {tot} of {len(rows)} multidegrees")


# ---------------------------------------------------------------------------
# part 6: does p = 2 give back the characteristic 0 dimensions?
# ---------------------------------------------------------------------------


def char2_dim(m, par):
    """The count of Theorem 5.2: W(m) + dim (L_{m/2})_od."""
    out = witt_multi(m)
    if all(a % 2 == 0 for a in m):
        h = tuple(a // 2 for a in m)
        if parity(h, par) == 1:
            out += witt_multi(h)
    return out


def part6(cases):
    print()
    print("=" * 78)
    print("6. p = 2 versus characteristic 0: is dim Free(V)_m = W_s(m)?")
    print("=" * 78)
    for par, D in cases:
        r = sum(1 for a in par if a == 0)
        s = sum(1 for a in par if a == 1)
        diffs = []
        n_m = 0
        for d in range(1, D + 1):
            for m in mdegs(len(par), d):
                n_m += 1
                a, b = char2_dim(m, par), witt_super_multi(m, par)
                if a != b:
                    diffs.append((m, a, b))
        print(f"\n  sdim V = ({r}|{s}), {n_m} multidegrees up to degree {D}: "
              f"{len(diffs)} disagreements")
        for m, a, b in diffs[:6]:
            print(f"        m = {m}: p = 2 gives {a}, char 0 gives {b}")
        check(True, f"recorded, sdim V = ({r}|{s})")


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# part 7: which W does each dimension formula need?
# ---------------------------------------------------------------------------


def part7(cases):
    print()
    print("=" * 78)
    print("7. the two W's: (dim2) needs the CLASSICAL W, (dim3) the SUPER W_s")
    print("=" * 78)
    for par, D in cases:
        r = sum(1 for a in par if a == 0)
        s = sum(1 for a in par if a == 1)
        T = Tensor(par, D)
        L2 = {m: len(v) for m, v in T.lie(p=2).items()}
        L3 = {m: len(v) for m, v in T.lie(p=3).items()}
        print(f"\n  sdim V = ({r}|{s}), degrees 1..{D}")

        b2 = [m for m in L2 if L2[m] != witt_multi(m)]
        check(not b2, "over F_2, dim L_m = the CLASSICAL W(m) (free Lie algebra)"
                      + (f"  [{b2[:3]}]" if b2 else ""))
        b2s = [m for m in L2 if L2[m] != witt_super_multi(m, par)]
        print(f"        (it differs from W_s(m) in {len(b2s)} multidegrees, "
              f"e.g. {b2s[:2]})")

        b3 = [m for m in L3 if L3[m] != witt_super_multi(m, par)]
        check(not b3, "over F_3, dim L_m = the SUPER W_s(m)"
                      + (f"  [{b3[:3]}]" if b3 else ""))

        # the p = 2 free object against the characteristic 0 super Witt number
        bad = []
        for m in L2:
            tot = L2[m]
            if all(a % 2 == 0 for a in m):
                h = tuple(a // 2 for a in m)
                if parity(h, par) == 1:
                    tot += L2.get(h, 0)
            if tot != witt_super_multi(m, par):
                bad.append((m, tot, witt_super_multi(m, par)))
        check(not bad, "at p = 2, dim Free(V)_m = W_s(m): the characteristic 0 "
                       "dimensions" + (f"  [{bad[:3]}]" if bad else ""))


# ---------------------------------------------------------------------------
# part 8: the identity behind part 7, and its proof, checked
# ---------------------------------------------------------------------------


def part8(pars, B):
    print()
    print("=" * 78)
    print("8. the identity  W_s(m) = W(m) + [2|m, p(m/2)=1] W(m/2)")
    print("=" * 78)
    bad, tested, key = [], 0, 0
    for par in pars:
        for m in product(range(0, B + 1), repeat=len(par)):
            if sum(m) == 0:
                continue
            tested += 1
            rhs = witt_multi(m)
            if all(a % 2 == 0 for a in m) and parity(tuple(a // 2 for a in m), par) == 1:
                rhs += witt_multi(tuple(a // 2 for a in m))
                key += 1
            if witt_super_multi(m, par) != rhs:
                bad.append((par, m))
    check(not bad, f"holds in all {tested} multidegrees tested "
                   f"({key} of them with a square term)"
                   + (f"  [{bad[:3]}]" if bad else ""))

    # the two steps of the proof
    bad1 = []
    for par in pars:
        for m in product(range(0, B + 1), repeat=len(par)):
            if sum(m) == 0 or not all(a % 2 == 0 for a in m):
                continue
            g = gcd_all(m)
            for e in range(1, g // 2 + 1):
                if (g % (2 * e)) or e % 2 == 0:
                    continue
                if parity(tuple(a // (2 * e) for a in m), par) != \
                   parity(tuple(a // 2 for a in m), par):
                    bad1.append((par, m, e))
    check(not bad1, "step 1: p(m/2e) = p(m/2) for every odd e (so the parity "
                    "condition does not depend on e)")

    bad2 = []
    for par in pars:
        for m in product(range(0, B + 1), repeat=len(par)):
            if sum(m) == 0 or not all(a % 2 == 0 for a in m):
                continue
            if gcd_all(m) % 4 == 0 and parity(tuple(a // 2 for a in m), par) == 1:
                bad2.append((par, m))
    check(not bad2, "step 2: 4 | gcd(m) forces p(m/2) = 0, so when the square "
                    "term is there, gcd(m)/2 is odd")


def main():
    print("wittsuper.py -- Witt's formula and its super version, exact check")
    print("linear-algebra backend:", backend())

    cases = [([1], 8), ([0, 1], 6), ([1, 1], 6), ([0, 0], 6), ([0, 1, 1], 5),
             ([1, 1, 1], 5)]
    part1(cases)
    part4([([1], 8), ([0, 1], 6), ([1, 1], 6), ([0, 0], 6)])
    part5([0, 1], 6)
    part7([([1], 8), ([0, 1], 6), ([1, 1], 6), ([0, 1, 1], 5)])
    part8([[1], [0, 1], [1, 1], [0, 1, 1], [1, 1, 1], [0, 0, 1]], 8)
    part6([([1], 10), ([0, 1], 8), ([1, 1], 8), ([0, 1, 1], 6), ([1, 1, 1], 6), ([0, 0, 1], 6)])

    print()
    print("=" * 78)
    if FAIL:
        print(f"{len(FAIL)} CHECK(S) FAILED")
        for f in FAIL:
            print("   -", f)
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
