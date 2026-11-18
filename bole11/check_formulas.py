#!/usr/bin/env python3
"""Independent check of the singular-vector computations of the text
"Gordan-Rankin-Cohen operators on the spaces of weighted densities in superdimension 1|1".

Section 2 (k(1|1), osp(1|2)-singular vectors), Section 3 (vect(1|1), pgl(2|1)).

Everything is built from the *elementary* action rules on the induced modules,
verified by hand from the contact bracket / vector-field bracket; the linear
systems of the text are NOT reused.  Exact arithmetic over Q.
"""
from fractions import Fraction as F
from itertools import product

# ---------------------------------------------------------------- linear algebra
def nullity(rows, ncols):
    """rank/nullity of a matrix given as list of dict{col: Fraction}."""
    M = [dict(r) for r in rows]
    rank = 0
    pivots = set()
    for col in range(ncols):
        piv = None
        for r in M:
            if r.get(col):
                piv = r
                break
        if piv is None:
            continue
        M.remove(piv)
        p = piv[col]
        for r in M:
            if r.get(col):
                f = r[col] / p
                for c, v in piv.items():
                    r[c] = r.get(c, F(0)) - f * v
                    if r[c] == 0:
                        del r[c]
        rank += 1
        pivots.add(col)
    return ncols - rank


# ================================================================ Section 2
# Basis of I(V1*) (x) I(V2*): (e1, i, e2, j) == Kth^e1 K1^i v (x) Kth^e2 K1^j w
# degree = 2i + e1 + 2j + e2.   nabla_+ = K_{t theta}:
#   nabla_+(K1^i v)        = -2 i Kth K1^{i-1} v
#   nabla_+(Kth K1^i v)    = (mu - i) K1^i v
# and nabla_+(A (x) B) = nabla_+(A)(x)B + (-1)^{p(A)} A (x) nabla_+(B).
def basis_k11(deg):
    out = []
    for e1, e2 in product((0, 1), repeat=2):
        rest = deg - e1 - e2
        if rest < 0 or rest % 2:
            continue
        for i in range(rest // 2 + 1):
            out.append((e1, i, e2, rest // 2 - i))
    return sorted(out)


def nabla_plus(el, mu1, mu2):
    e1, i, e2, j = el
    out = {}
    def add(k, c):
        out[k] = out.get(k, F(0)) + c
    # first factor
    if e1 == 0:
        if i > 0:
            add((1, i - 1, e2, j), F(-2 * i))
    else:
        add((0, i, e2, j), F(mu1 - i))
    sign = -1 if e1 else 1
    # second factor
    if e2 == 0:
        if j > 0:
            add((e1, i, 1, j - 1), F(sign * -2 * j))
    else:
        add((e1, i, 0, j), F(sign) * F(mu2 - j))
    return {k: v for k, v in out.items() if v}


def dim_sing_k11(deg, mu1, mu2):
    src = basis_k11(deg)
    tgt = basis_k11(deg - 1)
    idx = {b: n for n, b in enumerate(src)}
    tix = {b: n for n, b in enumerate(tgt)}
    rows = [dict() for _ in tgt]
    for b in src:
        for t, c in nabla_plus(b, mu1, mu2).items():
            rows[tix[t]][idx[b]] = rows[tix[t]].get(idx[b], F(0)) + c
    return nullity(rows, len(src))


# ================================================================ Section 3
# Basis of I(V1)(x)I(V2): (k, d1, m, d2) == d^k delta^d1 v (x) d^m delta^d2 w
#   X_+ (d^k v)        = -k d^{k-1} delta v ;      X_+ (d^k delta v) = 0
#   s_xi(d^k v)        = 0 ;                       s_xi(d^k delta v) = (mu/2 - k) d^k v
def basis_v11(n):
    out = []
    for d1, d2 in product((0, 1), repeat=2):
        rest = n - d1 - d2
        if rest < 0:
            continue
        for k in range(rest + 1):
            out.append((k, d1, rest - k, d2))
    return sorted(out)


def act(el, mu1, mu2, which):
    k, d1, m, d2 = el
    out = {}
    def add(key, c):
        out[key] = out.get(key, F(0)) + c
    if which == "X":
        if d1 == 0 and k > 0:
            add((k - 1, 1, m, d2), F(-k))
        sign = -1 if d1 else 1
        if d2 == 0 and m > 0:
            add((k, d1, m - 1, 1), F(sign * -m))
    else:  # s_xi
        if d1 == 1:
            add((k, 0, m, d2), F(mu1, 2) - k)
        sign = -1 if d1 else 1
        if d2 == 1:
            add((k, d1, m, 0), F(sign) * (F(mu2, 2) - m))
    return {key: v for key, v in out.items() if v}


def dim_sing_v11(n, mu1, mu2):
    """returns (even_dim, odd_dim) of ker X_+ cap ker s_xi on level -n"""
    res = {}
    for par in (0, 1):
        src = [b for b in basis_v11(n) if (b[1] + b[3]) % 2 == par]
        idx = {b: i for i, b in enumerate(src)}
        rows = []
        for which in ("X", "s"):
            tgt = {}
            for b in src:
                for t, c in act(b, mu1, mu2, which).items():
                    tgt.setdefault(t, {})[idx[b]] = c
            rows.extend(tgt.values())
        res[par] = nullity(rows, len(src)) if src else 0
    return res[0], res[1]


# ================================================================ predictions
def pred_k11_even(n, m1, m2):   # v_{2n}
    ok = lambda m: isinstance(m, int) and 1 <= m <= n - 1
    return 2 if ok(m1) and ok(m2) and m1 + m2 >= n else 1


def pred_k11_odd(n, m1, m2):    # v_{2n+1}
    ok = lambda m: isinstance(m, int) and 0 <= m <= n
    return 2 if ok(m1) and ok(m2) and m1 + m2 >= n else 1


def pred_paper_thm(n, m1, m2):  # the Theorem on the superdimension of the space of pgl(2|1)-singular vectors
    ev = lambda m: isinstance(m, int) and m >= 0 and m % 2 == 0
    if ev(m1) and ev(m2) and m1 + m2 == 2 * n - 4:
        return (1, 1)
    if ev(m1) and ev(m2) and m1 <= 2 * n - 2 and m2 <= 2 * n - 2 and m1 + m2 >= 2 * n - 2:
        return (0, 2)
    return (0, 1)


if __name__ == "__main__":
    rng = list(range(-2, 9)) + [F(1, 2), F(5, 3), F(-7, 2)]
    print("=== Section 2:  dim of osp(1|2)-singular vectors in I(V1*)(x)I(V2*) ===")
    bad = 0
    for n in range(0, 7):
        for m1, m2 in product(rng, repeat=2):
            d = dim_sing_k11(2 * n, m1, m2)
            p = pred_k11_even(n, m1, m2)
            if d != p and n >= 1:
                bad += 1
                print(f"  MISMATCH even n={n} mu=({m1},{m2}): computed {d}, predicted {p}")
            d = dim_sing_k11(2 * n + 1, m1, m2)
            p = pred_k11_odd(n, m1, m2)
            if d != p:
                bad += 1
                print(f"  MISMATCH odd  n={n} mu=({m1},{m2}): computed {d}, predicted {p}")
    print(f"  mismatches with the corrected criteria: {bad}")

    print("\n  sample dimensions (2n = v_{2n}, 2n+1 = v_{2n+1}):")
    for n in (0, 1, 2, 3):
        for m1, m2 in ((0, 0), (1, 1), (2, 1), (1, 2), (2, 2), (3, 0)):
            print(f"    n={n} mu=({m1},{m2}):  dim v_2n={dim_sing_k11(2*n,m1,m2)}"
                  f"  dim v_2n+1={dim_sing_k11(2*n+1,m1,m2)}")

    print("\n=== Section 3:  Theorem 3.6.1 (pgl(2|1)-singular vectors) ===")
    bad = 0
    for n in range(1, 8):
        for m1, m2 in product(rng, repeat=2):
            d = dim_sing_v11(n, m1, m2)
            p = pred_paper_thm(n, m1, m2)
            if d != p:
                bad += 1
                if bad < 25:
                    print(f"  MISMATCH n={n} mu=({m1},{m2}): computed {d}, the text says {p}")
    print(f"  mismatches with the Theorem of the text: {bad}")
    print("  n=0 check:", dim_sing_v11(0, 3, 5), "(the Theorem is stated for n >= 1)")
