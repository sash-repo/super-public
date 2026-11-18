#!/usr/bin/env python3
"""Bilinear osp(1|2)-invariant operators on weighted densities over S^{1|1}_c.

Everything is computed on the *operator* side (no induced modules), from the
definitions:

    F_w = C[t,theta] alpha^{w/2},   alpha = dt + theta dtheta,
    K_f = (2-E)(f) d_t - (-1)^{p(f)} (d_theta f - theta d_t f) d_theta,
    L_{K_f}(u alpha^{w/2}) = (K_f(u) + w d_t(f) u) alpha^{w/2}   (since L_{K_f} alpha = 2 f_t alpha).

osp(1|2) = <K_theta, K_{t theta}> (they generate: [K_th,K_th]=K_1, [K_th,K_tth]=K_t,
[K_tth,K_tth]=K_{t^2}), so invariance under those two odd fields is invariance
under osp(1|2).  Exact rational arithmetic.

No dependencies:  python3 grc_ops.py
"""
from fractions import Fraction as F
from itertools import product

Z = F(0)

# --------------------------------------------------------------- C[t,theta]
# element = dict {(a, e): coeff},  a >= 0, e in {0,1}   <->  coeff * t^a theta^e


def clean(d):
    return {k: v for k, v in d.items() if v}


def add(dst, src, c=1):
    for k, v in src.items():
        dst[k] = dst.get(k, Z) + c * v
        if dst[k] == 0:
            del dst[k]
    return dst


def mul(u, v):
    out = {}
    for (a1, e1), c1 in u.items():
        for (a2, e2), c2 in v.items():
            if e1 and e2:
                continue                      # theta^2 = 0
            k = (a1 + a2, e1 + e2)            # t is even: no sign
            out[k] = out.get(k, Z) + c1 * c2
    return clean(out)


def dt(u):
    out = {}
    for (a, e), c in u.items():
        if a:
            out[(a - 1, e)] = out.get((a - 1, e), Z) + a * c
    return clean(out)


def dth(u):                                   # left derivative d/d theta
    out = {}
    for (a, e), c in u.items():
        if e:
            out[(a, 0)] = out.get((a, 0), Z) + c
    return clean(out)


def thmul(u):                                 # theta . u
    return clean({(a, 1): c for (a, e), c in u.items() if e == 0})


def tmul(u):
    return clean({(a + 1, e): c for (a, e), c in u.items()})


# ------------------------------------------------------- the two odd fields
# K_theta   = theta d_t + d_theta                    (weight-independent)
# K_{t th}  = t theta d_t + t d_theta  + w theta     on F_w
def LX(u, X, w):
    if X == "th":
        out = thmul(dt(u))
        return add(out, dth(u))
    out = tmul(thmul(dt(u)))
    add(out, tmul(dth(u)))
    add(out, thmul(u), w)
    return clean(out)


# ------------------------------------------------------------- the operators
# The general bidifferential operator invariant under translations K_1 = 2 d_t
# has coefficients independent of t, i.e. of the form  a + b theta.  Since
# theta^2 = 0, a basis of such operators of K_t-degree k is given by the words
#
#   key (d,e1,i,e2,j):  B(f,g) = theta^d . (d_t^i d_theta^e1 f).(d_t^j d_theta^e2 g)
#
# with d in {0,1} and k = e1 + 2i + e2 + 2j - d  (deg t = 2, deg theta = 1).
# NB: this space is strictly bigger than the constant-coefficient one -- e.g.
# D_theta f . g = (theta d_t f).g - (d_theta f).g needs the d = 1 words.
# Putting the theta in the middle gives nothing new: u.theta.v = (-1)^p(u) theta.u.v,
# and theta.u = 0 whenever u is odd, so both words define the same bilinear map.
def op_keys(k):
    keys = []
    for d in (0, 1):
        for e1, e2 in product((0, 1), repeat=2):
            r = k + d - e1 - e2
            if r < 0 or r % 2:
                continue
            n = r // 2
            for i in range(n + 1):
                keys.append((d, e1, i, e2, n - i))
    return sorted(keys)


def apply_key(key, f, g):
    d, e1, i, e2, j = key
    u = dth(f) if e1 else dict(f)
    for _ in range(i):
        u = dt(u)
    v = dth(g) if e2 else dict(g)
    for _ in range(j):
        v = dt(v)
    out = mul(u, v)
    return thmul(out) if d else out


def apply_op(op, keys, f, g):
    out = {}
    for c, key in zip(op, keys):
        if c:
            add(out, apply_key(key, f, g), c)
    return clean(out)


# ------------------------------------------------------------ linear algebra
def rref(rows, ncols):
    M = [list(r) for r in rows if any(r)]
    piv = []
    r = 0
    for c in range(ncols):
        p = None
        for i in range(r, len(M)):
            if M[i][c]:
                p = i
                break
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        pv = M[r][c]
        M[r] = [x / pv for x in M[r]]
        for i in range(len(M)):
            if i != r and M[i][c]:
                f0 = M[i][c]
                M[i] = [x - f0 * y for x, y in zip(M[i], M[r])]
        piv.append(c)
        r += 1
        if r == len(M):
            break
    return M[:r], piv


def nullspace(rows, ncols):
    R, piv = rref(rows, ncols)
    free = [c for c in range(ncols) if c not in piv]
    basis = []
    for fc in free:
        v = [Z] * ncols
        v[fc] = F(1)
        for i, pc in enumerate(piv):
            v[pc] = -R[i][fc]
        basis.append(v)
    return basis


# -------------------------------------------- invariant operators of order k
def invariant_ops(k, w1, w2, sigma, A=None):
    """basis of the space of osp(1|2)-invariant order-k bilinear operators
    F_w1 (x) F_w2 -> F_{w1+w2+k}, with the sign convention
        L_X B(f,g) = sigma * ( B(L_X f, g) + (-1)^{p(f)} B(f, L_X g) )   (X odd)."""
    w3 = w1 + w2 + k
    keys = op_keys(k)
    if A is None:
        A = k + 3
    tests = [{(a, e): F(1)} for a in range(A + 1) for e in (0, 1)]
    rows = []
    for X in ("th", "tth"):
        for f in tests:
            pf = list(f.keys())[0][1]
            for g in tests:
                res = {}
                for n, key in enumerate(keys):
                    tot = dict(LX(apply_key(key, f, g), X, w3))
                    add(tot, apply_key(key, LX(f, X, w1), g), -sigma)
                    add(tot, apply_key(key, f, LX(g, X, w2)), -sigma * (-1) ** pf)
                    for mono, c in tot.items():
                        res.setdefault(mono, [Z] * len(keys))[n] = c
                rows.extend(res.values())
    return keys, nullspace(rows, len(keys))


def verify_invariance(op, keys, k, w1, w2, sigma, A):
    """re-check a found operator on a larger set of test monomials"""
    w3 = w1 + w2 + k
    tests = [{(a, e): F(1)} for a in range(A + 1) for e in (0, 1)]
    for X in ("th", "tth"):
        for f in tests:
            pf = list(f.keys())[0][1]
            for g in tests:
                tot = dict(LX(apply_op(op, keys, f, g), X, w3))
                add(tot, apply_op(op, keys, LX(f, X, w1), g), -sigma)
                add(tot, apply_op(op, keys, f, LX(g, X, w2)), -sigma * (-1) ** pf)
                if clean(tot):
                    return False
    return True


# ------------------------------------------------------------------ report
def pred_even(n, m1, m2):        # corrected Prop. (even case), v_{2n}, order k=2n
    ok = lambda m: isinstance(m, int) and 1 <= m <= n - 1
    return 2 if ok(m1) and ok(m2) and m1 + m2 >= n else 1


def pred_odd(n, m1, m2):         # corrected Prop. (odd case), v_{2n+1}, order k=2n+1
    ok = lambda m: isinstance(m, int) and 0 <= m <= n
    return 2 if ok(m1) and ok(m2) and m1 + m2 >= n else 1


if __name__ == "__main__":
    print("=== which sign convention sigma gives a non-zero answer, per order k ===")
    for k in range(0, 6):
        line = []
        for sigma in (1, -1):
            keys, B = invariant_ops(k, F(0), F(0), sigma)
            line.append(f"sigma={sigma:+d}: dim={len(B)}")
        print(f"  k={k}  w1=w2=0   " + "   ".join(line))

    print("\n=== dim of invariant operators, sigma=(-1)^k, vs corrected criteria ===")
    print("    (the text's mu_i vs the module weight w_i: mu_i = -w_i)")
    rng = list(range(-3, 6)) + [F(1, 2), F(5, 3), F(-7, 2)]
    bad = 0
    for k in range(0, 8):
        sigma = 1 if k % 2 == 0 else -1
        for w1, w2 in product(rng, repeat=2):
            keys, B = invariant_ops(k, w1, w2, sigma)
            m1, m2 = -w1, -w2
            m1 = int(m1) if isinstance(m1, int) or m1.denominator == 1 else m1
            m2 = int(m2) if isinstance(m2, int) or m2.denominator == 1 else m2
            p = pred_even(k // 2, m1, m2) if k % 2 == 0 else pred_odd(k // 2, m1, m2)
            if k == 0:
                p = 1
            if len(B) != p:
                bad += 1
                if bad < 20:
                    print(f"  MISMATCH k={k} w=({w1},{w2}) mu=({m1},{m2}): "
                          f"dim={len(B)}, predicted {p}")
    print(f"  mismatches: {bad}")

    print("\n=== the invariant operators themselves (generic weights) ===")
    for k in range(0, 5):
        sigma = 1 if k % 2 == 0 else -1
        w1, w2 = F(1, 3), F(1, 5)
        keys, B = invariant_ops(k, w1, w2, sigma)
        print(f"  k={k}  w=({w1},{w2})  dim={len(B)}")
        for v in B:
            print("      " + "  ".join(f"{c}*{key}"
                                       for c, key in zip(v, keys) if c))
