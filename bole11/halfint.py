#!/usr/bin/env python3
"""Why the odd brackets are indexed by {1/2, 3/2, 5/2, ...}.

The text sets  F_mu := F alpha_1^{mu/2}  (Subsec. 2.1), so the weight mu is
counted in units of the SQUARE ROOT of the contact form.  The bracket

    [[ , ]]_k :  F_{w1} (x) F_{w2}  -->  F_{w1+w2+k}

therefore raises the exponent of alpha_1 itself by k/2: by an integer when
k = 2n, and by n + 1/2 when k = 2n+1.  The half-integers so obtained are
exactly {1/2, 3/2, 5/2, ...}.

Two things are checked here, both from grc_ops.py (exact rational arithmetic):

  (1) the shift is EXACTLY k, i.e. the target weight w1+w2+k is the only one
      for which a non-zero osp(1|2)-invariant operator of order k exists.
      (Without this the "k/2" would be an assertion about a normalization,
      not about the operator.)

  (2) the bracket [[ , ]]_k is an ODD operator iff k is odd -- so "odd bracket"
      and "half-integer shift of the alpha_1-exponent" are one and the same
      condition, which is why the second line of eq. (A11) is a product on
      Pi(F_.) and not on F_. .

    python3 halfint.py
"""
from fractions import Fraction as F

import grc_ops as G


def ops_for_target(k, w1, w2, w3, sigma=1, A=None):
    """basis of the order-k invariant operators F_w1 (x) F_w2 -> F_w3.

    grc_ops.invariant_ops hard-wires w3 = w1+w2+k; here w3 is free, so that
    the shift can be measured rather than assumed."""
    keys = G.op_keys(k)
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
                    tot = dict(G.LX(G.apply_key(key, f, g), X, w3))
                    G.add(tot, G.apply_key(key, G.LX(f, X, w1), g), -sigma)
                    G.add(tot, G.apply_key(key, f, G.LX(g, X, w2)),
                          -sigma * (-1) ** pf)
                    for mono, c in tot.items():
                        res.setdefault(mono, [G.Z] * len(keys))[n] = c
                rows.extend(res.values())
    return keys, G.nullspace(rows, len(keys))


def parity_of(op, keys, k):
    """parity of the bilinear operator: the parity it adds to p(f)+p(g).

    Returns 0, 1, or None if the operator is not homogeneous.  The test
    monomials must carry enough powers of t for every word of the operator to
    survive -- with f = t the order-7 words die and the surviving ones answer
    for a shorter operator."""
    par = set()
    for pf in (0, 1):
        for pg in (0, 1):
            f = {(k, pf): F(1)}          # t^k . theta^{pf}
            g = {(k, pg): F(1)}
            out = G.apply_op(op, keys, f, g)
            for (a, e) in out:
                par.add((e - pf - pg) % 2)
    if len(par) != 1:
        return None
    return par.pop()


def main():
    checks = 0
    fails = []

    # generic weights: nothing special happens there, so the bracket is unique
    # up to a factor and the dimension count of Theorem 2.2.4 gives 1.
    W1, W2 = F(3, 7), F(5, 11)

    print("(1) the weight shift of [[ , ]]_k is exactly k")
    print("    (dim of the space of invariant operators F_{w1} (x) F_{w2} -> F_{w3},")
    print("     w1 = %s, w2 = %s, w3 = w1+w2+s)" % (W1, W2))
    print()
    print("      k |  " + "  ".join("s=%-2d" % s for s in range(0, 7)))
    print("    ----+-" + "-".join(["-----"] * 7))
    for k in range(0, 6):
        row = []
        for s in range(0, 7):
            keys, basis = ops_for_target(k, W1, W2, W1 + W2 + s)
            d = len(basis)
            row.append(d)
            checks += 1
            # the shift must be k and nothing else
            if s == k:
                if d == 0:
                    fails.append("k=%d: no operator at the predicted shift" % k)
            else:
                if d != 0:
                    fails.append("k=%d: an operator at shift %d" % (k, s))
        print("      %d |  " % k + "  ".join("%-4d" % d for d in row))
    print()

    print("(2) [[ , ]]_k is odd iff k is odd")
    for k in range(0, 8):
        keys, basis = ops_for_target(k, W1, W2, W1 + W2 + k)
        assert basis, "k=%d: empty" % k
        for op in basis:
            p = parity_of(op, keys, max(k, 1))
            checks += 1
            if p != k % 2:
                fails.append("k=%d: parity %s, expected %d" % (k, p, k % 2))
        shift = F(k, 2)
        print("      k=%d: dim %d, parity %d, alpha_1-exponent raised by %s%s"
              % (k, len(basis), k % 2, shift,
                 "   <-- half-integer" if k % 2 else ""))
    print()

    # (3) the set of half-integer shifts is the printed one
    odd_shifts = sorted({F(2 * n + 1, 2) for n in range(0, 6)})
    checks += 1
    if odd_shifts != [F(1, 2), F(3, 2), F(5, 2), F(7, 2), F(9, 2), F(11, 2)]:
        fails.append("the set of odd shifts is not {1/2, 3/2, 5/2, ...}")
    print("(3) the shifts of the odd brackets, k = 2n+1, n = 0, 1, 2, ... :")
    print("      " + ", ".join(str(x) for x in odd_shifts) + ", ...   "
          "= N - 1/2 as printed")
    print()

    print("%d checks, %d failures" % (checks, len(fails)))
    for f in fails:
        print("  FAIL " + f)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
