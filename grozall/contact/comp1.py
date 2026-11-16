"""Are the exceptional operators over k(1|1) compositions?

The linear invariant operators are the identity (every weight) and D: F_0 -> F_{1/2}
(unarysuper.py), so the only compositions available are

    D o B    (when B lands in F_0, i.e. on the line lam + mu + 1 = 0 for W = 2),
    B(D., .) and B(., D.)   (when an argument has weight 0).

Each exceptional operator found in runsuper.py is tested against the span of these.
"""

import os, sys
from fractions import Fraction as Fr

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import rank                                 # noqa: E402

from super import add, scal, mul, parity, mono, D          # noqa: E402
from gensuper import pencil, apply_term, show_op, rand_fun  # noqa: E402
from affine import dim_at, ops_at                          # noqa: E402

n = 1
Fs = [rand_fun(n, p, 300 + i, maxdeg=4) for p in (0, 1) for i in range(3)]
Gs = [rand_fun(n, p, 400 + i, maxdeg=4) for p in (0, 1) for i in range(3)]


def as_vector(op):
    """The values of a bilinear operator on the fixed test pairs, as one long vector."""
    keys, vals = [], []
    for F in Fs:
        for G in Gs:
            v = op(F, G)
            for k in sorted(v):
                keys.append((F is F, k))
                vals.append((F, G, k, v[k]))
    return vals


def to_row(op, index):
    row = [Fr(0)] * len(index)
    for iF, F in enumerate(Fs):
        for iG, G in enumerate(Gs):
            v = op(F, G)
            for k, c in v.items():
                row[index[(iF, iG, k)]] = c
    return row


def build_index(ops):
    keys = set()
    for op in ops:
        for iF, F in enumerate(Fs):
            for iG, G in enumerate(Gs):
                for k in op(F, G):
                    keys.add((iF, iG, k))
    return {k: i for i, k in enumerate(sorted(keys, key=str))}


def op_from(terms, vec):
    def B(F, G):
        out = {}
        for c, t in zip(vec, terms):
            if c:
                out = add(out, scal(c, apply_term(t, F, G)))
        return out
    return B


def get_ops(W, lam, mu):
    terms, V0, N0, N1, N2 = pencil(n, W)
    d = dim_at(V0, N0, N1, N2, lam, mu)
    return [op_from(terms, v) for v in ops_at(terms, V0, N0, N1, N2, lam, mu)], terms, \
           [v for v in ops_at(terms, V0, N0, N1, N2, lam, mu)]


def in_span(target, cands):
    idx = build_index([target] + cands)
    rows = [to_row(c, idx) for c in cands]
    r0 = rank(rows) if rows else 0
    r1 = rank(rows + [to_row(target, idx)])
    return r0 == r1, r0, r1


print("=" * 74)
print("W = 3 on the line lam + mu + 1 = 0:  is it  D o B_2 ?")
for lam in (Fr(1), Fr(-2), Fr(1, 2), Fr(-3)):
    mu = -1 - lam
    ops3, t3, v3 = get_ops(3, lam, mu)
    if not ops3:
        print("   lam=%-5s mu=%-5s : no operator" % (lam, mu))
        continue
    B2, t2, v2 = get_ops(2, lam, mu)          # lands in F_{lam+mu+1} = F_0
    comp = [lambda F, G, b=b: D(1, b(F, G)) for b in B2]
    ok, r0, r1 = in_span(ops3[0], comp)
    print("   lam=%-5s mu=%-5s : dim W3 = %d, D o B_2 spans it: %s   (%d -> %d)"
          % (lam, mu, len(ops3), ok, r0, r1))

print("=" * 74)
print("W = 4 at the three isolated points:  composition of D and the families?")
for (lam, mu) in ((Fr(-3, 2), Fr(0)), (Fr(0), Fr(-3, 2)), (Fr(0), Fr(0))):
    ops4, t4, v4 = get_ops(4, lam, mu)
    if not ops4:
        print("   (%s,%s): none" % (lam, mu))
        continue
    cands = []
    # D o B_3, when the W=3 operator at (lam,mu) lands in F_0
    if lam + mu + Fr(3, 2) == 0:
        B3s, _, _ = get_ops(3, lam, mu)
        cands += [lambda F, G, b=b: D(1, b(F, G)) for b in B3s]
    # B(F, DG) and B(DF, G) followed, where possible, by D
    if mu == 0:
        B3, _, _ = get_ops(3, lam, Fr(1, 2))
        cands += [lambda F, G, b=b: b(F, D(1, G)) for b in B3]
        B2, _, _ = get_ops(2, lam, Fr(1, 2))
        if lam + Fr(1, 2) + 1 == 0:
            cands += [lambda F, G, b=b: D(1, b(F, D(1, G))) for b in B2]
        else:
            cands += [lambda F, G, b=b: D(1, b(F, D(1, G))) for b in B2]
    if lam == 0:
        B3, _, _ = get_ops(3, Fr(1, 2), mu)
        cands += [lambda F, G, b=b: b(D(1, F), G) for b in B3]
        B2, _, _ = get_ops(2, Fr(1, 2), mu)
        cands += [lambda F, G, b=b: D(1, b(D(1, F), G)) for b in B2]
    if lam == 0 and mu == 0:
        B2, _, _ = get_ops(2, Fr(1, 2), Fr(1, 2))
        cands += [lambda F, G, b=b: b(D(1, F), D(1, G)) for b in B2]
        B1, _, _ = get_ops(1, Fr(1, 2), Fr(1, 2))
        cands += [lambda F, G, b=b: D(1, b(D(1, F), D(1, G))) for b in B1]
    if not cands:
        print("   (%s,%s): dim = %d, no candidate composition" % (lam, mu, len(ops4)))
        continue
    ok, r0, r1 = in_span(ops4[0], cands)
    print("   (%s,%s): dim = %d, %d candidate compositions span it: %s   (%d -> %d)"
          % (lam, mu, len(ops4), len(cands), ok, r0, r1))

print("=" * 74)
print("W = 1 at (0,0), where the dimension jumps from 1 to 2:")
ops1, t1, v1 = get_ops(1, Fr(0), Fr(0))
print("   dim = %d" % len(ops1))
# written through apply_term, so that the Koszul sign is the same on both sides
mults = [lambda F, G: apply_term(((1,), 0, (), 0), F, G),
         lambda F, G: apply_term(((), 0, (1,), 0), F, G)]
ok, r0, r1 = in_span(ops1[0], mults)
print("   both are  (D x id) and (id x D) followed by the multiplication: %s" % ok)
