"""
Space of weight-0 2-cocycles of k^M(1|4), computed from scratch, and the
comparison with equation 275.
"""

from fractions import Fraction as Q
from itertools import product

from stringy import mono, parity
import kM4 as M
from solveL import reduce_rows, nullspace

B = 3
A = 2 * B + 1
keysA = M.monomials(range(-A, A + 1))
keysB = M.monomials(range(-B, B + 1))

def par(key):
    return len(key[1]) % 2

class U:
    idx = {}
    keys = []

    @classmethod
    def get(cls, k1, k2):
        if M.weight(k1) + M.weight(k2) != 0:
            return None
        if k1 == k2 and par(k1) == 0:
            return None
        if k1 <= k2:
            key, sgn = (k1, k2), 1
        else:
            key, sgn = (k2, k1), -((-1) ** (par(k1) * par(k2)))
        if key not in cls.idx:
            cls.idx[key] = len(cls.keys)
            cls.keys.append(key)
        return cls.idx[key], sgn


WA = M.by_weight(keysA)
for k1 in keysB:
    for k2 in WA.get(-M.weight(k1), ()):
        U.get(k1, k2)
nB = len(U.keys)

def lin_c(f, g):
    out = {}
    for k1, c1 in f.items():
        for k2, c2 in g.items():
            r = U.get(k1, k2)
            if r is None:
                continue
            i, s = r
            out[i] = out.get(i, 0) + s * c1 * c2
    return {i: v for i, v in out.items() if v}

rows, seen = [], set()
W = M.by_weight(keysB)
for kf, kg in product(keysB, repeat=2):
    for kh in W.get(-(M.weight(kf) + M.weight(kg)), ()):
        f, g, h = mono(*kf), mono(*kg), mono(*kh)
        row = lin_c(f, M.bracket(g, h))
        for i, v in lin_c(M.bracket(f, g), h).items():
            row[i] = row.get(i, 0) - v
        sgn = (-1) ** (parity(f) * parity(g))
        for i, v in lin_c(g, M.bracket(f, h)).items():
            row[i] = row.get(i, 0) - sgn * v
        row = {i: v for i, v in row.items() if v}
        if not row:
            continue
        key = tuple(sorted(row.items()))
        if key not in seen:
            seen.add(key)
            rows.append(row)

print("B=%d  unknowns=%d (%d in the window)  equations=%d" % (B, len(U.keys), nB, len(rows)))
piv = reduce_rows(rows)
free, basis = nullspace(piv, len(U.keys))
sols = [tuple(v[:nB]) for v in basis]
piv2 = reduce_rows([{i: x for i, x in enumerate(v) if x} for v in sols])
print("dim of weight-0 cocycles in the window =", len(piv2))

# coboundaries: even weight-0 elements
w0 = [k for k in keysB if M.weight(k) == 0 and par(k) == 0]
print("even weight-0 elements:", [M.name_of(k)[:2] for k in w0])
cob = []
for phi in w0:
    v = [Q(0)] * nB
    for i, (k1, k2) in enumerate(U.keys):
        v[i] = Q(M.bracket(mono(*k1), mono(*k2)).get(phi, 0))
    cob.append(v)
pivC = reduce_rows([{i: x for i, x in enumerate(v) if x} for v in cob])
print("dim of coboundaries =", len(pivC))
print("=> dim H^2 (weight 0) =", len(piv2) - len(pivC))

# is equation 275 in the space, and is it nonzero modulo coboundaries?
tv = [M.c_mono(k1, k2) for (k1, k2) in U.keys]
def reduce_mod(pivots, vec):
    row = {i: x for i, x in enumerate(vec) if x}
    while row:
        p = min(row)
        if p in pivots:
            c = row[p]
            for i, val in pivots[p].items():
                row[i] = row.get(i, 0) - c * val
            row = {i: x for i, x in row.items() if x}
        else:
            break
    return row
print("equation 275 is a cocycle:", not reduce_mod(piv2, tv))
print("equation 275 is a coboundary:", not reduce_mod(pivC, tv))

# --- inspect the classes modulo coboundaries
BAS = []
for p in sorted(piv2):
    v = [Q(0)] * nB
    for i, x in piv2[p].items():
        v[i] = x
    BAS.append(v)

def types(vec):
    tp = {}
    for i in range(nB):
        if vec[i]:
            n1, n2 = M.name_of(U.keys[i][0]), M.name_of(U.keys[i][1])
            tp.setdefault((n1[0], n2[0]), []).append((n1[1], n2[1], vec[i]))
    return tp

# reduce the basis modulo coboundaries and modulo equation 275
piv3 = dict(pivC)
extra = []
for v in BAS:
    r = reduce_mod(piv3, v)
    if r:
        piv3[min(r)] = {i: x / r[min(r)] for i, x in r.items()}
        extra.append(v)
print("\nclasses beyond the coboundaries: %d" % len(extra))
for j, v in enumerate(extra):
    print(" class %d entry types: %s" % (j, sorted(types(v))))
