"""
The space of weight-0 2-cocycles of k^L'(1|4) in the theta-realisation, from
scratch: no shape assumed.  Unknowns are c(x,y) for all pairs of monomials
K_{t^a M} of total weight 0; equations are the 2-cocycle identity for all
triples inside a window.  Coboundaries are computed separately, so the output
is dim H^2 together with an explicit basis read off on the named basis vectors
of (monBas).
"""

from fractions import Fraction as Q
from itertools import product
import sys

from stringy import mono, parity
from nsTheta import bracket, monomials, weight, name_of, ODD, ODD_NAMES

S0KEY = (-1, (1, 2, 3, 4))


def par(key):
    return len(key[1]) % 2


def by_weight(keys):
    d = {}
    for k in keys:
        d.setdefault(weight(k), []).append(k)
    return d


class Unknowns:
    def __init__(self):
        self.idx = {}
        self.keys = []

    def get(self, k1, k2):
        if weight(k1) + weight(k2) != 0:
            return None
        if k1 == k2 and par(k1) == 0:
            return None                       # c(x,x) = -c(x,x) for x even
        if k1 <= k2:
            key, sgn = (k1, k2), 1
        else:
            key, sgn = (k2, k1), -((-1) ** (par(k1) * par(k2)))
        if key not in self.idx:
            self.idx[key] = len(self.keys)
            self.keys.append(key)
        return self.idx[key], sgn


def lin_c(U, f, g):
    out = {}
    for k1, c1 in f.items():
        for k2, c2 in g.items():
            r = U.get(k1, k2)
            if r is None:
                continue
            i, s = r
            out[i] = out.get(i, 0) + s * c1 * c2
    return {i: v for i, v in out.items() if v}


def eqs(U, keysB, kappa=Q(1)):
    W = by_weight(keysB)
    seen = set()
    for kf, kg in product(keysB, repeat=2):
        for kh in W.get(-(weight(kf) + weight(kg)), ()):
            f, g, h = mono(*kf), mono(*kg), mono(*kh)
            row = lin_c(U, f, bracket(g, h, kappa))
            for i, v in lin_c(U, bracket(f, g, kappa), h).items():
                row[i] = row.get(i, 0) - v
            sgn = (-1) ** (parity(f) * parity(g))
            for i, v in lin_c(U, g, bracket(f, h, kappa)).items():
                row[i] = row.get(i, 0) - sgn * v
            row = {i: v for i, v in row.items() if v}
            if not row:
                continue
            key = tuple(sorted(row.items()))
            if key not in seen:
                seen.add(key)
                yield row


def coboundaries(U, keysB, nB, kappa=Q(1)):
    """d(phi)(x,y) = phi([x,y]) for phi a weight-0 linear functional, i.e. phi
    supported on the monomials of weight 0."""
    zero_wt = [k for k in keysB if weight(k) == 0]
    out = []
    for kz in zero_wt:
        vec = [Q(0)] * nB
        for i in range(nB):
            k1, k2 = U.keys[i]
            b = bracket(mono(*k1), mono(*k2), kappa)
            if kz in b:
                vec[i] = Q(b[kz])
        if any(vec):
            out.append(vec)
    return out


def reduce_rows(rows):
    piv = {}
    for row in rows:
        row = dict(row)
        while row:
            p = min(row)
            if p in piv:
                c = row[p]
                for i, v in piv[p].items():
                    row[i] = row.get(i, 0) - c * v
                row = {i: v for i, v in row.items() if v}
            else:
                piv[p] = {i: Q(v) / row[p] for i, v in row.items()}
                break
    return piv


def nullspace(piv, n):
    free = [j for j in range(n) if j not in piv]
    basis = []
    for j in free:
        v = [Q(0)] * n
        v[j] = Q(1)
        for p in sorted(piv, reverse=True):
            s = sum(piv[p][i] * v[i] for i in piv[p] if i != p)
            v[p] = -s
        basis.append(v)
    return basis


def rank(vecs, n):
    return len(reduce_rows([{i: x for i, x in enumerate(v) if x} for v in vecs]))


def run(B=2, kappa=Q(1), drop_S0=True):
    A = 2 * B + 1
    keysA = monomials(range(-A, A + 1))
    keysB = monomials(range(-B, B + 1))
    if drop_S0:
        keysA = [k for k in keysA if k != S0KEY]
        keysB = [k for k in keysB if k != S0KEY]
    U = Unknowns()
    WA = by_weight(keysA)
    for k1 in keysB:
        for k2 in WA.get(-weight(k1), ()):
            U.get(k1, k2)
    nB = len(U.keys)
    rows = list(eqs(U, keysB, kappa))
    n = len(U.keys)
    piv = reduce_rows(rows)
    basis = nullspace(piv, n)
    proj = [v[:nB] for v in basis]
    dimZ = rank(proj, nB)
    cob = coboundaries(U, keysB, nB, kappa)
    dimB = rank(cob, nB)
    dimBinZ = rank(list(proj) + cob, nB)
    print("B=%d kappa=%s: unknowns %d (%d in window), equations %d"
          % (B, kappa, n, nB, len(rows)))
    print("  dim Z^2 (weight 0, restricted to the window) = %d" % dimZ)
    print("  dim B^2 = %d, and Z + B has dim %d  =>  B is inside Z: %s"
          % (dimB, dimBinZ, dimBinZ == dimZ))
    print("  dim H^2 (weight 0) = %d" % (dimZ - dimB))
    return U, nB, proj, cob


def show_named(U, vec, nB, only=None):
    out = {}
    for i in range(nB):
        if vec[i]:
            k1, k2 = U.keys[i]
            n1, n2 = name_of(k1), name_of(k2)
            out[((n1[0], n1[1]), (n2[0], n2[1]))] = vec[i] * n1[2] * n2[2]
    return out


if __name__ == "__main__":
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    kappa = Q(int(sys.argv[2])) if len(sys.argv) > 2 else Q(1)
    run(B, kappa)
