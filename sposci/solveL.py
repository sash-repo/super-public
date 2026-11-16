"""
Determine the space of weight-0 2-cocycles of k^L(1|4) (xi-eta realisation)
from scratch, by exact linear algebra, and compare with the table
equation 273 of the text,.

Unknowns: c(x,y) for all pairs of monomials K_{t^a M} of total weight 0 with
|a| <= A; equations: the 2-cocycle identity for all triples with |a| <= B,
A = 2B+1 so that no bracket leaves the unknown window.
"""

from fractions import Fraction as Q
from itertools import product
import sys

from stringy import mono, add, smul, parity
from kL4 import (bracket, monomials, weight, by_weight, name_of, ODD_NAMES,
                 par_of_name, c_mono, table_raw)

def par(key):
    return len(key[1]) % 2

# ------------------------------------------------------------- unknowns

class Unknowns:
    def __init__(self):
        self.idx = {}
        self.keys = []

    def get(self, k1, k2):
        """returns (index, sign) with c(k1,k2) = sign * u[index]; None if the
        pair is forced to vanish."""
        if weight(k1) + weight(k2) != 0:
            return None
        if k1 <= k2:
            key, sgn = (k1, k2), 1
        else:
            key, sgn = (k2, k1), -((-1) ** (par(k1) * par(k2)))
        if k1 == k2 and par(k1) == 0:
            return None                      # c(x,x) = -c(x,x)
        if key not in self.idx:
            self.idx[key] = len(self.keys)
            self.keys.append(key)
        return self.idx[key], sgn


def lin_c(U, f, g):
    """c(f,g) as a sparse dict {unknown index: coefficient}."""
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
            if key in seen:
                continue
            seen.add(key)
            yield row


def reduce_rows(rows):
    """sparse Gaussian elimination; returns {pivot: normalised row}."""
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
                c = row[p]
                piv[p] = {i: Q(v) / c for i, v in row.items()}
                break
    return piv


def nullspace(piv, n):
    free = [j for j in range(n) if j not in piv]
    basis = []
    for j in free:
        v = [Q(0)] * n
        v[j] = Q(1)
        for p in sorted(piv, reverse=True):
            row = piv[p]
            s = sum(row[i] * v[i] for i in row if i != p)
            v[p] = -s
        basis.append(v)
    return free, basis


# ---------------------------------------------------------------- run

def run(B=2, kappa=Q(1), drop_S0=True):
    A = 2 * B + 1
    keysA = monomials(range(-A, A + 1))
    keysB = monomials(range(-B, B + 1))
    if drop_S0:
        S0 = (-1, (1, 2, 3, 4))
        keysA = [k for k in keysA if k != S0]
        keysB = [k for k in keysB if k != S0]
    U = Unknowns()
    # pre-register every unknown inside the B-window so the projection below
    # is over a fixed set
    WA = by_weight(keysA)
    for k1 in keysB:
        for k2 in WA.get(-weight(k1), ()):
            U.get(k1, k2)
    nB = len(U.keys)
    rows = list(eqs(U, keysB, kappa))
    n = len(U.keys)
    print("B=%d  unknowns=%d (of which %d inside the B-window) equations=%d"
          % (B, n, nB, len(rows)))
    piv = reduce_rows(rows)
    free, basis = nullspace(piv, n)
    # project on the B-window unknowns
    proj = [tuple(v[:nB]) for v in basis]
    piv2 = reduce_rows([{i: x for i, x in enumerate(v) if x} for v in proj])
    dim = len(piv2)
    print("dim of the space of weight-0 cocycles, restricted to the B-window: %d" % dim)
    return U, nB, proj


def named(U, vec, nB):
    out = {}
    for i in range(nB):
        if vec[i]:
            k1, k2 = U.keys[i]
            out[(name_of(k1), name_of(k2))] = vec[i]
    return out


if __name__ == "__main__":
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    U, nB, proj = run(B)
