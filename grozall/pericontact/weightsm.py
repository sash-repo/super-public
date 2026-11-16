"""How many weights does a rank-1 module over m(n) carry?  Derived, not guessed.

A rank-1-fiber module structure on the functions is  L_f F = M_f(F) + rho(f) F.
Restricted to f, g of non-positive weighted degree the cocycle condition says exactly

    rho vanishes on m_{<0}  (there are no functions of negative weighted degree)
    and rho|_{m_0} is a character, i.e. kills [m_0, m_0];

the conditions with an argument in m_{<0} are then automatic, m_{<0} acting by
constant-coefficient fields and rho(m_0) being constants.  So the number of weights is
at most dim(m_0/[m_0,m_0]), and exactly that whenever every character extends.

Computed here: dim m_0, dim [m_0,m_0], the quotient, and the two characters
f_tau and Delta^m -- each of which DOES extend (test_mperi.py checks that F_{a,b} is
a module for every (a,b)), so the bound is attained.

For comparison the engine also prints the same quotient for the contact algebra's
zero part, which is one-dimensional: c(sp(2n)) has a simple derived algebra.
"""

import os, sys
from fractions import Fraction as Fr

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import rank  # noqa: E402

from mperi import (  # noqa: E402
    add, Delta_m, d_tau, m_piece, mbracket, mono, monomials, parity, scal,
)


def coords(n, f, basis_index):
    v = [Fr(0)] * len(basis_index)
    for k, c in f.items():
        v[basis_index[k]] = c
    return v


def quotient(n):
    """dim m_0, dim [m_0,m_0] and the quotient, as generating functions."""
    g0 = m_piece(n, 0)
    keys = sorted({k for f in g0 for k in f}, key=repr)
    idx = {k: i for i, k in enumerate(keys)}
    brs = []
    for f in g0:
        for g in g0:
            b = mbracket(n, f, g)
            if b:
                brs.append(coords(n, b, idx))
    d0 = len(g0)
    d1 = rank(brs) if brs else 0
    return d0, d1, d0 - d1


def characters(n):
    """The two named characters of m_0, as vectors over the basis of m_0."""
    g0 = m_piece(n, 0)
    chi1, chi2 = [], []
    for f in g0:
        sp = (-1) ** parity(f)
        t = d_tau(f)
        dl = Delta_m(n, f)
        # both must be CONSTANTS on m_0 (weighted degree 0)
        c1 = t.get(((0,) * n, 0), Fr(0)) if len(t) <= 1 else None
        c2 = dl.get(((0,) * n, 0), Fr(0)) if len(dl) <= 1 else None
        assert c1 is not None and c2 is not None, "not a constant on m_0"
        chi1.append(sp * c1)
        chi2.append(sp * c2)
    return chi1, chi2


def main():
    print("=" * 74)
    print("The number of weights of a rank-1 module over m(n)")
    print("=" * 74)
    for n in (1, 2, 3, 4):
        d0, d1, dq = quotient(n)
        chi1, chi2 = characters(n)
        r = rank([chi1, chi2])
        # do the two characters kill the derived algebra?
        g0 = m_piece(n, 0)
        keys = sorted({k for f in g0 for k in f}, key=repr)
        idx = {k: i for i, k in enumerate(keys)}
        ok = True
        for f in g0:
            for g in g0:
                b = mbracket(n, f, g)
                if not b:
                    continue
                # expand b over the basis g0 (m_0 is spanned by monomials)
                for chi in (chi1, chi2):
                    val = Fr(0)
                    for j, bas in enumerate(g0):
                        k = list(bas)[0]
                        val += chi[j] * b.get(k, Fr(0))
                    if val:
                        ok = False
        print("n = %d:  dim m_0 = %2d,  dim [m_0,m_0] = %2d,  "
              "dim m_0/[m_0,m_0] = %d;  rank of the two named characters = %d;  "
              "they kill [m_0,m_0]: %s" % (n, d0, d1, dq, r, ok))
    print()
    print("So a density over m(n) carries TWO weights (a,b) -- the exponents of")
    print("alpha_0 and of Vol -- against ONE over the contact algebra k(2n+1).")


if __name__ == "__main__":
    main()
