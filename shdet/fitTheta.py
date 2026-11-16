"""
Solve for the coefficients of the table (70) of the text.

We keep the shape printed in the text -- which named pairs are non-zero and
their dependence on the index -- and leave every scalar free:

  c(L_m, L_n)      = A1 * m(m^2-1) d0
  c(G^i_m, G^i_n)  = A2 * m(m+1)   d1
  c(T^ij_m,T^ij_n) = A3 * m        d0
  c(F^i_m, F^i_n)  = A4            d1
  c(S_m, S_n)      = A5 * 1/m      d0
  c(L_m, S_n)      = (A6 + A7 m)   d0
  c(G^i_m, F^i_n)  = (A8 + A9 m)   d1
  c(T^ij_m,T^kl_n) = A10 * m       d0   for (i,j,k,l) in A_4

(d0 = delta_{m+n,0}, d1 = delta_{m+n+1,0}), and impose the 2-cocycle identity
on every triple of monomials in a window.  The solution space is printed.
"""

from fractions import Fraction as Q
from itertools import product
import sys

from stringy import mono, parity
from nsTheta import (bracket, monomials, weight, name_of, ODD_NAMES,
                     even_perm, ODD)

N = 10   # number of unknowns


def zerov():
    return [Q(0)] * N


def entry(nx, ny):
    """the ansatz value of c(named_x, named_y) as a vector of coefficients of
    A1..A10; None if the ordered pair is not one of the listed ones."""
    (n1, m), (n2, n) = nx, ny
    d0 = (m + n == 0)
    d1 = (m + n + 1 == 0)
    v = zerov()
    if n1 == "L" and n2 == "L":
        if d0:
            v[0] = Q(m * (m * m - 1))
        return v
    if n1 == n2 and n1.startswith("G"):
        if d1:
            v[1] = Q(m * (m + 1))
        return v
    if n1 == n2 and n1.startswith("T"):
        if d0:
            v[2] = Q(m)
        return v
    if n1 == n2 and n1.startswith("F"):
        if d1:
            v[3] = Q(1)
        return v
    if n1 == "S" and n2 == "S":
        if d0 and m != 0:
            v[4] = Q(1, m)
        return v
    if n1 == "L" and n2 == "S":
        if d0:
            v[5], v[6] = Q(1), Q(m)
        return v
    if n1[0] == "G" and n2[0] == "F" and n1[1] == n2[1]:
        if d1:
            v[7], v[8] = Q(1), Q(m)
        return v
    if n1.startswith("T") and n2.startswith("T") and n1 != n2:
        i, j = int(n1[1]), int(n1[2])
        k, l = int(n2[1]), int(n2[2])
        if {i, j, k, l} == {1, 2, 3, 4} and d0 and even_perm((i, j, k, l)):
            v[9] = Q(m)
        return v
    return None


def c_mono(k1, k2):
    nx, ny = name_of(k1), name_of(k2)
    s = nx[2] * ny[2]
    ax, ay = (nx[0], nx[1]), (ny[0], ny[1])
    v = entry(ax, ay)
    if v is None:
        w = entry(ay, ax)
        if w is None:
            return zerov()
        sg = -((-1) ** (int(ODD_NAMES(ax)) * int(ODD_NAMES(ay))))
        v = [sg * x for x in w]
    return [s * x for x in v]


def c_fun(f, g):
    out = zerov()
    for k1, c1 in f.items():
        for k2, c2 in g.items():
            v = c_mono(k1, k2)
            for i in range(N):
                if v[i]:
                    out[i] += c1 * c2 * v[i]
    return out


def rows(keys, kappa=Q(1), drop_S0=True):
    W = {}
    for k in keys:
        W.setdefault(weight(k), []).append(k)
    seen = set()
    for kf, kg in product(keys, repeat=2):
        for kh in W.get(-(weight(kf) + weight(kg)), ()):
            if drop_S0 and any(name_of(k)[:2] == ("S", 0) for k in (kf, kg, kh)):
                continue
            f, g, h = mono(*kf), mono(*kg), mono(*kh)
            lhs = c_fun(f, bracket(g, h, kappa))
            r1 = c_fun(bracket(f, g, kappa), h)
            sgn = (-1) ** (parity(f) * parity(g))
            r2 = c_fun(g, bracket(f, h, kappa))
            row = tuple(lhs[i] - r1[i] - sgn * r2[i] for i in range(N))
            if any(row) and row not in seen:
                seen.add(row)
                yield row


def solve(rowlist):
    """returns a basis of the nullspace of the rows."""
    piv = {}
    for row in rowlist:
        row = {i: v for i, v in enumerate(row) if v}
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
    free = [j for j in range(N) if j not in piv]
    basis = []
    for j in free:
        v = [Q(0)] * N
        v[j] = Q(1)
        for p in sorted(piv, reverse=True):
            s = sum(piv[p][i] * v[i] for i in piv[p] if i != p)
            v[p] = -s
        basis.append(v)
    return free, basis


NAMES = ["A1 c(L,L)/m(m^2-1)", "A2 c(G,G)/m(m+1)", "A3 c(T,T)/m",
         "A4 c(F,F)", "A5 c(S,S)*m", "A6 c(L,S) const", "A7 c(L,S)/m",
         "A8 c(G,F) const", "A9 c(G,F)/m", "A10 c(T^ij,T^kl)/m"]


if __name__ == "__main__":
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    kappa = Q(int(sys.argv[2])) if len(sys.argv) > 2 else Q(1)
    keys = monomials(range(-B, B + 1))
    rl = list(rows(keys, kappa))
    print("window |a| <= %d, kappa = %s : %d independent equations on %d unknowns"
          % (B, kappa, len(rl), N))
    free, basis = solve(rl)
    print("solution space has dimension %d; free unknowns: %s"
          % (len(basis), [NAMES[j].split()[0] for j in free]))
    for v in basis:
        print("   " + "  ".join("%s=%s" % (NAMES[i].split()[0], v[i])
                                for i in range(N) if v[i]))
    print()
    print("The printed table is the vector")
    printed = [Q(1), Q(1), Q(-1), Q(-1), Q(1),     # alpha part (alpha=1)
               Q(1), Q(1), Q(2), Q(2), Q(-1)]      # beta=gamma=1
    print("   " + "  ".join("%s=%s" % (NAMES[i].split()[0], printed[i])
                            for i in range(N)))
